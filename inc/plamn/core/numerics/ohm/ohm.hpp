#ifndef PLAMN_OHM_HPP
#define PLAMN_OHM_HPP

#include "damn/core/geometry/point.hpp"
#include "damn/core/def/damn_config.hpp"

#include "plamn/core/data/grid/layout_defs.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"
#include "plamn/core/data/vecfield/vecfield_component.hpp"
#include "plamn/core/numerics/differential_operators.hpp"
#include "plamn/initializer/data_provider.hpp"

#include "dict.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

namespace plamn::core
{
enum class HyperMode : std::uint8_t { constant = 0, spatial, count };

struct OhmInfo
{
    double const eta;
    double const nu;
    HyperMode const hyper_mode = HyperMode::spatial;

    bool isResistive() const { return eta > 0.0; }
    bool isHyperResistive() const { return nu > 0.0; }

    OhmInfo static FROM(plamn::initializer::Dict const& dict)
    {
        return {dict["resistivity"].template to<double>(),
                dict["hyper_resistivity"].template to<double>(),
                cppdict::get_value(dict, "hyper_mode", HyperMode::constant)};
    }
};

template<typename Layout>
class Ohm : public OhmInfo
{
    using Super                     = OhmInfo;
    constexpr static auto dimension = Layout::dimension;
    using Scalar                    = std::decay_t<decltype(Layout::options)>::FieldOptions::Scalar;

public:
    explicit Ohm(OhmInfo const& info, Layout const& layout)
        : Super{info}
        , layout_{layout}
    {
    }

    // E component c at one point: gradPe_c is that component of grad(layout, Pe), layout the
    //  one of the fields at ijk (patch or tile). The resistive / hyper-resistive runtime flags
    //  and the hyper mode come in as compile-time tags: no branch on them per point (the loops
    //  vectorise), the projection / laplacian are skipped when eta or nu is zero. Where and how
    //  it is evaluated (boxes, tiles, GPU) is up to the caller, see
    //  core/numerics/ohm/ohm_transformer.hpp
    //  n: the ion charge density, F: the total ion charge flux sum_s q_s F_s (moment nodes)
    template<auto c, bool isResistive, bool isHyperResistive, HyperMode hyper>
    void operator()(auto const& ijk, auto const& n, auto const& F, auto const& gradPe_c,
                    auto const& B, auto const& J, auto& E) const DAMN_ALL
    {
        static_assert(Components::check<c>());

        auto E_ = ideal_<c>(n, F, B, J, ijk) + pressure_<c>(n, gradPe_c, ijk);

        if constexpr (isResistive)
            E_ += resistive_<c>(J, ijk);
        if constexpr (isHyperResistive)
            E_ += hyperresistive_<c, hyper>(J, B, n, ijk);

        E(c)(ijk) = E_;
    }

private:
    Layout layout_; // GPU copy needs object not pointer!

    // The ideal term -Ve x B = -G x B / n, G = n Ve = F - J the electron flux, written so that it
    //  does no work on the electrons: sum_i G_i E_i = 0 over a periodic grid. G is formed on the E
    //  nodes, w = B / n on the B nodes, each product averaged to the B node and back:
    //   E_x = -P_{Bz->Ex}(wz P_{Ey->Bz} Gy) + P_{By->Ex}(wy P_{Ez->By} Gz), cyclic
    //  The 2 point averages P_{a->b} and P_{b->a} are transposes of each other, so G -> E is
    //  antisymmetric. Averaging Ve and B to the E node and multiplying there is not: particle
    //  noise then feeds a steady energy gain (grid scale B grows). Reads n and F one node beyond
    //  the box E is computed on, J one edge beyond.
    template<auto component, typename VecField>
    auto ideal_(auto const& n, VecField const& F, VecField const& B, VecField const& J,
                damn::core::Point<std::size_t, dimension> const index) const DAMN_ALL
    {
        using Point_t = damn::core::Point<std::size_t, dimension>;

        // G component c on its E node
        auto const G = [&]<auto c, auto Fq, auto Eq>() {
            return [&](Point_t const& ijk) { return project<Fq, Eq>(F(c), ijk) - J(c)(ijk); };
        };
        auto const Gx = G.template operator()<Component::X, Scalar::Vx, Scalar::Ex>();
        auto const Gy = G.template operator()<Component::Y, Scalar::Vy, Scalar::Ey>();
        auto const Gz = G.template operator()<Component::Z, Scalar::Vz, Scalar::Ez>();

        // B_c / n on the B_c node times the G component Gq (on Eq nodes) averaged to it
        auto const wG = [&]<auto c, auto Bq, auto Eq>(auto const& Gq) {
            return [&, Gq](Point_t const& ijk) {
                return B(c)(ijk) / project<Scalar::rho, Bq>(n, ijk) * project<Eq, Bq>(Gq, ijk);
            };
        };

        if constexpr (component == Component::X)
            return -project<Scalar::Bz, Scalar::Ex>(
                       wG.template operator()<Component::Z, Scalar::Bz, Scalar::Ey>(Gy), index)
                   + project<Scalar::By, Scalar::Ex>(
                       wG.template operator()<Component::Y, Scalar::By, Scalar::Ez>(Gz), index);

        if constexpr (component == Component::Y)
            return project<Scalar::Bz, Scalar::Ey>(
                       wG.template operator()<Component::Z, Scalar::Bz, Scalar::Ex>(Gx), index)
                   - project<Scalar::Bx, Scalar::Ey>(
                       wG.template operator()<Component::X, Scalar::Bx, Scalar::Ez>(Gz), index);

        if constexpr (component == Component::Z)
            return -project<Scalar::By, Scalar::Ez>(
                       wG.template operator()<Component::Y, Scalar::By, Scalar::Ex>(Gx), index)
                   + project<Scalar::Bx, Scalar::Ez>(
                       wG.template operator()<Component::X, Scalar::Bx, Scalar::Ey>(Gy), index);
    }


    template<auto component, typename Field>
    auto pressure_(Field const& n, auto const& gradPe,
                   damn::core::Point<std::size_t, Field::dimension> index) const DAMN_ALL
    {
        if constexpr (component == Component::X)
        {
            auto const nOnEx = project<Scalar::rho, Scalar::Ex>(n, index);

            return -gradPe(index) / nOnEx;
        }

        else if constexpr (component == Component::Y)
        {
            if constexpr (dimension >= 2)
            {
                auto const nOnEy = project<Scalar::rho, Scalar::Ey>(n, index);

                return -gradPe(index) / nOnEy;
            }
            else
            {
                return 0.;
            }
        }

        else if constexpr (component == Component::Z)
        {
            if constexpr (dimension >= 3)
            {
                auto const nOnEz = project<Scalar::rho, Scalar::Ez>(n, index);

                return -gradPe(index) / nOnEz;
            }
            else
            {
                return 0.;
            }
        }
    }




    template<auto component>
    auto resistive_(auto const& J, damn::core::Point<std::size_t, dimension> index) const DAMN_ALL
    {
        auto const& Jxyx = J(component);

        if constexpr (component == Component::X)
        {
            auto const jxOnEx = project<Scalar::Jx, Scalar::Ex>(Jxyx, index);
            return eta * jxOnEx;
        }

        if constexpr (component == Component::Y)
        {
            auto const jyOnEy = project<Scalar::Jy, Scalar::Ey>(Jxyx, index);
            return eta * jyOnEy;
        }

        if constexpr (component == Component::Z)
        {
            auto const jzOnEz = project<Scalar::Jz, Scalar::Ez>(Jxyx, index);
            return eta * jzOnEz;
        }
    }

    // one overload per HyperMode (a new mode: a compile error here, add its overload)
    static_assert(static_cast<std::underlying_type_t<HyperMode>>(HyperMode::count) == 2);

    template<auto component, HyperMode hyper, typename VecField>
        requires(hyper == HyperMode::constant)
    auto hyperresistive_(VecField const& J, VecField const&, auto const&,
                         damn::core::Point<std::size_t, VecField::dimension> index) const DAMN_ALL
    {
        return constant_hyperresistive_<component>(J, index);
    }

    template<auto component, HyperMode hyper, typename VecField>
        requires(hyper == HyperMode::spatial)
    auto hyperresistive_(VecField const& J, VecField const& B, auto const& n,
                         damn::core::Point<std::size_t, VecField::dimension> index) const DAMN_ALL
    {
        return spatial_hyperresistive_<component>(J, B, n, index);
    }


    template<auto component, typename VecField>
    auto constant_hyperresistive_(VecField const& J,
                                  damn::core::Point<std::size_t, VecField::dimension> index) const DAMN_ALL
    { // TODO : https://github.com/PHAREHUB/PHARE/issues/3
        return -nu * laplacian(layout_, J(component), index);
    }

    template<auto component, typename VecField>
    auto spatial_hyperresistive_(
        VecField const& J, VecField const& B, auto const& n,
        damn::core::Point<std::size_t, VecField::dimension> index) const DAMN_ALL
    {
        // 1 / 4^level, exact (no pow call per point: it blocks vectorisation)
        auto const lvlCoeff = 1. / static_cast<double>(std::uint64_t{1} << (2 * layout_.level));

        auto constexpr min_density = 0.1;
        auto computeHR             = [&]<auto dst>() {
            auto const BxOnE = project<Scalar::Bx, dst>(B(Component::X), index);
            auto const ByOnE = project<Scalar::By, dst>(B(Component::Y), index);
            auto const BzOnE = project<Scalar::Bz, dst>(B(Component::Z), index);
            auto const nOnE  = project<Scalar::rho, dst>(n, index);
            auto b          = std::sqrt(BxOnE * BxOnE + ByOnE * ByOnE + BzOnE * BzOnE);

            return -nu * (b / (nOnE + min_density) + 1) * lvlCoeff
                   * laplacian(layout_, J(component), index);
        };
        if constexpr (component == Component::X)
        {
            return computeHR.template operator()<Scalar::Ex>();
        }
        if constexpr (component == Component::Y)
        {
            return computeHR.template operator()<Scalar::Ey>();
        }
        if constexpr (component == Component::Z)
        {
            return computeHR.template operator()<Scalar::Ez>();
        }
    }
};

} // namespace plamn::core

#endif
