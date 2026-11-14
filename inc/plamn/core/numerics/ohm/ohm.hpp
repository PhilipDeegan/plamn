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
    //  come in as compile-time tags: no branch on them per point, the projection / laplacian
    //  are skipped when eta or nu is zero. Where and how it is evaluated (boxes, tiles, GPU) is
    //  up to the caller, see core/numerics/ohm/ohm_transformer.hpp
    template<auto c, bool isResistive, bool isHyperResistive>
    void operator()(auto const& ijk, auto const& n, auto const& Ve, auto const& gradPe_c,
                    auto const& B, auto const& J, auto& E) const DAMN_ALL
    {
        static_assert(Components::check<c>());

        auto E_ = ideal_<c>(Ve, B, ijk) + pressure_<c>(n, gradPe_c, ijk);

        if constexpr (isResistive)
            E_ += resistive_<c>(J, ijk);
        if constexpr (isHyperResistive)
            E_ += hyperresistive_<c>(J, B, n, ijk);

        E(c)(ijk) = E_;
    }

private:
    Layout layout_; // GPU copy needs object not pointer!

    template<auto component, typename VecField>
    auto ideal_(VecField const& Ve, VecField const& B,
                damn::core::Point<std::size_t, dimension> index) const DAMN_ALL
    {
        if constexpr (component == Component::X)
        {
            auto const& Vy = Ve(Component::Y);
            auto const& Vz = Ve(Component::Z);

            auto const& By = B(Component::Y);
            auto const& Bz = B(Component::Z);

            auto const vyOnEx = project<Scalar::Vy, Scalar::Ex>(Vy, index);
            auto const vzOnEx = project<Scalar::Vz, Scalar::Ex>(Vz, index);
            auto const byOnEx = project<Scalar::By, Scalar::Ex>(By, index);
            auto const bzOnEx = project<Scalar::Bz, Scalar::Ex>(Bz, index);

            return -vyOnEx * bzOnEx + vzOnEx * byOnEx;
        }

        if constexpr (component == Component::Y)
        {
            auto const& Vx = Ve(Component::X);
            auto const& Vz = Ve(Component::Z);
            auto const& Bx = B(Component::X);
            auto const& Bz = B(Component::Z);

            auto const vxOnEy = project<Scalar::Vx, Scalar::Ey>(Vx, index);
            auto const vzOnEy = project<Scalar::Vz, Scalar::Ey>(Vz, index);
            auto const bxOnEy = project<Scalar::Bx, Scalar::Ey>(Bx, index);
            auto const bzOnEy = project<Scalar::Bz, Scalar::Ey>(Bz, index);

            return -vzOnEy * bxOnEy + vxOnEy * bzOnEy;
        }

        if constexpr (component == Component::Z)
        {
            auto const& Vx = Ve(Component::X);
            auto const& Vy = Ve(Component::Y);
            auto const& Bx = B(Component::X);
            auto const& By = B(Component::Y);

            auto const vxOnEz = project<Scalar::Vx, Scalar::Ez>(Vx, index);
            auto const vyOnEz = project<Scalar::Vy, Scalar::Ez>(Vy, index);
            auto const bxOnEz = project<Scalar::Bx, Scalar::Ez>(Bx, index);
            auto const byOnEz = project<Scalar::By, Scalar::Ez>(By, index);

            return -vxOnEz * byOnEz + vyOnEz * bxOnEz;
        }
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

    template<auto component, typename VecField>
    auto hyperresistive_(VecField const& J, VecField const& B, auto const& n,
                         damn::core::Point<std::size_t, VecField::dimension> index) const DAMN_ALL
    {
        // if compile error, fix this function
        static_assert(static_cast<std::underlying_type_t<HyperMode>>(HyperMode::count) == 2);

        if (hyper_mode == HyperMode::constant)
            return constant_hyperresistive_<component>(J, index);

        // else
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
        auto const lvlCoeff = 1. / std::pow(4, layout_.level);

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
