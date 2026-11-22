#ifndef PLAMN_ELECTRONS_HPP
#define PLAMN_ELECTRONS_HPP

#include "damn/core/def/damn_config.hpp"

#include "plamn/core/def.hpp"
#include "plamn/core/data/grid/grid_tiles.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"
#include "plamn/core/data/grid/detail/detail.hpp"
#include "plamn/core/data/field/field_tiles.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"
#include "plamn/core/data/tensorfield/tensorfield.hpp"
#include "plamn/core/data/vecfield/vecfield_component.hpp"
#include "plamn/initializer/data_provider.hpp"

#include "dict.hpp"

#include "damn/core/exec/kernels/region_chunks.hpp" // IWYU pragma: keep (run_velocity_gpu)
#include <array>
#include <tuple>
#include <cstddef>
#include <stdexcept>
#include <format>
#include <string>
#include <vector>
#include <type_traits>

namespace plamn::core
{


// what one patch's Ve needs on the device (make_velocity_item): plain views
template<typename Field_t, typename VecFieldJ_t, typename VecField_t, typename Box_t>
struct VelocityItem
{
    Field_t Ne;
    VecFieldJ_t J;
    VecField_t F, Ve;
    std::array<Box_t, 3> boxes;
};

template<typename Ions>
class StandardHybridElectronFluxComputer
{
public:
    using VecField   = Ions::vecfield_type;
    using Field      = Ions::field_type;
    using Layout     = Ions::layout_type;
    using Scalar     = std::decay_t<decltype(Layout::options)>::FieldOptions::Scalar;

    StandardHybridElectronFluxComputer(Ions& ions, VecField& J)
        : ions_{ions}
        , J_{J}
        , Ve_{"StandardHybridElectronFluxComputer_Ve", HybridQuantity::Vector::V}
    {
    }

    //-------------------------------------------------------------------------
    //                  start the ResourcesUser interface
    //-------------------------------------------------------------------------


    NO_DISCARD bool isUsable() const { return ions_.isUsable() && J_.isUsable() && Ve_.isUsable(); }

    NO_DISCARD bool isSettable() const
    {
        return Ve_.isSettable() && ions_.isSettable() && J_.isSettable();
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        return std::forward_as_tuple(Ve_, ions_, J_);
    }

    NO_DISCARD auto getCompileTimeResourcesViewList()
    {
        return std::forward_as_tuple(Ve_, ions_, J_);
    }

    //-------------------------------------------------------------------------
    //                  ends the ResourcesUser interface
    //-------------------------------------------------------------------------




    NO_DISCARD Field const& density() const { return ions_.chargeDensity(); }

    NO_DISCARD Field& density() { return ions_.chargeDensity(); }


    NO_DISCARD VecField& velocity() { return Ve_; }


    void computeChargeDensity() {}



    void static Vxyz(auto const& layout, auto const& Ne, auto&&... args)
    {
        // Ve = (sum_s q_s F_s - J) / Ne, with Ne the total ion charge density.
        // The ion bulk velocity is mass weighted and cannot be used here (the multi species
        // momentum fix)
        auto&& [J, F, Ve] = std::forward_as_tuple(args...);
        auto const& Jx    = J(Component::X);
        auto const& Jy    = J(Component::Y);
        auto const& Jz    = J(Component::Z);
        auto const& Fx    = F(Component::X);
        auto const& Fy    = F(Component::Y);
        auto const& Fz    = F(Component::Z);
        auto& Vex         = Ve(Component::X);
        auto& Vey         = Ve(Component::Y);
        auto& Vez         = Ve(Component::Z);

        eval_on_box(Ne, domain_box(layout, Ne), [=] DAMN_ALL(auto const& ijk) mutable {
            auto const JxOnVx = project<Scalar::Jx, Scalar::Vx>(Jx, ijk);
            auto const JyOnVy = project<Scalar::Jy, Scalar::Vy>(Jy, ijk);
            auto const JzOnVz = project<Scalar::Jz, Scalar::Vz>(Jz, ijk);

            Vex(ijk) = (Fx(ijk) - JxOnVx) / Ne(ijk);
            Vey(ijk) = (Fy(ijk) - JyOnVy) / Ne(ijk);
            Vez(ijk) = (Fz(ijk) - JzOnVz) / Ne(ijk);
        });
    }

    void computeBulkVelocity(Layout const& layout)
    {
        check_field(ions_.chargeDensity(), layout);
        check_tensor_field(ions_.chargeFlux(), layout);
        check_tensor_field(J_, layout);

        if constexpr (is_field_tile_set_v<Field>)
            core::tile_exec_with_layout([&](auto&&... args) { Vxyz(args...); },
                                        ions_.chargeDensity(), J_, ions_.chargeFlux(), Ve_);
        else
            Vxyz(layout, ions_.chargeDensity(), J_, ions_.chargeFlux(), Ve_);

        check_tensor_field(Ve_, layout);
    }

    // GPU_UNIFIED, an item per untiled patch collected across a level (run_velocity_gpu: one
    //  launch): the patch's views, the charge density's domain box for each Ve component
    NO_DISCARD auto make_velocity_item(Layout const& layout)
    {
        auto const Ne  = untiled_view(ions_.chargeDensity());
        auto const box = domain_box(layout, ions_.chargeDensity());
        return VelocityItem<decltype(Ne), decltype(untiled_view(J_)), decltype(untiled_view(Ve_)),
                            std::decay_t<decltype(box)>>{
            Ne, untiled_view(J_), untiled_view(ions_.chargeFlux()), untiled_view(Ve_),
            {box, box, box}};
    }

    // Vxyz's Ve for every item on the device, one launch (damn's region chunks)
    static void run_velocity_gpu([[maybe_unused]] auto& items)
    {
#if DAMN_HAVE_GPU
        damn::core::run_region_chunks(items, [] DAMN_ALL(auto& it, auto const c,
                                                         auto const& ijk) {
            if (c == 0)
                it.Ve(Component::X)(ijk)
                    = (it.F(Component::X)(ijk)
                       - project<Scalar::Jx, Scalar::Vx>(it.J(Component::X), ijk))
                      / it.Ne(ijk);
            else if (c == 1)
                it.Ve(Component::Y)(ijk)
                    = (it.F(Component::Y)(ijk)
                       - project<Scalar::Jy, Scalar::Vy>(it.J(Component::Y), ijk))
                      / it.Ne(ijk);
            else
                it.Ve(Component::Z)(ijk)
                    = (it.F(Component::Z)(ijk)
                       - project<Scalar::Jz, Scalar::Vz>(it.J(Component::Z), ijk))
                      / it.Ne(ijk);
        });
#else
        throw std::runtime_error("electrons: run_velocity_gpu needs a GPU build");
#endif
    }

    void computeBulkVelocity(std::size_t const tile_idx)
        requires(is_field_tile_set_v<Field>)
    {
        core::tile_exec_with_layout_at(
            tile_idx, [&](auto&&... args) { Vxyz(args...); }, ions_.chargeDensity(), J_,
            ions_.chargeFlux(), Ve_);
    }


    auto& getIons() const { return ions_; }

private:
    Ions ions_;
    VecField J_;
    VecField Ve_;
};



template<typename Ions>
class IsothermalElectronPressureClosure
{
    using Layout     = Ions::layout_type;
    using VecField   = Ions::vecfield_type;
    using Field      = Ions::field_type;

public:
    using field_type = Field;

    IsothermalElectronPressureClosure(plamn::initializer::Dict const& dict, Ions const& ions)
        : ions_{ions}
        , Te_{dict["Te"].template to<double>()}
        , Pe_{"Pe", HybridQuantity::Scalar::P}
    {
    }


    //-------------------------------------------------------------------------
    //                  start the ResourcesUser interface
    //-------------------------------------------------------------------------


    NO_DISCARD bool isUsable() const { return Pe_.isUsable() and ions_.isUsable(); }

    NO_DISCARD bool isSettable() const { return Pe_.isSettable(); }

    struct PressureProperty
    {
        std::string name;
        HybridQuantity::Scalar qty;
    };

    using PressureProperties = std::vector<PressureProperty>;

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        return std::forward_as_tuple(ions_, Pe_);
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() { return std::forward_as_tuple(ions_, Pe_); }

    //-------------------------------------------------------------------------
    //                  ends the ResourcesUser interface
    //-------------------------------------------------------------------------



    NO_DISCARD Field& pressure() { return Pe_; }
    NO_DISCARD Field const& pressure() const { return Pe_; }
    NO_DISCARD double Te() const { return Te_; }

    void computePressure(Layout const& /*layout*/)
    {
        // static_assert(Field::is_contiguous, "Error - assumes Field date is contiguous");

        if (!Pe_.isUsable())
            throw std::runtime_error("Error - isothermal closure pressure not usable");

        auto const& Ne_ = ions_.chargeDensity();
        transform(Ne_, Pe_, [this](auto n) { return n * Te_; });
    }

    void computePressure(std::size_t const tile_idx)
        requires(is_field_tile_set_v<Field>)
    {
        if (!Pe_.isUsable())
            throw std::runtime_error("Error - isothermal closure pressure not usable");

        auto const& Ne_ = ions_.chargeDensity();
        transform(Ne_[tile_idx](), Pe_[tile_idx](), [this](auto n) { return n * Te_; });
    }

private:
    Ions ions_;
    double const Te_ = 0;
    Field Pe_;
};




template<typename Ions>
class ElectronMomentModel
{
    using VecField     = Ions::vecfield_type;
    using Field        = Ions::field_type;
    using Layout       = Ions::layout_type;
    using FluxComputer = StandardHybridElectronFluxComputer<Ions>;

public:
    ElectronMomentModel(plamn::initializer::Dict const& dict, Ions& ions, VecField& J)
        : fluxComput_{ions, J}
        , pressureClosure_{dict["pressure_closure"], fluxComput_.getIons()}
    {
    }

    //-------------------------------------------------------------------------
    //                  start the ResourcesUser interface
    //-------------------------------------------------------------------------

    NO_DISCARD bool isUsable() const
    {
        return fluxComput_.isUsable() and pressureClosure_.isUsable();
    }

    NO_DISCARD bool isSettable() const { return fluxComput_.isSettable(); }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        return std::forward_as_tuple(fluxComput_, pressureClosure_);
    }

    NO_DISCARD auto getCompileTimeResourcesViewList()
    {
        return std::forward_as_tuple(fluxComput_, pressureClosure_);
    }

    //-------------------------------------------------------------------------
    //                  ends the ResourcesUser interface
    //-------------------------------------------------------------------------


    NO_DISCARD Field const& density() const { return fluxComput_.density(); }

    NO_DISCARD Field& density() { return fluxComput_.density(); }


    NO_DISCARD VecField const& velocity() const { return fluxComput_.velocity(); }

    NO_DISCARD VecField& velocity() { return fluxComput_.velocity(); }


    NO_DISCARD Field const& pressure() const { return pressureClosure_.pressure(); }
    NO_DISCARD Field& pressure() { return pressureClosure_.pressure(); }
    NO_DISCARD double Te() const { return pressureClosure_.Te(); }



    void computeChargeDensity() { fluxComput_.computeChargeDensity(); }
    NO_DISCARD auto make_velocity_item(auto const& layout)
    {
        return fluxComput_.make_velocity_item(layout);
    }
    void computeBulkVelocity(auto const& layout_or_tile_idx)
    {
        fluxComput_.computeBulkVelocity(layout_or_tile_idx);
    }
    void computePressure(auto const& layout_or_tile_idx)
    {
        pressureClosure_.computePressure(layout_or_tile_idx);
    }

private:
    FluxComputer fluxComput_;
    IsothermalElectronPressureClosure<Ions> pressureClosure_;
};



template<typename Ions>
class Electrons
{
    using VecField   = Ions::vecfield_type;
    using Field      = Ions::field_type;
    using Layout     = Ions::layout_type;

public:
    Electrons(plamn::initializer::Dict const& dict, Ions& ions, VecField& J)
        : momentModel_{dict, ions, J}
    {
    }


    void update(Layout const& layout) { update_(layout); }
    // Ve only: the pressure from the level's charge density at once (Pe = Ne Te, Te())
    void update_velocity(Layout const& layout)
    {
        momentModel_.computeChargeDensity();
        momentModel_.computeBulkVelocity(layout);
    }
    NO_DISCARD double Te() const { return momentModel_.Te(); }
    // update_velocity's Ve for a level at once (GPU_UNIFIED, untiled): an item per patch, one
    //  launch for the items (run_velocity_gpu)
    NO_DISCARD auto make_velocity_item(Layout const& layout)
    {
        return momentModel_.make_velocity_item(layout);
    }
    static void run_velocity_gpu(auto& items)
    {
        StandardHybridElectronFluxComputer<Ions>::run_velocity_gpu(items);
    }

    // one tile's worth of work, tiles can be updated concurrently
    void update(std::size_t const tile_idx)
        requires(is_field_tile_set_v<Field>)
    {
        update_(tile_idx);
    }


    //-------------------------------------------------------------------------
    //                  start the ResourcesUser interface
    //-------------------------------------------------------------------------

    NO_DISCARD bool isUsable() const { return momentModel_.isUsable(); }

    NO_DISCARD bool isSettable() const { return momentModel_.isSettable(); }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        return std::forward_as_tuple(momentModel_);
    }

    NO_DISCARD auto getCompileTimeResourcesViewList()
    {
        return std::forward_as_tuple(momentModel_);
    }



    //-------------------------------------------------------------------------
    //                  ends the ResourcesUser interface
    //-------------------------------------------------------------------------




    NO_DISCARD Field const& density() const { return momentModel_.density(); }
    NO_DISCARD VecField const& velocity() const { return momentModel_.velocity(); }
    NO_DISCARD Field const& pressure() const { return momentModel_.pressure(); }


    NO_DISCARD Field& density() { return momentModel_.density(); }
    NO_DISCARD VecField& velocity() { return momentModel_.velocity(); }
    NO_DISCARD Field& pressure() { return momentModel_.pressure(); }

private:
    void update_(auto const& layout_or_tile_idx)
    {
        if (!isUsable())
            throw std::runtime_error("Error - Electron is not usable");

        momentModel_.computeChargeDensity();
        momentModel_.computeBulkVelocity(layout_or_tile_idx);
        momentModel_.computePressure(layout_or_tile_idx);
    }

    ElectronMomentModel<Ions> momentModel_;
};

} // namespace plamn::core


#endif
