#ifndef PLAMN_SIMULATOR_HYBRID_PATCH_HPP
#define PLAMN_SIMULATOR_HYBRID_PATCH_HPP

// HybridPatch: everything one patch of a hybrid level owns
//   model:  B, E (em), J, ions (a runtime list of populations), electrons (Ve, Pe)
//   solver: predicted and averaged B and E (PPC), the ion updater's boxing
//
// not movable: the electrons hold copies of the ions and J views, which point at members;
// LevelData keeps patches behind unique_ptr
// nothing is filled on construction: the simulator writes everything before reading it
//
// for_each_resource names what damn's schedules may move:
//   EM_B_x... EM_E_x..., EMPred_B_x..., EMAvg_E_x...,
//   <pop>_density, <pop>_charge_density, <pop>_flux_x...,
//   <pop>_domain, <pop>_patch_ghost (particles)

#include "plamn/core/def.hpp"
#include "plamn/plamn_core.hpp"
#include "plamn/amr/resources.hpp"
#include "plamn/core/data/ions/usable_ions.hpp"
#include "plamn/core/data/electromag/usable_electromag.hpp"
#include "plamn/core/data/tensorfield/usable_tensorfield.hpp"
#include "plamn/core/numerics/ion_updater/ion_updater.hpp"
#include "plamn/initializer/data_provider.hpp"

#include "damn/amr/hierarchy.hpp"

#include <array>
#include <memory>
#include <string>
#include <vector>
#include <cstddef>

namespace plamn::simulator
{
template<auto opts>
struct HybridPatch
{
    using Types    = core::PLAMN_Types<opts>::Hybrid;
    using Layout_t = Types::Layout_t;
    using Grid_t   = Types::Grid_t;

    auto static constexpr dimension            = Layout_t::dimension;
    auto static constexpr field_ghost_width    = Layout_t::ghost_width;
    auto static constexpr particle_ghost_width = Layout_t::options.particle_ghost_width;

    auto static constexpr field_opts = core::TensorFieldOptions<Types>{};
    using Electromag_t               = core::UsableElectromag<field_opts>;
    using VecField_t                 = core::UsableVecField<field_opts>;
    using Ions_t                     = core::UsableIons_t<opts>;
    using Electrons_t                = Types::Electrons_t;
    using IonUpdater_t               = core::IonUpdater<typename Types::Particles_t, Layout_t>;
    using Boxing_t                   = IonUpdater_t::Boxing_t;
    using Box_t                      = damn::core::Box<int, dimension>;

    // dict: the "simulation" node; non_level_ghost: the cells around the patch other
    // patches of the level cover (see damn::amr::covered_by_level)
    HybridPatch(Layout_t const& layout_, initializer::Dict const& dict,
                std::vector<Box_t> const& non_level_ghost)
        : layout{layout_}
        , em{layout, dict["electromag"], core::no_fill}
        , J{"J", layout, core::HybridQuantity::Vector::J}
        , ions{layout, dict["ions"], core::no_fill}
        , Ve{"StandardHybridElectronFluxComputer_Ve", layout, core::HybridQuantity::Vector::V}
        , Pe{"Pe", layout, core::HybridQuantity::Scalar::P}
        , electrons{dict["electrons"], *ions, *J}
        , emPred{layout, dict["electromag"], core::no_fill} // storage, not initialized
        , emAvg{layout, dict["electromag"], core::no_fill}
        , boxing{layout, non_level_ghost}
    {
        bind_electrons_();
    }

    HybridPatch(HybridPatch const&)            = delete;
    HybridPatch(HybridPatch&&)                 = delete;
    HybridPatch& operator=(HybridPatch const&) = delete;
    HybridPatch& operator=(HybridPatch&&)      = delete;

    Layout_t layout;
    Electromag_t em;
    VecField_t J;
    Ions_t ions;
    VecField_t Ve;
    Grid_t Pe;
    Electrons_t electrons;
    Electromag_t emPred, emAvg;
    Boxing_t boxing;

private:
    // the electrons' own fields: Ve and Pe
    void bind_electrons_()
    {
        auto&& [moments]          = electrons.getCompileTimeResourcesViewList();
        auto&& [flux, closure]    = moments.getCompileTimeResourcesViewList();
        auto&& [ve, ions_, J_]    = flux.getCompileTimeResourcesViewList();
        auto&& [ions__, pressure] = closure.getCompileTimeResourcesViewList();
        Ve.set_on(ve);
        pressure.setBuffer(&Pe);
    }
};


namespace detail
{
    std::array<char const*, 3> constexpr inline xyz{"_x", "_y", "_z"};

    template<auto gw>
    void vecfield_resources(std::string const& name, auto& vecfield, auto&& fn)
    {
        for (std::size_t i = 0; i < 3; ++i)
        {
            auto r = amr::field_resource<gw>(vecfield[i]);
            fn(name + xyz[i], r);
        }
    }
} // namespace detail

template<auto opts>
void for_each_resource(HybridPatch<opts>& patch, auto&& fn)
{
    using Patch    = HybridPatch<opts>;
    auto const gw  = Patch::field_ghost_width;
    auto const pgw = Patch::particle_ghost_width;

    detail::vecfield_resources<gw>("EM_B", patch.em.B, fn);
    detail::vecfield_resources<gw>("EM_E", patch.em.E, fn);
    detail::vecfield_resources<gw>("EMPred_B", patch.emPred.B, fn);
    detail::vecfield_resources<gw>("EMPred_E", patch.emPred.E, fn);
    detail::vecfield_resources<gw>("EMAvg_B", patch.emAvg.B, fn);
    detail::vecfield_resources<gw>("EMAvg_E", patch.emAvg.E, fn);

    for (auto& pop : patch.ions.populations)
    {
        auto const& name = pop.name();
        auto density     = amr::field_resource<gw>(pop.rhoP);
        auto charge      = amr::field_resource<gw>(pop.rhoC);
        fn(name + "_density", density);
        fn(name + "_charge_density", charge);
        detail::vecfield_resources<gw>(name + "_flux", pop.F, fn);

        auto domain      = amr::particle_resource<pgw>(pop.particles.domain_particles);
        auto patch_ghost = amr::particle_resource<pgw>(pop.particles.patch_ghost_particles);
        fn(name + "_domain", domain);
        fn(name + "_patch_ghost", patch_ghost);
    }
}

// LevelData holds patches by unique_ptr
template<auto opts>
void for_each_resource(std::unique_ptr<HybridPatch<opts>>& patch, auto&& fn)
{
    for_each_resource(*patch, fn);
}

} // namespace plamn::simulator

#endif // PLAMN_SIMULATOR_HYBRID_PATCH_HPP
