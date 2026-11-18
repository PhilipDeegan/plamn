#ifndef PLAMN_SIMULATOR_HYBRID_PATCH_HPP
#define PLAMN_SIMULATOR_HYBRID_PATCH_HPP

// HybridPatch: one patch of a hybrid level, its data and the core views over it, as a
// SAMRAI patch's data would be
//   views:   em, J, ions (a runtime list of populations), electrons; and the PPC solver's
//            predicted and averaged electromag (emPred, emAvg)
//   storage: the grids and particle arrays the views point at, named as the views, not
//            filled: the simulator writes everything before reading it
//
// not movable: the views, and the electrons' copies of the ions and J views, point at
// members; LevelData keeps patches behind unique_ptr
//
// for_each_resource names what damn's schedules may move:
//   EM_B_x... EM_E_x..., EMPred_B_x..., EMAvg_E_x...,
//   <pop>_density, <pop>_charge_density, <pop>_flux_x...,
//   ions_charge_density, ions_charge_flux_x... (the totals, what a synchronization coarsens),
//   EM_E_old_x..., EM_B_old_x..., EM_fluxSum_E_x... (AMR),
//   <pop>_domain, <pop>_patch_ghost, <pop>_level_ghost[_old, _new] (particles)

#include "mkn/kul/for.hpp"
#include "damn/core/geometry/box.hpp"
#include "mkn/kul/env.hpp"
#include "plamn/core/def.hpp"
#include "plamn/plamn_core.hpp"
#include "plamn/amr/resources.hpp"
#include "plamn/core/data/ions/ion_population/particle_pack.hpp"
#include "plamn/core/numerics/particles_synchronised_mover/particles_synchronised_mover.hpp"
#include "plamn/initializer/data_provider.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <cstddef>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

namespace plamn::simulator
{
template<auto opts>
struct HybridPatch
{
    using Types       = core::PLAMN_Types<opts>::Hybrid;
    using Layout_t    = Types::Layout_t;
    using Grid_t      = Types::Grid_t;
    using Particles_t = Types::Particles_t;

    auto static constexpr dimension            = Layout_t::dimension;
    auto static constexpr field_ghost_width    = Layout_t::ghost_width;
    auto static constexpr particle_ghost_width = Layout_t::options.particle_ghost_width;

    using Electromag_t = Types::Electromag_t;
    using VecField_t   = Types::VecField_t;
    using Ions_t       = Types::Ions_t;
    using Electrons_t  = Types::Electrons_t;
    // AoSMapped: the serial per patch mover, else a level's (Simulator::move_ions_)
    using ParticlesSynchronisedMover_t
        = core::ParticlesSynchronisedMoverFor_t<Particles_t, Layout_t>;
    static constexpr bool per_patch_mover
        = std::is_same_v<ParticlesSynchronisedMover_t,
                         core::ParticlesSynchronisedMover<Particles_t, Layout_t>>;
    using Boxing_t = ParticlesSynchronisedMover_t::Boxing_t;
    using Box_t    = damn::core::Box<int, dimension>;

    template<std::size_t N>
    using Grids = std::array<Grid_t, N>;

    // a grid for a field view: same name and quantity, not filled
    static Grid_t grid(auto const& field, Layout_t const& layout)
    {
        return Grid_t{field.name(), layout, field.physicalQuantity()};
    }

    // grids for the components of a tensor field view
    template<typename TensorField>
    static auto grids(TensorField const& tf, Layout_t const& layout)
    {
        auto constexpr N = std::tuple_size_v<std::decay_t<decltype(tf.components())>>;
        return mkn::kul::for_N<N, mkn::kul::for_N_R_mode::make_array>(
            [&](auto i) { return grid(tf[i], layout); });
    }

    // grids like a vector's, named name_x, name_y, name_z: storage without a view
    static Grids<3> like(Grids<3> const& vector, std::string const& name, Layout_t const& layout)
    {
        std::array<char const*, 3> constexpr xyz{"_x", "_y", "_z"};
        return mkn::kul::for_N<3, mkn::kul::for_N_R_mode::make_array>(
            [&](auto i) { return Grid_t{name + xyz[i], layout, vector[i].physicalQuantity()}; });
    }

    template<typename TensorField, std::size_t N>
    static void bind(TensorField& tf, Grids<N>& grids)
    {
        for (std::size_t i = 0; i < N; ++i)
            tf[i].setBuffer(&grids[i]);
    }

    // one population's data
    struct PopulationData
    {
        PopulationData(auto& pop, Layout_t const& layout)
            : density{grid(pop.particleDensity(), layout)}
            , charge_density{grid(pop.chargeDensity(), layout)}
            , flux{grids(pop.flux(), layout)}
            , domain{core::make_particles<Particles_t>(layout)}
            , patch_ghost{core::make_particles<Particles_t>(layout)}
            , level_ghost{core::make_particles<Particles_t>(layout)}
            , level_ghost_old{core::make_particles<Particles_t>(layout)}
            , level_ghost_new{core::make_particles<Particles_t>(layout)}
            , pack{pop.name(),   &domain,          &patch_ghost,
                   &level_ghost, &level_ghost_old, &level_ghost_new}
        {
            auto&& [F, M, pd, cd, particles] = pop.getCompileTimeResourcesViewList();
            if constexpr (Particles_t::layout_mode == core::LayoutMode::SoAVXChunked)
            { // the domain's chunks with gaps between them (particles_soavx_chunked.hpp): changes
              //  of cell without a sort (PLAMN_CHUNK_SLACK, a fraction of the particles,
              //  PLAMN_CHUNK_BORDER, the border cells' weight to 1 inside)
                static double const slack  = mkn::kul::env::get_as("PLAMN_CHUNK_SLACK", .1);
                static double const border = mkn::kul::env::get_as("PLAMN_CHUNK_BORDER", 3.);
                domain.set_slack(slack, layout.cells, border);
            }
            bind(F, flux); // M: over scratch grids when computed (simulator_computed.hpp)
            pd.setBuffer(&density);
            cd.setBuffer(&charge_density);
            particles.setBuffer(&pack);
        }

        PopulationData(PopulationData const&) = delete;
        PopulationData(PopulationData&&)      = delete;

        Grid_t density, charge_density;
        Grids<3> flux;
        Particles_t domain, patch_ghost, level_ghost, level_ghost_old, level_ghost_new;
        core::ParticlesPack<Particles_t> pack;
    };

    // the patch's particle ghost box and the level's other patches, periodic images included
    //   non_level_ghost:  the cells other patches cover (damn::amr::covered_by_level)
    //   owned_elsewhere:  the cells another patch holding them owns (the nearest): their
    //                     level ghost particles are its alone (Simulator's own_level_ghosts_;
    //                     damn::amr::owned_elsewhere)
    struct Surroundings
    {
        std::vector<Box_t> non_level_ghost;
        std::vector<Box_t> owned_elsewhere;
    };

    // dict: the "simulation" node
    HybridPatch(Layout_t const& layout_, initializer::Dict const& dict,
                Surroundings const& surroundings)
        : layout{layout_}
        , em{dict["electromag"]}
        , emPred{"EMPred"}
        , emAvg{"EMAvg"}
        , J{"J", core::HybridQuantity::Vector::J}
        , Bold{"EM_B_old", core::HybridQuantity::Vector::B}
        , ions{dict["ions"]}
        , boxing{std::in_place, layout, surroundings.non_level_ghost}
        , level_ghost_not_owned{surroundings.owned_elsewhere}
        , emE{grids(em.E, layout)}
        , emB{grids(em.B, layout)}
        , predE{grids(emPred.E, layout)}
        , predB{grids(emPred.B, layout)}
        , avgE{grids(emAvg.E, layout)}
        , avgB{grids(emAvg.B, layout)}
        , Jxyz{grids(J, layout)}
        , massDensity{grid(ions.massDensity(), layout)}
        , chargeDensity{grid(ions.chargeDensity(), layout)}
        , bulkVelocity{grids(ions.velocity(), layout)}
        , chargeFlux{grids(ions.chargeFlux(), layout)}
        , oldE{like(emE, "EM_E_old", layout)}
        , oldB{like(emB, "EM_B_old", layout)}
        , fluxSumE{like(emE, "EM_fluxSum_E", layout)}
    {
        bind(em.E, emE);
        bind(em.B, emB);
        bind(emPred.E, predE);
        bind(emPred.B, predB);
        bind(emAvg.E, avgE);
        bind(emAvg.B, avgB);
        bind(J, Jxyz);
        bind(Bold, oldB);

        auto&& [bV, M, cd, md, cF] = ions.getCompileTimeResourcesViewList();
        bind(bV, bulkVelocity); // M: over scratch grids when computed
        cd.setBuffer(&chargeDensity);
        md.setBuffer(&massDensity);
        bind(cF, chargeFlux);
        for (auto& pop : ions)
            populations.emplace_back(std::make_unique<PopulationData>(pop, layout));

        // the electrons copy the ions and J views: made once those point at data
        electrons_.emplace(dict["electrons"], ions, J);
        auto&& [moments]          = electrons_->getCompileTimeResourcesViewList();
        auto&& [flux, closure]    = moments.getCompileTimeResourcesViewList();
        auto&& [ve, ions_, J_]    = flux.getCompileTimeResourcesViewList();
        auto&& [ions__, pressure] = closure.getCompileTimeResourcesViewList();
        electronVelocity.emplace(grids(ve, layout));
        electronPressure.emplace(grid(pressure, layout));
        bind(ve, *electronVelocity);
        pressure.setBuffer(&*electronPressure);

        // debug: every value NaN until set, so a point nothing filled is found (Simulator's
        //  check_set_) rather than read
        if constexpr (DAMN_DEBUG)
            for_each_grid([](auto& grid) { grid.fill(std::numeric_limits<double>::quiet_NaN()); });
    }

    // fn(grid) for every grid of the patch
    void for_each_grid(auto&& fn)
    {
        for (auto* grids : {&emE, &emB, &predE, &predB, &avgE, &avgB, &Jxyz, &bulkVelocity,
                            &chargeFlux, &oldE, &oldB, &fluxSumE, &*electronVelocity})
            for (auto& g : *grids)
                fn(g);
        fn(massDensity);
        fn(chargeDensity);
        fn(*electronPressure);
        for (auto& pop : populations)
        {
            fn(pop->density);
            fn(pop->charge_density);
            for (auto& g : pop->flux)
                fn(g);
        }
    }

    HybridPatch(HybridPatch const&)            = delete;
    HybridPatch(HybridPatch&&)                 = delete;
    HybridPatch& operator=(HybridPatch const&) = delete;
    HybridPatch& operator=(HybridPatch&&)      = delete;

    NO_DISCARD auto& electrons() { return *electrons_; }

    Layout_t layout;

    // views
    Electromag_t em, emPred, emAvg;
    VecField_t J;
    VecField_t Bold; // over oldB
    Ions_t ions;
    std::optional<Boxing_t> boxing; // the mover's: rebox() when the level's patches change
    // the level ghost cells another patch owns: their level ghosts are its (Surroundings)
    std::vector<Box_t> level_ghost_not_owned;

    // the level's other patches changed (a regrid)
    void rebox(Surroundings const& surroundings)
    {
        boxing.emplace(layout, surroundings.non_level_ghost);
        level_ghost_not_owned = surroundings.owned_elsewhere;
    }

    // storage
    Grids<3> emE, emB, predE, predB, avgE, avgB, Jxyz;
    Grid_t massDensity, chargeDensity;
    Grids<3> bulkVelocity, chargeFlux;
    // computed quantities' grids (e.g. the momentum tensor), by name, made on first use and kept
    // with the patch (simulator_computed.hpp)
    std::map<std::string, Grid_t> scratch;
    Grid_t& scratch_grid(std::string const& name, core::HybridQuantity::Scalar const qty)
    {
        auto it = scratch.find(name);
        if (it == scratch.end())
            it = scratch.try_emplace(name, name, layout, qty).first;
        return it->second;
    }
    // AMR: E and B at the start of the level's step (prepare_step), for the finer level's time
    // interpolation and the reflux; the time average of E over a coarser step (reflux)
    Grids<3> oldE, oldB, fluxSumE;
    std::vector<std::unique_ptr<PopulationData>> populations;
    std::optional<Grids<3>> electronVelocity;
    std::optional<Grid_t> electronPressure;

private:
    std::optional<Electrons_t> electrons_;
};


namespace detail
{
    std::array<char const*, 3> constexpr inline xyz{"_x", "_y", "_z"};

    template<auto gw>
    void vector_resources(std::string const& name, auto& grids, auto&& fn)
    {
        for (std::size_t i = 0; i < 3; ++i)
        {
            auto r = amr::field_resource<gw>(grids[i]);
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

    detail::vector_resources<gw>("EM_B", patch.emB, fn);
    detail::vector_resources<gw>("EM_E", patch.emE, fn);
    detail::vector_resources<gw>("EMPred_B", patch.predB, fn);
    detail::vector_resources<gw>("EMPred_E", patch.predE, fn);
    detail::vector_resources<gw>("EMAvg_B", patch.avgB, fn);
    detail::vector_resources<gw>("EMAvg_E", patch.avgE, fn);
    detail::vector_resources<gw>("EM_E_old", patch.oldE, fn);
    detail::vector_resources<gw>("EM_B_old", patch.oldB, fn);
    detail::vector_resources<gw>("EM_fluxSum_E", patch.fluxSumE, fn);

    auto charge_density = amr::field_resource<gw>(patch.chargeDensity);
    fn("ions_charge_density", charge_density);
    detail::vector_resources<gw>("ions_charge_flux", patch.chargeFlux, fn);

    for (auto& pop : patch.populations)
    {
        auto const& name = pop->pack._name;
        auto density     = amr::field_resource<gw>(pop->density);
        auto charge      = amr::field_resource<gw>(pop->charge_density);
        fn(name + "_density", density);
        fn(name + "_charge_density", charge);
        detail::vector_resources<gw>(name + "_flux", pop->flux, fn);

        auto domain          = amr::particle_resource<pgw>(pop->domain);
        auto patch_ghost     = amr::particle_resource<pgw>(pop->patch_ghost);
        auto level_ghost     = amr::particle_resource<pgw>(pop->level_ghost);
        auto level_ghost_old = amr::particle_resource<pgw>(pop->level_ghost_old);
        auto level_ghost_new = amr::particle_resource<pgw>(pop->level_ghost_new);
        fn(name + "_domain", domain);
        fn(name + "_patch_ghost", patch_ghost);
        fn(name + "_level_ghost", level_ghost);
        fn(name + "_level_ghost_old", level_ghost_old);
        fn(name + "_level_ghost_new", level_ghost_new);
    }
    for (auto& [name, grid] : patch.scratch)
    {
        auto r = amr::field_resource<gw>(grid);
        fn(name, r);
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
