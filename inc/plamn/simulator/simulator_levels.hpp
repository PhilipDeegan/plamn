#ifndef PLAMN_SIMULATOR_SIMULATOR_LEVELS_HPP
#define PLAMN_SIMULATOR_SIMULATOR_LEVELS_HPP

// plamn::Simulator: making and initializing levels (PHARE's HybridLevelInitializer)
// IWYU pragma: private, include "plamn/simulator/simulator.hpp"

#include "damn/amr/copy.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/level_map.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/amr/partition.hpp"
#include "damn/amr/topology.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/logger.hpp"
#include "mkn/kul/std/array.hpp"

#include "plamn/simulator/simulator.hpp" // IWYU pragma: keep (the class)
#include "plamn/amr/level_pairs.hpp"
#include "plamn/amr/refine.hpp"
#include "plamn/amr/resources.hpp"
#include "plamn/amr/split/splitter.hpp"
#include "plamn/amr/tagging.hpp"
#include "plamn/core/models/electromag_initializer.hpp"
#include "plamn/core/numerics/ampere/ampere_transformer.hpp"
#include "plamn/core/numerics/moments/moments.hpp"

#include "dict.hpp"
#include "phlop/timing/scope.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
// plamn::Simulator: making and initializing levels (PHARE's HybridLevelInitializer)
// IWYU pragma: private, include "plamn/simulator/simulator.hpp"

namespace plamn
{
// level ln from its boxes (that level's cells): chopped into the largest patch size,
// partitioned along the Hilbert curve by the particles they will hold, allocated, with their
// neighbours and the map to the coarser level
template<auto opts>
damn::amr::Level<Simulator<opts>::dimension>
Simulator<opts>::level_of_(int const ln, std::vector<Box_t> const& boxes)
{
    // the cells shared evenly (no patch left with a few cells, SAMRAI's patches on one rank);
    // refined levels: patches of whole coarser cells, the size a multiple of the ratio
    auto size = patch_size_;
    if (ln > 0)
        for (auto& s : size)
            s = std::max(hierarchy_.ratio, s / hierarchy_.ratio * hierarchy_.ratio);
    std::vector<Box_t> patches;
    for (auto const& box : boxes)
        for (auto const& b : damn::amr::chop_evenly(box, size, ln > 0 ? hierarchy_.ratio : 1))
            patches.push_back(b);
    return damn::amr::make_level(
        ln, patches,
        damn::amr::partition(
            patches, isInitialized_ ? particle_weights_(ln, patches) : box_weights_(ln, patches),
            damn::amr::node_share_capacity(topology_)));
}

template<auto opts>
void Simulator<opts>::make_level_(int const ln, std::vector<Box_t> const& boxes)
{
    damn::amr::set_level(hierarchy_, level_of_(ln, boxes));
    allocate_level_(ln);
}

template<auto opts>
void Simulator<opts>::allocate_level_(int const ln)
{
    auto const& sim = dict_["simulation"];
    resize_levels_(static_cast<std::size_t>(ln) + 1);
    auto& ld = data_(ln);
    ld       = damn::amr::allocate(hierarchy_.levels[ln], rank_, [&](auto const& p) {
        Layout_t const layout{p.box, damn::amr::mesh_size(hierarchy_, ln),
                              damn::amr::position(hierarchy_, ln, p.box.lower), ln};
        return std::make_unique<Patch_t>(layout, sim, surroundings_(ln, p.box));
    });
    link_level_(ln);
}

template<auto opts>
typename Simulator<opts>::Patch_t::Surroundings
Simulator<opts>::surroundings_(int const ln, Box_t const& box) const
{
    auto constexpr pgw = static_cast<int>(Patch_t::particle_ghost_width);
    return {damn::amr::covered_by_level(
                hierarchy_, ln, damn::core::grow(box, mkn::kul::ConstArray<int, dimension>(pgw))),
            damn::amr::owned_elsewhere(hierarchy_, ln, box, pgw)};
}

template<auto opts>
void Simulator<opts>::link_level_(int const ln)
{
    auto const pgw    = Patch_t::particle_ghost_width;
    auto constexpr gw = Patch_t::field_ghost_width;
    auto& l           = levels_[ln];
    auto& ld          = data_(ln);

    // the level's neighbours, as far as any resource reaches outside its patch: field ghosts
    // plus primal nodes, or particle ghosts
    auto constexpr reach = std::max<int>(gw + 1, pgw);
    l.neighbours         = damn::amr::make_neighbours(hierarchy_, ln, rank_, topology_, reach);
    if (ln == 0)
        return;

    // to the coarser level: as far as a refined field ghost's stencil, or a split particle's
    //  parent, reaches
    l.to_coarser = damn::amr::make_level_map(hierarchy_, ln - 1, rank_, topology_,
                                             reach + 2 * amr::splitBoxGrowth<Splitter_t>() + 4);
    l.level_ghost_cells.clear();
    for (auto const& p : ld.patches)
    {
        auto& cells = l.level_ghost_cells.emplace_back();
        cells.push_back(damn::core::grow(p.box, mkn::kul::ConstArray<int, dimension>(gw)));
        for (auto const& covered : damn::amr::covered_by_level(hierarchy_, ln, cells.front()))
        {
            std::vector<Box_t> rest;
            for (auto const& c : cells)
                for (auto const& r : c.remove(covered))
                    rest.push_back(r);
            cells = std::move(rest);
        }
    }
}

template<auto opts>
void Simulator<opts>::initialize()
{
    DAMN_LOG_SCOPE(1, "Simulator::initialize");
    if (isInitialized_)
        throw std::runtime_error("plamn Simulator already initialized");

    // the steps from the start (or the restart's time), the first dt (adaptive: from the
    //  initialized levels, as PHARE's for its first dump)
    auto const start_steps = [&]() {
        timestepper_.start_at(startTime_);
        dt_ = timestepper_.next([&](auto const& cfl) { return stable_dt_(cfl); });
    };

    if (!restarts_.load_path.empty()) // pyphare's restart_time: every level from the file
    {
        load_restart_();
        start_steps();
        isInitialized_ = true;
        return;
    }

    std::optional<std::vector<std::vector<Box_t>>> pretag_boxes;
    if (tagging_ and pretag_enabled_)
        pretag_boxes = pretag_();

    make_level_(0, {hierarchy_.domain});
    initialize_root_();
    for (std::size_t ln = 1; ln <= refinement_boxes_.size(); ++ln)
    {
        make_level_(static_cast<int>(ln), refinement_boxes_[ln - 1]);
        initialize_refined_(ln);
    }
    // tagging: each new level from the tags of the one under it, as long as some are (pretag:
    //  those boxes are known already)
    if (tagging_)
        for (int ln = 0; ln + 1 < hierarchy_.max_levels; ++ln)
        {
            if (pretag_boxes and static_cast<std::size_t>(ln) >= pretag_boxes->size())
                break;
            auto const boxes = pretag_boxes ? (*pretag_boxes)[ln] : cluster_(ln);
            if (boxes.empty())
                break;
            clustered_.resize(static_cast<std::size_t>(ln) + 2);
            clustered_[ln + 1] = boxes;
            make_level_(ln + 1, boxes);
            initialize_refined_(static_cast<std::size_t>(ln) + 1);
        }
    // every level's E and B at t0, for the finer levels' time interpolation
    for (std::size_t ln = 0; ln < levels_.size(); ++ln)
        prepare_step_(static_cast<int>(ln), startTime_);

    currentTime_ = startTime_;
    start_steps();
    isInitialized_ = true;
}

// PHARE's HybridLevelInitializer, root level: the user's functions
template<auto opts>
void Simulator<opts>::initialize_root_()
{
    auto const& sim = dict_["simulation"];
    per_patch_(0, [&](auto& patch) {
        core::ElectromagInitializer<dimension>{sim["electromag"]}(patch.em, patch.layout);
        for (auto& pop : patch.ions)
        {
            auto initializer = Patch_t::Types::ParticleInitializerFactory_t::create(
                pop.particleInitializerInfo(), pop.name());
            initializer->loadParticles(pop.domainParticles(), patch.layout);
        }
    });
    copy_(0, damn::amr::fill_ghosts, "EM_B");
    copy_(0, damn::amr::fill_ghosts, "EM_E");
    check_set_(0, "EM_B", "root initialization"); // E: from Ohm's law below

    // moments from the particles, completed across patch borders
    per_patch_(0, [&](auto& patch) {
        core::resetMoments(patch.ions);
        core::depositParticles(patch.ions, patch.layout, Interpolator{}, core::DomainDeposit{});
    });
    solver::hybrid::fill_ion_borders(level_services_(0));
    solver::hybrid::compute_moments(level_services_(0));

    // J from B, then E from Ohm's law
    solver::hybrid::ampere_ohm<&Patch_t::em>(hybrid_, level_services_(0));
    copy_(0, damn::amr::fill_ghosts, "EM_E");
    check_set_(0, "EM_E", "root initialization");
}

// PHARE's HybridLevelInitializer, a refined level: B, E and the particles from the coarser
// level, the level ghost particles split from it, then the moments and J
template<auto opts>
void Simulator<opts>::initialize_refined_(std::size_t const ln)
{
    using damn::amr::LevelRegion;
    refine_magnetic_(ln);
    refine_(ln, amr::FieldRefiner{}, LevelRegion::Interior, amr::vector_pairs("EM_E", "EM_E"));
    fill_magnetic_ghosts_(ln, "EM_B", startTime_);
    fill_electric_ghosts_(ln, "EM_E", startTime_);

    for (auto const& name : population_names_)
    {
        refine_(ln, amr::ParticleSplitter<Splitter_t>{}, LevelRegion::Interior,
                {{name + "_domain", name + "_domain", ""}});
        refine_(ln, amr::ParticleSplitter<Splitter_t>{}, LevelRegion::LevelGhosts,
                {{name + "_domain", name + "_level_ghost_old", ""}});
    }
    own_level_ghosts_(ln, /*new_ghosts=*/false);
    per_patch_(ln, [](auto& patch) {
        for (auto& pop : patch.ions)
            pop.levelGhostParticles() = pop.levelGhostParticlesOld();
    });
    sort_refined_particles_(ln);

    refined_moments_(ln);
}

// a new level's B from the coarser level's: refined, then the divergence free touch up (its
//  ghosts are filled after, by fill_magnetic_ghosts_)
template<auto opts>
void Simulator<opts>::refine_magnetic_(std::size_t const ln)
{
    refine_(ln, amr::FieldRefiner{}, damn::amr::LevelRegion::Interior,
            amr::vector_pairs("EM_B", "EM_B"));
    auto& ld = data_(ln);
    for (std::size_t i = 0; i < damn::amr::size(ld); ++i)
    {
        auto& patch = *ld.data[i];
        auto bx     = amr::field_resource<Patch_t::field_ghost_width>(patch.emB[0]);
        auto by     = amr::field_resource<Patch_t::field_ghost_width>(patch.emB[1]);
        auto bz     = amr::field_resource<Patch_t::field_ghost_width>(patch.emB[2]);
        amr::divergence_free_touch_up(std::array{&bx, &by, &bz}, ld.patches[i].box,
                                      ld.patches[i].box,
                                      damn::amr::mesh_size(hierarchy_, static_cast<int>(ln)));
    }
}

// refinement="tagging" with pretag: every level's boxes before any particle, from B alone as the
//  run computes it (level 0's from the functions, a finer level's refined from the coarser one,
//  with its ghosts), tagged and clustered as initialize does; the levels are dropped after, the
//  run then makes them with these boxes knowing all of them. Per level above 0, its boxes
template<auto opts>
std::vector<std::vector<typename Simulator<opts>::Box_t>> Simulator<opts>::pretag_()
{
    DAMN_LOG_SCOPE(1, "Simulator::pretag_");
    auto const& sim = dict_["simulation"];
    make_level_(0, {hierarchy_.domain});
    per_patch_(0, [&](auto& patch) {
        core::ElectromagInitializer<dimension>{sim["electromag"]}(patch.em, patch.layout);
    });
    copy_(0, damn::amr::fill_ghosts, "EM_B");

    std::vector<std::vector<Box_t>> boxes;
    for (int ln = 0; ln + 1 < hierarchy_.max_levels; ++ln)
    {
        auto level_boxes = cluster_(ln);
        if (level_boxes.empty())
            break;
        make_level_(ln + 1, level_boxes);
        refine_magnetic_(static_cast<std::size_t>(ln) + 1);
        fill_magnetic_ghosts_(ln + 1, "EM_B", startTime_);
        boxes.push_back(std::move(level_boxes));
    }
    resize_levels_(0);
    hierarchy_.levels.clear();
    return boxes;
}

// the particles refined into a level (new or regridded) are appended unsorted: SoAVXChunked lays
//  them out in chunks here, so the level's first pushes take the per cell kernels too (they
//  need sorted arrays)
template<auto opts>
void Simulator<opts>::sort_refined_particles_(std::size_t const ln)
{
    if constexpr (opts.layout_mode == core::LayoutMode::SoAVXChunked)
        per_patch_(ln, [](auto& patch) {
            for (auto& pop : patch.ions)
                for (auto* ps : {&pop.domainParticles(), &pop.levelGhostParticles(),
                                 &pop.levelGhostParticlesOld()})
                    if (!ps->sorted())
                        ps->sort();
        });
}

template<auto opts>
void Simulator<opts>::refined_moments_(std::size_t const ln)
{
    // as the ions moved (solver/hybrid/advance.hpp): the domain particles, the level ghosts (each
    // held by one patch), then the
    //  border sums
    per_patch_(ln, [&](auto& patch) {
        core::deposit_domain<dimension, interp_order>(patch.ions, patch.layout);
        core::depositParticles(patch.ions, patch.layout, Interpolator{}, core::LevelGhostDeposit{});
    });
    solver::hybrid::fill_ion_borders(level_services_(ln));
    solver::hybrid::compute_moments(level_services_(ln));

    // J from B; E was refined
    per_patch_(ln,
               [&](auto& patch) { core::AmpereTransformer{}(patch.layout, patch.em.B, patch.J); });
}

} // namespace plamn

#endif // PLAMN_SIMULATOR_SIMULATOR_LEVELS_HPP
