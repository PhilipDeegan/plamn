#ifndef PLAMN_SIMULATOR_SIMULATOR_LEVELS_HPP
#define PLAMN_SIMULATOR_SIMULATOR_LEVELS_HPP

// plamn::Simulator: making and initializing levels (PHARE's HybridLevelInitializer)
// IWYU pragma: private, include "plamn/simulator/simulator.hpp"

#include "mkn/kul/env.hpp"
#include "plamn/amr/untiled.hpp"
#include "damn/amr/copy.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/level_map.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/amr/decompose.hpp"
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
// IWYU pragma: private, include "plamn/simulator/simulator.hpp"

namespace plamn
{
template<auto opts>
damn::amr::Decompose<Simulator<opts>::dimension>
Simulator<opts>::decompose_in_(std::string const& partition)
{
    auto weigh = [this](int const l, std::vector<Box_t> const& patches) {
        if constexpr (Models_t::hybrid_on)
            if (!models_.is_mhd(static_cast<std::size_t>(l)))
                return isInitialized_ ? particle_weights_(l, patches) : box_weights_(l, patches);
        std::vector<double> cells; // MHD: the work is per cell
        for (auto const& p : patches)
            cells.push_back(static_cast<double>(p.size()));
        return cells;
    };
    // the level 0 cells about level 1 its coarse step reads: where its level ghost particles
    //  split from (a particle ghost width, in coarse cells at most); running level 0's far part
    //  out of order would need the stages' ghost exchanges on top
    int near_band = lb_.near_band;
    if constexpr (Models_t::hybrid_on)
        if (near_band < 0)
            near_band = static_cast<int>(solver::HybridModel<opts>::Patch_t::particle_ghost_width);
    return {.topology   = topology_,
            .weigh      = weigh,
            .patch_size = patch_size_,
            .partition  = partition,
            .spill      = lb_.spill,
            .block      = rebalance_block,
            .near_band  = std::max(near_band, 0)};
}

// level ln from its boxes (that level's cells): planned_[ln] if decomposed with the others
// (initialization, rebalance: finest / aligned), else damn::amr::decompose_level (lb_.partition,
// lb_.spill; finest / aligned, a level alone: to the owners of the coarser cells under it),
// weighed by the particles each patch will hold (MHD levels: cells); refined levels' patches in
// whole coarse cells, so their tiles are even, starting on even indices (tile_set_mapper.hpp)
template<auto opts>
damn::amr::Level<Simulator<opts>::dimension>
Simulator<opts>::level_of_(int const ln, std::vector<Box_t> const& boxes)
{
    auto const l = static_cast<std::size_t>(ln);
    if (l < planned_.size())
    {
        std::size_t planned = 0, given = 0;
        for (auto const& p : planned_[l].patches)
            planned += p.box.size();
        for (auto const& b : boxes)
            given += b.size();
        if (planned != given)
            throw std::runtime_error("plamn Simulator: level " + std::to_string(ln)
                                     + " planned on other boxes");
        return planned_[l];
    }
    return damn::amr::decompose_level(hierarchy_, ln, boxes, decompose_in_(lb_.partition));
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
    resize_levels_(static_cast<std::size_t>(ln) + 1);
    models_.visit(static_cast<std::size_t>(ln), [&](auto& model) {
        model.levels[ln] = damn::amr::allocate(hierarchy_.levels[ln], rank_, [&](auto const& p) {
            return make_patch_(model, ln, p.box);
        });
    });
    link_level_(ln);
}

template<auto opts>
auto Simulator<opts>::layout_of_(int const ln, Box_t const& box) const
{
    using Layout_t = solver::HybridModel<opts>::Patch_t::Layout_t;
    return Layout_t{box, damn::amr::mesh_size(hierarchy_, ln),
                    damn::amr::position(hierarchy_, ln, box.lower), ln};
}

template<auto opts>
auto Simulator<opts>::make_patch_(auto const& model, int const ln, Box_t const& box) const
{
    using Model_t = std::remove_cvref_t<decltype(model)>;
    if constexpr (solver::is_mhd_model_v<Model_t>)
    {
        using Patch_t  = Model_t::Patch_t;
        using Layout_t = Patch_t::Layout_t;
        auto patch     = std::make_unique<Patch_t>(
            Layout_t{box, damn::amr::mesh_size(hierarchy_, ln),
                     damn::amr::position(hierarchy_, ln, box.lower), ln},
            model.n_states());
        // the finest MHD level, a hybrid level over it (fixed by the index: max_mhd_level)
        if constexpr (Models_t::hybrid_on)
            if (ln + 1 < static_cast<int>(hierarchy_.max_levels) and !models_.is_mhd(ln + 1))
                simulator::make_under_hybrid(*patch);
        return patch;
    }
    else
        return solver::hybrid::make_patch(model, layout_of_(ln, box), dict_["simulation"],
                                          solver::hybrid::surroundings(model, hierarchy_, ln, box));
}

template<auto opts>
void Simulator<opts>::link_level_(int const ln)
{
    if constexpr (Models_t::mhd_on)
        if (models_.is_mhd(static_cast<std::size_t>(ln)))
            return link_mhd_level_(ln);
    if constexpr (Models_t::hybrid_on)
        link_hybrid_level_(ln);
}

// the level's neighbours, as far as its fields reach outside a patch: ghosts plus primal nodes;
//  to the coarser level, as far as a refined ghost's stencil reaches
template<auto opts>
void Simulator<opts>::link_mhd_level_(int const ln)
{
    auto constexpr gw = solver::MHDModel<opts>::Patch_t::field_ghost_width;
    link_geometry_(ln, gw + 1, gw + 1 + 4, gw);
}

template<auto opts>
void Simulator<opts>::link_hybrid_level_(int const ln)
{
    using Patch_t     = solver::HybridModel<opts>::Patch_t;
    using Splitter_t  = solver::HybridModel<opts>::Splitter_t;
    auto const pgw    = Patch_t::particle_ghost_width;
    auto constexpr gw = Patch_t::field_ghost_width;
    // the level's neighbours, as far as any resource reaches outside its patch: field ghosts
    // plus primal nodes, or particle ghosts; to the coarser level: as far as a refined field
    // ghost's stencil, or a split particle's parent, reaches
    auto constexpr reach = std::max<int>(gw + 1, pgw);
    link_geometry_(ln, reach, reach + 2 * amr::splitBoxGrowth<Splitter_t>() + 4, gw);
}

template<auto opts>
void Simulator<opts>::link_geometry_(int const ln, int const reach, int const coarser_reach,
                                     int const gw)
{
    auto& l      = levels_[ln];
    ++l.generation; // a new decomposition: what is kept per decomposition is stale
    l.neighbours = damn::amr::make_neighbours(hierarchy_, ln, rank_, topology_, reach);
    if (ln == 0)
        return;
    l.to_coarser = damn::amr::make_level_map(hierarchy_, ln - 1, rank_, topology_, coarser_reach);
    l.level_ghost_cells.clear();
    models_.visit(static_cast<std::size_t>(ln), [&](auto& model) {
        for (auto const& p : model.levels[ln].patches)
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
    });
}

template<auto opts>
void Simulator<opts>::initialize()
{
    DAMN_LOG_SCOPE(1, "Simulator::initialize");
    if (isInitialized_)
        throw std::runtime_error("plamn Simulator already initialized");
    auto const init = [&] {
        if constexpr (Models_t::mhd_on)
            if (models_.mhd_levels > 0)
                return initialize_mhd_();
        if constexpr (Models_t::hybrid_on)
            initialize_hybrid_();
    };
    init();
    record_decomposition_("init", currentTime_, /*always=*/true);
}

// level 0 from the user's functions (MHD alone, one level: MHD refinement and coupling to hybrid
//  levels not yet)
template<auto opts>
void Simulator<opts>::initialize_mhd_()
{
    if (!restarts_.load_path.empty()) // pyphare's restart_time: the levels from the file
        load_restart_();
    else
    {
        make_level_(0, {hierarchy_.domain});
        MHDLevel level{*this, 0};
        level.per_patch([&](auto& patch) {
            solver::mhd::initialize_state(models_.mhd, patch, dict_["simulation"]);
        });
        level.fill_state_ghosts(0, startTime_);
        // a face on a periodic boundary is initialized twice (at 0 and at L): every patch
        //  holding it takes the max of its copies, once, after the ghost fill (PHARE's
        //  MHDMessenger fillRootGhosts)
        level.copy(damn::amr::max_borders, "MHD_B");
        // the refinement boxes' levels, refined (PHARE's MHDMessenger initLevel)
        for (std::size_t ln = 1; ln <= refinement_boxes_.size(); ++ln)
        {
            make_level_(static_cast<int>(ln), refinement_boxes_[ln - 1]);
            initialize_refined_(ln, startTime_);
        }
        // tagging: each new level from the tags of the one under it, as long as some are (or
        //  from the supplied tag boxes); pretag is a hybrid one (B before any particle): the
        //  MHD state is already there to tag on
        if (tagging_)
            for (int ln = 0; ln + 1 < hierarchy_.max_levels; ++ln)
            {
                bool const supplied = !supplied_tag_boxes_.empty();
                if (supplied and static_cast<std::size_t>(ln) >= supplied_tag_boxes_.size())
                    break;
                auto const boxes = supplied ? supplied_tag_boxes_[ln] : cluster_(ln);
                if (boxes.empty())
                    break;
                clustered_.resize(static_cast<std::size_t>(ln) + 2);
                clustered_[ln + 1] = boxes;
                make_level_(ln + 1, boxes);
                initialize_refined_(static_cast<std::size_t>(ln) + 1, startTime_);
            }
        // every level's old state at t0, for the finer levels' time interpolation
        for (std::size_t ln = 0; ln < levels_.size(); ++ln)
            prepare_step_(static_cast<int>(ln), startTime_);
        currentTime_ = startTime_;
    }
    restart_stepping_();
    start_timestepper_();
    dt_            = timestepper_.next([&](auto const& cfl) { return stable_dt_(cfl); });
    isInitialized_ = true;
}

template<auto opts>
void Simulator<opts>::initialize_refined_mhd_(std::size_t const ln, double const t)
{
    using damn::amr::LevelRegion;
    std::vector<damn::amr::LevelPair> pairs{{"rho", "rho", ""}, {"Etot", "Etot", ""}};
    for (auto const& p : amr::vector_pairs("rhoV", "rhoV"))
        pairs.push_back(p);
    refine_(ln, amr::FieldRefiner{}, LevelRegion::Interior, pairs);
    refine_(ln, amr::FieldRefiner{}, LevelRegion::Interior, amr::vector_pairs("B", "B"));
    auto& ld      = models_.mhd.levels[static_cast<int>(ln)];
    auto const dx = damn::amr::mesh_size(hierarchy_, static_cast<int>(ln));
    using Patch_t = solver::MHDModel<opts>::Patch_t;
    for (std::size_t i = 0; i < damn::amr::size(ld); ++i)
    {
        amr::with_untiled_resources<Patch_t::field_ghost_width>(
            ld.data[i]->states[0].B.grids, [&](auto const& B) { // tiled: untiled copies
                amr::divergence_free_touch_up(B, ld.patches[i].box, ld.patches[i].box, dx);
            });
    }
    MHDLevel{*this, ln}.fill_state_ghosts(0, t);
}

template<auto opts>
void Simulator<opts>::initialize_refined_(std::size_t const ln, double const t)
{
    if (models_.is_mhd(ln))
        return initialize_refined_mhd_(ln, t);
    if constexpr (Models_t::hybrid_on)
    {
        if (models_.is_mhd(ln - 1))
            return initialize_hybrid_over_mhd_(ln, t);
        auto level = level_services_(ln);
        solver::hybrid::initialize_refined(models_.hybrid, level, t);
        // the initial hierarchy: the closure's state from the input, as the root's
        solver::hybrid::seed_level(models_.hybrid, level, dict_["simulation"]);
    }
}

template<auto opts>
void Simulator<opts>::mhd_prims_over_(std::size_t const ln, bool const interior)
{
    if constexpr (Models_t::hybrid_on and Models_t::mhd_on)
    {
        using damn::amr::LevelRegion;
        MHDLevel{*this, ln - 1}.per_patch([&](auto& patch) {
            auto& u = patch.states[0];
            core::PrimitiveConservativeTransformer{models_.mhd.gamma}.to_primitive(
                patch.layout, u.rho, u.rhoV, u.B, u.Etot, u.V, u.P);
        });
        for (auto& patch : models_.hybrid.levels[static_cast<int>(ln)].data)
            simulator::coupling::make_prim_scratch(*patch);
        if (interior)
            refine_(ln, amr::CellInjector{}, LevelRegion::Interior,
                    simulator::coupling::prim_pairs());
        refine_(ln, amr::CellInjector{}, LevelRegion::LevelGhosts,
                simulator::coupling::prim_pairs());
    }
}

template<auto opts>
void Simulator<opts>::spawn_level_ghosts_(std::size_t const ln, bool const new_ghosts,
                                          std::uint64_t const salt)
{
    if constexpr (Models_t::hybrid_on and Models_t::mhd_on)
    {
        using Patch_t     = solver::HybridModel<opts>::Patch_t;
        auto const pgw    = Patch_t::particle_ghost_width;
        auto const domain = damn::amr::level_domain(hierarchy_, static_cast<int>(ln));
        double const Te   = core::hybrid::coupling_Te(*models_.hybrid.electron_closure);
        auto& ld = models_.hybrid.levels[static_cast<int>(ln)];
        for (std::size_t i = 0; i < damn::amr::size(ld); ++i)
        {
            auto const& cell = ld.patches[i].box;
            static thread_local std::vector<Box_t> ghosts, rest; // used up per patch
            ghosts.assign(1, damn::core::grow(cell, mkn::kul::ConstArray<int, dimension>(pgw)));
            for (std::size_t d = 0; d < dimension; ++d)
                if (!hierarchy_.periodic[d])
                {
                    ghosts[0].lower[d] = std::max(ghosts[0].lower[d], domain.lower[d]);
                    ghosts[0].upper[d] = std::min(ghosts[0].upper[d], domain.upper[d]);
                }
            for (auto const& covered :
                 damn::amr::covered_by_level(hierarchy_, static_cast<int>(ln), ghosts.front()))
            {
                rest.clear();
                for (auto const& g : ghosts)
                    for (auto const& r : g.remove(covered))
                        rest.push_back(r);
                std::swap(ghosts, rest);
            }
            simulator::coupling::spawn(
                *ld.data[i], ghosts,
                [&](auto& pop) -> auto& {
                    return new_ghosts ? pop.levelGhostParticlesNew() : pop.levelGhostParticlesOld();
                },
                Te, salt);
        }
        solver::hybrid::own_level_ghosts(level_services_(ln), new_ghosts);
    }
}

template<auto opts>
void Simulator<opts>::initialize_hybrid_over_mhd_(std::size_t const ln, double const t)
{
    if constexpr (Models_t::hybrid_on and Models_t::mhd_on)
    {
        using Patch_t = solver::HybridModel<opts>::Patch_t;
        // the ions' internal energy is (3/2) Pi: MHD's (gamma - 1)^-1 P agrees with it at 5/3 only
        if (std::abs(models_.mhd.gamma - 5. / 3.) > 1e-12)
            throw std::runtime_error("plamn coupling: MHD under hybrid levels needs gamma 5/3");
        double const Te = core::hybrid::coupling_Te(*models_.hybrid.electron_closure);

        mhd_prims_over_(ln, /*interior=*/true);
        auto level = level_services_(ln);
        solver::hybrid::refine_magnetic(models_.hybrid, level); // MHD's B (coarse_pairs_)
        level.fill_magnetic_ghosts("EM_B", t);

        auto& ld = models_.hybrid.levels[static_cast<int>(ln)];
        for (std::size_t i = 0; i < damn::amr::size(ld); ++i)
            simulator::coupling::spawn(
                *ld.data[i], {ld.patches[i].box},
                [](auto& pop) -> auto& { return pop.domainParticles(); }, Te,
                simulator::coupling::Salt::domain);
        spawn_level_ghosts_(ln, /*new_ghosts=*/false, simulator::coupling::Salt::ghosts_at_init);
        level.per_patch([](auto& patch) {
            for (auto& pop : patch.ions)
                pop.levelGhostParticles() = pop.levelGhostParticlesOld();
        });
        solver::hybrid::sort_refined_particles(models_.hybrid, level);
        solver::hybrid::refined_moments(models_.hybrid, level);

        // E from Ohm's law over the moments, its level ghosts MHD's
        solver::hybrid::ampere_ohm<&Patch_t::em>(models_.hybrid, level);
        level.fill_electric_ghosts("EM_E", t);
    }
}

template<auto opts>
void Simulator<opts>::initialize_hybrid_()
{
    // the steps from the start (or the restart's time), the first dt (adaptive: from the
    //  initialized levels, as PHARE's for its first dump)
    auto const start_steps = [&]() {
        restart_stepping_();
        start_timestepper_();
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
    if (tagging_ and !supplied_tag_boxes_.empty())
        pretag_boxes = supplied_tag_boxes_;
    else if (tagging_ and pretag_enabled_)
        pretag_boxes = pretag_();

    // every level known (pretag, tag_boxes) and a partition that needs them all: decomposed
    //  together before any level is made, on the particles they will hold (none exist yet)
    if (pretag_boxes and refinement_boxes_.empty()
        and (lb_.partition == "finest" or lb_.partition == "aligned"))
    {
        std::vector<std::vector<Box_t>> boxes{{hierarchy_.domain}};
        for (auto const& b : *pretag_boxes)
        {
            if (b.empty() or static_cast<int>(boxes.size()) >= hierarchy_.max_levels)
                break;
            boxes.push_back(b);
        }
        planned_ = damn::amr::decompose(hierarchy_, 0, boxes, decompose_in_(lb_.partition));
    }

    make_level_(0, {hierarchy_.domain});
    solver::hybrid::initialize_root(models_.hybrid, level_services_(0), dict_["simulation"]);
    for (std::size_t ln = 1; ln <= refinement_boxes_.size(); ++ln)
    {
        make_level_(static_cast<int>(ln), refinement_boxes_[ln - 1]);
        auto level = level_services_(ln);
        solver::hybrid::initialize_refined(models_.hybrid, level, startTime_);
        solver::hybrid::seed_level(models_.hybrid, level, dict_["simulation"]); // the closure's
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
            auto level = level_services_(static_cast<std::size_t>(ln) + 1);
            solver::hybrid::initialize_refined(models_.hybrid, level, startTime_);
            solver::hybrid::seed_level(models_.hybrid, level, dict_["simulation"]);
        }
    planned_.clear(); // made: later levels (regrids) level by level
    // the totals under a finer level the finer one's from the start, as after a synchronization
    //  (PLAMN_INIT_SYNC=0: each level's own deposit until the first synchronization, as PHARE
    //  master's initialization with refinement boxes: for comparisons)
    static bool const init_sync = mkn::kul::env::get_as("PLAMN_INIT_SYNC", 1) != 0;
    if (init_sync)
        rebuild_moments_(0);
    // every level's E and B at t0, for the finer levels' time interpolation
    for (std::size_t ln = 0; ln < levels_.size(); ++ln)
        prepare_step_(static_cast<int>(ln), startTime_);

    currentTime_ = startTime_;
    start_steps();
    isInitialized_ = true;
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
        solver::hybrid::refine_magnetic(models_.hybrid,
                                        level_services_(static_cast<std::size_t>(ln) + 1));
        fill_magnetic_ghosts_(ln + 1, "EM_B", startTime_);
        boxes.push_back(std::move(level_boxes));
    }
    resize_levels_(0);
    hierarchy_.levels.clear();
    return boxes;
}

} // namespace plamn

#endif // PLAMN_SIMULATOR_SIMULATOR_LEVELS_HPP
