#ifndef PLAMN_SIMULATOR_SIMULATOR_LEVELS_HPP
#define PLAMN_SIMULATOR_SIMULATOR_LEVELS_HPP

// plamn::Simulator: making and initializing levels (PHARE's HybridLevelInitializer)
// IWYU pragma: private, include "plamn/simulator/simulator.hpp"

#include "plamn/amr/untiled.hpp"
#include "damn/amr/copy.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/level_map.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/amr/partition.hpp"
#include "damn/amr/keep_partition.hpp"
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
// level ln from its boxes (that level's cells): chopped into the largest patch size, given
// ranks (lb_.partition: along the Hilbert curve by the particles they will hold, or where the
// data is, then spilled), allocated, with their neighbours and the map to the coarser level
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
    auto const chop = [&](std::vector<Box_t> const& bs) {
        std::vector<Box_t> patches;
        for (auto const& box : bs)
            for (auto const& b : damn::amr::chop_evenly(box, size, ln > 0 ? hierarchy_.ratio : 1))
                patches.push_back(b);
        return patches;
    };
    auto const weights = [&](std::vector<Box_t> const& patches) {
        if constexpr (Models_t::hybrid_on)
            if (!models_.is_mhd(static_cast<std::size_t>(ln)))
                return isInitialized_ ? particle_weights_(ln, patches) : box_weights_(ln, patches);
        std::vector<double> cells; // MHD: the work is per cell
        for (auto const& p : patches)
            cells.push_back(static_cast<double>(p.size()));
        return cells;
    };
    auto const capacity = damn::amr::node_share_capacity(topology_);
    auto const curve    = [&](std::vector<Box_t> const& patches) {
        return damn::amr::partition(patches, weights(patches), capacity);
    };

    auto const l          = static_cast<std::size_t>(ln);
    auto const* const old = l < hierarchy_.levels.size() ? &hierarchy_.levels[l].patches : nullptr;
    auto const* const coarser = ln > 0 ? &hierarchy_.levels[l - 1].patches : nullptr;
    // a refined level's patches in whole coarse cells (lower and shape multiples of the
    //  ratio): its tiles are then even, starting on even indices (tile_set_mapper.hpp)
    auto const whole_coarse = [&](std::vector<Box_t> const& patches) {
        if (ln == 0)
            return;
        auto const r = static_cast<int>(hierarchy_.ratio);
        for (auto const& p : patches)
            for (std::size_t d = 0; d < dimension; ++d)
                if (p.lower[d] % r != 0 or p.shape()[d] % r != 0)
                    throw std::runtime_error("plamn Simulator: a level " + std::to_string(ln)
                                             + " patch not in whole coarse cells");
    };
    if (lb_.partition != "keep" or (!old and !coarser))
    {
        auto const patches = chop(boxes);
        whole_coarse(patches);
        return damn::amr::make_level(ln, patches, curve(patches));
    }

    using Owned   = damn::amr::OwnedBox<dimension>;
    using Patches = std::vector<damn::amr::Patch<dimension>>;
    auto const pieces
        = damn::amr::keep(boxes, old ? *old : Patches{}, coarser ? *coarser : Patches{},
                          hierarchy_.ratio, [&](std::vector<Box_t> const& rest) {
                              auto const patches = chop(rest);
                              auto const ranks   = curve(patches);
                              std::vector<Owned> out;
                              for (std::size_t i = 0; i < patches.size(); ++i)
                                  out.push_back({patches[i], ranks[i]});
                              return out;
                          });
    std::vector<Box_t> patches;
    std::vector<int> ranks;
    auto const per_rank = damn::amr::owned_boxes(pieces, n_ranks_);
    for (std::size_t r = 0; r < per_rank.size(); ++r)
        for (auto const& b : chop(per_rank[r]))
        {
            patches.push_back(b);
            ranks.push_back(static_cast<int>(r));
        }
    if (lb_.spill >= 0)
        ranks = damn::amr::spill(patches, ranks, weights(patches), topology_, lb_.spill);
    whole_coarse(patches);
    return damn::amr::make_level(ln, patches, ranks);
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
        return std::make_unique<Patch_t>(Layout_t{box, damn::amr::mesh_size(hierarchy_, ln),
                                                  damn::amr::position(hierarchy_, ln, box.lower),
                                                  ln},
                                         model.n_states());
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
    timestepper_.start_at(startTime_);
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
        solver::hybrid::initialize_refined(models_.hybrid, level_services_(ln), t);
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
        double const Te
            = dict_["simulation"]["electrons"]["pressure_closure"]["Te"].template to<double>();
        auto& ld = models_.hybrid.levels[static_cast<int>(ln)];
        for (std::size_t i = 0; i < damn::amr::size(ld); ++i)
        {
            auto const& cell = ld.patches[i].box;
            std::vector<Box_t> ghosts{
                damn::core::grow(cell, mkn::kul::ConstArray<int, dimension>(pgw))};
            for (std::size_t d = 0; d < dimension; ++d)
                if (!hierarchy_.periodic[d])
                {
                    ghosts[0].lower[d] = std::max(ghosts[0].lower[d], domain.lower[d]);
                    ghosts[0].upper[d] = std::min(ghosts[0].upper[d], domain.upper[d]);
                }
            for (auto const& covered :
                 damn::amr::covered_by_level(hierarchy_, static_cast<int>(ln), ghosts.front()))
            {
                std::vector<Box_t> rest;
                for (auto const& g : ghosts)
                    for (auto const& r : g.remove(covered))
                        rest.push_back(r);
                ghosts = std::move(rest);
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
        double const Te
            = dict_["simulation"]["electrons"]["pressure_closure"]["Te"].template to<double>();

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
    if (tagging_ and !supplied_tag_boxes_.empty())
        pretag_boxes = supplied_tag_boxes_;
    else if (tagging_ and pretag_enabled_)
        pretag_boxes = pretag_();

    make_level_(0, {hierarchy_.domain});
    solver::hybrid::initialize_root(models_.hybrid, level_services_(0), dict_["simulation"]);
    for (std::size_t ln = 1; ln <= refinement_boxes_.size(); ++ln)
    {
        make_level_(static_cast<int>(ln), refinement_boxes_[ln - 1]);
        solver::hybrid::initialize_refined(models_.hybrid, level_services_(ln), startTime_);
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
            solver::hybrid::initialize_refined(
                models_.hybrid, level_services_(static_cast<std::size_t>(ln) + 1), startTime_);
        }
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
