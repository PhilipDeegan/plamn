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
    solver::hybrid::initialize_root(hybrid_, level_services_(0), dict_["simulation"]);
    for (std::size_t ln = 1; ln <= refinement_boxes_.size(); ++ln)
    {
        make_level_(static_cast<int>(ln), refinement_boxes_[ln - 1]);
        solver::hybrid::initialize_refined(hybrid_, level_services_(ln), startTime_);
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
                hybrid_, level_services_(static_cast<std::size_t>(ln) + 1), startTime_);
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
        solver::hybrid::refine_magnetic(hybrid_, level_services_(static_cast<std::size_t>(ln) + 1));
        fill_magnetic_ghosts_(ln + 1, "EM_B", startTime_);
        boxes.push_back(std::move(level_boxes));
    }
    resize_levels_(0);
    hierarchy_.levels.clear();
    return boxes;
}

} // namespace plamn

#endif // PLAMN_SIMULATOR_SIMULATOR_LEVELS_HPP
