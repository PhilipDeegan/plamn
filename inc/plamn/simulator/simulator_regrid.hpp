#ifndef PLAMN_SIMULATOR_SIMULATOR_REGRID_HPP
#define PLAMN_SIMULATOR_SIMULATOR_REGRID_HPP

// plamn::Simulator: tagging, clustering, regrid, load balancing
// IWYU pragma: private, include "plamn/simulator/simulator.hpp"

#include "plamn/amr/untiled.hpp"
#include "damn/amr/clustering.hpp"
#include "damn/amr/copy.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/level_map.hpp"
#include "damn/amr/decompose.hpp"
#include "damn/amr/regrid.hpp"
#include "damn/amr/topology.hpp"
#include "damn/core/def.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/logger.hpp"
#include "mkn/kul/std/array.hpp"
#include "damn/mpi/clustering.hpp"
#include "damn/mpi/regrid.hpp"

#include "plamn/simulator/simulator.hpp" // IWYU pragma: keep (the class)
#include "plamn/amr/level_pairs.hpp"
#include "plamn/amr/load_balancing.hpp"
#include "plamn/amr/masked.hpp"
#include "plamn/amr/refine.hpp"
#include "plamn/amr/resources.hpp"
#include "plamn/amr/tagging.hpp"

#include "mpi.h"
#include "phlop/timing/scope.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
// plamn::Simulator: tagging, clustering, regrid, load balancing
// IWYU pragma: private, include "plamn/simulator/simulator.hpp"


namespace plamn
{
template<auto opts>
std::vector<typename Simulator<opts>::Box_t> Simulator<opts>::cluster_(int const ln)
{
    DAMN_LOG_SCOPE(1, "Simulator::cluster_");
    auto const dx = damn::amr::mesh_size(hierarchy_, ln);
    std::vector<damn::core::Point<int, dimension>> tagged;
    models_.visit(static_cast<std::size_t>(ln), [&](auto& model) {
        using Model_t   = std::remove_cvref_t<decltype(model)>;
        using Patch_t   = Model_t::Patch_t;
        using Resource  = amr::FieldResource<typename resource_field_<Patch_t>::type,
                                             Patch_t::field_ghost_width>;
        bool const mhd  = solver::is_mhd_model_v<Model_t>;
        auto const& tag = mhd ? *mhd_tagging_ : *tagging_;
        auto& ld        = model.levels[ln];
        std::vector<std::string> tagged_on; // tiled fields: read on untiled temporaries
        for (auto const& q : tag.quantities)
            for (auto const& component : q.components)
                tagged_on.push_back(mhd ? component : "EM_" + component);
        Leases leases;
        leases.add(ld, tagged_on, amr::Lease::read);
        for (std::size_t i = 0; i < damn::amr::size(ld); ++i)
        {
            auto const resource = [&](std::string const& component) {
                auto const name = mhd ? component : "EM_" + component; // hybrid: EM_B_x...
                std::optional<Resource> r;
                damn::amr::with_resource(ld.data[i], name, [&](auto& x) {
                    if constexpr (std::is_same_v<std::decay_t<decltype(x)>, Resource>)
                        r = x;
                });
                if (!r)
                    throw std::runtime_error("plamn Simulator: no field " + name + " to tag on");
                return *r;
            };
            for (auto const& c : amr::tagged_cells(tag, ld.patches[i].box, dx, resource))
                tagged.push_back(c);
        }
    });
    if (n_ranks_ > 1)
        return damn::mpi::cluster(hierarchy_, ln, tagged, clustering_);
    return damn::amr::cluster(
        hierarchy_, ln, damn::amr::tagged_tiles(hierarchy_, ln, tagged, clustering_), clustering_);
}

template<auto opts>
std::vector<double> Simulator<opts>::particle_weights_(int const ln,
                                                       std::vector<Box_t> const& boxes)
{
    DAMN_LOG_SCOPE(1, "Simulator::particle_weights_");
    std::vector<double> weights(boxes.size(), 0.);
    if (lb_.mode == "homogeneous")
    {
        for (std::size_t b = 0; b < boxes.size(); ++b)
            weights[b] = static_cast<double>(boxes[b].size());
        return weights;
    }

    auto const fine = static_cast<std::size_t>(ln);
    // the level's own particles, where it is
    if (fine < levels_.size())
        solver::hybrid::add_particles(models_.hybrid, level_services_(fine), boxes, weights);
    // where it is not (yet): the coarser level's particles, split
    if (ln > 0)
        solver::hybrid::add_split_particles(
            models_.hybrid, level_services_(fine - 1), boxes, hierarchy_.ratio,
            fine < levels_.size() ? hierarchy_.levels[ln].patches
                                  : std::vector<damn::amr::Patch<dimension>>{},
            weights);
    if (n_ranks_ > 1)
        MPI_Allreduce(MPI_IN_PLACE, weights.data(), static_cast<int>(weights.size()), MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
    return weights;
}

template<auto opts>
bool Simulator<opts>::should_rebalance_()
{
    if (!lb_.active or !tagging_ or n_ranks_ == 1)
        return false;
    if (!lb_.automatic)
        return (steps_ == 0 and lb_.on_init)
               or (lb_.every > 0 and steps_ > 0 and steps_ % static_cast<int>(lb_.every) == 0);
    return (*rebalance_check_)([&] {
        DAMN_LOG_SCOPE(1, "Simulator::should_rebalance_::automatic");
        // every rank's level 0 workload
        double const mine = lb_.mode == "homogeneous" ? [&] {
            double cells = 0;
            for (auto const& p : data_(0).patches)
                cells += static_cast<double>(p.box.size());
            return cells;
        }()
                                                      : static_cast<double>(particle_count(0));
        std::vector<double> all(static_cast<std::size_t>(n_ranks_));
        MPI_Allgather(&mine, 1, MPI_DOUBLE, all.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);
        return all;
    });
}

// after a regrid or a rebalance: every level's totals from the finest down to `lowest` (the
//  coarsest whose finer level may have changed) from its own populations, then coarsened from
//  the finer level as a synchronization does, then their ghosts: under a finer level the
//  totals are the finer level's (PHARE's chargeDensitySynchronizers_ after a regrid). The
//  populations' moments are deposits of the particles, moved or split (regrid_hybrid_data_)
template<auto opts>
void Simulator<opts>::rebuild_moments_(int const lowest)
{
    if constexpr (Models_t::hybrid_on)
        for (auto ln = static_cast<int>(levels_.size()) - 1; ln >= std::max(lowest, 0); --ln)
        {
            auto const l = static_cast<std::size_t>(ln);
            if (models_.is_mhd(l))
                continue;
            solver::hybrid::compute_moments<opts>(level_services_(l));
            if (l + 1 < levels_.size() and !models_.is_mhd(l + 1))
            { // with the closure's part (polytropic: Pe), as a synchronization
                auto finer = level_services_(l + 1);
                solver::hybrid::coarsen_moments(finer);
                solver::hybrid::synchronize_level(models_.hybrid, finer);
            }
            copy_(ln, damn::amr::fill_ghosts, "IonMoments");
        }
}

// level 0 partitioned again by the particles it holds: the domain's patches (as at setup) cut
// on blocks where a rank's share ends inside one (damn::amr::decompose_level, "levels"), so
// level 0's boxes may change; its data moved to the new owners, its ghosts and moments
// refilled; then each finer level the same (regrid_level_ onto its own boxes: nothing new to
// refine)
template<auto opts>
void Simulator<opts>::rebalance_(double const t)
{
    DAMN_LOG_SCOPE(1, "Simulator::rebalance_");

    // level 0 afresh ("levels": cut where a rank's share ends in a patch), by its particles
    auto const weigh = [this](int const ln, std::vector<Box_t> const& boxes) {
        return particle_weights_(ln, boxes);
    };
    damn::amr::Decompose<dimension> const in{.topology   = topology_,
                                             .weigh      = weigh,
                                             .patch_size = patch_size_,
                                             .partition  = "levels",
                                             .block      = rebalance_block};
    DAMN_SUPER_DEBUG_DO(auto const debug_before = rebalance_state_();)
    auto const old0 = hierarchy_.levels[0];
    // "finest" / "aligned": every level decomposed together (their boxes known: clustered_), on
    //  the particles they hold; the finer levels below take theirs through level_of_
    if (lb_.partition == "finest" or lb_.partition == "aligned")
    {
        std::vector<std::vector<Box_t>> boxes{{hierarchy_.domain}};
        for (std::size_t ln = 1; ln < levels_.size(); ++ln)
            boxes.push_back(clustered_[ln]);
        planned_ = damn::amr::decompose(hierarchy_, 0, boxes, decompose_in_(lb_.partition));
    }
    auto level0 = planned_.empty()
                      ? damn::amr::decompose_level(hierarchy_, 0, {hierarchy_.domain}, in)
                      : planned_[0];

    if (!damn::amr::same_decomposition(old0, level0))
    {
        auto const finer = std::vector<damn::amr::Level<dimension>>(hierarchy_.levels.begin() + 1,
                                                                    hierarchy_.levels.end());
        damn::amr::set_level(hierarchy_, level0); // drops the finer: put back below
        for (auto const& level : finer)
            damn::amr::set_level(hierarchy_, level);

        auto const map = damn::amr::make_regrid_map(old0, hierarchy_.levels[0], rank_,
                                                    damn::amr::periodic_shifts(hierarchy_, 0));
        auto& ld       = data_(0);
        ld             = regrid_leased_(
            map, std::move(ld),
            [&](auto const& p) { return make_patch_(models_.hybrid, 0, p.box); }, /*mpi=*/true);
        for (std::size_t i = 0; i < damn::amr::size(ld); ++i)
        {
            ld.data[i]->rebox(
                solver::hybrid::surroundings(models_.hybrid, hierarchy_, 0, ld.patches[i].box));
            for (auto& pop : ld.data[i]->ions)
                pop.patchGhostParticles().clear();
        }
        link_level_(0);
        fill_magnetic_ghosts_(0, "EM_B", t);
        fill_electric_ghosts_(0, "EM_E", t);
        // the moments deposited again from the moved particles, the totals under level 1
        //  coarsened again after the finer levels (rebuild_moments_)
        solver::hybrid::refined_moments(models_.hybrid, level_services_(0));
        copy_(0, damn::amr::fill_ghosts, "IonPopMoments");
        prepare_step_(0, t);
    }

    // every finer level, coarser to finer, each onto its own boxes, the finer kept: moved if its
    //  owners changed, else only linked to its coarser level again (which may have moved)
    for (std::size_t ln = 1; ln < levels_.size(); ++ln)
    {
        auto const l = static_cast<int>(ln);
        if (damn::amr::same_decomposition(hierarchy_.levels[ln], level_of_(l, clustered_[ln])))
            link_level_(l);
        else
            regrid_level_(l, clustered_[ln], t, /*keep_finer=*/true, /*moved=*/true);
    }
    planned_.clear();
    rebuild_moments_(0);
    DAMN_SUPER_DEBUG_DO(check_rebalance_(debug_before, t);)
    record_decomposition_("rebalance", t);
}


// every level but the finest allowed tagged and clustered again: a finer level whose boxes
// changed is regridded (made, if new; removed, with the finer, if nothing is tagged)
template<auto opts>
void Simulator<opts>::regrid_(double const t, int const from, bool const always)
{
    DAMN_LOG_SCOPE(1, "Simulator::regrid_");
    // coarser to finer: a level regridded keeps its finer levels (keep_finer), each regridded in
    //  turn onto its new boxes, its data moved from the old level where they overlap (SAMRAI's
    //  regrid: the old finer data is not refined again from the new coarser level)
    bool coarser_regridded = false;
    for (int ln = from; ln + 1 < hierarchy_.max_levels and ln < static_cast<int>(levels_.size());
         ++ln)
    {
        auto const boxes = cluster_(ln);
        auto const fine  = static_cast<std::size_t>(ln) + 1;
        clustered_.resize(std::max(clustered_.size(), fine + 1));
        if (fine < levels_.size() and boxes == clustered_[fine] and !always and !coarser_regridded)
            continue;
        if (boxes.empty())
        {
            resize_levels_(fine);
            clustered_.resize(fine);
            damn::amr::remove_finer_levels(hierarchy_, ln);
            break;
        }
        clustered_[fine]  = boxes;
        coarser_regridded = true;
        if (fine < levels_.size())
            regrid_level_(ln + 1, boxes, t, /*keep_finer=*/true);
        else
        {
            make_level_(ln + 1, boxes);
            initialize_refined_level_(ln + 1, startTime_);
            prepare_step_(ln + 1, t);
        }
    }
    rebuild_moments_(from);
    record_decomposition_("regrid", t);
}

// level ln onto new boxes, at t (every level synchronized): its data moved where the old level
// was (damn::amr::regrid), refined from the coarser level elsewhere, as PHARE's
// HybridHybridMessengerStrategy::regrid (moments: copied where the old level was, not deposited
// again, unlike PHARE's); finer levels go, the caller makes them again, unless keep_finer (a
// rebalance or a tagging regrid: the caller regrids each finer level onto its new boxes after
// this one, coarser to finer, which also links them to their new coarser level)
template<auto opts>
void Simulator<opts>::regrid_level_(int const ln, std::vector<Box_t> const& boxes, double const t,
                                    bool const keep_finer, bool const moved)
{
    DAMN_LOG_SCOPE(1, "Simulator::regrid_level_");
    auto const fine = static_cast<std::size_t>(ln);

    // the old level's cells, periodic images included
    std::vector<Box_t> old_cells;
    {
        auto shifts = damn::amr::periodic_shifts(hierarchy_, ln);
        shifts.insert(shifts.begin(), damn::core::Point<int, dimension>{});
        for (auto const& p : hierarchy_.levels[ln].patches)
            for (auto const& shift : shifts)
                old_cells.push_back(damn::core::shift(p.box, shift));
    }
    auto const old_level = hierarchy_.levels[ln];
    auto const finer = keep_finer ? std::vector<damn::amr::Level<dimension>>(
                                        hierarchy_.levels.begin() + ln + 1, hierarchy_.levels.end())
                                  : std::vector<damn::amr::Level<dimension>>{};
    resize_levels_(keep_finer ? std::max(levels_.size(), fine + 1) : fine + 1);
    damn::amr::set_level(hierarchy_, level_of_(ln, boxes)); // drops the finer: put back below
    for (auto const& level : finer)
        damn::amr::set_level(hierarchy_, level);
    bool const hybrid_moved = moved and !models_.is_mhd(fine);
    auto const map          = damn::amr::make_regrid_map(
        old_level, hierarchy_.levels[ln], rank_, damn::amr::periodic_shifts(hierarchy_, ln),
        hybrid_moved ? moved_ghosts_(ln) : damn::amr::RegridGhosts{});

    if (models_.is_mhd(fine))
    {
        if constexpr (Models_t::mhd_on)
            regrid_mhd_data_(ln, map, old_cells, t);
    }
    else if constexpr (Models_t::hybrid_on)
        regrid_hybrid_data_(ln, map, old_cells, t, hybrid_moved);
}

template<auto opts>
damn::amr::RegridGhosts Simulator<opts>::moved_ghosts_(int const ln)
{
    damn::amr::RegridGhosts out;
    if constexpr (Models_t::hybrid_on)
    {
        using Patch_t = solver::HybridModel<opts>::Patch_t;
        out.reach     = std::max(static_cast<int>(Patch_t::field_ghost_width) + 1,
                                 static_cast<int>(Patch_t::particle_ghost_width));
        auto& ld      = data_(static_cast<std::size_t>(ln));
        if (damn::amr::size(ld) > 0)
            simulator::for_each_field_grid(
                *ld.data[0], [&](std::string const& name, auto&) { out.fields.push_back(name); });
        for (auto const& name : models_.hybrid.population_names)
            out.items.push_back(name + "_level_ghost_old");
    }
    return out;
}

template<auto opts>
void Simulator<opts>::regrid_hybrid_data_(int const ln, auto const& map,
                                          std::vector<Box_t> const& old_cells, double const t,
                                          bool const moved)
{
    using Patch_t    = solver::HybridModel<opts>::Patch_t;
    using Splitter_t = solver::HybridModel<opts>::Splitter_t;
    using damn::amr::LevelRegion;
    auto const fine   = static_cast<std::size_t>(ln);
    auto constexpr gw = Patch_t::field_ghost_width;
    auto const make   = [&](auto const& p) { return make_patch_(models_.hybrid, ln, p.box); };
    auto& l           = levels_[fine];
    auto& ld          = data_(fine);
    ld                = regrid_leased_(map, std::move(ld), make);
    for (std::size_t i = 0; i < damn::amr::size(ld); ++i)
        ld.data[i]->rebox(
            solver::hybrid::surroundings(models_.hybrid, hierarchy_, ln, ld.patches[i].box));
    link_level_(ln);
    l.coarse_t  = t;
    l.coarse_dt = 0; // the coarser level's data at t

    if (moved) // a rebalance: no new cells, every field, ghosts included, and the old level
    {          //  ghosts moved; the copies of a level ghost the owner's alone, the moments again
        per_patch_(fine, [](auto& patch) {
            for (auto& pop : patch.ions)
            {
                pop.patchGhostParticles().clear();
                pop.levelGhostParticlesNew().clear();
            }
        });
        solver::hybrid::own_level_ghosts(level_services_(fine), /*new_ghosts=*/false);
        per_patch_(fine, [](auto& patch) {
            for (auto& pop : patch.ions)
                pop.levelGhostParticles() = pop.levelGhostParticlesOld();
        });
        solver::hybrid::sort_refined_particles(models_.hybrid, level_services_(fine));
        solver::hybrid::refined_moments(models_.hybrid, level_services_(fine));
        copy_(ln, damn::amr::fill_ghosts, "IonPopMoments");
        prepare_step_(ln, t);
        return;
    }

    // the new cells from the coarser level: B divergence free, E, the domain particles split
    using MaskedField = amr::Masked<amr::FieldRefiner, dimension>;
    MaskedField const masked_field{{}, &old_cells};
    refine_(fine, masked_field, LevelRegion::Interior, amr::vector_pairs("EM_B", "EM_B"));
    for (std::size_t i = 0; i < damn::amr::size(ld); ++i)
    {
        auto& patch      = *ld.data[i];
        auto const cells = ld.patches[i].box;
        amr::with_untiled_resources<gw>(patch.emB, [&](auto const& B) { // tiled: untiled copies
            for (auto const& region :
                 masked_field.uncovered(cells, [](auto const& b) { return b; }))
                amr::divergence_free_touch_up(B, cells, region,
                                              damn::amr::mesh_size(hierarchy_, ln));
        });
    }
    refine_(fine, masked_field, LevelRegion::Interior, amr::vector_pairs("EM_E", "EM_E"));
    per_patch_(fine, [](auto& patch) {
        for (auto& pop : patch.ions)
        {
            pop.patchGhostParticles().clear();
            pop.levelGhostParticles().clear();
            pop.levelGhostParticlesOld().clear();
            pop.levelGhostParticlesNew().clear();
        }
    });
    using MaskedSplit  = amr::Masked<amr::ParticleSplitter<Splitter_t>, dimension>;
    auto const& routes = *models_.hybrid.particle_routes;
    refine_(fine, MaskedSplit{{}, &old_cells}, LevelRegion::Interior, routes.interior_splits);

    // ghosts: patch ghosts from the new neighbours, level ghosts from the coarser level now
    fill_magnetic_ghosts_(fine, "EM_B", t);
    fill_electric_ghosts_(fine, "EM_E", t);
    refine_(fine, amr::ParticleSplitter<Splitter_t>{}, LevelRegion::LevelGhosts,
            routes.old_ghost_splits);
    solver::hybrid::own_level_ghosts(level_services_(fine), /*new_ghosts=*/false);
    per_patch_(fine, [](auto& patch) {
        for (auto& pop : patch.ions)
            pop.levelGhostParticles() = pop.levelGhostParticlesOld();
    });
    solver::hybrid::sort_refined_particles(models_.hybrid, level_services_(fine));

    // the moments deposited again from the particles moved and split (domain and old level
    //  ghosts: the last deposit of a step's end), the totals under a finer level coarsened again
    //  once the hierarchy is regridded (rebuild_moments_)
    solver::hybrid::refined_moments(models_.hybrid, level_services_(fine));
    copy_(ln, damn::amr::fill_ghosts, "IonPopMoments");
    // the closure's state: its own where the old level was, the coarser level's on the new cells
    auto level = level_services_(fine);
    solver::hybrid::refine_state_level(models_.hybrid, level, masked_field, t);
    prepare_step_(ln, t);
}

// PHARE's MHDMessenger regrid (stateInitRefiners_): the advanced state moved where the old level
//  was, refined from the coarser level elsewhere (B divergence free there), then its ghosts
template<auto opts>
void Simulator<opts>::regrid_mhd_data_(int const ln, auto const& map,
                                       std::vector<Box_t> const& old_cells, double const t)
{
    using damn::amr::LevelRegion;
    using Patch_t     = solver::MHDModel<opts>::Patch_t;
    auto constexpr gw = Patch_t::field_ghost_width;
    auto const fine   = static_cast<std::size_t>(ln);
    auto const make   = [&](auto const& p) { return make_patch_(models_.mhd, ln, p.box); };
    auto& ld          = models_.mhd.levels[ln];
    ld                = regrid_leased_(map, std::move(ld), make);
    link_level_(ln);
    auto& l     = levels_[fine];
    l.coarse_t  = t;
    l.coarse_dt = 0; // the coarser level's data at t

    using MaskedField = amr::Masked<amr::FieldRefiner, dimension>;
    MaskedField const masked{{}, &old_cells};
    std::vector<damn::amr::LevelPair> pairs{{"rho", "rho", ""}, {"Etot", "Etot", ""}};
    for (auto const& p : amr::vector_pairs("rhoV", "rhoV"))
        pairs.push_back(p);
    refine_(fine, masked, LevelRegion::Interior, pairs);
    refine_(fine, masked, LevelRegion::Interior, amr::vector_pairs("B", "B"));
    for (std::size_t i = 0; i < damn::amr::size(ld); ++i)
    {
        auto const cells = ld.patches[i].box;
        amr::with_untiled_resources<gw>(ld.data[i]->states[0].B.grids, [&](auto const& B) {
            for (auto const& region : masked.uncovered(cells, [](auto const& b) { return b; }))
                amr::divergence_free_touch_up(B, cells, region,
                                              damn::amr::mesh_size(hierarchy_, ln));
        });
    }
    MHDLevel{*this, fine}.fill_state_ghosts(0, t);
    prepare_step_(ln, t);
}

// a new level ln (regrid_), from the coarser level
template<auto opts>
void Simulator<opts>::initialize_refined_level_(int const ln, double const t)
{
    auto const fine = static_cast<std::size_t>(ln);
    if (models_.is_mhd(fine))
    {
        if constexpr (Models_t::mhd_on)
            initialize_refined_mhd_(fine, t);
    }
    else if constexpr (Models_t::hybrid_on)
    {
        auto level = level_services_(fine);
        solver::hybrid::initialize_refined(models_.hybrid, level, t);
        // made by a regrid: the closure's state from the coarser level now
        solver::hybrid::refine_state_level(models_.hybrid, level, amr::FieldRefiner{}, t);
    }
}


} // namespace plamn

#endif // PLAMN_SIMULATOR_SIMULATOR_REGRID_HPP
