#ifndef PLAMN_SIMULATOR_SIMULATOR_REGRID_HPP
#define PLAMN_SIMULATOR_SIMULATOR_REGRID_HPP

// plamn::Simulator: tagging, clustering, regrid, load balancing
// IWYU pragma: private, include "plamn/simulator/simulator.hpp"

#include "damn/amr/clustering.hpp"
#include "damn/amr/copy.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/level_map.hpp"
#include "damn/amr/partition.hpp"
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

namespace plamn::simulator::detail
{
// a patch's domain particles per cell, all populations
template<std::size_t dim>
struct CellCounts
{
    damn::core::Box<int, dim> cells;
    std::vector<double> n = std::vector<double>(cells.size(), 0.);

    NO_DISCARD std::size_t index(damn::core::Point<int, dim> const& c) const
    {
        std::size_t k = 0, stride = 1;
        for (std::size_t d = 0; d < dim; ++d)
        {
            k += static_cast<std::size_t>(c[d] - cells.lower[d]) * stride;
            stride *= static_cast<std::size_t>(cells.upper[d] - cells.lower[d] + 1);
        }
        return k;
    }

    // the particles in box's cells (those keep(cell) keeps)
    NO_DISCARD double in(damn::core::Box<int, dim> const& box, auto&& keep) const
    {
        double sum = 0;
        if (auto const overlap = box * cells)
            damn::core::for_each_point(*overlap, [&](auto const& c) {
                if (keep(c))
                    sum += n[index(c)];
            });
        return sum;
    }
    NO_DISCARD double in(damn::core::Box<int, dim> const& box) const
    {
        return in(box, [](auto const&) { return true; });
    }
};

template<typename Patch>
NO_DISCARD auto cell_counts(Patch const& patch)
{
    auto constexpr dim = Patch::dimension;
    CellCounts<dim> counts{patch.layout.cells};
    for (auto const& pop : patch.ions)
    {
        auto const& particles = pop.domainParticles();
        for (std::size_t p = 0; p < particles.size(); ++p)
        {
            damn::core::Point<int, dim> const c{particles[p].iCell()};
            if (damn::core::isIn(c, counts.cells))
                counts.n[counts.index(c)] += 1;
        }
    }
    return counts;
}

// true if coarse cell c is refined into one of the patches (whole coarse cells)
template<std::size_t dim>
NO_DISCARD bool refined_into(damn::core::Point<int, dim> const& c, int const ratio,
                             std::vector<damn::amr::Patch<dim>> const& patches)
{
    auto const fine = c * ratio;
    for (auto const& p : patches)
        if (damn::core::isIn(fine, p.box))
            return true;
    return false;
}
} // namespace plamn::simulator::detail

namespace plamn
{
template<auto opts>
std::vector<typename Simulator<opts>::Box_t> Simulator<opts>::cluster_(int const ln)
{
    DAMN_LOG_SCOPE(1, "Simulator::cluster_");
    using Resource = amr::FieldResource<typename Patch_t::Grid_t, Patch_t::field_ghost_width>;
    auto& ld       = data_(ln);
    auto const dx  = damn::amr::mesh_size(hierarchy_, ln);

    std::vector<damn::core::Point<int, dimension>> tagged;
    for (std::size_t i = 0; i < damn::amr::size(ld); ++i)
    {
        auto const resource = [&](std::string const& name) {
            std::optional<Resource> r;
            damn::amr::with_resource(ld.data[i], name, [&](auto& x) {
                if constexpr (std::is_same_v<std::decay_t<decltype(x)>, Resource>)
                    r = x;
            });
            if (!r)
                throw std::runtime_error("plamn Simulator: no field " + name + " to tag on");
            return *r;
        };
        for (auto const& c : amr::tagged_cells(*tagging_, ld.patches[i].box, dx, resource))
            tagged.push_back(c);
    }
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
        for (auto const& patch : data_(fine).data)
        {
            auto const counts = simulator::detail::cell_counts(*patch);
            for (std::size_t b = 0; b < boxes.size(); ++b)
                weights[b] += counts.in(boxes[b]);
        }
    // where it is not (yet): the coarser level's particles, split
    if (ln > 0)
    {
        auto const old = fine < levels_.size() ? hierarchy_.levels[ln].patches
                                               : std::vector<damn::amr::Patch<dimension>>{};
        for (auto const& patch : data_(fine - 1).data)
        {
            auto const counts = simulator::detail::cell_counts(*patch);
            for (std::size_t b = 0; b < boxes.size(); ++b)
                weights[b]
                    += nbRefinedPart
                       * counts.in(
                           damn::core::coarsen(boxes[b], hierarchy_.ratio), [&](auto const& c) {
                               return !simulator::detail::refined_into(c, hierarchy_.ratio, old);
                           });
        }
    }
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

// level 0 partitioned again by the particles it holds: the domain's patches (as at setup) cut
// on blocks where a rank's share ends inside one (damn::amr::cut_partition), so level 0's boxes
// may change; its data moved to the new owners, its ghosts and moments refilled; then each finer
// level the same (regrid_level_ onto its own boxes: nothing new to refine)
template<auto opts>
void Simulator<opts>::rebalance_(double const t)
{
    DAMN_LOG_SCOPE(1, "Simulator::rebalance_");

    auto const boxes = damn::amr::chop_evenly(hierarchy_.domain, patch_size_); // as level_of_
    auto const block = mkn::kul::ConstArray<int, dimension>(rebalance_block);
    auto const cut = damn::amr::cut_partition(boxes, block,
                                              particle_weights_(0, damn::amr::blocks(boxes, block)),
                                              damn::amr::node_share_capacity(topology_));
    auto const old0 = hierarchy_.levels[0];
    auto level0     = damn::amr::make_level(0, cut.boxes, cut.ranks);

    if (!damn::amr::same_decomposition(old0, level0))
    {
        auto const finer = std::vector<damn::amr::Level<dimension>>(hierarchy_.levels.begin() + 1,
                                                                    hierarchy_.levels.end());
        damn::amr::set_level(hierarchy_, level0); // drops the finer: put back below
        for (auto const& level : finer)
            damn::amr::set_level(hierarchy_, level);

        auto const map  = damn::amr::make_regrid_map(old0, hierarchy_.levels[0], rank_,
                                                     damn::amr::periodic_shifts(hierarchy_, 0));
        auto const& sim = dict_["simulation"];
        auto& l         = levels_[0];
        auto& ld        = data_(0);
        ld              = damn::mpi::regrid(map, std::move(ld), [&](auto const& p) {
            Layout_t const layout{p.box, damn::amr::mesh_size(hierarchy_, 0),
                                  damn::amr::position(hierarchy_, 0, p.box.lower), 0};
            return std::make_unique<Patch_t>(layout, sim, surroundings_(0, p.box));
        });
        for (std::size_t i = 0; i < damn::amr::size(ld); ++i)
        {
            ld.data[i]->rebox(surroundings_(0, ld.patches[i].box));
            for (auto& pop : ld.data[i]->ions)
                pop.patchGhostParticles().clear();
        }
        link_level_(0);
        fill_magnetic_ghosts_(0, "EM_B", t);
        fill_electric_ghosts_(0, "EM_E", t);
        copy_(0, damn::amr::fill_ghosts, "IonPopMoments");
        solver::hybrid::compute_moments(level_services_(0));
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
            regrid_level_(l, clustered_[ln], t, /*keep_finer=*/true);
    }
}


// every level but the finest allowed tagged and clustered again: a finer level whose boxes
// changed is regridded (made, if new; removed, with the finer, if nothing is tagged)
template<auto opts>
void Simulator<opts>::regrid_(double const t, int const from, bool const always)
{
    DAMN_LOG_SCOPE(1, "Simulator::regrid_");
    for (int ln = from; ln + 1 < hierarchy_.max_levels and ln < static_cast<int>(levels_.size());
         ++ln)
    {
        auto const boxes = cluster_(ln);
        auto const fine  = static_cast<std::size_t>(ln) + 1;
        clustered_.resize(std::max(clustered_.size(), fine + 1));
        if (fine < levels_.size() and boxes == clustered_[fine] and !always)
            continue;
        if (boxes.empty())
        {
            resize_levels_(fine);
            clustered_.resize(fine);
            damn::amr::remove_finer_levels(hierarchy_, ln);
            break;
        }
        clustered_[fine] = boxes;
        if (fine < levels_.size())
            regrid_level_(ln + 1, boxes, t);
        else
        {
            make_level_(ln + 1, boxes);
            initialize_refined_(fine);
            prepare_step_(ln + 1, t);
        }
    }
}

// level ln onto new boxes, at t (every level synchronized): its data moved where the old level
// was (damn::amr::regrid), refined from the coarser level elsewhere, as PHARE's
// HybridHybridMessengerStrategy::regrid; finer levels go, the caller makes them again, unless
// keep_finer (a rebalance, or a regrid onto the same boxes: the caller regrids each finer level
// onto its own boxes after this one, coarser to finer, which also links them to their new
// coarser level)
template<auto opts>
void Simulator<opts>::regrid_level_(int const ln, std::vector<Box_t> const& boxes, double const t,
                                    bool const keep_finer)
{
    DAMN_LOG_SCOPE(1, "Simulator::regrid_level_");
    using damn::amr::LevelRegion;
    auto const fine   = static_cast<std::size_t>(ln);
    auto constexpr gw = Patch_t::field_ghost_width;

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
    auto const map = damn::amr::make_regrid_map(old_level, hierarchy_.levels[ln], rank_,
                                                damn::amr::periodic_shifts(hierarchy_, ln));

    auto const& sim = dict_["simulation"];
    auto const make = [&](auto const& p) {
        Layout_t const layout{p.box, damn::amr::mesh_size(hierarchy_, ln),
                              damn::amr::position(hierarchy_, ln, p.box.lower), ln};
        return std::make_unique<Patch_t>(layout, sim, surroundings_(ln, p.box));
    };
    auto& l = levels_[fine];
    auto& ld = data_(fine);
    ld       = n_ranks_ > 1 ? damn::mpi::regrid(map, std::move(ld), make)
                            : damn::amr::regrid(map, std::move(ld), make);
    for (std::size_t i = 0; i < damn::amr::size(ld); ++i)
        ld.data[i]->rebox(surroundings_(ln, ld.patches[i].box));
    link_level_(ln);
    l.coarse_t  = t;
    l.coarse_dt = 0; // the coarser level's data at t

    // the new cells from the coarser level: B divergence free, E, the domain particles split
    using MaskedField = amr::Masked<amr::FieldRefiner, dimension>;
    MaskedField const masked_field{{}, &old_cells};
    refine_(fine, masked_field, LevelRegion::Interior, amr::vector_pairs("EM_B", "EM_B"));
    for (std::size_t i = 0; i < damn::amr::size(ld); ++i)
    {
        auto& patch      = *ld.data[i];
        auto const cells = ld.patches[i].box;
        auto bx          = amr::field_resource<gw>(patch.emB[0]);
        auto by          = amr::field_resource<gw>(patch.emB[1]);
        auto bz          = amr::field_resource<gw>(patch.emB[2]);
        for (auto const& region : masked_field.uncovered(cells, [](auto const& b) { return b; }))
            amr::divergence_free_touch_up(std::array{&bx, &by, &bz}, cells, region,
                                          damn::amr::mesh_size(hierarchy_, ln));
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
    using MaskedSplit = amr::Masked<amr::ParticleSplitter<Splitter_t>, dimension>;
    for (auto const& name : population_names_)
        refine_(fine, MaskedSplit{{}, &old_cells}, LevelRegion::Interior,
                {{name + "_domain", name + "_domain", ""}});

    // ghosts: patch ghosts from the new neighbours, level ghosts from the coarser level now
    fill_magnetic_ghosts_(fine, "EM_B", t);
    fill_electric_ghosts_(fine, "EM_E", t);
    for (auto const& name : population_names_)
        refine_(fine, amr::ParticleSplitter<Splitter_t>{}, LevelRegion::LevelGhosts,
                {{name + "_domain", name + "_level_ghost_old", ""}});
    own_level_ghosts_(fine, /*new_ghosts=*/false);
    per_patch_(fine, [](auto& patch) {
        for (auto& pop : patch.ions)
            pop.levelGhostParticles() = pop.levelGhostParticlesOld();
    });
    sort_refined_particles_(fine);

    if (!keep_finer)
        refined_moments_(fine);
    else
        moved_moments_(fine);
    prepare_step_(ln, t);
}


// a level moved onto new owners (keep_finer: the same boxes, a rebalance): the moments moved
//  with it are kept on the domain (under a finer level they are the finer level's, coarsened at
//  the last synchronization: a deposit would replace them), the deposit of its particles only
//  gives the ghost nodes the move does not bring (outside the level: level ghost deposits),
//  then the patch borders from the neighbours, as level 0's rebalance
template<auto opts>
void Simulator<opts>::moved_moments_(std::size_t const ln)
{
    auto const each_moment = [](auto& patch, auto&& fn) {
        for (auto& pop : patch.populations)
        {
            fn(pop->density);
            fn(pop->charge_density);
            for (auto& f : pop->flux)
                fn(f);
        }
    };
    std::vector<std::vector<std::vector<double>>> kept; // per patch, per moment: its values
    per_patch_(ln, [&](auto& patch) {
        auto& mine = kept.emplace_back();
        each_moment(patch,
                    [&](auto const& f) { mine.emplace_back(f.data(), f.data() + f.size()); });
    });
    refined_moments_(ln); // the ghost nodes; the domain put back below
    std::size_t i = 0;
    per_patch_(ln, [&](auto& patch) {
        std::size_t m = 0;
        each_moment(patch, [&](auto& f) {
            auto const& values = kept[i][m++];
            damn::core::for_each_point(domain_box(patch.layout, f), [&](auto const& ijk) {
                auto& v = f(ijk);
                v       = values[static_cast<std::size_t>(&v - f.data())];
            });
        });
        ++i;
    });
    copy_(ln, damn::amr::fill_ghosts, "IonPopMoments");
    solver::hybrid::compute_moments(level_services_(ln));
}

} // namespace plamn

#endif // PLAMN_SIMULATOR_SIMULATOR_REGRID_HPP
