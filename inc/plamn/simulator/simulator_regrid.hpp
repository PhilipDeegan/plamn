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
#include "damn/core/def.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/logger.hpp"
#include "damn/core/math/array.hpp"
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
    auto& l        = levels_[ln];
    auto const dx  = damn::amr::mesh_size(hierarchy_, ln);

    std::vector<damn::core::Point<int, dimension>> tagged;
    for (std::size_t i = 0; i < damn::amr::size(l.data); ++i)
    {
        auto const resource = [&](std::string const& name) {
            std::optional<Resource> r;
            damn::amr::with_resource(l.data.data[i], name, [&](auto& x) {
                if constexpr (std::is_same_v<std::decay_t<decltype(x)>, Resource>)
                    r = x;
            });
            if (!r)
                throw std::runtime_error("plamn Simulator: no field " + name + " to tag on");
            return *r;
        };
        for (auto const& c : amr::tagged_cells(*tagging_, l.data.patches[i].box, dx, resource))
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
        for (auto const& patch : levels_[fine].data.data)
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
        for (auto const& patch : levels_[fine - 1].data.data)
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
            for (auto const& p : levels_[0].data.patches)
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
    auto const pgw = Patch_t::particle_ghost_width;

    auto const boxes = damn::amr::chop(hierarchy_.domain, patch_size_);
    auto const block = damn::core::ConstArray<int, dimension>(rebalance_block);
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
        l.data          = damn::mpi::regrid(map, std::move(l.data), [&](auto const& p) {
            Layout_t const layout{p.box, damn::amr::mesh_size(hierarchy_, 0),
                                  damn::amr::position(hierarchy_, 0, p.box.lower), 0};
            auto const ghost = damn::core::grow(p.box, damn::core::ConstArray<int, dimension>(pgw));
            return std::make_unique<Patch_t>(layout, sim,
                                             damn::amr::covered_by_level(hierarchy_, 0, ghost));
        });
        for (std::size_t i = 0; i < damn::amr::size(l.data); ++i)
        {
            auto const ghost = damn::core::grow(l.data.patches[i].box,
                                                damn::core::ConstArray<int, dimension>(pgw));
            l.data.data[i]->rebox(damn::amr::covered_by_level(hierarchy_, 0, ghost));
            for (auto& pop : l.data.data[i]->ions)
                pop.patchGhostParticles().clear();
        }
        link_level_(0);
        fill_magnetic_ghosts_(0, "EM_B", t);
        fill_electric_ghosts_(0, "EM_E", t);
        copy_(0, damn::amr::fill_ghosts, "IonPopMoments");
        compute_moments_(0);
        prepare_step_(0, t);
    }

    for (std::size_t ln = 1; ln < levels_.size(); ++ln)
        regrid_level_(static_cast<int>(ln), clustered_[ln], t);
}


// every level but the finest allowed tagged and clustered again: a finer level whose boxes
// changed is regridded (made, if new; removed, with the finer, if nothing is tagged)
template<auto opts>
void Simulator<opts>::regrid_(double const t)
{
    DAMN_LOG_SCOPE(1, "Simulator::regrid_");
    for (int ln = 0; ln + 1 < hierarchy_.max_levels and ln < static_cast<int>(levels_.size()); ++ln)
    {
        auto const boxes = cluster_(ln);
        auto const fine  = static_cast<std::size_t>(ln) + 1;
        clustered_.resize(std::max(clustered_.size(), fine + 1));
        if (fine < levels_.size() and boxes == clustered_[fine])
            continue;
        if (boxes.empty())
        {
            levels_.resize(fine);
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
// HybridHybridMessengerStrategy::regrid; finer levels go, the caller makes them again
template<auto opts>
void Simulator<opts>::regrid_level_(int const ln, std::vector<Box_t> const& boxes, double const t)
{
    DAMN_LOG_SCOPE(1, "Simulator::regrid_level_");
    using damn::amr::LevelRegion;
    auto const fine   = static_cast<std::size_t>(ln);
    auto const pgw    = Patch_t::particle_ghost_width;
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
    levels_.resize(fine + 1);
    damn::amr::set_level(hierarchy_, level_of_(ln, boxes));
    auto const map = damn::amr::make_regrid_map(old_level, hierarchy_.levels[ln], rank_,
                                                damn::amr::periodic_shifts(hierarchy_, ln));

    auto const& sim = dict_["simulation"];
    auto const make = [&](auto const& p) {
        Layout_t const layout{p.box, damn::amr::mesh_size(hierarchy_, ln),
                              damn::amr::position(hierarchy_, ln, p.box.lower), ln};
        auto const ghost = damn::core::grow(p.box, damn::core::ConstArray<int, dimension>(pgw));
        return std::make_unique<Patch_t>(layout, sim,
                                         damn::amr::covered_by_level(hierarchy_, ln, ghost));
    };
    auto& l = levels_[fine];
    l.data  = n_ranks_ > 1 ? damn::mpi::regrid(map, std::move(l.data), make)
                           : damn::amr::regrid(map, std::move(l.data), make);
    for (std::size_t i = 0; i < damn::amr::size(l.data); ++i)
    {
        auto const ghost
            = damn::core::grow(l.data.patches[i].box, damn::core::ConstArray<int, dimension>(pgw));
        l.data.data[i]->rebox(damn::amr::covered_by_level(hierarchy_, ln, ghost));
    }
    link_level_(ln);
    l.coarse_t  = t;
    l.coarse_dt = 0; // the coarser level's data at t

    // the new cells from the coarser level: B divergence free, E, the domain particles split
    using MaskedField = amr::Masked<amr::FieldRefiner, dimension>;
    MaskedField const masked_field{{}, &old_cells};
    refine_(fine, masked_field, LevelRegion::Interior, amr::vector_pairs("EM_B", "EM_B"));
    for (std::size_t i = 0; i < damn::amr::size(l.data); ++i)
    {
        auto& patch      = *l.data.data[i];
        auto const cells = l.data.patches[i].box;
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
    per_patch_(fine, [](auto& patch) {
        for (auto& pop : patch.ions)
            pop.levelGhostParticles() = pop.levelGhostParticlesOld();
    });

    refined_moments_(fine);
    prepare_step_(ln, t);
}

} // namespace plamn

#endif // PLAMN_SIMULATOR_SIMULATOR_REGRID_HPP
