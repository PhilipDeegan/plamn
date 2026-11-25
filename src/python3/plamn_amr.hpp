#ifndef PLAMN_PYTHON3_PLAMN_AMR_HPP
#define PLAMN_PYTHON3_PLAMN_AMR_HPP

// plamn_etc's data free hierarchy tools (plamn.amr.Chopper): damn's chopping, partitioning and
// decomposition metrics on boxes given from python, no simulation

#include "damn/amr/cost.hpp"
#include "damn/amr/decompose.hpp"
#include "damn/amr/estimate.hpp"
#include "damn/amr/metrics.hpp"
#include "damn/amr/topology.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/partition.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/data/tiles/tile_set.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <algorithm>
#include <cstddef>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace plamn::python3
{
namespace nb = nanobind;

template<std::size_t dim>
using IntArray = std::array<int, dim>;
template<std::size_t dim>
using PyBox = std::pair<IntArray<dim>, IntArray<dim>>; // lower, upper
template<std::size_t dim>
using PyPatch = std::tuple<IntArray<dim>, IntArray<dim>, int>; // lower, upper, rank

template<std::size_t dim>
damn::core::Box<int, dim> to_box(IntArray<dim> const& lower, IntArray<dim> const& upper)
{
    return {damn::core::Point<int, dim>{lower}, damn::core::Point<int, dim>{upper}};
}

template<std::size_t dim>
PyBox<dim> from_box(damn::core::Box<int, dim> const& b)
{
    return {b.lower.toArray(), b.upper.toArray()};
}

// ranks given as each rank's node, and its memory domain on the node (none: one per node)
inline damn::amr::Topology topology_of(std::vector<int> const& node_of_rank,
                                       std::vector<int> const& numa_of_rank = {})
{
    if (!numa_of_rank.empty() and numa_of_rank.size() != node_of_rank.size())
        throw std::runtime_error("plamn: numa_of_rank: one per rank, as node_of_rank");
    auto t = damn::amr::single_node(static_cast<int>(node_of_rank.size()));
    for (std::size_t r = 0; r < node_of_rank.size(); ++r)
    {
        t.ranks[r].node = node_of_rank[r];
        if (!numa_of_rank.empty())
            t.ranks[r].numa = numa_of_rank[r];
    }
    return t;
}

// a data free hierarchy of levels given as (lower, upper, rank) per patch
template<std::size_t dim>
damn::amr::Hierarchy<dim> hierarchy_of(IntArray<dim> const& domain_lower,
                                       IntArray<dim> const& domain_upper,
                                       std::array<bool, dim> const& periodic, int const ratio,
                                       std::vector<std::vector<PyPatch<dim>>> const& levels)
{
    damn::amr::Hierarchy<dim> h;
    h.domain     = to_box<dim>(domain_lower, domain_upper);
    h.periodic   = periodic;
    h.ratio      = ratio;
    h.max_levels = static_cast<int>(levels.size());
    for (std::size_t ln = 0; ln < levels.size(); ++ln)
    {
        std::vector<damn::core::Box<int, dim>> boxes;
        std::vector<int> ranks;
        boxes.reserve(levels[ln].size());
        ranks.reserve(levels[ln].size());
        for (auto const& [lo, up, rank] : levels[ln])
        {
            boxes.push_back(to_box<dim>(lo, up));
            ranks.push_back(rank);
        }
        h.levels.push_back(damn::amr::make_level(static_cast<int>(ln), boxes, ranks));
    }
    return h;
}

// per distance name, a count
inline nb::dict by_distance(std::array<std::size_t, damn::amr::n_distances> const& a)
{
    nb::dict d;
    for (std::size_t i = 0; i < damn::amr::n_distances; ++i)
        d[std::string{damn::amr::to_string(static_cast<damn::amr::Distance>(i))}.c_str()] = a[i];
    return d;
}

template<std::size_t dim>
void declare_amr(nb::module_& m)
{
    auto const d = std::to_string(dim);

    m.def(("chop_" + d).c_str(),
          [](IntArray<dim> const& lower, IntArray<dim> const& upper, IntArray<dim> const& size,
             int const granularity, std::string const& mode) {
              auto const box = to_box<dim>(lower, upper);
              std::vector<damn::core::Box<int, dim>> boxes;
              if (mode == "chop")
                  boxes = damn::amr::chop(box, size);
              else if (mode == "even")
                  boxes = damn::amr::chop_evenly(box, size, granularity);
              else
                  throw std::runtime_error("plamn chop: mode " + mode + ": chop or even");
              std::vector<PyBox<dim>> out;
              out.reserve(boxes.size());
              for (auto const& b : boxes)
                  out.push_back(from_box(b));
              return out;
          },
          "box lower..upper cut into patches of at most size cells: mode 'chop' (from the lower "
          "corner) or 'even' (cells shared evenly in units of granularity, SAMRAI's on one rank)");

    m.def(("tile_cells_" + d).c_str(),
          [](std::vector<PyBox<dim>> const& boxes) {
              std::vector<std::size_t> tiles_per_box, cells_per_tile;
              tiles_per_box.reserve(boxes.size());
              for (auto const& [lo, up] : boxes)
              {
                  damn::core::TileSet<damn::core::Box<int, dim>> const tiles{to_box<dim>(lo, up)};
                  tiles_per_box.push_back(tiles.size());
                  for (auto const& tile : tiles())
                      cells_per_tile.push_back(tile.size());
              }
              return std::make_tuple(tiles_per_box, cells_per_tile);
          },
          nb::arg("boxes"),
          "damn's tiles of each box, no data (TileSet of boxes, DAMN_TILING_* as a run): "
          "(tiles per box, cells per tile, box after box)");

    m.def(("partition_" + d).c_str(),
          [](std::vector<PyBox<dim>> const& boxes, std::vector<double> const& weights,
             std::vector<int> const& node_of_rank, std::vector<int> const& numa_of_rank) {
              std::vector<damn::core::Box<int, dim>> bs;
              bs.reserve(boxes.size());
              for (auto const& [lo, up] : boxes)
                  bs.push_back(to_box<dim>(lo, up));
              auto const t = topology_of(node_of_rank, numa_of_rank);
              return damn::amr::partition(bs, weights, damn::amr::node_share_capacity(t));
          },
          nb::arg("boxes"), nb::arg("weights"), nb::arg("node_of_rank"),
          nb::arg("numa_of_rank") = std::vector<int>{},
          "each box's rank: the boxes along the Hilbert curve cut by weight, each node an equal "
          "share split over its ranks (node_of_rank: each rank's node; numa_of_rank: its memory "
          "domain on the node)");

    m.def(("metrics_" + d).c_str(),
          [](IntArray<dim> const& domain_lower, IntArray<dim> const& domain_upper,
             std::array<bool, dim> const& periodic, int const ratio,
             std::vector<std::vector<PyPatch<dim>>> const& levels,
             std::vector<int> const& node_of_rank, int const reach,
             std::vector<std::vector<double>> const& weights,
             std::vector<int> const& numa_of_rank) {
              auto const h = hierarchy_of<dim>(domain_lower, domain_upper, periodic, ratio, levels);
              nb::list out;
              for (auto const& lm :
                   damn::amr::metrics(h, topology_of(node_of_rank, numa_of_rank), reach, weights))
              {
                  nb::dict dm;
                  dm["level"]          = lm.level;
                  dm["patches"]        = lm.patches;
                  dm["cells"]          = lm.cells;
                  dm["smallest_patch"] = lm.smallest_patch;
                  dm["weight"]         = lm.weight;
                  dm["imbalance"]      = lm.imbalance();
                  nb::dict ghost, coarse;
                  for (std::size_t i = 0; i < damn::amr::n_distances; ++i)
                  {
                      auto const name
                          = std::string{damn::amr::to_string(static_cast<damn::amr::Distance>(i))};
                      ghost[name.c_str()]  = lm.ghost_cells[i];
                      coarse[name.c_str()] = lm.coarse_cells[i];
                  }
                  dm["ghost_cells"]  = ghost;
                  dm["coarse_cells"] = coarse;
                  out.append(dm);
              }
              return out;
          },
          nb::arg("domain_lower"), nb::arg("domain_upper"), nb::arg("periodic"), nb::arg("ratio"),
          nb::arg("levels"), nb::arg("node_of_rank"), nb::arg("reach"), nb::arg("weights"),
          nb::arg("numa_of_rank") = std::vector<int>{},
          "per level: patches, cells, weight per rank, imbalance, and the cells exchanged by "
          "distance (same level ghosts within reach, coarser cells under each grown patch)");

    m.def(("migration_" + d).c_str(),
          [](IntArray<dim> const& domain_lower, IntArray<dim> const& domain_upper,
             std::array<bool, dim> const& periodic, int const ratio,
             std::vector<std::vector<PyPatch<dim>>> const& from,
             std::vector<std::vector<PyPatch<dim>>> const& to, std::vector<int> const& node_of_rank,
             std::vector<int> const& numa_of_rank) {
              auto const a = hierarchy_of<dim>(domain_lower, domain_upper, periodic, ratio, from);
              auto const b = hierarchy_of<dim>(domain_lower, domain_upper, periodic, ratio, to);
              nb::list out;
              for (auto const& lm :
                   damn::amr::migration(a, b, topology_of(node_of_rank, numa_of_rank)))
              {
                  nb::dict dm;
                  dm["level"]   = lm.level;
                  dm["moved"]   = by_distance(lm.moved);
                  dm["refined"] = by_distance(lm.refined);
                  out.append(dm);
              }
              return out;
          },
          nb::arg("domain_lower"), nb::arg("domain_upper"), nb::arg("periodic"), nb::arg("ratio"),
          nb::arg("from"), nb::arg("to"), nb::arg("node_of_rank"),
          nb::arg("numa_of_rank") = std::vector<int>{},
          "per level of `to`, the cells moving from `from`'s owners by distance: held by the "
          "old level (moved) or refined from the old coarser level (refined)");
}

// a Weigh from python: weigh(ln, boxes) -> a weight per box when given, else the cost model
// (costs: particle, cell, ghost_cell, patch, reach; defaults CostModel's) over particles counted
// as cells times ppc[ln] (the last for the finer levels)
template<std::size_t dim>
damn::amr::Weigh<dim> weigh_of(nb::object const& weigh, std::map<std::string, double> const& costs,
                               std::vector<double> const& ppc)
{
    using Box_t = damn::core::Box<int, dim>;
    if (!weigh.is_none())
        return [weigh](int const ln, std::vector<Box_t> const& boxes) {
            std::vector<PyBox<dim>> bs;
            bs.reserve(boxes.size());
            for (auto const& b : boxes)
                bs.push_back(from_box(b));
            return nb::cast<std::vector<double>>(weigh(ln, bs));
        };
    damn::amr::CostModel model;
    for (auto const& [k, v] : costs)
        if (k == "particle") // damn's counted item: a particle here
            model.item = v;
        else if (k == "cell")
            model.cell = v;
        else if (k == "ghost_cell")
            model.ghost_cell = v;
        else if (k == "patch")
            model.patch = v;
        else if (k == "reach")
            model.reach = static_cast<int>(v);
        else
            throw std::runtime_error("plamn decompose: cost " + k
                                     + ": particle, cell, ghost_cell, patch or reach");
    if (ppc.empty())
        throw std::runtime_error("plamn decompose: weigh or ppc (particles per cell per level)");
    return damn::amr::weigh_by<dim>(model, [ppc](int const ln, std::vector<Box_t> const& boxes) {
        auto const per = ppc[std::min(static_cast<std::size_t>(ln), ppc.size() - 1)];
        std::vector<double> out;
        out.reserve(boxes.size());
        for (auto const& b : boxes)
            out.push_back(static_cast<double>(b.size()) * per);
        return out;
    });
}

template<std::size_t dim>
void declare_decompose(nb::module_& m)
{
    auto const d = std::to_string(dim);
    m.def(("decompose_" + d).c_str(),
          [](IntArray<dim> const& domain_lower, IntArray<dim> const& domain_upper,
             std::array<bool, dim> const& periodic, int const ratio,
             std::vector<std::vector<PyPatch<dim>>> const& levels, int const from,
             std::vector<std::vector<PyBox<dim>>> const& boxes,
             std::vector<int> const& node_of_rank, std::vector<int> const& numa_of_rank,
             IntArray<dim> const& patch_size, std::string const& partition, double const spill,
             int const block, bool const cut, std::vector<IntArray<dim>> const& level_size,
             std::vector<IntArray<dim>> const& under_size, nb::object const& weigh,
             std::map<std::string, double> const& costs, std::vector<double> const& ppc) {
              auto const h = hierarchy_of<dim>(domain_lower, domain_upper, periodic, ratio, levels);
              auto const topology = topology_of(node_of_rank, numa_of_rank);
              damn::amr::Decompose<dim> in{.topology   = topology,
                                           .weigh      = weigh_of<dim>(weigh, costs, ppc),
                                           .patch_size = patch_size,
                                           .partition  = partition,
                                           .spill      = spill,
                                           .block      = block,
                                           .cut        = cut,
                                           .level_size = level_size,
                                           .under_size = under_size};
              std::vector<std::vector<damn::core::Box<int, dim>>> bs(boxes.size());
              for (std::size_t k = 0; k < boxes.size(); ++k)
                  for (auto const& [lo, up] : boxes[k])
                      bs[k].push_back(to_box<dim>(lo, up));
              std::vector<std::vector<PyPatch<dim>>> out;
              for (auto const& level : damn::amr::decompose(h, from, bs, in))
              {
                  auto& o = out.emplace_back();
                  for (auto const& p : level.patches)
                      o.emplace_back(p.box.lower.toArray(), p.box.upper.toArray(), p.rank);
              }
              return out;
          },
          nb::arg("domain_lower"), nb::arg("domain_upper"), nb::arg("periodic"), nb::arg("ratio"),
          nb::arg("levels"), nb::arg("from_level"), nb::arg("boxes"), nb::arg("node_of_rank"),
          nb::arg("numa_of_rank") = std::vector<int>{}, nb::arg("patch_size"),
          nb::arg("partition") = "levels", nb::arg("spill") = -1., nb::arg("block") = 8,
          nb::arg("cut") = true, nb::arg("level_size") = std::vector<IntArray<dim>>{},
          nb::arg("under_size") = std::vector<IntArray<dim>>{}, nb::arg("weigh") = nb::none(),
          nb::arg("costs") = std::map<std::string, double>{},
          nb::arg("ppc")   = std::vector<double>{},
          "damn::amr::decompose, as a run does it: levels from_level, from_level + 1, ... from "
          "boxes (each level's cells), on `levels` (the hierarchy so far, (lower, upper, rank) per "
          "patch: the fixed coarser levels and, partition 'keep', the ones replaced); partition "
          "levels | keep | finest | aligned. Weights: weigh(ln, boxes) if given, else costs "
          "(particle, cell, ghost_cell, patch, reach) over cells times ppc[ln] particles. Returns "
          "each new level's (lower, upper, rank)");
}

template<std::size_t dim>
void declare_estimate(nb::module_& m)
{
    auto const d = std::to_string(dim);
    m.def(("estimate_" + d).c_str(),
          [](IntArray<dim> const& domain_lower, IntArray<dim> const& domain_upper,
             std::array<bool, dim> const& periodic, int const ratio,
             std::vector<std::vector<PyPatch<dim>>> const& levels,
             std::vector<int> const& node_of_rank, std::vector<int> const& numa_of_rank,
             int const reach, std::vector<std::vector<double>> const& weights,
             double const bytes_per_cell, std::optional<double> const node_latency,
             std::optional<double> const node_s_per_byte, int const threads, double const main_busy,
             double const thread_eff) {
              auto const h = hierarchy_of<dim>(domain_lower, domain_upper, periodic, ratio, levels);
              auto const costs
                  = node_latency and node_s_per_byte
                        ? damn::amr::exchange_costs(*node_latency, *node_s_per_byte, bytes_per_cell)
                        : damn::amr::bootstrapped_exchange_costs(bytes_per_cell);
              auto const e = damn::amr::estimate(
                  h, topology_of(node_of_rank, numa_of_rank), reach, weights, costs,
                  damn::amr::Threads{
                      .per_rank = threads, .main_busy = main_busy, .efficiency = thread_eff});
              nb::dict out;
              out["step"] = e.step;
              nb::list lvls;
              for (auto const& l : e.levels)
              {
                  nb::dict dl;
                  dl["level"]    = l.level;
                  dl["time"]     = l.time;
                  dl["compute"]  = l.compute;
                  dl["exchange"] = l.exchange;
                  lvls.append(dl);
              }
              out["levels"] = lvls;
              return out;
          },
          nb::arg("domain_lower"), nb::arg("domain_upper"), nb::arg("periodic"), nb::arg("ratio"),
          nb::arg("levels"), nb::arg("node_of_rank"), nb::arg("numa_of_rank") = std::vector<int>{},
          nb::arg("reach") = 0, nb::arg("weights") = std::vector<std::vector<double>>{},
          nb::arg("bytes_per_cell") = 8., nb::arg("node_latency") = nb::none(),
          nb::arg("node_s_per_byte") = nb::none(), nb::arg("threads") = 1,
          nb::arg("main_busy") = 0., nb::arg("thread_eff") = 1.,
          "damn::amr::estimate: a decomposition's step time, (lower, upper, rank) per patch per "
          "level, weights per patch (seconds), exchange costs per Distance from node_latency and "
          "node_s_per_byte when given, else this machine's bootstrap; threads per rank (their "
          "tiles scheduled, main_busy, thread_eff): {step, levels: [{level, "
          "time, compute (per rank), exchange (per rank)}]}");
}

inline void declare_amr(nb::module_& m)
{
    declare_estimate<1>(m);
    declare_estimate<2>(m);
    declare_estimate<3>(m);
    declare_decompose<1>(m);
    declare_decompose<2>(m);
    declare_decompose<3>(m);
    declare_amr<1>(m);
    declare_amr<2>(m);
    declare_amr<3>(m);
}

} // namespace plamn::python3

#endif // PLAMN_PYTHON3_PLAMN_AMR_HPP
