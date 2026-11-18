#ifndef PLAMN_PYTHON3_PLAMN_AMR_HPP
#define PLAMN_PYTHON3_PLAMN_AMR_HPP

// plamn_etc's data free hierarchy tools (plamn.amr.Chopper): damn's chopping, partitioning and
// decomposition metrics on boxes given from python, no simulation

#include "damn/amr/metrics.hpp"
#include "damn/amr/topology.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/partition.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstddef>
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

// ranks per node given as each rank's node
inline damn::amr::Topology topology_of(std::vector<int> const& node_of_rank)
{
    auto t = damn::amr::single_node(static_cast<int>(node_of_rank.size()));
    for (std::size_t r = 0; r < node_of_rank.size(); ++r)
        t.ranks[r].node = node_of_rank[r];
    return t;
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

    m.def(("partition_" + d).c_str(),
          [](std::vector<PyBox<dim>> const& boxes, std::vector<double> const& weights,
             std::vector<int> const& node_of_rank) {
              std::vector<damn::core::Box<int, dim>> bs;
              bs.reserve(boxes.size());
              for (auto const& [lo, up] : boxes)
                  bs.push_back(to_box<dim>(lo, up));
              auto const t = topology_of(node_of_rank);
              return damn::amr::partition(bs, weights, damn::amr::node_share_capacity(t));
          },
          "each box's rank: the boxes along the Hilbert curve cut by weight, each node an equal "
          "share split over its ranks (node_of_rank: each rank's node)");

    m.def(("metrics_" + d).c_str(),
          [](IntArray<dim> const& domain_lower, IntArray<dim> const& domain_upper,
             std::array<bool, dim> const& periodic, int const ratio,
             std::vector<std::vector<PyPatch<dim>>> const& levels,
             std::vector<int> const& node_of_rank, int const reach,
             std::vector<std::vector<double>> const& weights) {
              damn::amr::Hierarchy<dim> h;
              h.domain     = to_box<dim>(domain_lower, domain_upper);
              h.periodic   = periodic;
              h.ratio      = ratio;
              h.max_levels = static_cast<int>(levels.size());
              for (std::size_t ln = 0; ln < levels.size(); ++ln)
              {
                  std::vector<damn::core::Box<int, dim>> boxes;
                  std::vector<int> ranks;
                  for (auto const& [lo, up, rank] : levels[ln])
                  {
                      boxes.push_back(to_box<dim>(lo, up));
                      ranks.push_back(rank);
                  }
                  h.levels.push_back(damn::amr::make_level(static_cast<int>(ln), boxes, ranks));
              }
              nb::list out;
              for (auto const& lm :
                   damn::amr::metrics(h, topology_of(node_of_rank), reach, weights))
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
          "per level: patches, cells, weight per rank, imbalance, and the cells exchanged by "
          "distance (same level ghosts within reach, coarser cells under each grown patch)");
}

inline void declare_amr(nb::module_& m)
{
    declare_amr<1>(m);
    declare_amr<2>(m);
    declare_amr<3>(m);
}

} // namespace plamn::python3

#endif // PLAMN_PYTHON3_PLAMN_AMR_HPP
