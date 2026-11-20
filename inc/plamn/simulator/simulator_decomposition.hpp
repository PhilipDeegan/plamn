#ifndef PLAMN_SIMULATOR_SIMULATOR_DECOMPOSITION_HPP
#define PLAMN_SIMULATOR_SIMULATOR_DECOMPOSITION_HPP

// what the running hierarchy's decomposition costs, for load balancing (not a PHARE comparison):
//  simulation/AMR/decomposition/log (or PLAMN_DECOMPOSITION) a file rank 0 writes one JSON line
//  to per event, truncated at setup:
//
//   "init"       after initialize
//   "regrid"     after a regrid that changed some level's boxes or owners
//   "rebalance"  after a rebalance that changed them
//   "sample"     every simulation/AMR/decomposition/every (PLAMN_DECOMPOSITION_EVERY) steps,
//                changed or not (the particles move)
//
// a line: event, t, step, the hierarchy's frame (domain, periodic, ratio, dl, interp, reach) and
//  each rank's node, then per level
//   patches    [lower, upper, rank] each: the decomposition itself
//   weights    each patch's particles (cells: an MHD level, or load balancing "homogeneous")
//   metrics    damn::amr::metrics with those weights: weight per rank, imbalance, the cells
//              exchanged by Distance within reach (the field ghost width, as PHARE's run
//              summary): ghost_cells (same level, patch ghosts), coarse_cells (the coarser cells
//              under each grown patch), and ghost_box_cells (every patch's grown box less the
//              patch: level ghosts and physical boundary cells = ghost_box_cells less the
//              ghost_cells)
//   busy       hybrid levels: seconds of work in the level's advance since the last line
//              (per_patch_ timed: outside the exchanges; the push too only with a per patch
//              mover: "busy_push") of each patch of the PREVIOUS line's level (the
//              decomposition that did the work; none if the level is new), summed over the
//              ranks; busy_rank: each rank's
//   migration  damn::amr::migration from the previous line's decomposition (moved, refined
//              cells by Distance), when there is one
//  and busy_steps: the steps since the last line (a line within a step: 0)
// the costs (critical path, distance weighted exchanges, migration) are derived from it in
//  python (plamn.amr.read_decomposition), so their per cell costs can change without a build.
// Collective: the weights are summed over the ranks, so every rank calls it at the same points.
// IWYU pragma: private, include "plamn/simulator/simulator.hpp"

#include "damn/amr/metrics.hpp"
#include "damn/amr/regrid.hpp"
#include "plamn/simulator/simulator.hpp" // IWYU pragma: keep (the class)

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace plamn
{
namespace detail
{
    // a JSON array of a range of numbers
    void json_array(std::ostream& os, auto const& values)
    {
        os << '[';
        bool first = true;
        for (auto const& v : values)
        {
            os << (first ? "" : ",") << v;
            first = false;
        }
        os << ']';
    }

    // {"Self": n, ...} of a per Distance count
    void json_distances(std::ostream& os, auto const& counts)
    {
        os << '{';
        for (std::size_t i = 0; i < counts.size(); ++i)
            os << (i ? "," : "") << '"' << damn::amr::to_string(static_cast<damn::amr::Distance>(i))
               << "\":" << counts[i];
        os << '}';
    }
} // namespace detail

template<auto opts>
void Simulator<opts>::setup_decomposition_log_()
{
    auto& d = decomposition_;
    d.path  = cppdict::get_value(dict_, "simulation/AMR/decomposition/log", std::string{});
    if (auto const* env = std::getenv("PLAMN_DECOMPOSITION"); env and *env)
        d.path = env;
    d.every = cppdict::get_value(dict_, "simulation/AMR/decomposition/every", 0);
    if (auto const* env = std::getenv("PLAMN_DECOMPOSITION_EVERY"); env and *env)
        d.every = std::stoi(env);
    if (!d.path.empty() and rank_ == 0)
        std::ofstream{d.path, std::ios::trunc};
}

template<auto opts>
int Simulator<opts>::decomposition_reach_()
{
    if constexpr (Models_t::hybrid_on)
        return solver::HybridModel<opts>::Patch_t::field_ghost_width;
    else
        return solver::MHDModel<opts>::Patch_t::field_ghost_width;
}

template<auto opts>
bool Simulator<opts>::busy_push_()
{
    if constexpr (Models_t::hybrid_on)
        return solver::HybridModel<opts>::Patch_t::per_patch_mover;
    else
        return false;
}

template<auto opts>
std::array<int, 2 * Simulator<opts>::dimension> Simulator<opts>::d_key_(Box_t const& box)
{
    std::array<int, 2 * dimension> key{};
    for (std::size_t i = 0; i < dimension; ++i)
    {
        key[i]             = box.lower[i];
        key[dimension + i] = box.upper[i];
    }
    return key;
}

template<auto opts>
void Simulator<opts>::record_decomposition_(std::string const& event, double const t,
                                            bool const always)
{
    auto& d = decomposition_;
    if (d.path.empty() or !isInitialized_) // regrids while initializing: the "init" line
        return;
    auto const same = [&] {
        if (!d.last or d.last->levels.size() != hierarchy_.levels.size())
            return false;
        for (std::size_t ln = 0; ln < hierarchy_.levels.size(); ++ln)
            if (!damn::amr::same_decomposition(d.last->levels[ln], hierarchy_.levels[ln]))
                return false;
        return true;
    };
    if (!always and same())
        return;
    DAMN_LOG_SCOPE(1, "Simulator::record_decomposition_");

    // collective: every rank, the same levels
    // busy time per patch (its owner's) and per rank, summed / gathered over the ranks
    std::vector<std::vector<double>> busy(hierarchy_.levels.size()), busy_rank(busy.size());
    for (std::size_t ln = 0; ln < hierarchy_.levels.size(); ++ln)
    {
        static std::vector<damn::amr::Patch<dimension>> const none;
        auto const& patches // the decomposition the work ran on
            = d.last and ln < d.last->levels.size() ? d.last->levels[ln].patches : none;
        busy[ln].assign(patches.size(), 0.);
        double mine = 0;
        if (ln < d.busy.size())
        {
            for (std::size_t i = 0; i < patches.size(); ++i)
                if (patches[i].rank == rank_)
                    if (auto const it = d.busy[ln].find(d_key_(patches[i].box));
                        it != d.busy[ln].end())
                        busy[ln][i] = it->second;
            for (auto const& [_, s] : d.busy[ln])
                mine += s;
        }
        busy_rank[ln].assign(static_cast<std::size_t>(n_ranks_), 0.);
        busy_rank[ln][static_cast<std::size_t>(rank_)] = mine;
        if (n_ranks_ > 1)
        {
            MPI_Allreduce(MPI_IN_PLACE, busy[ln].data(), static_cast<int>(busy[ln].size()),
                          MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, busy_rank[ln].data(), n_ranks_, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
        }
    }
    auto const busy_steps = steps_ - d.steps_then;
    d.busy.clear();
    d.steps_then = steps_;

    std::vector<std::vector<double>> weights(hierarchy_.levels.size());
    for (std::size_t ln = 0; ln < hierarchy_.levels.size(); ++ln)
    {
        std::vector<Box_t> boxes;
        for (auto const& p : hierarchy_.levels[ln].patches)
            boxes.push_back(p.box);
        auto& w = weights[ln];
        for (auto const& b : boxes)
            w.push_back(static_cast<double>(b.size()));
        models_.visit(ln, [&](auto& model) {
            if constexpr (solver::is_hybrid_model_v<std::remove_cvref_t<decltype(model)>>)
                w = particle_weights_(static_cast<int>(ln), boxes);
        });
    }

    if (rank_ == 0)
    {
        auto const reach   = decomposition_reach_();
        auto const metrics = damn::amr::metrics(hierarchy_, topology_, reach, weights);
        std::ostringstream os;
        os.precision(17);
        os << std::boolalpha;
        os << R"({"event":")" << event << R"(","t":)" << t << R"(,"step":)" << steps_
           << R"(,"domain":[)";
        detail::json_array(os, hierarchy_.domain.lower.toArray());
        os << ',';
        detail::json_array(os, hierarchy_.domain.upper.toArray());
        os << R"(],"periodic":)";
        detail::json_array(os, hierarchy_.periodic);
        os << R"(,"ratio":)" << hierarchy_.ratio << R"(,"dl":)";
        detail::json_array(os, cellWidth_);
        os << R"(,"interp":)" << interp_order << R"(,"reach":)" << reach << R"(,"busy_steps":)"
           << busy_steps << R"(,"busy_push":)" << busy_push_();
        os << R"(,"nodes":)";
        std::vector<int> nodes;
        for (auto const& r : topology_.ranks)
            nodes.push_back(r.node);
        detail::json_array(os, nodes);

        os << R"(,"levels":[)";
        for (std::size_t ln = 0; ln < hierarchy_.levels.size(); ++ln)
        {
            auto const& m = metrics[ln];
            os << (ln ? "," : "") << R"({"level":)" << ln << R"(,"patches":[)";
            std::size_t ghost_box_cells = 0;
            bool first                  = true;
            for (auto const& p : hierarchy_.levels[ln].patches)
            {
                os << (first ? "" : ",") << '[';
                detail::json_array(os, p.box.lower.toArray());
                os << ',';
                detail::json_array(os, p.box.upper.toArray());
                os << ',' << p.rank << ']';
                first = false;
                ghost_box_cells += damn::core::grow(p.box, reach).size() - p.box.size();
            }
            os << R"(],"weights":)";
            detail::json_array(os, weights[ln]);
            os << R"(,"metrics":{"cells":)" << m.cells << R"(,"smallest_patch":)"
               << m.smallest_patch << R"(,"imbalance":)" << m.imbalance() << R"(,"weight":)";
            detail::json_array(os, m.weight);
            os << R"(,"ghost_box_cells":)" << ghost_box_cells;
            os << R"(,"busy":)";
            detail::json_array(os, busy[ln]);
            os << R"(,"busy_rank":)";
            detail::json_array(os, busy_rank[ln]);
            os << R"(,"ghost_cells":)";
            detail::json_distances(os, m.ghost_cells);
            os << R"(,"coarse_cells":)";
            detail::json_distances(os, m.coarse_cells);
            os << "}}";
        }
        os << ']';

        if (d.last)
        {
            os << R"(,"migration":[)";
            auto const moves = damn::amr::migration(*d.last, hierarchy_, topology_);
            for (std::size_t i = 0; i < moves.size(); ++i)
            {
                os << (i ? "," : "") << R"({"level":)" << moves[i].level << R"(,"moved":)";
                detail::json_distances(os, moves[i].moved);
                os << R"(,"refined":)";
                detail::json_distances(os, moves[i].refined);
                os << '}';
            }
            os << ']';
        }
        os << "}\n";
        std::ofstream{d.path, std::ios::app} << os.str();
    }
    d.last = hierarchy_;
}

} // namespace plamn

#endif // PLAMN_SIMULATOR_SIMULATOR_DECOMPOSITION_HPP
