#ifndef PLAMN_SIMULATOR_SIMULATOR_SUMS_HPP
#define PLAMN_SIMULATOR_SIMULATOR_SUMS_HPP

// heavy debugging, compiled only with -DDAMN_SUPER_DEBUG (the calls are in DAMN_SUPER_DEBUG_DO):
//
//   sums_(ln, stage, t)      per level and stage, every resource of the level's model summed:
//                            fields over the level's nodes once each ("dom": the patch owning a
//                            node, its cells' range plus the level's last node row) and over
//                            the level ghost nodes once each ("gh": the first patch in box order
//                            holding it), whatever the patches; particle arrays as counts. One
//                            line each:  SUMS L<ln> t=<t> <stage> <name> dom <sum> gh <sum>
//                            The hybrid model's PHARE names (Bx, PredBx, AvgEx, rhoC, ...) as
//                            the PHARE debug branch's SolverPPC::debug_sums_ prints them, so the
//                            two runs' lines diff. Bit equal only on one rank (sums in node
//                            order); with more, each rank prints its own (R<rank>).
//   debug_regrid_(ln, t)     PLAMN_DEBUG_REGRID_TIMES="t0,t1,...": the finest level, just
//                            synchronized at one of those times, regridded onto its own boxes
//                            (as SAMRAI does with refinement boxes): to isolate regrid bugs
//   rebalance_state_(), check_rebalance_(before, t)
//                            a rebalance moves data: every level's E, B and moments on its
//                            domain nodes, and its domain and old level ghost particles (count,
//                            sums of weights, velocities, positions: order independent, every
//                            rank's) the same after it as before; the moments, deposited again,
//                            and the sums to round off. Throws naming what changed
// IWYU pragma: private, include "plamn/simulator/simulator.hpp"

#include "damn/core/def.hpp"
#include "plamn/simulator/simulator.hpp" // IWYU pragma: keep (the class)
#include "mkn/kul/range.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <stdexcept>
#include <numeric>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

namespace plamn
{
namespace detail
{
    // plamn's resource names as PHARE's debug branch labels them
    inline std::string sums_label(std::string const& name)
    {
        static std::map<std::string, std::string> const phare{{"EM_B_x", "Bx"},
                                                              {"EM_B_y", "By"},
                                                              {"EM_B_z", "Bz"},
                                                              {"EM_E_x", "Ex"},
                                                              {"EM_E_y", "Ey"},
                                                              {"EM_E_z", "Ez"},
                                                              {"EMPred_B_x", "PredBx"},
                                                              {"EMPred_B_y", "PredBy"},
                                                              {"EMPred_B_z", "PredBz"},
                                                              {"EMPred_E_x", "PredEx"},
                                                              {"EMPred_E_y", "PredEy"},
                                                              {"EMPred_E_z", "PredEz"},
                                                              {"EMAvg_B_x", "AvgBx"},
                                                              {"EMAvg_B_y", "AvgBy"},
                                                              {"EMAvg_B_z", "AvgBz"},
                                                              {"EMAvg_E_x", "AvgEx"},
                                                              {"EMAvg_E_y", "AvgEy"},
                                                              {"EMAvg_E_z", "AvgEz"},
                                                              {"ions_charge_density", "rhoC"}};
        auto const it = phare.find(name);
        return it == phare.end() ? name : it->second;
    }
} // namespace detail

template<auto opts>
void Simulator<opts>::sums_(std::size_t const ln, char const* const stage, double const t)
{
    using Pt = std::array<int, dimension>;
    models_.visit(ln, [&](auto& model) {
        auto& ld     = model.levels[static_cast<int>(ln)];
        auto const n = damn::amr::size(ld);
        if (n == 0)
            return;
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::sort(order.begin(), order.end(), [&](auto a, auto b) {
            return ld.patches[a].box.lower.toArray() < ld.patches[b].box.lower.toArray();
        });
        Pt llo, lhi;
        llo.fill(std::numeric_limits<int>::max());
        lhi.fill(std::numeric_limits<int>::min());
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t d = 0; d < dimension; ++d)
            {
                llo[d] = std::min(llo[d], ld.patches[i].box.lower[d]);
                lhi[d] = std::max(lhi[d], ld.patches[i].box.upper[d]);
            }
        std::vector<std::string> names; // the model's resources, as its patches list them
        for_each_resource(*ld.data[order[0]],
                          [&](std::string const& name, auto&&) { names.push_back(name); });
        std::string const where = n_ranks_ > 1 ? "R" + std::to_string(rank_) + " " : std::string{};

        for (auto const& name : names)
        {
            double domain = 0, ghost = 0;
            std::size_t count = 0;
            bool particles    = false;
            std::set<Pt> seen;
            for (auto const i : order)
            {
                auto const& cells = ld.patches[i].box;
                damn::amr::with_resource(ld.data[i], name, [&](auto& r) {
                    using R = std::decay_t<decltype(r)>;
                    if constexpr (damn::amr::BoxData<R, dimension>)
                    {
                        auto const cent = core::centering<dimension>(*r.field);
                        damn::core::for_each_point(ghost_box(r, cells), [&](auto const& p) {
                            bool level_node = true, owned = true;
                            Pt k;
                            for (std::size_t d = 0; d < dimension; ++d)
                            {
                                k[d]            = p[d];
                                bool const prim = cent[d] == core::QtyCentering::primal;
                                level_node
                                    = level_node and p[d] >= llo[d] and p[d] <= lhi[d] + prim;
                                owned = owned
                                        and ((p[d] >= cells.lower[d] and p[d] <= cells.upper[d])
                                             or (prim and p[d] == lhi[d] + 1
                                                 and cells.upper[d] == lhi[d]));
                            }
                            if (level_node and owned)
                                domain += value(r, cells, p);
                            else if (!level_node and seen.insert(k).second)
                                ghost += value(r, cells, p);
                        });
                    }
                    else if constexpr (requires { r.particles->size(); })
                    {
                        particles = true;
                        count += r.particles->size();
                    }
                });
            }
            auto const label = detail::sums_label(name);
            if (particles)
                std::printf("%sSUMS L%zu t=%.10f %-14s %-24s n %zu\n", where.c_str(), ln, t, stage,
                            label.c_str(), count);
            else
                std::printf("%sSUMS L%zu t=%.10f %-14s %-24s dom % .17e gh % .17e\n", where.c_str(),
                            ln, t, stage, label.c_str(), domain, ghost);
        }
        std::fflush(stdout);
    });
}

template<auto opts>
void Simulator<opts>::debug_regrid_(int const ln, double const t)
{
    static std::vector<double> const times = [] {
        std::vector<double> out;
        if (auto const* env = std::getenv("PLAMN_DEBUG_REGRID_TIMES"))
        {
            std::string const s{env};
            for (std::size_t at = 0; at < s.size();)
            {
                auto const end = s.find(',', at);
                out.push_back(std::stod(s.substr(at, end - at)));
                at = end == std::string::npos ? s.size() : end + 1;
            }
        }
        return out;
    }();
    if (static_cast<std::size_t>(ln) + 1 != levels_.size() // the finest only, as PHARE's
        or std::none_of(times.begin(), times.end(),
                        [&](double const x) { return std::abs(x - t) < 1e-12; }))
        return;
    std::vector<Box_t> boxes;
    for (auto const& p : hierarchy_.levels[ln].patches)
        boxes.push_back(p.box);
    std::printf("DEBUG_REGRID L%d t=%.10f %zu boxes\n", ln, t, boxes.size());
    std::fflush(stdout);
    regrid_level_(ln, boxes, t);
}

template<auto opts>
typename Simulator<opts>::RebalanceState Simulator<opts>::rebalance_state_()
{
    RebalanceState out;
    std::vector<std::string> names;
    for (auto const* v : {"EM_B_", "EM_E_"})
        for (auto const* c : {"x", "y", "z"})
            names.push_back(std::string{v} + c);
    if constexpr (Models_t::hybrid_on)
    {
        names.push_back("ions_charge_density");
        for (auto const* c : {"x", "y", "z"})
            names.push_back(std::string{"ions_charge_flux_"} + c);
        for (auto const& pop : models_.hybrid.population_names)
        {
            names.push_back(pop + "_density");
            for (auto const* c : {"x", "y", "z"})
                names.push_back(pop + "_flux_" + c);
        }
    }
    for (auto const ln : mkn::kul::range(levels_.size()))
    {
        if (models_.is_mhd(ln))
            continue;
        for (auto const& name : names)
            out.values.push_back(level_values(name, ln));
        if constexpr (Models_t::hybrid_on)
        {
            std::array<double, 8> sums{}; // domain, old level ghosts: count, w, v, x
            auto const add = [&](std::size_t const set, auto const& particles) {
                core::for_each_particle(particles, [&](auto const& p) {
                    sums[set * 4] += 1;
                    sums[set * 4 + 1] += p.weight();
                    for (auto const v : p.v())
                        sums[set * 4 + 2] += v;
                    for (auto const d : mkn::kul::range(dimension))
                        sums[set * 4 + 3] += p.iCell()[d] + p.delta()[d];
                });
            };
            per_patch_(static_cast<int>(ln), [&](auto& patch) {
                for (auto& pop : patch.ions)
                {
                    add(0, pop.domainParticles());
                    add(1, pop.levelGhostParticlesOld());
                }
            });
            if (n_ranks_ > 1)
                MPI_Allreduce(MPI_IN_PLACE, sums.data(), static_cast<int>(sums.size()), MPI_DOUBLE,
                              MPI_SUM, MPI_COMM_WORLD);
            out.particles.insert(out.particles.end(), sums.begin(), sums.end());
        }
    }
    return out;
}

template<auto opts>
void Simulator<opts>::check_rebalance_(RebalanceState const& before, double const t)
{
    auto const after = rebalance_state_();
    auto const close = [](double const a, double const b) {
        return std::abs(a - b) <= 1e-12 * std::max(1., std::abs(a));
    };
    std::string changed;
    if (after.values.size() != before.values.size())
        changed += " (the levels)";
    for (auto const k : mkn::kul::range(std::min(after.values.size(), before.values.size())))
    {
        auto const& a = before.values[k];
        auto const& b = after.values[k];
        bool same     = a.size() == b.size();
        for (auto const i : mkn::kul::range(same ? a.size() : 0))
            same = same and close(a[i], b[i]);
        if (!same)
            changed += " values#" + std::to_string(k);
    }
    for (auto const k : mkn::kul::range(std::min(after.particles.size(), before.particles.size())))
        if (!close(before.particles[k], after.particles[k]))
            changed += " particles#" + std::to_string(k);
    if (!changed.empty())
        throw std::runtime_error("plamn Simulator: the rebalance at t " + std::to_string(t)
                                 + " changed" + changed);
}

} // namespace plamn

#endif // PLAMN_SIMULATOR_SIMULATOR_SUMS_HPP
