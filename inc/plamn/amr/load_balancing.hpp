#ifndef PLAMN_AMR_LOAD_BALANCING_HPP
#define PLAMN_AMR_LOAD_BALANCING_HPP

// PHARE's load balancing schedule: when to rebalance level 0 (CoarsestRebalanceCheck)

#include "damn/core/def.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace plamn::amr
{
// PHARE's CoarsestRebalanceCheck: once per step, whether level 0 is rebalanced in automatic
//  mode. Workloads are only gathered (collective) when a check is due; every rank holds the
//  same state so they agree on when. After a balanced check the interval grows by backoff
//  up to max_interval, after a rebalance it restarts from interval
struct CoarsestRebalanceCheck
{
    std::size_t first = 200, backoff = 2, max_interval = 1000;
    double tolerance = 0.05;
    std::size_t interval{std::max<std::size_t>(first, 1)};
    std::size_t steps_to_next_check = 0;

    MKN_KUL_FN_SIG_RET static bool imbalanced(std::vector<double> const& workloads,
                                              double const tolerance)
    {
        if (workloads.empty())
            return false;
        double total = 0;
        for (auto const w : workloads)
            total += w;
        if (total <= 0)
            return false;
        auto const average = total / static_cast<double>(workloads.size());
        return *std::max_element(workloads.begin(), workloads.end()) / average - 1 > tolerance;
    }

    MKN_KUL_FN_SIG_RET bool operator()(auto&& gather_workloads)
    {
        if (steps_to_next_check > 0)
        {
            --steps_to_next_check;
            return false;
        }
        bool const rebalance = imbalanced(gather_workloads(), tolerance);
        if (rebalance)
            interval = std::max<std::size_t>(first, 1);
        steps_to_next_check = interval - 1;
        if (!rebalance)
            interval = std::min(interval * std::max<std::size_t>(backoff, 1),
                                std::max<std::size_t>(max_interval, 1));
        return rebalance;
    }
};

} // namespace plamn::amr

#endif // PLAMN_AMR_LOAD_BALANCING_HPP
