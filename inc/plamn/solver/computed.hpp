#ifndef PLAMN_SOLVER_COMPUTED_HPP
#define PLAMN_SOLVER_COMPUTED_HPP

// what a model does not keep (e.g. the momentum tensor), made on demand into its patches'
//  scratch grids, looked up by key (a diagnostic's quantity)
//   grids        components in file order
//   local        zeroes and fills a patch's grids (required)
//   border_sum   a resource key, the shared nodes summed (collective): local deposits what one
//                patch holds (a level ghost is held by one), the sum completes every patch's copy

#include <functional>
#include <string>
#include <vector>
#include <iterator>

namespace plamn::solver
{
template<typename Patch, typename Grid, typename Scalar>
struct Computed
{
    using Step = std::function<void(Patch&, std::vector<Grid*> const&)>;

    std::vector<std::pair<std::string, Scalar>> grids{};
    Step local{};
    std::string border_sum{}; // empty: none
};

} // namespace plamn::solver

#endif // PLAMN_SOLVER_COMPUTED_HPP
