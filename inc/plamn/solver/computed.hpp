#ifndef PLAMN_SOLVER_COMPUTED_HPP
#define PLAMN_SOLVER_COMPUTED_HPP

// a computed quantity of a model: what the model does not keep (e.g. the momentum tensor), made
//  on demand into scratch grids of its patches, looked up by key (a diagnostic's quantity) by the
//  diagnostics and the DataWrangler (the simulator's compute(ln, key), simulator_computed.hpp)
//
//   grids        the scratch grids' names and quantities, components in file order
//   local        per patch: zeroes and fills its grids (required)
//   border_sum   optional: a resource key, the grids' contributions to the nodes patches share
//                summed (collective): local deposits what one patch holds (a level ghost is
//                held by one), the sum completes every patch's copy

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
