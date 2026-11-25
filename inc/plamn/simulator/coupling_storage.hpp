#ifndef PLAMN_SIMULATOR_COUPLING_STORAGE_HPP
#define PLAMN_SIMULATOR_COUPLING_STORAGE_HPP

// what a level's patches hold for neighbouring levels of another model (multi model runs):
//   under  for a finer level of another model above it (MHD under hybrid: E at the step's
//          start; hybrid under full kinetic, later)
//   over   for a coarser level of another model below it (hybrid over MHD; full kinetic over
//          hybrid, later)
// Each an optional, emplaced on the boundary level's patches only (a level's model and its
//  neighbours' are fixed by its index): the other levels' patches hold an empty optional. A
//  side's type void: the build has no such neighbour model, no member at all (a one model build
//  carries nothing)

#include <optional>
#include <type_traits>

namespace plamn::simulator
{
struct NoCoupling
{
};

template<typename T>
using CouplingSlot = std::conditional_t<std::is_void_v<T>, NoCoupling, std::optional<T>>;

template<typename Under, typename Over>
struct CouplingStorage
{
    using Under_t = Under;
    using Over_t  = Over;

    [[no_unique_address]] CouplingSlot<Under> under{};
    [[no_unique_address]] CouplingSlot<Over> over{};
};

} // namespace plamn::simulator

#endif // PLAMN_SIMULATOR_COUPLING_STORAGE_HPP
