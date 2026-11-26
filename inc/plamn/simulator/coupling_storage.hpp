#ifndef PLAMN_SIMULATOR_COUPLING_STORAGE_HPP
#define PLAMN_SIMULATOR_COUPLING_STORAGE_HPP

// what a level's patches hold for a neighbouring level of another model: `under` for a finer
//  one above (MHD under hybrid: E at the step's start), `over` for a coarser one below (hybrid
//  over MHD). Emplaced on the boundary level's patches only (a level's model is fixed by its
//  index); a void side (no such model in the build) has no member at all

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
