#ifndef PLAMN_SIMULATOR_SIMULATOR_COMPUTED_HPP
#define PLAMN_SIMULATOR_SIMULATOR_COMPUTED_HPP

// computed quantities (solver/computed.hpp), the models' (solver/hybrid/computed.hpp):
//   compute(ln, key)            runs key's steps on level ln (collective if it sums borders)
//   computed(patch, key)        the patch's grids of key, as compute left them
// IWYU pragma: private, include "plamn/simulator/simulator.hpp"

#include "damn/core/logger.hpp"

#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"
#include "plamn/core/numerics/interpolator/interpolating.hpp"
#include "plamn/simulator/simulator.hpp" // IWYU pragma: keep (the class)

#include <array>
#include <cstddef>
#include <stdexcept>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <functional>
#include <vector>

namespace plamn
{
template<auto opts>
auto const& Simulator<opts>::computed_for_(std::string const& key) const
{
    auto const it = models_.hybrid.computed.find(key);
    if (it == models_.hybrid.computed.end())
        throw std::runtime_error("plamn: no computed quantity " + key);
    return it->second;
}

template<auto opts>
bool Simulator<opts>::is_computed(std::string const& key) const
{
    return models_.hybrid.computed.contains(key);
}

template<auto opts>
auto Simulator<opts>::computed(auto& patch, std::string const& key)
{
    using Grid_t = solver::HybridModel<opts>::Grid_t;
    std::vector<Grid_t*> out;
    for (auto const& [name, qty] : computed_for_(key).grids)
        out.push_back(std::addressof(patch.scratch_grid(name, qty))); // Grid overloads operator&
    return out;
}

template<auto opts>
void Simulator<opts>::compute(std::size_t const ln, std::string const& key)
{
    DAMN_LOG_SCOPE(1, "Simulator::compute");
    auto const& c = computed_for_(key);
    per_patch_(ln, [&](auto& patch) { c.local(patch, computed(patch, key)); });
    if (!c.border_sum.empty())
        copy_(ln, damn::amr::sum_borders, c.border_sum);
    if (c.after)
        per_patch_(ln, [&](auto& patch) { c.after(patch, computed(patch, key)); });
}

} // namespace plamn

#endif // PLAMN_SIMULATOR_SIMULATOR_COMPUTED_HPP
