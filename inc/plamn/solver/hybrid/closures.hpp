#ifndef PLAMN_SOLVER_HYBRID_CLOSURES_HPP
#define PLAMN_SOLVER_HYBRID_CLOSURES_HPP

// the electron pressure closures (core/data/electrons/closures.hpp) on a whole level: an overload
//  per closure, std::visit on the model's closure picks it (pressure_level). Unified memory
//  (untiled): one launch for the level; host memory or tiled fields: per patch on the pool
//
//   pressure(closure, level)   every patch's Pe from its Ne (after the electrons' Ve)

#include "plamn/core/data/electrons/closures.hpp"

#include "plamn/core/data/field/field_algorithm.hpp"
#include "plamn/core/data/field/level_fields.hpp"
#include "plamn/solver/hybrid/model.hpp"
#include "damn/core/def/damn_config.hpp"
#include "plamn/core/data/particles/particles_def.hpp"

#include <variant>

namespace plamn::solver::hybrid
{
// isothermal: Pe = Ne Te
template<auto opts>
    requires(opts.alloc_mode == damn::AllocatorMode::GPU_UNIFIED
             and !core::is_tiled(opts.layout_mode))
void pressure(core::IsothermalClosure const& closure, auto& level)
{
    auto constexpr am = opts.alloc_mode;
    auto const Ne     = core::level_of<am>(
        level, [](auto& patch, auto& fields) { fields.add(patch.electrons().density()); });
    core::level_of<am>(level, [](auto& patch, auto& fields) {
        fields.add(patch.electrons().pressure());
    }).transform([Te = closure.Te] DAMN_ALL(double& pe, double const n) { pe = n * Te; }, Ne);
}
template<auto opts>
    requires(opts.alloc_mode == damn::AllocatorMode::CPU or core::is_tiled(opts.layout_mode))
void pressure(core::IsothermalClosure const& closure, auto& level)
{
    level.per_patch_threaded([Te = closure.Te](auto& patch) {
        auto& electrons = patch.electrons();
        core::transform(electrons.density(), electrons.pressure(),
                        [Te](auto const n) { return n * Te; });
    });
}

// the model's closure's Pe on every patch of the level
template<auto opts>
void pressure_level(HybridModel<opts> const& model, auto& level)
{
    std::visit([&](auto const& closure) { pressure<opts>(closure, level); }, *model.closure);
}

} // namespace plamn::solver::hybrid

#endif // PLAMN_SOLVER_HYBRID_CLOSURES_HPP
