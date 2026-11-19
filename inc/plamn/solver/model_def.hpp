#ifndef PLAMN_SOLVER_MODEL_DEF_HPP
#define PLAMN_SOLVER_MODEL_DEF_HPP

// the models a build enables, from its SimOpts (simulator_def.hpp): a disabled model's types and
//  storage are empty specializations (models, stacks.hpp), never naming its patches, particles...
//  Levels are MHD..., then hybrid..., then fully kinetic...: above MHD MHD or hybrid, above
//  hybrid hybrid or fully kinetic

namespace plamn::solver
{
template<auto opts>
inline constexpr bool has_hybrid_v = opts.hybrid_enabled;

template<auto opts>
inline constexpr bool has_mhd_v = opts.mhd_enabled;

} // namespace plamn::solver

#endif // PLAMN_SOLVER_MODEL_DEF_HPP
