#ifndef PLAMN_CORE_BOUNDARY_BOUNDARY_DEFS_HPP
#define PLAMN_CORE_BOUNDARY_BOUNDARY_DEFS_HPP

// physical boundary kinds, as pyphare writes them (simulation/grid/domain_boundaries)
// only the enum so far: boundary conditions are not ported yet

namespace plamn::core
{
enum class BoundaryType { None, Reflective, SuperMagnetofastInflow, Open };

} // namespace plamn::core

#endif // PLAMN_CORE_BOUNDARY_BOUNDARY_DEFS_HPP
