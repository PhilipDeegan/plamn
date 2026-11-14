#ifndef PLAMN_CORE_DATA_GRID_GHOST_WIDTH_CALCULATOR_HPP
#define PLAMN_CORE_DATA_GRID_GHOST_WIDTH_CALCULATOR_HPP

#include <cstdint>

namespace plamn::core
{

// ============================================================================
// Utility: Round up to nearest even number
// ============================================================================

constexpr inline std::uint32_t roundUpToEven(std::uint32_t n)
{
    return (n % 2 == 0) ? n : n + 1;
}


// ============================================================================
// Ghost Width Computation Functions
// ============================================================================

/**
 * @brief Most cells a domain particle may move in one advance (two half pushes): a CFL for
 * particles.
 *
 * A domain particle moving further throws (PHAREHUB/PHARE#1565); level ghost particles leaving the
 * particle ghost box are dropped. The particle and field ghost widths below follow from it.
 */
inline constexpr std::uint32_t particle_allowed_cell_jump = 2;

/**
 * @brief Cells the interpolation stencil reaches outside a particle's cell, the worst of primal
 * and dual, either side: a particle in cell c touches nodes c - shift ... c - shift + order, with
 * shift from Interpolator::computeStartLeftShift.
 */
template<std::uint32_t interp_order>
constexpr std::uint32_t stencilReach()
{
    static_assert(interp_order > 0 and interp_order < 4);
    return interp_order == 3 ? 2 : 1;
}

/**
 * @brief Ghost cells level ghost particles must cover for their deposits to reach every domain
 * node of the level.
 */
template<std::uint32_t interp_order>
constexpr std::uint32_t levelGhostParticleWidth()
{
    static_assert(interp_order > 0 and interp_order < 4);
    return interp_order == 1 ? 1 : 2;
}

/**
 * @brief Particle ghost width: room for level ghost particles, and for domain particles that moved
 * out by up to the allowed cell jump.
 */
template<std::uint32_t interp_order, std::uint32_t jump = particle_allowed_cell_jump>
constexpr std::uint32_t particleGhostWidth()
{
    auto constexpr level_ghost = levelGhostParticleWidth<interp_order>();
    return jump > level_ghost ? jump : level_ghost;
}

/**
 * @brief Compute ghost width for Hybrid PIC model based on interpolation order.
 *
 * Ghost cells are needed for:
 * - Particle-mesh interpolation of particles in the particle ghost box: particleGhostWidth plus
 *   the stencil reach
 * - Rounded to even so the ghost box stays a whole-coarse-cell union (lower even / upper odd) at
 *   refinement ratio 2 — the invariant the magnetic touch-up's fill-box round-out clips
 *   against (see coarse_cell_round_out.hpp and
 *   MagneticRefinePatchStrategy::reconstructionRegion)
 *
 * With a jump of 1: 2, 4, 4 for interp orders 1, 2, 3, PHARE's widths. With 2: 4, 4, 4.
 */
template<std::uint32_t interp_order, std::uint32_t jump = particle_allowed_cell_jump>
constexpr std::uint32_t nbrGhostsFromInterpOrder()
{
    return roundUpToEven(particleGhostWidth<interp_order, jump>() + stencilReach<interp_order>());
}

static_assert(nbrGhostsFromInterpOrder<1, 1>() == 2 and nbrGhostsFromInterpOrder<2, 1>() == 4
              and nbrGhostsFromInterpOrder<3, 1>() == 4); // PHARE's widths, jump 1


/**
 * @brief Compute ghost width for MHD model based on reconstruction stencil.
 *
 * Ghost cells are needed for:
 * - Reconstruction stencil width
 * - One layer because ampere computes J on the ghost box shrinked by one, so J is only valid on
 *   ghost_width - 1 layers while the reconstruction reads it out to the stencil width
 * - Rounded to even so the ghost box stays a whole-coarse-cell union (lower even / upper odd) at
 *   refinement ratio 2 — the invariant the magnetic touch-up's fill-box round-out clips
 *   against (see coarse_cell_round_out.hpp and
 *   MagneticRefinePatchStrategy::reconstructionRegion)
 *
 * There is no hyper-resistivity layer: the dissipative electric field takes its Laplacian on J at
 * its native edge location, and the flux contributions project it to the face, so they stay within
 * the reconstruction reach. See DissipativeElectricField and Godunov::transverse_on_face_.
 */
template<std::uint32_t reconstruction_nghosts>
constexpr std::uint32_t nbrGhostsFromReconstruction()
{
    return roundUpToEven(reconstruction_nghosts + 1);
}


} // namespace plamn::core

#endif // PLAMN_CORE_DATA_GRID_GHOST_WIDTH_CALCULATOR_HPP
