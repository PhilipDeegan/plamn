#ifndef PLAMN_CORE_DATA_GRID_GHOST_WIDTH_CALCULATOR_HPP
#define PLAMN_CORE_DATA_GRID_GHOST_WIDTH_CALCULATOR_HPP

#include <cstdint>

namespace plamn::core
{

// ghost widths of both models: doc/core/data/grid/ghost_width_calculator.md

constexpr inline std::uint32_t roundUpToEven(std::uint32_t n)
{
    return (n % 2 == 0) ? n : n + 1;
}


// most cells a domain particle may move in one advance (two half pushes): a CFL for particles.
//  1: ghost widths 2, 4, 4 (interp orders 1, 2, 3), for etc/compare/phare
#ifndef PLAMN_PARTICLE_ALLOWED_CELL_JUMP
#define PLAMN_PARTICLE_ALLOWED_CELL_JUMP 2
#endif
inline constexpr std::uint32_t particle_allowed_cell_jump = PLAMN_PARTICLE_ALLOWED_CELL_JUMP;

// |delta| past its cell's start after a half push before the Boris pusher throws: 2, or the
//  allowed cell jump if larger
inline constexpr std::uint32_t particle_half_push_limit
    = particle_allowed_cell_jump > 2 ? particle_allowed_cell_jump : 2;

// cells the interpolation stencil reaches outside a particle's cell, the worst of primal and
//  dual, either side
template<std::uint32_t interp_order>
constexpr std::uint32_t stencilReach()
{
    static_assert(interp_order > 0 and interp_order < 4);
    return interp_order == 3 ? 2 : 1;
}

// ghost cells level ghost particles must cover for their deposits to reach every domain node
template<std::uint32_t interp_order>
constexpr std::uint32_t levelGhostParticleWidth()
{
    static_assert(interp_order > 0 and interp_order < 4);
    return interp_order == 1 ? 1 : 2;
}

// room for level ghost particles, and for domain particles that moved out by up to the jump
template<std::uint32_t interp_order, std::uint32_t jump = particle_allowed_cell_jump>
constexpr std::uint32_t particleGhostWidth()
{
    auto constexpr level_ghost = levelGhostParticleWidth<interp_order>();
    return jump > level_ghost ? jump : level_ghost;
}

// even: the ghost box stays a union of whole coarse cells at ratio 2
template<std::uint32_t interp_order, std::uint32_t jump = particle_allowed_cell_jump>
constexpr std::uint32_t nbrGhostsFromInterpOrder()
{
    return roundUpToEven(particleGhostWidth<interp_order, jump>() + stencilReach<interp_order>());
}

static_assert(nbrGhostsFromInterpOrder<1, 1>() == 2 and nbrGhostsFromInterpOrder<2, 1>() == 4
              and nbrGhostsFromInterpOrder<3, 1>() == 4); // jump 1


// + 1: Ampere's J is only valid on ghost_width - 1 layers; even, as nbrGhostsFromInterpOrder
template<std::uint32_t reconstruction_nghosts>
constexpr std::uint32_t nbrGhostsFromReconstruction()
{
    return roundUpToEven(reconstruction_nghosts + 1);
}


} // namespace plamn::core

#endif // PLAMN_CORE_DATA_GRID_GHOST_WIDTH_CALCULATOR_HPP
