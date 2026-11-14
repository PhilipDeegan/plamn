#ifndef PLAMN_OPTIONS_HYBRID_OPTIONS_HPP
#define PLAMN_OPTIONS_HYBRID_OPTIONS_HPP

#include "plamn/core/data/grid/ghost_width_calculator.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"

namespace plamn
{

template<auto opts_>
struct HybridFieldOptions
{
    using Quantity = core::HybridQuantity;
    using Scalar   = Quantity::Scalar;
    using Vector   = Quantity::Vector;
    using Tensor   = Quantity::Tensor;

    auto static constexpr opts         = opts_;
    auto static constexpr dimension    = opts.dimension;
    auto static constexpr interp_order = opts.interp_order;
    // core/data/grid/ghost_width_calculator.hpp: both follow from the allowed cell jump
    auto static constexpr particle_allowed_cell_jump = core::particle_allowed_cell_jump;
    auto static constexpr particle_ghost_width
        = core::particleGhostWidth<interp_order, particle_allowed_cell_jump>();
    auto static constexpr field_ghost_width
        = core::nbrGhostsFromInterpOrder<interp_order, particle_allowed_cell_jump>();

    static_assert(field_ghost_width % 2 == 0); // whole coarse cells at refinement ratio 2
    static_assert(field_ghost_width >= particle_ghost_width + core::stencilReach<interp_order>());
};

template<HybridFieldOptions opts>
struct HybridOptions
{
    using FieldOptions   = decltype(opts);

    auto static constexpr field_options        = opts;
    auto static constexpr dimension            = opts.dimension;
    auto static constexpr interp_order         = opts.interp_order;
    auto static constexpr field_ghost_width    = opts.field_ghost_width;
    auto static constexpr particle_ghost_width = opts.particle_ghost_width;
    auto static constexpr particle_allowed_cell_jump = opts.particle_allowed_cell_jump;
    auto static constexpr nbRefinedPart        = opts.opts.nbRefinedPart;
};




} // namespace plamn

#endif // PLAMN_OPTIONS_HYBRID_OPTIONS_HPP
