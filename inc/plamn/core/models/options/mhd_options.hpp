#ifndef PLAMN_OPTIONS_MHD_OPTIONS_HPP
#define PLAMN_OPTIONS_MHD_OPTIONS_HPP

#include "plamn/core/def.hpp"
#include "plamn/core/models/quantities/mhd_quantities.hpp"
#include "plamn/core/data/grid/ghost_width_calculator.hpp"
#include "plamn/core/numerics/reconstructions/reconstruction_nghosts.hpp"

namespace plamn
{

template<auto opts_>
struct MHDFieldOptions
{
    using Quantity = core::MHDQuantity;
    using Scalar   = Quantity::Scalar;
    using Vector   = Quantity::Vector;
    using Tensor   = Quantity::Tensor;

    auto static constexpr opts      = opts_;
    auto static constexpr dimension = opts.dimension;

    auto static constexpr reconstruction_nghosts
        = MHDOpts::reconstruction_nghosts_v<opts.reconstruction_type>;

    auto static constexpr field_ghost_width
        = core::nbrGhostsFromReconstruction<reconstruction_nghosts>();
};


template<MHDFieldOptions opts>
struct MHDOptions
{
    using FieldOptions   = decltype(opts);

    auto static constexpr field_options     = opts;
    auto static constexpr dimension         = opts.dimension;
    auto static constexpr field_ghost_width = opts.field_ghost_width;
};


} // namespace plamn

#endif // PLAMN_OPTIONS_MHD_OPTIONS_HPP
