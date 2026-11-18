#ifndef PLAMN_SOLVER_STACKS_HPP
#define PLAMN_SOLVER_STACKS_HPP

// the types of each model a build enables (PHARE's HybridStack / MHDStack): empty when the model
//  is not, so a build never names a disabled model's types

#include "damn/core/def/constants.hpp"

#include "plamn/amr/split/split.hpp" // IWYU pragma: keep (the Splitter specializations)
#include "plamn/amr/split/splitter.hpp"
#include "plamn/simulator/hybrid_patch.hpp"
#include "plamn/solver/hybrid/model.hpp"
#include "plamn/solver/model_def.hpp"
#include "plamn/solver/mhd/model.hpp"

namespace plamn::solver
{
template<auto opts, bool enabled = has_hybrid_v<opts>>
struct HybridStack
{
};

template<auto opts>
struct HybridStack<opts, true>
{
    using Patch_t     = simulator::HybridPatch<opts>;
    using Model_t     = HybridModel<opts>;
    using LevelData_t = Model_t::LevelData_t;
    using Splitter_t  = amr::Splitter<damn::core::DimConst<opts.dimension>,
                                      damn::core::InterpConst<opts.interp_order>,
                                      amr::RefinedParticlesConst<opts.nbRefinedPart>>;
};

template<auto opts, bool enabled = has_mhd_v<opts>>
struct MHDStack
{
};

template<auto opts>
struct MHDStack<opts, true>
{
    using Model_t          = MHDModel<opts>;
    using Reconstruction_t = Model_t::Reconstruction;
};

} // namespace plamn::solver

#endif // PLAMN_SOLVER_STACKS_HPP
