#ifndef PLAMN_SOLVER_HYBRID_COMPUTED_HPP
#define PLAMN_SOLVER_HYBRID_COMPUTED_HPP

#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"
#include "plamn/core/numerics/interpolator/interpolating.hpp"
#include "plamn/solver/hybrid/model.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <format>
#include <iterator>

namespace plamn::solver::hybrid
{
// the hybrid model's computed quantities: the momentum tensor (PHARE's FluidDiagnosticWriter),
//  of all populations ("/ions/momentum_tensor", all deposited into one) and of each
//  ("/ions/pop/<name>/momentum_tensor", with its mass): domain particles, the border sum, then
//  the level ghosts at the end of the coarser step (levelGhostParticlesOld, after last_step).
//  resource(key, names): the grids a border sum adds up
template<auto opts>
void setup_computed(HybridModel<opts>& model, auto&& resource)
{
    using Patch_t                      = HybridModel<opts>::Patch_t;
    using Grid_t                       = Patch_t::Grid_t;
    using Computed_t                   = HybridModel<opts>::Computed_t;
    auto constexpr static dimension    = opts.dimension;
    auto constexpr static interp_order = opts.interp_order;
    using Qty                          = core::HybridQuantity::Scalar;
    std::array<char const*, 6> ijk     = {"_xx", "_xy", "_xz", "_yy", "_yz", "_zz"};
    auto const qtys   = core::HybridQuantity::componentsQuantities(core::HybridQuantity::Tensor::M);
    auto const tensor = [&](std::string const& name) {
        std::vector<std::pair<std::string, Qty>> out;
        out.reserve(ijk.size());
        for (std::size_t c = 0; c < ijk.size(); ++c)
            out.emplace_back(name + ijk[c], qtys[c]);
        return out;
    };

    // deposits particles into the tensor view over grids, if the layout can
    auto const deposit = [](auto type, auto& pop, auto& particles, auto& view,
                            std::vector<Grid_t*> const& grids, auto const& layout) {
        core::MomentumTensorInterpolating<dimension, interp_order> interpolate;
        for (std::size_t c = 0; c < 6; ++c)
            view[c].setBuffer(grids[c]);
        if constexpr (requires {
                          interpolate.template particleToMesh<decltype(type)::value>(
                              particles, view, layout, 1.);
                      })
            interpolate.template particleToMesh<decltype(type)::value>(particles, view, layout,
                                                                       pop.mass());
        else
            throw std::runtime_error("plamn: the momentum tensor is not deposited for this "
                                     "particle layout");
    };
    using Domain     = std::integral_constant<core::ParticleType, core::ParticleType::Domain>;
    using LevelGhost = std::integral_constant<core::ParticleType, core::ParticleType::LevelGhost>;
    auto const zero  = [](std::vector<Grid_t*> const& grids) {
        for (auto* g : grids)
            g->fill(0.);
    };

    // pops(patch): the populations deposited; view(patch, pop): the tensor view to deposit
    //  through, named as the grids (Field::setBuffer checks)
    auto const add_tensor
        = [&](std::string const& key, std::string const& name, auto pops, auto view) {
              auto const grids = tensor(name);
              std::vector<std::string> names;
              for (auto const& [n, _] : grids)
                  names.push_back(n);
              resource(key, names);
              model.computed[key] = Computed_t{
                  grids,
                  [=](Patch_t& patch, std::vector<Grid_t*> const& g) {
                      zero(g);
                      for (auto& pop : pops(patch))
                          deposit(Domain{}, pop.get(), pop.get().domainParticles(),
                                  view(patch, pop.get()), g, patch.layout);
                  },
                  key,
                  [=](Patch_t& patch, std::vector<Grid_t*> const& g) {
                      for (auto& pop : pops(patch))
                          deposit(LevelGhost{}, pop.get(), pop.get().levelGhostParticlesOld(),
                                  view(patch, pop.get()), g, patch.layout);
                  }};
          };

    using PopRef = std::reference_wrapper<
        std::remove_reference_t<decltype(*std::declval<Patch_t&>().ions.begin())>>;
    add_tensor(
        "/ions/momentum_tensor", "momentumTensor",
        [](Patch_t& patch) {
            std::vector<PopRef> out;
            for (auto& pop : patch.ions)
                out.emplace_back(pop);
            return out;
        },
        [](Patch_t& patch, auto&) -> auto& { return patch.ions.momentumTensor(); });
    for (std::size_t i = 0; i < model.population_names.size(); ++i)
        add_tensor(
            "/ions/pop/" + model.population_names[i] + "/momentum_tensor",
            model.population_names[i] + "_momentumTensor",
            [i](Patch_t& patch) {
                return std::vector<PopRef>{std::ref(*(patch.ions.begin() + i))};
            },
            [](Patch_t&, auto& pop) -> auto& { return pop.momentumTensor(); });
}

} // namespace plamn::solver::hybrid

#endif // PLAMN_SOLVER_HYBRID_COMPUTED_HPP
