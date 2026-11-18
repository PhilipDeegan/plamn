#ifndef PLAMN_SIMULATOR_SIMULATOR_COMPUTED_HPP
#define PLAMN_SIMULATOR_SIMULATOR_COMPUTED_HPP

// computed quantities: what the model does not keep (e.g. the momentum tensor), made on demand
// into scratch grids of the patches (HybridPatch::scratch, kept for the next time), looked up by
// key (a diagnostic's quantity: "/ions/momentum_tensor", "/ions/pop/<name>/momentum_tensor")
// by the diagnostics and the DataWrangler
//
//   Computed{grids, local, border_sum, after}
//     grids        the scratch grids' names and quantities, components in file order
//     local        per patch: zeroes and fills its grids (required)
//     border_sum   optional: a resources_ key, the grids' contributions to the nodes patches
//                  share summed (collective)
//     after        optional: per patch, after the sum (e.g. the level ghosts' deposit, which
//                  only the patch at the level border gets)
//
//   compute(ln, key)            runs key's steps on level ln (collective if it sums borders)
//   computed(patch, key)        the patch's grids of key, as compute left them
//
// the momentum tensor (PHARE's FluidDiagnosticWriter): domain particles, the border sum, then
// the level ghosts at the end of the coarser step (levelGhostParticlesOld, after last_step_);
// per population with its mass, the total as all populations deposited into one
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
void Simulator<opts>::setup_computed_()
{
    using Qty                      = core::HybridQuantity::Scalar;
    std::array<char const*, 6> ijk = {"_xx", "_xy", "_xz", "_yy", "_yz", "_zz"};
    auto const qtys   = core::HybridQuantity::componentsQuantities(core::HybridQuantity::Tensor::M);
    auto const tensor = [&](std::string const& name) {
        std::vector<std::pair<std::string, Qty>> out;
        for (std::size_t c = 0; c < 6; ++c)
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
    auto const add_tensor = [&](std::string const& key, std::string const& name, auto pops,
                                auto view) {
        auto const grids = tensor(name);
        std::vector<std::string> names;
        for (auto const& [n, _] : grids)
            names.push_back(n);
        resources_[key] = damn::amr::same_names(names);
        computed_[key]
            = Computed{grids,
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
    for (std::size_t i = 0; i < hybrid_.population_names.size(); ++i)
        add_tensor(
            "/ions/pop/" + hybrid_.population_names[i] + "/momentum_tensor",
            hybrid_.population_names[i] + "_momentumTensor",
            [i](Patch_t& patch) {
                return std::vector<PopRef>{std::ref(*(patch.ions.begin() + i))};
            },
            [](Patch_t&, auto& pop) -> auto& { return pop.momentumTensor(); });
}

template<auto opts>
typename Simulator<opts>::Computed const&
Simulator<opts>::computed_for_(std::string const& key) const
{
    auto const it = computed_.find(key);
    if (it == computed_.end())
        throw std::runtime_error("plamn: no computed quantity " + key);
    return it->second;
}

template<auto opts>
bool Simulator<opts>::is_computed(std::string const& key) const
{
    return computed_.contains(key);
}

template<auto opts>
std::vector<typename Simulator<opts>::Grid_t*> Simulator<opts>::computed(Patch_t& patch,
                                                                         std::string const& key)
{
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
