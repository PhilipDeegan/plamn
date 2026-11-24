#ifndef PLAMN_SOLVER_HYBRID_PATCHES_HPP
#define PLAMN_SOLVER_HYBRID_PATCHES_HPP

// a hybrid level's patches: made (with the cells around them other patches hold), and their
//  particles counted for load balancing, as the model's initial functions will put them (before
//  any) or as the levels hold them (running)

#include "damn/amr/hierarchy.hpp"
#include "damn/core/def.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/geometry/point.hpp"

#include "plamn/core/data/particles/particle_iteration.hpp"
#include "plamn/solver/hybrid/model.hpp"

#include "mkn/kul/std/array.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <format>
#include <utility>

namespace plamn::solver::hybrid::detail
{
// a patch's domain particles per cell, all populations
template<std::size_t dim>
struct CellCounts
{
    damn::core::Box<int, dim> cells;
    std::vector<double> n = std::vector<double>(cells.size(), 0.);

    MKN_KUL_FN_SIG_RET std::size_t index(damn::core::Point<int, dim> const& c) const
    {
        std::size_t k = 0, stride = 1;
        for (std::size_t d = 0; d < dim; ++d)
        {
            k += static_cast<std::size_t>(c[d] - cells.lower[d]) * stride;
            stride *= static_cast<std::size_t>(cells.upper[d] - cells.lower[d] + 1);
        }
        return k;
    }

    // the particles in box's cells (those keep(cell) keeps)
    MKN_KUL_FN_SIG_RET double in(damn::core::Box<int, dim> const& box, auto&& keep) const
    {
        double sum = 0;
        if (auto const overlap = box * cells)
            damn::core::for_each_point(*overlap, [&](auto const& c) {
                if (keep(c))
                    sum += n[index(c)];
            });
        return sum;
    }
    MKN_KUL_FN_SIG_RET double in(damn::core::Box<int, dim> const& box) const
    {
        return in(box, [](auto const&) { return true; });
    }
};

template<typename Patch>
MKN_KUL_FN_SIG_RET auto cell_counts(Patch const& patch)
{
    auto constexpr dim = Patch::dimension;
    CellCounts<dim> counts{patch.layout.cells};
    for (auto const& pop : patch.ions)
    {
        core::for_each_particle(pop.domainParticles(), [&](auto const& particle) {
            damn::core::Point<int, dim> const c{particle.iCell()};
            if (damn::core::isIn(c, counts.cells))
                counts.n[counts.index(c)] += 1;
        });
    }
    return counts;
}

// true if coarse cell c is refined into one of the patches (whole coarse cells)
template<std::size_t dim>
MKN_KUL_FN_SIG_RET bool refined_into(damn::core::Point<int, dim> const& c, int const ratio,
                                     std::vector<damn::amr::Patch<dim>> const& patches)
{
    auto const fine = c * ratio;
    for (auto const& p : patches)
        if (damn::core::isIn(fine, p.box))
            return true;
    return false;
}
} // namespace plamn::solver::hybrid::detail

namespace plamn::solver::hybrid
{
// a patch of level ln (its cells): its particle ghost box among the level's other patches
template<auto opts, std::size_t dim>
MKN_KUL_FN_SIG_RET auto surroundings(HybridModel<opts> const&, damn::amr::Hierarchy<dim> const& h,
                                     int const ln, damn::core::Box<int, dim> const& box)
{
    using Patch_t      = HybridModel<opts>::Patch_t;
    auto constexpr pgw = static_cast<int>(Patch_t::particle_ghost_width);
    using Surroundings = Patch_t::Surroundings;
    return Surroundings{damn::amr::covered_by_level(
                            h, ln, damn::core::grow(box, mkn::kul::ConstArray<int, dim>(pgw))),
                        damn::amr::owned_elsewhere(h, ln, box, pgw)};
}

// sim: the input's "simulation"
template<auto opts>
MKN_KUL_FN_SIG_RET auto make_patch(HybridModel<opts> const& model, auto const& layout,
                                   auto const& sim, auto const& surroundings)
{
    using Patch_t = HybridModel<opts>::Patch_t;
    auto patch    = std::make_unique<Patch_t>(layout, sim, surroundings);
    // the requested computed quantities' scratch grids from the start: every patch then holds
    //  the same resources, whatever was computed where (a level moved between ranks, regrid or
    //  rebalance, packs and unpacks every resource of a patch in order); none for the others
    for (auto const& key : model.requested)
        for (auto const& [name, qty] : model.computed.at(key).grids)
            std::ignore = patch->scratch_grid(name, qty);
    return patch;
}

// layout -> the particles the populations' initial functions put in its cells (populated
//  cells: density above each population's cut off), before any exists
template<auto opts>
MKN_KUL_FN_SIG_RET auto initial_particle_counter(HybridModel<opts> const&, auto const& sim)
{
    using Patch_t       = HybridModel<opts>::Patch_t;
    using Initializers  = Patch_t::Types::ParticleInitializerFactory_t;
    auto const& ions    = sim["ions"];
    auto const nbr_pops = ions["nbrPopulations"].template to<std::size_t>();
    using Initializer_t = decltype(Initializers::create(ions["pop0"]["particle_initializer"]));
    std::vector<Initializer_t> initializers;
    for (std::size_t i = 0; i < nbr_pops; ++i)
    {
        auto const& pop = ions["pop" + std::to_string(i)];
        initializers.push_back(Initializers::create(pop["particle_initializer"],
                                                    pop["name"].template to<std::string>()));
    }
    return [initializers = std::move(initializers)](auto const& layout) {
        double n = 0;
        for (auto const& initializer : initializers)
            n += static_cast<double>(initializer->nbrParticles(layout));
        return n;
    };
}

// weights[b] += the particles the level holds in boxes[b]
template<auto opts>
void add_particles(HybridModel<opts> const&, auto&& level, auto const& boxes,
                   std::vector<double>& weights)
{
    for (auto const& patch : level.data().data)
    {
        auto const counts = detail::cell_counts(*patch);
        for (std::size_t b = 0; b < boxes.size(); ++b)
            weights[b] += counts.in(boxes[b]);
    }
}

// weights[b] += the particles the coarser level would split into boxes[b] (cells of the finer
//  level), but where the finer level was (old: its patches, whose own particles are counted)
template<auto opts, std::size_t dim>
void add_split_particles(HybridModel<opts> const&, auto&& coarser, auto const& boxes,
                         int const ratio, std::vector<damn::amr::Patch<dim>> const& old,
                         std::vector<double>& weights)
{
    for (auto const& patch : coarser.data().data)
    {
        auto const counts = detail::cell_counts(*patch);
        for (std::size_t b = 0; b < boxes.size(); ++b)
            weights[b] += opts.nbRefinedPart
                          * counts.in(damn::core::coarsen(boxes[b], ratio), [&](auto const& c) {
                                return !detail::refined_into(c, ratio, old);
                            });
    }
}

} // namespace plamn::solver::hybrid

#endif // PLAMN_SOLVER_HYBRID_PATCHES_HPP
