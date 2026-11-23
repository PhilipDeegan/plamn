#ifndef PLAMN_SOLVER_HYBRID_LEVEL_STEPS_HPP
#define PLAMN_SOLVER_HYBRID_LEVEL_STEPS_HPP

// a hybrid level and its coarser hybrid level, around the finer level's substeps (PHARE's
//  HybridHybridMessengerStrategy, as damn::amr::advance's hooks call them)
//   prepare_step   E and B at the start of a step: the finer level's time interpolation, reflux
//   first_step     the finer level's level ghost particles of the coarser level's step end
//   last_step      those become the old ones
//   synchronize    the finer level onto the coarser one, its B refluxed, its ghosts refilled
//
// level: the simulator's services for one level (solver/hybrid/advance.hpp), and
//   refine(op, region, pairs)            from the coarser level into it
//   coarsen(op, pairs)                   from it onto the coarser level
//   coarser()                            the coarser level's services
//   coarse_dt()                          the coarser level's step it subcycles in
//   time_ratio()                         the coarser level's dt over a new level's first substep
//   substep_weight()                     its last step's share of its coarser level's step

#include "plamn/core/data/field/field_tiles.hpp"
#include "damn/amr/copy.hpp"
#include "damn/amr/level_map.hpp"
#include "damn/core/logger.hpp"

#include "plamn/amr/coarsen.hpp"
#include "plamn/amr/level_pairs.hpp"
#include "plamn/amr/refine.hpp"
#include "plamn/core/data/particles/exporting/particles_exporting.hpp"
#include "plamn/core/numerics/faraday/faraday_transformer.hpp"
#include "plamn/solver/hybrid/model.hpp"

#include "phlop/timing/scope.hpp"

#include <cstddef>
#include <utility>
#include <string>
#include <vector>

namespace plamn::solver::hybrid
{
template<auto opts>
void prepare_step(HybridModel<opts> const&, auto&& level)
{
    level.per_patch([](auto& patch) {
        for (std::size_t i = 0; i < 3; ++i)
        {
            patch.oldB[i].copyData(patch.emB[i]);
            patch.oldE[i].copyData(patch.emE[i]);
        }
    });
}

// a level ghost particle lives in one patch, its cell's owner (Surroundings): the others' copies
//  of a split dropped, old (initialization, regrid, restart) or new (coarse step)
void own_level_ghosts(auto&& level, bool const new_ghosts)
{
    level.per_patch([&](auto& patch) {
        if (patch.level_ghost_not_owned.empty())
            return;
        for (auto& pop : patch.ions)
            core::delete_particles_in(new_ghosts ? pop.levelGhostParticlesNew()
                                                 : pop.levelGhostParticlesOld(),
                                      patch.level_ghost_not_owned);
    });
}

// the coarser level is at the end of its step: the level ghost particles of then, split from its
//  domain particles
template<auto opts>
void first_step(HybridModel<opts> const& model, auto&& level)
{
    using Splitter_t = HybridModel<opts>::Splitter_t;
    level.per_patch([](auto& patch) {
        for (auto& pop : patch.ions)
            pop.levelGhostParticlesNew().clear();
        for (auto& e : patch.fluxSumE)
            e.zero();
    });
    level.refine(amr::ParticleSplitter<Splitter_t>{}, damn::amr::LevelRegion::LevelGhosts,
                 model.particle_routes->new_ghost_splits);
    own_level_ghosts(level, /*new_ghosts=*/true);
}

template<auto opts>
void last_step(HybridModel<opts> const&, auto&& level)
{
    level.per_patch([](auto& patch) {
        for (auto& pop : patch.ions)
        {
            std::swap(pop.levelGhostParticlesNew(), pop.levelGhostParticlesOld());
            pop.levelGhostParticlesNew().clear();
            pop.levelGhostParticles() = pop.levelGhostParticlesOld();
        }
    });
}

// the finest level's share of the coarser level's reflux, per substep: the time average of E over
//  them (coarser finer levels accumulate theirs when synchronized)
void accumulate_flux_sum(auto&& level, double const coef)
{
    level.per_patch([&](auto& patch) {
        for (std::size_t i = 0; i < 3; ++i)
        { // the same tiling: their buffers pair up
            std::vector<double const*> avg;
            core::for_each_buffer(std::as_const(patch.avgE[i]),
                                  [&](auto const* data, auto) { avg.push_back(data); });
            std::size_t b = 0;
            core::for_each_buffer(patch.fluxSumE[i], [&](auto* sum, auto const values) {
                auto const* a = avg[b++];
                for (std::size_t k = 0; k < values; ++k)
                    sum[k] += coef * a[k];
            });
        }
    });
}

// the level onto the coarser one, both at t: E and the total ion charge density and flux
// coarsened (as PHARE's synchronize: the populations' moments, the mass density and the bulk
// velocity stay the coarser level's own), then the coarser level's B recomputed from the fine
// time average of E (reflux), then its ghosts refilled
// the totals a synchronization coarsens onto the coarser level under this one (PHARE's
//  chargeDensitySynchronizers_, ionChargeFluxSynchronizers_); their ghosts are the caller's
void coarsen_moments(auto&& level)
{
    auto moments = amr::vector_pairs("ions_charge_flux", "ions_charge_flux");
    moments.push_back({"ions_charge_density", "ions_charge_density", ""});
    level.coarsen(amr::MomentsCoarsener{}, moments);
}

template<auto opts>
void synchronize(HybridModel<opts> const&, auto&& level, double const t)
{
    DAMN_LOG_SCOPE(1, "Simulator::synchronize_");
    auto coarse = level.coarser();

    level.coarsen(amr::ElectricFieldCoarsener{}, amr::vector_pairs("EM_E", "EM_E"));
    coarsen_moments(level);

    // reflux: the coarser Eavg under the fine level is the fine one's time average
    level.coarsen(amr::ElectricFieldCoarsener{}, amr::vector_pairs("EMAvg_E", "EM_fluxSum_E"));
    coarse.copy(damn::amr::fill_ghosts, "EMAvg_E");
    auto const dt = level.coarse_dt();
    coarse.per_patch([&](auto& patch) {
        core::FaradayTransformer{}(patch.layout, patch.Bold, patch.emAvg.E, patch.em.B, dt);
    });
    coarse.fill_magnetic_ghosts("EM_B", t);
    if (coarse.number() > 0)
        accumulate_flux_sum(coarse, coarse.substep_weight());

    // ghosts after the coarsening
    coarse.fill_electric_ghosts("EM_E", t);
    coarse.copy(damn::amr::fill_ghosts, "IonMoments");
}

} // namespace plamn::solver::hybrid

#endif // PLAMN_SOLVER_HYBRID_LEVEL_STEPS_HPP
