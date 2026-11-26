#ifndef PLAMN_SIMULATOR_SIMULATOR_SETUP_HPP
#define PLAMN_SIMULATOR_SIMULATOR_SETUP_HPP

// plamn::Simulator: the input: hierarchy, refinement, tagging, load balancing, resources
// IWYU pragma: private, include "plamn/simulator/simulator.hpp"

#include "damn/core/exec/thread_pool.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/amr/topology.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/mpi/scope_timing.hpp"
#include "damn/mpi/topology.hpp"
#include "mkn/kul/env.hpp"

#include "plamn/simulator/simulator.hpp" // IWYU pragma: keep (the class)
#include "plamn/amr/load_balancing.hpp"
#include "plamn/amr/tagging.hpp"
#include "plamn/core/numerics/ohm/ohm.hpp"
#include "plamn/initializer/data_provider.hpp"
#include "plamn/simulator/simulator_def.hpp"
#include "plamn/solver/models.hpp"
#include "plamn/solver/hybrid/electron/closures.hpp"
#include "plamn/core/data/electrons/closures.hpp"
#include "plamn/core/utilities/variants.hpp"

#include "dict.hpp"
#include "mpi.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <cstdlib>
#include <type_traits>
#include <variant>
#include <vector>
// IWYU pragma: private, include "plamn/simulator/simulator.hpp"

namespace plamn
{
template<auto opts>
void Simulator<opts>::setup_()
{
    auto const& sim = dict_["simulation"];

    // the ranks, if MPI is running (plamn starts it with pyphare's startMPI)
    int mpi_on = 0, mpi_done = 0;
    MPI_Initialized(&mpi_on);
    MPI_Finalized(&mpi_done);
    if (mpi_on and !mpi_done)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &n_ranks_);
    }
    topology_ = n_ranks_ > 1 ? damn::mpi::topology() : damn::amr::single_node();

    if (mpi_on and !mpi_done and damn::mpi::env_flag("PHARE_SCOPE_TIMING")
        and !damn::mpi::scope_timing_active())
    {
        damn::mpi::start_scope_timing(timings_prefix);
        timing_ = true;
    }

    // damn's thread pools (process wide): pyphare's threading, PHARE_THREAD_POOLS and
    //  PHARE_THREADS_PER_POOL first if set (as PHARE). A pool works a patch, its threads the
    //  patch's tiles; reconfigured only when it differs
    {
        auto const setting = [&](char const* env, std::string const& key) {
            if (auto const* v = std::getenv(env); v and *v)
                return static_cast<std::size_t>(std::stoul(v));
            return static_cast<std::size_t>(
                cppdict::get_value(dict_, "simulation/threading/" + key, 1));
        };
        auto const pools   = setting("PHARE_THREAD_POOLS", "pools");
        auto const threads = setting("PHARE_THREADS_PER_POOL", "threads_per_pool");
        using Pool         = damn::core::ThreadPool;
        if (pools != Pool::n_pools or threads != Pool::threads_per_pool)
            Pool::INSTANCE().configure(pools, threads);
    }

    timestepper_ = Timestepper::FROM(dict_);
    dt_          = timestepper_.dt();
    timeStepNbr_ = cppdict::get_value(dict_, "simulation/time_step_nbr", 0);
    finalTime_   = timestepper_.final_time();

    // the hierarchy: level 0 cells from 0, chopped into the largest patch size
    std::array<char const*, 3> constexpr xyz{"x", "y", "z"};
    std::array<int, dimension> cells{};
    for (std::size_t i = 0; i < dimension; ++i)
    {
        cells[i]                   = sim["grid"]["nbr_cells"][xyz[i]].template to<int>();
        hierarchy_.dx[i]           = sim["grid"]["meshsize"][xyz[i]].template to<double>();
        hierarchy_.periodic[i]     = sim["grid"]["periodicities"][xyz[i]].template to<bool>();
        patch_size_[i]             = cells[i];
        hierarchy_.domain.lower[i] = 0;
        hierarchy_.domain.upper[i] = cells[i] - 1;
        domainBox_.push_back(cells[i] - 1);
        cellWidth_.push_back(hierarchy_.dx[i]);
    }
    auto const& amr = sim["AMR"];
    {
        auto mode = cppdict::get_value(dict_, "simulation/AMR/regrid_mode", std::string{"normal"});
        if (auto const* env = std::getenv("PLAMN_REGRID_MODE"))
            mode = env;
        if (mode != "normal" and mode != "high")
            throw std::runtime_error("plamn Simulator: regrid_mode normal or high, not " + mode);
        regrid_mode_ = mode == "high" ? RegridMode::high : RegridMode::normal;
    }
    lb_.partition = cppdict::get_value(dict_, "simulation/AMR/loadbalancing/partition",
                                       std::string{"levels"});
    if (auto const* env = std::getenv("PLAMN_PARTITION"); env and *env)
        lb_.partition = env;
    if (lb_.partition != "levels" and lb_.partition != "keep" and lb_.partition != "finest"
        and lb_.partition != "aligned")
        throw std::runtime_error("plamn Simulator: partition levels, keep, finest or aligned, not "
                                 + lb_.partition);
    lb_.spill = cppdict::get_value(dict_, "simulation/AMR/loadbalancing/spill", -1.);
    if (auto const* env = std::getenv("PLAMN_SPILL"); env and *env)
        lb_.spill = std::stod(env);
    lb_.near_band = mkn::kul::env::get_as<int>(
        "PLAMN_NEAR_BAND", cppdict::get_value(dict_, "simulation/AMR/loadbalancing/near_band", -1));

    pretag_enabled_ = cppdict::get_value(dict_, "simulation/AMR/refinement/tagging/pretag", false);
    if (auto const* env = std::getenv("PLAMN_PRETAG"))
        pretag_enabled_ = std::string_view{env} == "1";
    if (amr.contains("largest_patch_size"))
    {
        auto const largest = amr["largest_patch_size"].template to<std::vector<int>>();
        for (std::size_t i = 0; i < dimension; ++i)
            patch_size_[i] = largest[i];
    }

    // finer levels: the user's refinement boxes (cells of the coarser level), refined
    hierarchy_.max_levels = cppdict::get_value(dict_, "simulation/AMR/max_nbr_levels", 1);
    if (!amr.contains("largest_patch_size"))
        for (auto [ln, box] = std::pair{0, hierarchy_.domain}; ln < hierarchy_.max_levels;
             ++ln, box = damn::core::refine(box, hierarchy_.ratio))
            level_size_.push_back(box.shape().toArray());
    models_.set_levels(sim, hierarchy_.max_levels);
    if constexpr (Models_t::mhd_on)
        if (models_.has_mhd_levels()) // a coupled build's hybrid only input has no MHD options
        {
            models_.mhd = solver::MHDModel<opts>::from(sim);
            register_mhd_resources_();
        }
    if (hierarchy_.max_levels > 1 and amr.contains("refinement")
        and amr["refinement"].contains("tagging")
        and amr["refinement"]["tagging"].contains("threshold"))
    {
        tagging_.emplace(amr::tagging_from(amr["refinement"]["tagging"]));
        if constexpr (Models_t::mhd_on)
            mhd_tagging_.emplace(amr::tagging_from(amr["refinement"]["tagging"], "mhd_method"));
        clustering_.tag_buffer = cppdict::get_value(dict_, "simulation/AMR/tag_buffer", 1);
        if (amr.contains("nesting_buffer"))
            clustering_.nesting_buffer
                = std::max(1, amr["nesting_buffer"].template to<std::vector<int>>().front());
        if (amr.contains("clustering") and amr["clustering"].template to<std::string>() != "tile")
            throw std::runtime_error("plamn Simulator: tile clustering only");
    }
    else if (hierarchy_.max_levels > 1)
    {
        if (!amr.contains("refinement") or !amr["refinement"].contains("boxes"))
            throw std::runtime_error("plamn Simulator: refinement boxes or tagging");
        auto const& boxes = amr["refinement"]["boxes"];
        for (int ln = 0; ln + 1 < hierarchy_.max_levels; ++ln)
        {
            auto const key = "L" + std::to_string(ln);
            if (!boxes.contains(key))
                break;
            auto const& level = boxes[key];
            auto const n      = level["nbr_boxes"].template to<int>();
            auto& fine        = refinement_boxes_.emplace_back();
            for (int b = 0; b < n; ++b)
            {
                auto const& box = level["B" + std::to_string(b)];
                Box_t coarse;
                for (std::size_t i = 0; i < dimension; ++i)
                {
                    coarse.lower[i] = box["lower"][xyz[i]].template to<int>();
                    coarse.upper[i] = box["upper"][xyz[i]].template to<int>();
                }
                fine.push_back(damn::core::refine(coarse, hierarchy_.ratio));
            }
        }
    }

    // tag boxes supplied (e.g. pyphare's Tagger(sim).tag()): each level's boxes (L1, L2..., in
    //  their own level's cells), the levels made from them at initialization instead of from
    //  their coarser level's tags; each nested in its coarser level
    if (tagging_ and amr["refinement"]["tagging"].contains("tag_boxes"))
    {
        auto const& levels = amr["refinement"]["tagging"]["tag_boxes"];
        for (int ln = 1; ln < hierarchy_.max_levels; ++ln)
        {
            auto const key = "L" + std::to_string(ln);
            if (!levels.contains(key))
                break;
            auto const& level = levels[key];
            auto& boxes       = supplied_tag_boxes_.emplace_back();
            for (int b = 0, n = level["nbr_boxes"].template to<int>(); b < n; ++b)
            {
                auto const& box = level["B" + std::to_string(b)];
                Box_t& fine     = boxes.emplace_back();
                for (std::size_t i = 0; i < dimension; ++i)
                {
                    fine.lower[i] = box["lower"][xyz[i]].template to<int>();
                    fine.upper[i] = box["upper"][xyz[i]].template to<int>();
                }
            }
            auto const& coarser
                = ln == 1 ? std::vector<Box_t>{hierarchy_.domain} : supplied_tag_boxes_[ln - 2];
            for (auto const& fine : boxes)
            {
                std::vector<Box_t> outside{damn::core::coarsen(fine, hierarchy_.ratio)};
                for (auto const& c : coarser)
                {
                    std::vector<Box_t> rest;
                    for (auto const& o : outside)
                        for (auto const& r : o.remove(c))
                            rest.push_back(r);
                    outside = std::move(rest);
                }
                if (!outside.empty())
                    throw std::runtime_error("plamn Simulator: a supplied tag box of level "
                                             + std::to_string(ln)
                                             + " is not nested in its coarser level");
            }
        }
    }

    if (tagging_ and amr.contains("loadbalancing"))
    {
        auto const& lb = amr["loadbalancing"];
        auto const get = [&](std::string const& key, auto const fallback) {
            return lb.contains(key) ? lb[key].template to<std::decay_t<decltype(fallback)>>()
                                    : fallback;
        };
        lb_.active             = get("active", false);
        lb_.automatic          = get("auto", false);
        lb_.on_init            = get("on_init", false);
        lb_.every              = get("every", std::size_t{0});
        lb_.mode               = get("mode", std::string{"nppc"});
        lb_.tolerance          = get("tolerance", 0.05);
        lb_.next_rebalance     = get("next_rebalance", std::size_t{200});
        lb_.backoff            = get("next_rebalance_backoff_multiplier", std::size_t{2});
        lb_.max_next_rebalance = get("max_next_rebalance", std::size_t{1000});
        amr::CoarsestRebalanceCheck check{lb_.next_rebalance, lb_.backoff, lb_.max_next_rebalance,
                                          lb_.tolerance};
        check.steps_to_next_check = lb_.on_init ? 0 : check.interval;
        rebalance_check_.emplace(check);
    }

    if constexpr (Models_t::hybrid_on)
        if (models_.has_hybrid_levels()) // a coupled build's MHD only input has no ions, no Ohm
            setup_hybrid_();
    setup_restarts_();
    setup_diagnostics_();
    setup_decomposition_log_();
}

template<auto opts>
void Simulator<opts>::setup_hybrid_()
{
    models_.hybrid.ohm.emplace(core::OhmInfo::FROM(dict_["simulation"]["algo"]["ohm"]));
    models_.hybrid.electron_closure.emplace(
        core::hybrid::make_electron_closure(dict_["simulation"]["electrons"]["pressure_closure"]));
    register_resources_();
    solver::hybrid::setup_computed(models_.hybrid, [&](std::string const& key, auto const& names) {
        resources_[key] = damn::amr::same_names(names);
    });
}

template<auto opts>
void Simulator<opts>::register_mhd_resources_()
{
    using Patch_t = solver::MHDModel<opts>::Patch_t;
    for (std::size_t s = 0; s < models_.mhd.n_states(); ++s)
    {
        std::vector<std::string> names{Patch_t::state_name("rho", s),
                                       Patch_t::state_name("Etot", s)};
        for (auto const* v : {"rhoV", "B"})
            for (auto const* c : {"_x", "_y", "_z"})
                names.push_back(Patch_t::state_name(v, s) + c);
        resources_[mhd_state_key(s)] = damn::amr::same_names(names);
    }
    resources_["MHD_E"] = damn::amr::same_names({"E_x", "E_y", "E_z"});
    resources_["MHD_B"] = damn::amr::same_names(
        {Patch_t::state_name("B", 0) + "_x", Patch_t::state_name("B", 0) + "_y",
         Patch_t::state_name("B", 0) + "_z"}); // the advanced state's B (the root's max)
    // a reflux's step fluxes and E, their patch ghosts refilled after (PHARE's
    //  patchGhostRefluxedAlgo_)
    std::vector<std::string> reflux{"butcherE_x", "butcherE_y", "butcherE_z"};
    for (std::size_t d = 0; d < dimension; ++d)
    {
        std::string const x{"xyz"[d]};
        for (auto const* f : {"F_rho_butcher_", "F_Etot_butcher_"})
            reflux.push_back(f + x);
        for (auto const* c : {"_x", "_y", "_z"})
            reflux.push_back("F_rhoV_butcher_" + x + c);
    }
    resources_["MHDReflux"] = damn::amr::same_names(reflux);
}

// what moves between patches, named as for_each_resource(HybridPatch) names it
template<auto opts>
void Simulator<opts>::register_resources_()
{
    auto const components = [](std::string const& name) {
        return damn::amr::same_names({name + "_x", name + "_y", name + "_z"});
    };
    auto const add = [&](std::string const& key, std::vector<damn::amr::ResourcePair> const& r) {
        auto& resources = resources_[key];
        resources.insert(resources.end(), r.begin(), r.end());
    };

    add("EM_B", components("EM_B"));
    add("EM_E", components("EM_E"));
    add("EMPred_B", components("EMPred_B"));
    add("EMAvg_E", components("EMAvg_E"));

    // the totals a synchronization coarsens, their ghosts refilled after it (PHARE's
    //  chargeDensitySynchronizers_, ionChargeFluxSynchronizers_)
    add("IonMoments", components("ions_charge_flux"));
    add("IonMoments", damn::amr::same_names({"ions_charge_density"}));

    // the electrons' own fields, each by its name (scalars), its components' (vectors, tensors)
    for (auto const& field : core::hybrid::electron_fields(*models_.hybrid.electron_closure))
        std::visit(
            core::overloads{[&](core::hybrid::ScalarFieldDescriptor const& f) {
                                add(f.name, damn::amr::same_names({f.name}));
                            },
                            [&](core::hybrid::VectorFieldDescriptor const& f) {
                                add(f.name, components(f.name));
                            },
                            [&](core::hybrid::TensorFieldDescriptor const& f) {
                                auto const names = core::detail::tensor_field_names<2>(f.name);
                                add(f.name, damn::amr::same_names({names.begin(), names.end()}));
                            }},
            field);

    auto const& ions    = dict_["simulation"]["ions"];
    auto const nbr_pops = ions["nbrPopulations"].template to<std::size_t>();
    for (std::size_t i = 0; i < nbr_pops; ++i)
    {
        auto const name = ions["pop" + std::to_string(i)]["name"].template to<std::string>();
        models_.hybrid.population_names.push_back(name);

        // every population's moments in one exchange, as PHARE
        add("IonPopBorderSum", components(name + "_flux"));
        add("IonPopBorderSum",
            damn::amr::same_names({name + "_density", name + "_charge_density"}));
        // the same, as ghost fills (after a synchronization)
        add("IonPopMoments", components(name + "_flux"));
        add("IonPopMoments", damn::amr::same_names({name + "_density", name + "_charge_density"}));
    }
    models_.hybrid.particle_routes.emplace(models_.hybrid.population_names);
}


template<auto opts>
std::vector<double> Simulator<opts>::box_weights_(int const ln, std::vector<Box_t> const& boxes)
{
    auto const count
        = solver::hybrid::initial_particle_counter(models_.hybrid, dict_["simulation"]);
    std::vector<double> weights(boxes.size(), 0.);
    for (std::size_t k = static_cast<std::size_t>(rank_); k < boxes.size();
         k += static_cast<std::size_t>(n_ranks_))
    {
        double w = static_cast<double>(boxes[k].size()); // field work, a particle's worth a cell
        // a refined level's particles are split from the coarser level's: the box (whole coarser
        //  cells) holds the children of the coarse cells under it, as many leaving a fine cell
        //  for a neighbour as entering it (exact but at the level's edges): the coarse box's
        //  particles (populated cells: density above each population's cutoff) times the
        //  children per particle
        auto const count_ln = ln > 0 ? ln - 1 : ln;
        auto const box      = ln > 0 ? damn::core::coarsen(boxes[k], hierarchy_.ratio) : boxes[k];
        double const per    = ln > 0 ? static_cast<double>(opts.nbRefinedPart) : 1.;
        weights[k]          = w + per * count(layout_of_(count_ln, box));
    }
    if (n_ranks_ > 1)
        MPI_Allreduce(MPI_IN_PLACE, weights.data(), static_cast<int>(weights.size()), MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
    return weights;
}

} // namespace plamn

#endif // PLAMN_SIMULATOR_SIMULATOR_SETUP_HPP
