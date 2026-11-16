#ifndef PLAMN_SIMULATOR_SIMULATOR_RESTARTS_HPP
#define PLAMN_SIMULATOR_SIMULATOR_RESTARTS_HPP

// restarts (pyphare's restart_options, dict simulation/restarts): written by dump_restarts at
// write_timestamps, or once each elapsed_timestamps seconds of wall time after setup, only
// between steps (every level at the same time); loaded by initialize() when loadPath is set
//
//   <filePath>/<time {:021.10f}>/restart.h5    one file, every rank writing its own patches
//     /restart      time, step, dt, levels, ranks, the serialized simulation, rebalance state
//     /level<ln>    a damn::hdf5 checkpoint of EM_B_*, EM_E_*, each population's domain
//                   particles and moments; /clustered<ln>: the boxes the level was clustered to
//
// only what the next step needs and cannot rebuild bit for bit (PHAREHUB/PHARE#1410): B, E and
// the populations' moments with their ghosts (a refined level's level ghosts were filled from
// the coarser level before its synchronization; the moments' border sums and coarsenings come
// in an order a fresh deposit does not repeat), and the domain particles. The total moments,
// J, the old copies and the level ghost particles are rebuilt after loading. Loading needs
// the ranks that wrote it (the same decomposition)
// IWYU pragma: private, include "plamn/simulator/simulator.hpp"

#include "damn/hdf5/damn_hdf5.hpp"
#include "plamn/simulator/restart_paths.hpp"
#include "plamn/simulator/simulator.hpp" // IWYU pragma: keep (the class)
#include "damn/amr/level_map.hpp"
#include "damn/core/logger.hpp"
#include "dict.hpp"
#include "phlop/timing/scope.hpp"
#include "plamn/amr/refine.hpp"
#include "plamn/core/numerics/ampere/ampere_transformer.hpp"

#if DAMN_HAS_HIGHFIVE
#include "damn/hdf5/checkpoint.hpp"
#include "damn/hdf5/level_table.hpp"
#endif

#include "mpi.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdlib>
#include <initializer_list>



namespace plamn
{
template<auto opts>
void Simulator<opts>::setup_restarts_()
{
    restarts_.start = std::chrono::steady_clock::now();
    auto const get  = [&](std::string const& key, auto const fallback) {
        return cppdict::get_value(dict_, "simulation/restarts/" + key, fallback);
    };
    restarts_.dir          = get("filePath", std::string{});
    restarts_.timestamps   = get("write_timestamps", std::vector<double>{});
    restarts_.elapsed      = get("elapsed_timestamps", std::vector<double>{});
    restarts_.serialized   = get("serialized_simulation", std::string{});
    restarts_.load_path    = get("loadPath", std::string{});
    restarts_.restart_time = get("restart_time", 0.);
    std::sort(restarts_.elapsed.begin(), restarts_.elapsed.end());
}

template<auto opts>
std::vector<std::string> Simulator<opts>::restart_names_() const
{
    std::vector<std::string> names;
    for (auto const* em : {"EM_B", "EM_E"})
        for (auto const* c : {"_x", "_y", "_z"})
            names.push_back(std::string{em} + c);
    // the totals a synchronization coarsened: not the populations' sum on a coarser level
    names.push_back("ions_charge_density");
    for (auto const* c : {"_x", "_y", "_z"})
        names.push_back(std::string{"ions_charge_flux"} + c);
    for (auto const& pop : population_names_)
    {
        names.push_back(pop + "_domain");
        for (auto const* m : {"_density", "_charge_density", "_flux_x", "_flux_y", "_flux_z"})
            names.push_back(pop + m);
    }
    return names;
}

template<auto opts>
bool Simulator<opts>::dump_restarts(double const timestamp, double const timestep)
{
    if (restarts_.dir.empty())
        return false;
    if (!restarts_.load_path.empty() and timestamp == startTime_)
        return false; // just loaded from it

    bool due = std::any_of(restarts_.timestamps.begin(), restarts_.timestamps.end(),
                           [&](auto const t) { return std::abs(t - timestamp) < timestep / 2; });
    if (restarts_.next_elapsed < restarts_.elapsed.size())
    {
        // rank 0's clock: every rank decides the same
        double seconds
            = std::chrono::duration<double>(std::chrono::steady_clock::now() - restarts_.start)
                  .count();
        MPI_Bcast(&seconds, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (seconds >= restarts_.elapsed[restarts_.next_elapsed])
        {
            due = true;
            while (restarts_.next_elapsed < restarts_.elapsed.size()
                   and seconds >= restarts_.elapsed[restarts_.next_elapsed])
                ++restarts_.next_elapsed;
        }
    }
    if (!due)
        return false;
    write_restart_(simulator::restart_dir_for_time(restarts_.dir, timestamp));
    return true;
}

template<auto opts>
void Simulator<opts>::write_restart_(std::string const& dir)
{
    DAMN_LOG_SCOPE(1, "Simulator::write_restart_");
#if DAMN_HAS_HIGHFIVE
    using namespace damn::hdf5;
    if (rank_ == 0)
        std::filesystem::create_directories(dir);
    MPI_Barrier(MPI_COMM_WORLD);

    h5::HighFiveFile file{simulator::restart_file_in(dir),
                          h5::HiFile::Create | h5::HiFile::Truncate};
    file.write_attribute("/restart", "time", currentTime_);
    file.write_attribute("/restart", "step", steps_);
    file.write_attribute("/restart", "dt", dt_);
    file.write_attribute("/restart", "levels", static_cast<int>(levels_.size()));
    file.write_attribute("/restart", "ranks", n_ranks_);
    file.write_attribute("/restart", "serialized_simulation", restarts_.serialized);
    file.write_attribute("/restart", "clustered", static_cast<int>(clustered_.size()));
    if (rebalance_check_)
    {
        file.write_attribute("/restart", "rebalance_interval",
                             static_cast<std::uint64_t>(rebalance_check_->interval));
        file.write_attribute("/restart", "rebalance_steps_to_next_check",
                             static_cast<std::uint64_t>(rebalance_check_->steps_to_next_check));
    }

    auto const names = restart_names_();
    for (std::size_t ln = 0; ln < levels_.size(); ++ln)
    {
        auto const group = "/level" + std::to_string(ln);
        write_checkpoint(file, group, hierarchy_.levels[ln], levels_[ln].data, names);
    }
    for (std::size_t ln = 0; ln < clustered_.size(); ++ln)
        write_boxes(file, "/clustered" + std::to_string(ln), clustered_[ln]);
#else
    throw std::runtime_error("plamn Simulator: restarts need HDF5 (damn_hdf5): " + dir);
#endif
}

template<auto opts>
void Simulator<opts>::load_restart_()
{
    DAMN_LOG_SCOPE(1, "Simulator::load_restart_");
#if DAMN_HAS_HIGHFIVE
    using namespace damn::hdf5;
    h5::HighFiveFile file{simulator::restart_file_in(restarts_.load_path), h5::HiFile::ReadOnly};

    if (auto const ranks = file.read_attribute<int>("/restart", "ranks"); ranks != n_ranks_)
        throw std::runtime_error("plamn Simulator: a restart written on " + std::to_string(ranks)
                                 + " ranks, loaded on " + std::to_string(n_ranks_) + ": not yet");
    startTime_   = file.read_attribute<double>("/restart", "time");
    currentTime_ = startTime_;
    steps_       = file.read_attribute<int>("/restart", "step");
    if (rebalance_check_ and file.file().getGroup("/restart").hasAttribute("rebalance_interval"))
    {
        rebalance_check_->interval = static_cast<std::size_t>(
            file.read_attribute<std::uint64_t>("/restart", "rebalance_interval"));
        rebalance_check_->steps_to_next_check = static_cast<std::size_t>(
            file.read_attribute<std::uint64_t>("/restart", "rebalance_steps_to_next_check"));
    }

    auto const names    = restart_names_();
    auto const n_levels = file.read_attribute<int>("/restart", "levels");
    for (int ln = 0; ln < n_levels; ++ln)
    {
        auto const group  = "/level" + std::to_string(ln);
        auto const stored = read_level<dimension>(file, group + "/level", ln);
        damn::amr::set_level(hierarchy_, stored.level);
        allocate_level_(ln);
        read_checkpoint(file, group, hierarchy_.levels[ln], levels_[ln].data, names);
    }
    clustered_.resize(static_cast<std::size_t>(file.read_attribute<int>("/restart", "clustered")));
    for (std::size_t ln = 0; ln < clustered_.size(); ++ln)
        clustered_[ln] = read_boxes<dimension>(file, "/clustered" + std::to_string(ln));

    rebuild_after_load_();
#else
    throw std::runtime_error("plamn Simulator: restarts need HDF5 (damn_hdf5)");
#endif
}

// what the running simulator holds between steps, from what was stored: the particles' maps,
// the level ghost particles (split from the coarser level, as first_step_ does), the mass
// density and bulk velocity (from the populations'), J (from B), and every level's old copies
template<auto opts>
void Simulator<opts>::rebuild_after_load_()
{
    using damn::amr::LevelRegion;
    auto const t = currentTime_;
    for (std::size_t ln = 0; ln < levels_.size(); ++ln)
    {
        per_patch_(ln, [](auto& patch) {
            for (auto& pop : patch.ions)
                if constexpr (requires { pop.domainParticles().remap(); })
                    pop.domainParticles().remap();
        });
        if (ln > 0)
        {
            for (auto const& name : population_names_)
                refine_(ln, amr::ParticleSplitter<Splitter_t>{}, LevelRegion::LevelGhosts,
                        {{name + "_domain", name + "_level_ghost_old", ""}});
            own_level_ghosts_(ln, /*new_ghosts=*/false);
            per_patch_(ln, [](auto& patch) {
                for (auto& pop : patch.ions)
                    pop.levelGhostParticles() = pop.levelGhostParticlesOld();
            });
        }
        // the mass density and bulk velocity from the populations' (stored); the charge
        //  density and flux as stored (coarsened where a finer level was synchronized)
        per_patch_(ln, [&](auto& patch) {
            std::vector<std::vector<double>> stored;
            auto const each = [&](auto&& fn) {
                fn(patch.chargeDensity);
                for (auto& c : patch.chargeFlux)
                    fn(c);
            };
            each([&](auto const& g) { stored.emplace_back(g.data(), g.data() + g.size()); });
            patch.ions.update();
            std::size_t i = 0;
            each([&](auto& g) { std::copy(stored[i].begin(), stored[i].end(), g.data()), ++i; });
        });
        per_patch_(
            ln, [&](auto& patch) { core::AmpereTransformer{}(patch.layout, patch.em.B, patch.J); });
        prepare_step_(static_cast<int>(ln), t);
    }
}

} // namespace plamn

#endif // PLAMN_SIMULATOR_SIMULATOR_RESTARTS_HPP
