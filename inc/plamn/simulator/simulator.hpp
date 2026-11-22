#ifndef PLAMN_SIMULATOR_SIMULATOR_HPP
#define PLAMN_SIMULATOR_SIMULATOR_HPP

// Simulator: PHARE's simulator shell (dict parsing, times, initialize/advance) over damn
//
// The models a build enables (solver/models.hpp) own the levels: MHD below max_mhd_level, hybrid
// above. The simulator's own types and members are model free; a model's paths compile only
// where the build enables it. Only hybrid levels run yet (an MHD level: initialize throws).
// Level 0 is the domain chopped into patches, partitioned over MPI's ranks by the
// particles they will hold (damn::amr::partition); finer levels come from the input's
// refinement boxes (pyphare's refinement="boxes"), fixed for the run. Each level's data is a
// LevelData of HybridPatch, communication damn::mpi's (damn::amr's on one rank).
//
// initialize() follows PHARE's HybridLevelInitializer: the root level from the user's
// functions, finer levels refined from the coarser one (fields, B divergence free, particles
// split). advance() is damn::amr::advance (time refinement): each level takes its model's step,
// solver::advance (hybrid: PHARE's SolverPPC, solver/hybrid/advance.hpp), finer levels ratio^2
// steps per coarser one (dt ~ dx^2), as PHARE's MultiPhysicsIntegrator and
// HybridHybridMessengerStrategy:
//   level ghosts (fine ghosts no fine patch covers): E and B refined from the coarser level,
//     linear in time between its start of step (EM_*_old) and its end (EM_*); B made divergence
//     free; moments from the deposit of level ghost particles split from the coarser level at
//     both ends of its step, weighted linearly in time
//   synchronize: the fine level's E and moments coarsened onto the coarser level, which then
//     recomputes B from the time average of the fine E over its substeps (reflux)

#include "damn/amr/clustering.hpp"
#include "damn/amr/copy.hpp"
#include "damn/amr/decompose.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/inter_level.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/level_map.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/amr/regrid.hpp"
#include "damn/amr/topology.hpp"
#include "damn/hdf5/damn_hdf5.hpp"
#include "damn/core/vector.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/mpi/copy.hpp"
#include "damn/mpi/damn_mpi.hpp"
#include "damn/mpi/inter_level.hpp"
#include "damn/mpi/regrid.hpp"
#include "damn/core/def/constants.hpp"
#include "damn/core/exec/thread_pool.hpp"
#include "damn/mpi/scope_timing.hpp"
#include "mkn/kul/range.hpp"
#include "plamn/diagnostic/vtkhdf.hpp"
#include "plamn/simulator/hybrid_patch_view.hpp"
#include "plamn/solver/hybrid/advance.hpp"
#include "plamn/solver/hybrid/computed.hpp"
#include "plamn/solver/hybrid/diagnostics.hpp"
#include "plamn/solver/hybrid/initialize.hpp"
#include "plamn/solver/hybrid/level_steps.hpp"
#include "plamn/solver/hybrid/model.hpp"
#include "plamn/solver/hybrid/patches.hpp"
#include "plamn/solver/hybrid/restarts.hpp"
#include "plamn/solver/mhd/initialize.hpp"
#include "plamn/solver/mhd/level_steps.hpp"
#include "plamn/solver/mhd/restarts.hpp"
#include "plamn/solver/models.hpp"
#include "plamn/solver/stacks.hpp"
#include "plamn/amr/load_balancing.hpp"
#include "plamn/core/def.hpp"
#include "plamn/core/numerics/interpolator/interpolating.hpp"
#include "plamn/core/numerics/ohm/ohm.hpp"
#include "plamn/initializer/data_provider.hpp"
#include "plamn/simulator/hybrid_patch.hpp"
#include "plamn/simulator/coupling.hpp"
#include "plamn/amr/field_access.hpp"
#include "plamn/simulator/simulator_def.hpp"
#include "plamn/simulator/timestepper.hpp"
#include "plamn/amr/tagging.hpp"
#include "plamn/amr/split/split.hpp" // IWYU pragma: keep (the Splitter specializations)
#include "plamn/amr/split/splitter.hpp"

#include <array>
#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <type_traits>
#include <set>
#include <chrono>
#include <cstddef>
#include <format>
#include <functional>
#include <tuple>
#include <any>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>
#include <cstdint>

namespace plamn::core
{
enum class UpdaterMode : std::uint16_t;
} // namespace plamn::core

#ifndef PLAMN_DIAG_DOUBLES // diagnostics in double precision (PHARE_DIAG_DOUBLES)
#define PLAMN_DIAG_DOUBLES 0
#endif

namespace plamn
{
// std::cout to a file per rank, as PHARE's simulator: by PHARE_LOG, which pyphare sets to
//  RANK_FILES (log_to_file, its default) after making .log/: .log/<rank>.out; NULL: discarded,
//  CLI or unset: the terminal. Eg with -DDAMN_LOG_SCOPE_PRINT (mkn.py --scope-print, a build
//  without NDEBUG): every scope's name, a rank stuck in a collective is the one whose file stops
NO_DISCARD inline std::unique_ptr<std::ofstream> rank_log_file()
{
    auto const* const env  = std::getenv("PHARE_LOG");
    std::string const mode = env ? env : "CLI";
    if (mode == "CLI")
        return nullptr;
    if (mode == "NULL")
        return std::make_unique<std::ofstream>("/dev/null");
    int initialized = 0, rank = 0;
    MPI_Initialized(&initialized);
    if (initialized)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (mode == "RANK_FILES")
        return std::make_unique<std::ofstream>(".log/" + std::to_string(rank) + ".out");
    throw std::runtime_error("PHARE_LOG: RANK_FILES, NULL or CLI, not " + mode);
}

namespace simulator::detail
{
    // one callable of several overloads (constrained lambdas: a visit's per case branches)
    template<typename... Fns>
    struct Overloaded : Fns...
    {
        using Fns::operator()...;
    };
} // namespace simulator::detail

// a patch whose fields are tiled (it has untiled temporaries for whole patch operations), a
//  patch whose particles are tiled: what the Simulator's per patch overloads branch on
template<typename Patch>
concept TiledFieldsPatch = requires {
    typename std::decay_t<Patch>::Grid_t;
    typename std::decay_t<Patch>::Untiled_t;
} and core::TiledField<typename std::decay_t<Patch>::Grid_t>;
template<typename Patch>
concept TiledParticlesPatch = requires { typename std::decay_t<Patch>::Particles_t; }
                              and core::TiledParticles<typename std::decay_t<Patch>::Particles_t>;

template<auto opts>
class Simulator : public ISimulator
{
public:
    auto static constexpr options = opts;

    std::size_t static constexpr dimension     = opts.dimension;
    std::size_t static constexpr interp_order  = opts.interp_order;
    std::size_t static constexpr nbRefinedPart = opts.nbRefinedPart;

    // model free: a model's types are its stack's (solver/stacks.hpp), named only where that
    //  model is enabled
    using Models_t     = solver::Models<opts>;
    using Hierarchy_t  = damn::amr::Hierarchy<dimension>;
    using Neighbours_t = damn::amr::Neighbours<dimension>;
    using LevelMap_t   = damn::amr::LevelMap<dimension>;
    using Resources_t  = std::map<std::string, std::vector<damn::amr::ResourcePair>>;
    using Box_t        = damn::core::Box<int, dimension>;

    // dict: the whole input, copied: patches keep references into it
    explicit Simulator(initializer::Dict const& dict)
        : dict_{dict}
    {
        setup_();
    }

    Simulator(Simulator const&)            = delete;
    Simulator& operator=(Simulator const&) = delete;

    // writes the scope timer files if timing (collective: every rank's simulator goes together)
    ~Simulator() override;

    // PHARE_SCOPE_TIMING=1 at setup: phlop's DAMN_LOG_SCOPE timers, a file per rank, written
    //  when the simulator goes (pyphare's reset): python3 -m phlop.timing.scope_timer ...
    inline static std::string const timings_prefix = ".phare/timings/rank.";

    NO_DISCARD double startTime() override { return startTime_; }
    NO_DISCARD double endTime() override { return finalTime_; }
    NO_DISCARD double timeStep() override { return dt_; }
    NO_DISCARD double currentTime() override { return currentTime_; }

    void initialize() override;
    double advance() override;
    bool dump_restarts(double timestamp, double timestep) override;    // simulator_restarts.hpp
    bool dump_diagnostics(double timestamp, double timestep) override; // simulator_diagnostics
    // the files flush_every left open, closed; the next dump reopens them (PHARE's)
    void flush_diagnostics() override
    {
#if DAMN_HAS_HIGHFIVE
        diagnostics_.files.clear();
#endif
    }

    // computed quantities by key (simulator_computed.hpp): compute is collective when the
    //  quantity sums borders
    NO_DISCARD bool is_computed(std::string const& key) const;
    void compute(std::size_t ln, std::string const& key);
    NO_DISCARD auto computed(auto& patch, std::string const& key); // the hybrid model's grids

    std::vector<int> const& domainBox() const override { return domainBox_; }
    std::vector<double> const& cellWidth() const override { return cellWidth_; }
    std::size_t interporder() const override { return interp_order; }

    NO_DISCARD std::string to_str() const override;

    // for tests and diagnostics
    NO_DISCARD auto& hierarchy() const { return hierarchy_; }
    NO_DISCARD auto& level(std::size_t const ln = 0) { return data_(ln); }
    NO_DISCARD std::size_t n_levels() const { return levels_.size(); }

    // a field resource's values on the domain nodes of every local patch of level ln, patch by
    // patch
    NO_DISCARD std::vector<double> domain_values(std::string const& name, std::size_t const ln = 0);

    // domain particles of every population of every local patch of level ln
    NO_DISCARD std::size_t particle_count(std::size_t const ln = 0)
    {
        std::size_t n = 0;
        per_patch_(ln, [&](auto& patch) {
            for (auto& pop : patch.ions)
                n += real_particles_(pop.domainParticles());
        });
        return n;
    }

    // an array's particles: SoAVXChunked's without its gaps (weight 0)
    NO_DISCARD static std::size_t real_particles_(auto const& particles)
    {
        if constexpr (requires { particles.nbr_particles(); })
        {
            if (particles.sorted())
                return particles.nbr_particles();
            std::size_t n = 0;
            for (std::size_t i = 0; i < particles.size(); ++i)
                n += particles.weight(i) != 0;
            return n;
        }
        else
            return particles.size();
    }

    NO_DISCARD int rank() const { return rank_; }
    NO_DISCARD int ranks() const { return n_ranks_; }

    // over every rank (collective)
    NO_DISCARD std::size_t total_particle_count(std::size_t const ln = 0)
    {
        auto const per_rank = particle_counts_per_rank(ln);
        return std::accumulate(per_rank.begin(), per_rank.end(), std::size_t{0});
    }

    // particle_count of every rank, on every rank (collective)
    NO_DISCARD std::vector<std::size_t> particle_counts_per_rank(std::size_t const ln = 0)
    {
        std::vector<unsigned long long> all(n_ranks_);
        unsigned long long const mine = particle_count(ln);
        if (n_ranks_ == 1)
            all[0] = mine;
        else
            MPI_Allgather(&mine, 1, MPI_UNSIGNED_LONG_LONG, all.data(), 1, MPI_UNSIGNED_LONG_LONG,
                          MPI_COMM_WORLD);
        return {all.begin(), all.end()};
    }

    // domain_values of every patch of level ln, in level order, on every rank (collective)
    NO_DISCARD std::vector<double> all_domain_values(std::string const& name,
                                                     std::size_t const ln = 0);
    // every rank's domain values of level ln, one per point (a point patches share once), in
    //  point order: the same for any decomposition (collective)
    NO_DISCARD std::vector<double> level_values(std::string const& name, std::size_t ln = 0);
    // level_values with where each value is: the points' indices in the level's AMR index space
    //  (dimension per point, the same order), and per direction whether the field is dual there
    //  (a point at (index + 1/2) cells) or primal (at index cells): what a plot needs (collective)
    struct LevelField
    {
        std::vector<int> points;
        std::vector<double> values;
        std::array<bool, dimension> dual{};
    };
    NO_DISCARD LevelField level_field(std::string const& name, std::size_t ln = 0);

    // per level, on rank 0 (collective, "" on other ranks), as PHARE's Simulator::summary:
    //   lvl:<n> (<patches>,<cells>,<patch ghost cells>,<level ghost cells>) parts(<particles>),
    //   tot:<particles>
    // ghost cells: field_ghost_width around each patch, patch ghosts where a neighbour (or a
    // periodic image) has them, level ghosts the rest
    NO_DISCARD std::string summary();

    // substeps of a level per step of the coarser one: whistler waves need dt ~ dx^2 (PHARE's
    //  getMaxFinerLevelDt)
    NO_DISCARD int time_ratio() const { return hierarchy_.ratio * hierarchy_.ratio; }

private:
    // one level, whatever its model: how its data moves (the data: the models')
    struct Level
    {
        Neighbours_t neighbours;
        std::optional<LevelMap_t> to_coarser; // levels > 0
        // per local patch: the fine ghost cells no patch of the level covers, whole coarser
        //  cells (B's touch up region)
        std::vector<std::vector<Box_t>> level_ghost_cells;
        double coarse_t = 0, coarse_dt = 0; // the coarser level's step this level subcycles in
        double old_t
            = 0; // the start of its step (prepare_step): an MHD reflux redoes it from there
    };

    // damn::amr::advance's hooks
    struct Hooks
    {
        Simulator& self;
        void advance(int ln, double t, double dt) { self.advance_level_(ln, t, dt); }
        void prepare_step(int ln, double t) { self.prepare_step_(ln, t); }
        void first_step(int ln, double t, double ct, double cdt)
        {
            self.first_step_(ln, t, ct, cdt);
        }
        void last_step(int ln) { self.last_step_(ln); }
        void synchronize(int ln, double t)
        {
            self.synchronize_(ln, t);
            DAMN_SUPER_DEBUG_DO(self.debug_regrid_(ln, t);)
        }
        void after_step(int ln, double t, bool last) { self.after_step_(ln, t, last); }
    };

    void setup_();
    void setup_hybrid_(); // the hybrid model's ohm, populations, resources, computed quantities
    void initialize_hybrid_(); // every level hybrid
    void initialize_mhd_();    // MHD levels: level 0 from the functions, the finer refined
    // an MHD level from the coarser one: the advanced state refined (B divergence free), its
    //  ghosts
    void initialize_refined_mhd_(std::size_t ln, double t);
    // a refined level of whichever model owns it, from the level under it
    void initialize_refined_(std::size_t ln, double t);
    // the hybrid level over an MHD one from it: B and E refined, the particles spawned from its
    //  primitives (domain, level ghosts), the moments, J and E (simulator/coupling.hpp)
    void initialize_hybrid_over_mhd_(std::size_t ln, double t);
    // the MHD level ln - 1's primitives now into hybrid level ln's scratch: its level ghosts, and
    //  its interior too if asked
    void mhd_prims_over_(std::size_t ln, bool interior);
    // hybrid level ln's level ghost particles (old or new) spawned from the scratch primitives,
    //  each kept by its cell's owner
    void spawn_level_ghosts_(std::size_t ln, bool new_ghosts, std::uint64_t salt);
    // MHD state s's level ghosts from the coarser level at t: rho, rhoV, Etot, B refined from the
    //  coarser level's old and advanced states, linear in time (PHARE's MHDMessenger
    //  stateGhostsRefiners_), B made divergence free
    void refine_mhd_level_ghosts_(std::size_t ln, std::size_t s, double t);

    // computed quantities, the models' (simulator_computed.hpp)
    auto const& computed_for_(std::string const& key) const;

    // diagnostics, see simulator_diagnostics.hpp
    void setup_diagnostics_();
    // only >= 0: that level only, the others written empty (PHARE's dump_level)
    void dump_vtkhdf_(diagnostic::Diagnostic& diag, double time, int only = -1);
    // only >= 0: that level only (PHARE's dump_level: the time holds that level)
    void dump_phareh5_(diagnostic::Diagnostic& diag, double time, int only = -1);
    void dump_(diagnostic::Diagnostic& diag, double time, int only = -1); // the format's
    // PHARE's fine_dump_lvl_max: every diagnostic, level ln only, at the end of its substeps
    //  that are not a level 0 step's end (the finest after advancing, coarser ones after their
    //  synchronization)
    void fine_dump_(std::size_t ln, double time);
    struct Diagnostics
    {
        using Float = std::conditional_t<PLAMN_DIAG_DOUBLES, double, float>;
        std::string dir;
        bool phareh5   = false;    // else pharevtkhdf
        bool overwrite = false;    // pyphare's mode "overwrite": truncated at the first dump only
        int fine_dump_lvl_max = 0; // PHARE's: levels 1 to it dumped at their substeps, 0: none
        std::vector<diagnostic::Diagnostic> list;
#if DAMN_HAS_HIGHFIVE
        std::map<std::string, std::unique_ptr<damn::hdf5::h5::HighFiveFile>> files; // open ones
#endif
        std::set<std::string> created;        // the files made (or truncated) so far
        damn::MinimizingVector<Float> buffer; // a level's rows on this rank
    } diagnostics_;

    // restarts, see simulator_restarts.hpp
    void setup_restarts_();
    void write_restart_(std::string const& dir);
    // the resources a restart stores of a level of the model's
    NO_DISCARD static auto restart_names_(auto const& model)
    {
        if constexpr (solver::is_mhd_model_v<std::remove_cvref_t<decltype(model)>>)
            return solver::mhd::restart_names(model);
        else
            return solver::hybrid::restart_names(model);
    }
    void load_restart_();
    void rebuild_after_load_();
    struct Restarts
    {
        std::string dir, serialized, load_path;  // load_path: set to restart from it
        std::vector<double> timestamps, elapsed; // elapsed: seconds of wall time after setup
        std::size_t next_elapsed = 0;
        double restart_time      = 0;
        std::chrono::steady_clock::time_point start;
    } restarts_;

    // the models and the levels each owns (their levels' patch data)
    solver::Models<opts> models_;
    // level ln's patch data (made if new)
    NO_DISCARD auto& data_(std::size_t const ln)
    {
        return models_.hybrid.levels[static_cast<int>(ln)];
    }
    // the levels from n on gone, with their data; the levels to n made (no data yet)
    void resize_levels_(std::size_t const n)
    {
        levels_.resize(n);
        models_.truncate(n);
    }
    // hybrid(the hybrid model) or mhd(the MHD model, its level), whichever owns level ln
    void step_model_(std::size_t ln, auto&& hybrid, auto&& mhd);

    // one patch's work, for every local patch of level ln
    void per_patch_(std::size_t const ln, auto&& fn)
    {
        auto& ld = data_(ln);
        if (!decomposition_.timing) // the decomposition log's busy time: a hybrid advance only
        {
            for (auto& patch : ld.data)
                fn(*patch);
            return;
        }
        for (std::size_t i = 0; i < ld.data.size(); ++i)
        {
            auto const t0 = std::chrono::steady_clock::now();
            fn(*ld.data[i]);
            decomposition_.busy_of(ln, ld.patches[i].box)
                += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
    }

    // per_patch_ with damn's pool threads taking the patches (untiled layouts: one pool, its
    //  threads across patches; tiled ones thread over tiles inside the kernels): fn touches only
    //  its patch and never python. Serial with one thread, and while scope timing (phlop's
    //  serial timer: scopes on one thread a rank)
    void per_patch_threaded_(std::size_t const ln, auto&& fn)
    {
        using Pool    = damn::core::ThreadPool;
        auto& ld      = data_(ln);
        using Patch_t = solver::HybridModel<opts>::Patch_t;
        auto const n  = ld.data.size();
        if constexpr (!Patch_t::per_patch_mover)
            return per_patch_(ln, fn);
        if (Pool::n_pools * Pool::threads_per_pool < 2 or n < 2
            or damn::mpi::scope_timing_active())
            return per_patch_(ln, fn);

        auto& pool = Pool::INSTANCE();
        std::vector<double> busy(decomposition_.timing ? n : 0, 0.);
        for (auto const i : mkn::kul::range(n))
            Pool::pool(i % Pool::n_pools).detach_task(pool.guard([&, i] {
                auto const t0 = std::chrono::steady_clock::now();
                fn(*ld.data[i]);
                if (!busy.empty())
                    busy[i] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                                  .count();
            }));
        pool.sync();
        for (auto const i : mkn::kul::range(busy.size()))
            decomposition_.busy_of(ln, ld.patches[i].box) += busy[i];
    }

    // tiled fields: damn's schedules work on untiled temporaries (amr/untiled.hpp), leased on
    //  every patch of a level for the names an operation reads (read) or writes (fill, sum:
    //  written back into every tile's ghost box when released, inner tile ghosts included)
    static void lease_(auto& ld, std::vector<std::string> const& names, amr::Lease const how)
    {
        for (auto& patch : ld.data)
            lease_patch_(*patch, names, how);
    }
    static void release_(auto& ld)
    {
        for (auto& patch : ld.data)
            release_patch_(*patch);
    }
    static void lease_patch_(TiledFieldsPatch auto& patch, std::vector<std::string> const& names,
                             amr::Lease const how)
    {
        simulator::for_each_field_grid(patch, [&](std::string const& name, auto& grid) {
            if (std::find(names.begin(), names.end(), name) != names.end())
                patch.untiled.lease(name, grid, patch.layout, how);
        });
    }
    static void lease_patch_(auto&, std::vector<std::string> const&, amr::Lease) {}
    static void release_patch_(TiledFieldsPatch auto& patch) { patch.untiled.release(); }
    static void release_patch_(auto&) {}
    // every tiled field of a patch leased (a regrid moves them all)
    static void lease_all_(TiledFieldsPatch auto& patch, amr::Lease const how)
    {
        simulator::for_each_field_grid(patch, [&](std::string const& name, auto& grid) {
            patch.untiled.lease(name, grid, patch.layout, how);
        });
    }
    static void lease_all_(auto&, amr::Lease) {}
    // the field type a patch's field resources hold: a tiled patch's leased untiled temporaries
    template<typename Patch>
    struct resource_field_
    {
        using type = Patch::Grid_t;
    };
    template<TiledFieldsPatch Patch>
    struct resource_field_<Patch>
    {
        using type = Patch::Untiled_t::field_type;
    };
    // damn's regrid of ld (MPI when there are ranks): tiled fields moved on untiled temporaries,
    //  the old patches' read, each new one's leased as made and written back after
    auto regrid_leased_(auto const& map, auto&& ld, auto const& make) const
    {
        return regrid_leased_(map, std::move(ld), make, n_ranks_ > 1);
    }
    auto regrid_leased_(auto const& map, auto&& ld, auto const& make, bool const mpi) const
    {
        for (auto& patch : ld.data)
            lease_all_(*patch, amr::Lease::read);
        auto const made = [&](auto const& p) {
            auto patch = make(p);
            lease_all_(*patch, amr::Lease::fill);
            return patch;
        };
        auto out = mpi ? damn::mpi::regrid(map, std::move(ld), made)
                       : damn::amr::regrid(map, std::move(ld), made);
        release_(out);
        return out;
    }

    // the leases of one operation, released when it ends (an exception too)
    struct Leases
    {
        // a level's release, any model's (a refine leases levels of two patch types)
        std::vector<std::function<void()>> releases;
        ~Leases()
        {
            for (auto const& release : releases)
                release();
        }
        void add(auto& ld, std::vector<std::string> const& names, amr::Lease const how)
        {
            lease_(ld, names, how);
            releases.emplace_back([&ld] { release_(ld); });
        }
    };
    template<typename How>
    static amr::Lease constexpr lease_of_()
    {
        if constexpr (std::is_same_v<How, std::decay_t<decltype(damn::amr::sum_borders)>>)
            return amr::Lease::sum;
        return amr::Lease::fill;
    }

    // copies key's resources between the patches of level ln (its model's), see
    //  register_resources_
    void copy_(std::size_t const ln, auto const how, std::string const& key)
    {
        auto& l               = levels_[ln];
        auto const& resources = resources_.at(key);
        models_.visit(ln, [&](auto& model) {
            auto& ld = model.levels[static_cast<int>(ln)];
            std::vector<std::string> dst, src;
            for (auto const& r : resources)
                dst.push_back(r.dst), src.push_back(r.src);
            Leases leases;
            leases.add(ld, dst, lease_of_<std::decay_t<decltype(how)>>()); // first: dst wins
            leases.add(ld, src, amr::Lease::read);
            if (n_ranks_ > 1)
                damn::mpi::copy(how, l.neighbours, resources, access_(model, ld, key, how), ld);
            else
                damn::amr::copy(how, l.neighbours, resources, access_(model, ld, key, how), ld);
        });
    }

    // how damn's copy of key reaches the resources: untiled field copies by accessors, resolved
    //  once per model (amr/field_access.hpp); tiled patches (leased temporaries) and particles
    //  by name
    template<typename Model, typename LD, damn::amr::Region region, damn::amr::Arrival arrival>
        requires(region != damn::amr::Region::Leaving
                 and !TiledFieldsPatch<std::decay_t<decltype(*std::declval<LD&>().data[0])>>)
    auto const& access_(Model& model, LD& ld, std::string const& key,
                        damn::amr::CopyOp<region, arrival>)
    {
        using Patch_t = std::decay_t<decltype(*ld.data[0])>;
        auto it       = model.field_access.find(key);
        if (it == model.field_access.end())
        {
            if (damn::amr::size(ld) == 0) // no patch here: never called, one per pair all the same
            {
                static thread_local std::vector<amr::FieldAccess<Patch_t::field_ghost_width>> none;
                none.resize(resources_.at(key).size());
                return none;
            }
            it = model.field_access
                     .emplace(key, amr::field_access<Patch_t::field_ghost_width>(
                                       *ld.data[0], resources_.at(key)))
                     .first;
        }
        return it->second;
    }
    static auto const& access_(auto&, auto&, std::string const&, auto)
    {
        return damn::amr::by_name;
    }

    // fn(level ln's data, level ln - 1's), each of its own model: across the model boundary
    //  (MHD under hybrid) two patch types, which damn's refine / coarsen take
    void with_level_pair_(std::size_t const ln, auto&& fn)
    {
        models_.visit(ln, [&](auto& fine) {
            models_.visit(ln - 1, [&](auto& coarse) {
                fn(fine.levels[static_cast<int>(ln)], coarse.levels[static_cast<int>(ln) - 1]);
            });
        });
    }

    // a fine level's pairs as the coarser level names them: the same model's, its own names; a
    //  hybrid level over an MHD one, MHD's (simulator/coupling.hpp)
    static auto const& coarse_pairs_(auto const&, auto const&,
                                     std::vector<damn::amr::LevelPair> const& pairs)
    {
        return pairs;
    }
    template<typename Fine, typename Coarse>
        requires(!simulator::coupling::is_mhd_level_v<Fine>
                 and simulator::coupling::is_mhd_level_v<Coarse>)
    static std::vector<damn::amr::LevelPair>
    coarse_pairs_(Fine const&, Coarse const&, std::vector<damn::amr::LevelPair> const& pairs)
    {
        return simulator::coupling::to_mhd_coarse(pairs);
    }

    // a level data's patch type; either level of a pair tiled (their schedules go by name)
    template<typename LD>
    using patch_of_t = std::decay_t<decltype(*std::declval<LD&>().data[0])>;
    template<typename FineLD, typename CoarseLD>
    static constexpr bool any_tiled_v
        = TiledFieldsPatch<patch_of_t<FineLD>> or TiledFieldsPatch<patch_of_t<CoarseLD>>;

    // from level ln - 1 into level ln
    void refine_(std::size_t const ln, auto const& op, damn::amr::LevelRegion const region,
                 std::vector<damn::amr::LevelPair> const& level_pairs, double const alpha = 0)
    {
        auto& l = levels_[ln];
        with_level_pair_(ln, [&](auto& ld, auto& coarse) {
            auto const& pairs = coarse_pairs_(ld, coarse, level_pairs);
            Leases leases;
            lease_level_pairs_(leases, ld, coarse, pairs, /*refine=*/true);
            auto const by = [&](auto const& access) {
                if (n_ranks_ > 1)
                    damn::mpi::refine(op, region, *l.to_coarser, pairs, access, coarse, ld, alpha);
                else
                    damn::amr::refine(op, region, *l.to_coarser, pairs, access, coarse, ld, alpha);
            };
            with_level_access_(ld, coarse, pairs, by);
        });
    }

    // fn(how refine / coarsen reach the pairs' resources): accessors when resolved (untiled
    //  fields), else by name; tiled patches by name only (their accessors are not even compiled:
    //  a tiled grid has no point access)
    template<typename FineLD, typename CoarseLD>
        requires any_tiled_v<FineLD, CoarseLD>
    void with_level_access_(FineLD&, CoarseLD&, std::vector<damn::amr::LevelPair> const&, auto&& fn)
    {
        fn(damn::amr::by_name);
    }
    template<typename FineLD, typename CoarseLD>
        requires(!any_tiled_v<FineLD, CoarseLD>)
    void with_level_access_(FineLD& fine, CoarseLD& coarse,
                            std::vector<damn::amr::LevelPair> const& pairs, auto&& fn)
    {
        if (auto const* access = level_access_(fine, coarse, pairs))
            fn(*access);
        else
            fn(damn::amr::by_name);
    }

    // tiled fields: the pairs' names leased on both levels (damn's plans read them on both),
    //  the written ones first: fill (refine: the fine ones, coarsen: the coarse ones)
    template<typename FineLD, typename CoarseLD>
        requires any_tiled_v<FineLD, CoarseLD>
    static void lease_level_pairs_(Leases& leases, FineLD& fine_ld, CoarseLD& coarse_ld,
                                   std::vector<damn::amr::LevelPair> const& pairs,
                                   bool const refine)
    {
        {
            std::vector<std::string> fine, coarser;
            for (auto const& p : pairs)
            {
                fine.push_back(p.fine);
                coarser.push_back(p.coarse);
                if (!p.coarse_new.empty())
                    coarser.push_back(p.coarse_new);
            }
            auto all = fine;
            all.insert(all.end(), coarser.begin(), coarser.end());
            if (refine)
                leases.add(fine_ld, fine, amr::Lease::fill);
            else
                leases.add(coarse_ld, coarser, amr::Lease::fill);
            leases.add(fine_ld, all, amr::Lease::read);
            leases.add(coarse_ld, all, amr::Lease::read);
        }
    }
    template<typename FineLD, typename CoarseLD>
        requires(!any_tiled_v<FineLD, CoarseLD>)
    static void lease_level_pairs_(Leases&, FineLD&, CoarseLD&,
                                   std::vector<damn::amr::LevelPair> const&, bool)
    {
    }

    // refine / coarsen's accessors of pairs between a level's data and the coarser level's:
    //  untiled fields, resolved once per models and pairs (amr/field_access.hpp); null: by
    //  name (particle arrays, tiled patches, or no patch of either level here yet)
    template<typename FineLD, typename CoarseLD>
    using level_access_t
        = std::vector<amr::LevelFieldAccess<patch_of_t<CoarseLD>::field_ghost_width,
                                            patch_of_t<FineLD>::field_ghost_width>>;

    template<typename FineLD, typename CoarseLD>
        requires(!any_tiled_v<FineLD, CoarseLD>)
    auto const* level_access_(FineLD& fine, CoarseLD& coarse,
                              std::vector<damn::amr::LevelPair> const& pairs)
    {
        using F        = patch_of_t<FineLD>;
        using C        = patch_of_t<CoarseLD>;
        using Access_t = level_access_t<FineLD, CoarseLD>;
        using Cached_t = std::optional<Access_t>;
        {
            auto const key = LevelAccessKey{simulator::coupling::is_mhd_patch_v<F>,
                                            simulator::coupling::is_mhd_patch_v<C>, pairs};
            auto it        = level_access_cache_.find(key);
            if (it == level_access_cache_.end())
            {
                if (damn::amr::size(fine) == 0 or damn::amr::size(coarse) == 0)
                    return static_cast<Access_t const*>(nullptr);
                it = level_access_cache_
                         .emplace(
                             key,
                             amr::level_field_access<C::field_ghost_width, F::field_ghost_width>(
                                 *coarse.data[0], *fine.data[0], pairs))
                         .first;
            }
            auto const& cached = std::any_cast<Cached_t const&>(it->second);
            return cached ? &*cached : static_cast<Access_t const*>(nullptr);
        }
    }
    using LevelAccessKey = std::tuple<bool, bool, std::vector<damn::amr::LevelPair>>;
    std::map<LevelAccessKey, std::any> level_access_cache_; // level_access_'s

    // from level ln onto level ln - 1
    void coarsen_(std::size_t const ln, auto const& op,
                  std::vector<damn::amr::LevelPair> const& pairs)
    {
        auto& l = levels_[ln];
        with_level_pair_(ln, [&](auto& ld, auto& coarse) {
            Leases leases;
            lease_level_pairs_(leases, ld, coarse, pairs, /*refine=*/false);
            auto const by = [&](auto const& access) {
                if (n_ranks_ > 1)
                    damn::mpi::coarsen(op, *l.to_coarser, pairs, access, ld, coarse);
                else
                    damn::amr::coarsen(op, *l.to_coarser, pairs, access, ld, coarse);
            };
            with_level_access_(ld, coarse, pairs, by);
        });
    }

    // a box's load balancing weight: the particles it will hold, plus its cells for the field
    // work. Each rank evaluates its share of the boxes, then they are summed over ranks
    std::vector<double> box_weights_(int ln, std::vector<Box_t> const& boxes);
    // once running: the particles each box (level ln's cells) holds now, as PHARE's nppc load
    //  balancing counts them (cells for mode homogeneous). A level being made or regridded: its
    //  old particles where it was, the coarser level's under the rest times the split. Each
    //  rank counts its patches', then they are summed over ranks
    std::vector<double> particle_weights_(int ln, std::vector<Box_t> const& boxes);
    // PHARE's: a rebalance of level 0 (and so of the finer levels) due at this step
    NO_DISCARD bool should_rebalance_();
    void rebalance_(double t); // level 0 onto new owners, then each finer level

    // the decomposition log (simulator_decomposition.hpp): one JSON line per init, changing
    //  regrid or rebalance, and every `every` steps; collective
    struct DecompositionLog
    {
        std::string path{}; // empty: off
        int every = 0;
        std::optional<damn::amr::Hierarchy<dimension>> last{}; // the last line's
        // busy time: seconds of each patch's work in a hybrid level's advance (per_patch_,
        //  outside the exchanges), by level and patch box, since the last line
        bool timing    = false;
        int steps_then = 0; // steps_ at the last line
        std::vector<std::map<std::array<int, 2 * dimension>, double>> busy{};
        double& busy_of(std::size_t const ln, Box_t const& box)
        {
            if (busy.size() <= ln)
                busy.resize(ln + 1);
            return busy[ln][Simulator::d_key_(box)];
        }
    } decomposition_;
    void setup_decomposition_log_();
    NO_DISCARD static int decomposition_reach_();
    NO_DISCARD static bool busy_push_();
    NO_DISCARD static std::array<int, 2 * dimension> d_key_(Box_t const& box);
    void record_decomposition_(std::string const& event, double t, bool always = false);

    void register_resources_();
    // the MHD states' (conservative) and E's resources: MHDState_<s>, MHD_E, MHD_B (state 0's B)
    void register_mhd_resources_();
    NO_DISCARD static std::string mhd_state_key(std::size_t const s)
    {
        return "MHDState_" + std::to_string(s);
    }
    // level ln's patches from its boxes (that level's cells): chopped into the largest patch
    //  size, partitioned along the Hilbert curve by the particles they will hold
    NO_DISCARD damn::amr::Level<dimension> level_of_(int ln, std::vector<Box_t> const& boxes);
    // damn::amr::decompose's settings: this run's topology, patch size, spill, its weights (the
    //  particles a patch will hold, or holds; MHD levels: cells), `partition`
    NO_DISCARD damn::amr::Decompose<dimension> decompose_in_(std::string const& partition);
    void make_level_(int ln, std::vector<Box_t> const& boxes);
    // the levels' total moments after a regrid / rebalance, finest down to lowest: from their
    //  populations, coarsened from the finer level (rebuild_moments_)
    void rebuild_moments_(int lowest);
    void allocate_level_(int ln); // the patch data of hierarchy_'s level ln, linked
    void link_level_(int ln);     // neighbours, map to the coarser level, level ghost cells
    void link_hybrid_level_(int ln);
    void link_mhd_level_(int ln);
    // neighbours reaching reach cells, the map to the coarser level coarser_reach, the level ghost
    //  cells of each local patch (gw: its field ghosts)
    void link_geometry_(int ln, int reach, int coarser_reach, int gw);
    // a patch of level ln (its cells): its particle ghost box among the level's other patches
    NO_DISCARD auto layout_of_(int ln, Box_t const& box) const;
    NO_DISCARD auto make_patch_(auto const& model, int ln, Box_t const& box) const;
    // the boxes of level ln + 1 from level ln's tagged cells (collective)
    NO_DISCARD std::vector<Box_t> cluster_(int ln);
    NO_DISCARD std::vector<std::vector<Box_t>> pretag_();
    // the levels finer than from re-clustered (tagging) at t; always: re-made even where their
    //  boxes did not change (SAMRAI's, see RegridMode)
    void regrid_(double t, int from = 0, bool always = false);
    // regrid_mode high: the finer levels of ln regridded at SAMRAI's regrid points, after a step
    //  of ln to t (last: ln's last substep of its coarser level's step)
    void after_step_(int ln, double t, bool last);
    void regrid_after_step_(int ln, double t); // after_step_'s regrid, at a regrid point
    // keep_finer: the finer levels stay (each to be regridded onto its own boxes in turn);
    //  moved: the same cells (a rebalance), the old ghosts valid: moved with the data
    //  (damn::amr::RegridGhosts), nothing refined or refilled from the coarser level
    void regrid_level_(int ln, std::vector<Box_t> const& boxes, double t, bool keep_finer = false,
                       bool moved = false);
    // heavy debugging (-DDAMN_SUPER_DEBUG only, simulator_sums.hpp): every resource of level ln
    //  summed and printed, at stage; the finest level regridded onto its boxes at given times
    void sums_(std::size_t ln, char const* stage, double t);
    void debug_regrid_(int ln, double t);
    // a rebalance moves data: what must survive it (rebalance_state_), checked after it
    struct RebalanceState
    {
        std::vector<std::vector<double>> values; // per level and quantity, its domain values
        std::vector<double> particles;           // per level and set: count, sums of w, v, x
    };
    NO_DISCARD RebalanceState rebalance_state_();
    void check_rebalance_(RebalanceState const& before, double t);
    // regrid_level_'s data, the level's model's: where the old level was, moved; the new cells
    //  (outside old_cells) refined from the coarser level, then the ghosts
    void regrid_hybrid_data_(int ln, auto const& map, std::vector<Box_t> const& old_cells, double t,
                             bool moved);
    // a hybrid level's resources whose ghosts a rebalance moves (every field grid, the old level
    //  ghost particles), from this rank's patches (a rank without any has no transfers)
    NO_DISCARD damn::amr::RegridGhosts moved_ghosts_(int ln);
    void regrid_mhd_data_(int ln, auto const& map, std::vector<Box_t> const& old_cells, double t);
    // a new level ln's data, from the coarser level (regrid_, initialization)
    void initialize_refined_level_(int ln, double t);

    // level ghost fills (patch ghosts, then level ghosts from the coarser level at time t)
    void fill_magnetic_ghosts_(std::size_t ln, std::string const& key, double t);
    void fill_electric_ghosts_(std::size_t ln, std::string const& key, double t);
    // debug (NDEBUG not set): throws if a point of key's components (key_x, key_y, key_z) on
    //  their ghost boxes is NaN, i.e. nothing filled it (HybridPatch starts NaN). Points past a
    //  non periodic domain boundary are skipped
    void check_set_(std::size_t ln, std::string const& key, std::string const& where);
    // B over the level ghost cells of every local patch made divergence free
    void touch_up_level_ghosts_(std::size_t ln, std::string const& key);
    NO_DISCARD double alpha_(std::size_t const ln, double const t) const
    {
        auto const& l = levels_[ln];
        // no coarser step in progress (initialization, regrid): the coarser level's data now
        return l.coarse_dt > 0 ? (t - l.coarse_t) / l.coarse_dt : 1;
    }

    // a level ghost particle lives in one patch, its cell's owner (Surroundings): the others'
    //  copies of a split dropped, old (initialization, regrid, restart) or new (coarse step)

    // what the models' solvers use of a level (solver/hybrid/advance.hpp)
    struct LevelServices
    {
        Simulator& self;
        std::size_t ln;

        void sums(char const* stage, double const t) { self.sums_(ln, stage, t); } // SUPER_DEBUG

        NO_DISCARD std::size_t number() const { return ln; }
        NO_DISCARD auto& data() { return self.data_(ln); }
        void per_patch(auto&& fn) { self.per_patch_(ln, fn); }
        // fn touching only its patch, never python: on the pool's threads (per_patch_threaded_)
        void per_patch_threaded(auto&& fn) { self.per_patch_threaded_(ln, fn); }
        void copy(auto const how, std::string const& key) { self.copy_(ln, how, key); }
        void fill_magnetic_ghosts(std::string const& key, double const t)
        {
            self.fill_magnetic_ghosts_(ln, key, t);
        }
        void fill_electric_ghosts(std::string const& key, double const t)
        {
            self.fill_electric_ghosts_(ln, key, t);
        }
        NO_DISCARD double alpha(double const t) const { return self.alpha_(ln, t); }
        void check_set(std::string const& key, std::string const& where)
        {
            self.check_set_(ln, key, where);
        }
        NO_DISCARD auto mesh_size() const
        {
            return damn::amr::mesh_size(self.hierarchy_, static_cast<int>(ln));
        }

        // with the coarser level (ln > 0)
        NO_DISCARD LevelServices coarser() const { return {self, ln - 1}; }
        NO_DISCARD double coarse_dt() const { return self.levels_[ln].coarse_dt; }
        NO_DISCARD int time_ratio() const { return self.time_ratio(); }
        void refine(auto const& op, damn::amr::LevelRegion const region,
                    std::vector<damn::amr::LevelPair> const& pairs, double const alpha = 0)
        {
            self.refine_(ln, op, region, pairs, alpha);
        }
        void coarsen(auto const& op, std::vector<damn::amr::LevelPair> const& pairs)
        {
            self.coarsen_(ln, op, pairs);
        }
    };
    NO_DISCARD LevelServices level_services_(std::size_t const ln) { return {*this, ln}; }

    // what the MHD model's advance uses of a level (solver/advance.hpp)
    struct MHDLevel
    {
        Simulator& self;
        std::size_t ln;

        void sums(char const* stage, double const t) { self.sums_(ln, stage, t); } // SUPER_DEBUG

        void per_patch(auto&& fn)
        {
            for (auto& patch : self.models_.mhd.levels[static_cast<int>(ln)].data)
                fn(*patch);
        }
        void fill_state_ghosts(std::size_t const s, double const t)
        {
            self.fill_mhd_state_ghosts_(ln, s, t);
        }
        void fill_electric_ghosts(double const t) { self.fill_mhd_electric_ghosts_(ln, t); }

        // with the coarser level (solver/mhd/level_steps.hpp)
        NO_DISCARD std::size_t number() const { return ln; }
        NO_DISCARD MHDLevel coarser() const { return {self, ln - 1}; }
        void coarsen(auto const& op, std::vector<damn::amr::LevelPair> const& pairs)
        {
            self.coarsen_(ln, op, pairs);
        }
        void copy(auto const how, std::string const& key) { self.copy_(ln, how, key); }
        NO_DISCARD double old_time() const { return self.levels_[ln].old_t; }
        NO_DISCARD int time_ratio() const { return self.time_ratio(); }
    };
    // patch ghosts, then the physical boundaries' (simulator_ghosts.hpp)
    void fill_mhd_state_ghosts_(std::size_t ln, std::size_t s, double t);
    void fill_mhd_electric_ghosts_(std::size_t ln, double t);
    // fn(patch, its physical boundary regions) for every local patch of MHD level ln
    void mhd_physical_boundaries_(std::size_t ln, auto&& fn);

    // damn::amr::advance's hooks, see damn/amr/time_refinement.hpp
    void advance_level_(int ln, double t, double dt);
    void prepare_step_(int ln, double t);
    void first_step_(int ln, double t, double coarse_t, double coarse_dt);
    void last_step_(int ln);
    void synchronize_(int ln, double t);

    damn::CoutRedirect cout_redirect_{rank_log_file()}; // first: for the whole lifetime
    initializer::Dict dict_;
    Hierarchy_t hierarchy_;
    std::vector<Level> levels_;
    Resources_t resources_;                            // what moves together, by name
    std::vector<std::vector<Box_t>> refinement_boxes_; // per level, in that level's cells
    // refinement="tagging": what to tag on, how to cluster, every how many level 0 steps
    std::optional<amr::Tagging> tagging_;     // hybrid levels' (hybrid_method); set: tagging on
    std::optional<amr::Tagging> mhd_tagging_; // MHD levels' (mhd_method)
    damn::amr::ClusteringOptions<dimension> clustering_;
    std::vector<std::vector<Box_t>> clustered_; // per level: the boxes it was clustered to
    int regrid_interval_ = 2;                   // SAMRAI's for ratio 2 with time refinement
    // when levels are regridded (simulation/AMR/regrid_mode, or PLAMN_REGRID_MODE):
    //   normal  tagging: every regrid_interval_ level 0 steps, a level only where its boxes
    //           changed; refinement boxes: never
    //   high    PHARE's (SAMRAI's TimeRefinementIntegrator and GriddingAlgorithm): after every
    //           regrid_interval_-th substep of a level within its coarser level's step, its finer
    //           levels re-made (unless the coarser level regrids then too); tagging: re-clustered,
    //           re-made even with unchanged boxes; refinement boxes: re-made with the same boxes,
    //           during level 0's first step only (SAMRAI marks the user's boxes changed for its
    //           first cycle). For comparisons with PHARE: a regrid is not a no-op (the level
    //           ghost fields are refilled after the synchronization)
    enum class RegridMode : std::uint8_t { normal, high };
    RegridMode regrid_mode_ = RegridMode::normal;
    // refinement="tagging": every level's boxes found from B alone before any particle is made
    //  (simulation/AMR/refinement/tagging/pretag, or PLAMN_PRETAG=1), see pretag_
    bool pretag_enabled_ = false;
    // simulation/AMR/refinement/tagging/tag_boxes: levels 1... made from these at initialization
    std::vector<std::vector<Box_t>> supplied_tag_boxes_{};
    // levels decomposed together (finest, aligned) where every level's boxes are known
    //  (initialization with pretag or tag_boxes, rebalance), taken by level_of_, then cleared
    std::vector<damn::amr::Level<dimension>> planned_{};
    std::vector<int> level_steps_; // per level > 0: its substeps in the coarser level's step
    int steps_ = 0;
    // simulation/AMR/loadbalancing (pyphare's LoadBalancer), with refinement="tagging"
    struct LoadBalancing
    {
        bool active = false, automatic = false, on_init = false;
        std::size_t every          = 0;
        std::string mode           = "nppc";
        double tolerance           = 0.05;
        std::size_t next_rebalance = 200, backoff = 2, max_next_rebalance = 1000;
        // how a level's patches get their ranks (level_of_; plamn's, not PHARE's: the PHARE
        //  comparisons keep "levels"): "levels" each level along the Hilbert curve on its own,
        //  "keep" where the data is (damn::amr::keep: the old level's owners, else the coarser
        //  patch's under it, else the curve); "finest" / "aligned" (damn::amr::decompose) every
        //  level together where all are known (initialization with pretag or tag_boxes,
        //  rebalance), a level alone to the owners of the coarser cells under it; spill >= 0:
        //  then no rank above (1 + spill) of its share
        //  (damn::amr::spill). simulation/AMR/loadbalancing/partition, spill, or
        //  PLAMN_PARTITION, PLAMN_SPILL. finest: near_band, the level 0 cells about level 1
        //  its step reads (default: a particle ghost width, where its level ghosts split
        //  from), with level 1's owners; near_band or PLAMN_NEAR_BAND
        std::string partition = "levels";
        double spill          = -1;
        int near_band         = -1; // -1: the default
    } lb_;
    std::optional<amr::CoarsestRebalanceCheck> rebalance_check_;
    int rank_    = 0;
    int n_ranks_ = 1;
    damn::amr::Topology topology_;


    std::array<int, dimension> patch_size_{};
    // cells per direction of the blocks level 0 is cut on when rebalanced
    static constexpr int rebalance_block = 8;
    std::vector<int> domainBox_;
    std::vector<double> cellWidth_;
    Timestepper timestepper_; // each step's dt, the time after it
    // the adaptive dt: every level's whistler bound as a level 0 dt, the smallest (all ranks)
    double stable_dt_(Timestepper::CFL const& cfl);
    double dt_          = 0; // this step's
    int timeStepNbr_    = 0;
    double startTime_   = 0;
    double finalTime_   = 0;
    double currentTime_ = 0;
    bool isInitialized_ = false;
    bool timing_        = false; // this simulator started the scope timers
};

} // namespace plamn

// the definitions, by concern
#include "plamn/simulator/simulator_access.hpp"      // IWYU pragma: export
#include "plamn/simulator/simulator_advance.hpp"     // IWYU pragma: export
#include "plamn/simulator/simulator_ghosts.hpp"      // IWYU pragma: export
#include "plamn/simulator/simulator_levels.hpp"      // IWYU pragma: export
#include "plamn/simulator/simulator_regrid.hpp"      // IWYU pragma: export
#include "plamn/simulator/simulator_restarts.hpp"    // IWYU pragma: export
#include "plamn/simulator/simulator_computed.hpp"    // IWYU pragma: export
#include "plamn/simulator/simulator_decomposition.hpp" // IWYU pragma: export
#include "plamn/simulator/simulator_diagnostics.hpp" // IWYU pragma: export
#include "plamn/simulator/simulator_setup.hpp"       // IWYU pragma: export
#include "plamn/simulator/simulator_sums.hpp"        // IWYU pragma: export

#endif // PLAMN_SIMULATOR_SIMULATOR_HPP
