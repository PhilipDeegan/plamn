#ifndef PLAMN_SIMULATOR_SIMULATOR_HPP
#define PLAMN_SIMULATOR_SIMULATOR_HPP

// Simulator: PHARE's simulator shell (dict parsing, times, initialize/advance) over damn; the
// models, levels, initialization and the advance's steps: doc/simulator/simulator.md

#include "damn/amr/clustering.hpp"
#include "damn/amr/copy.hpp"
#include "damn/amr/decompose.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/inter_level.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/level_map.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/amr/regrid.hpp"
#include "damn/amr/time_refinement.hpp"
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
#include "mkn/kul/env.hpp"
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
#include "plamn/amr/particle_access.hpp"
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
// std::cout to a file per rank by PHARE_LOG, as PHARE's simulator: doc/simulator/simulator.md
MKN_KUL_FN_SIG_RET inline std::unique_ptr<std::ofstream> rank_log_file()
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

// tiled fields come with untiled temporaries for whole patch operations
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

    // model free: a model's types are named only where the build enables it
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

    // PHARE_SCOPE_TIMING=1 at setup: a file per rank, written when the simulator goes
    inline static std::string const timings_prefix = ".phare/timings/rank.";

    MKN_KUL_FN_SIG_RET double startTime() override { return startTime_; }
    MKN_KUL_FN_SIG_RET double endTime() override { return finalTime_; }
    MKN_KUL_FN_SIG_RET double timeStep() override { return dt_; }
    MKN_KUL_FN_SIG_RET double currentTime() override { return currentTime_; }

    void initialize() override;
    double advance() override;
    bool dump_restarts(double timestamp, double timestep) override;
    bool dump_diagnostics(double timestamp, double timestep) override;
    // the files flush_every left open, closed; the next dump reopens them (PHARE's)
    void flush_diagnostics() override
    {
#if DAMN_HAS_HIGHFIVE
        diagnostics_.files.clear();
#endif
    }

    // compute is collective when the quantity sums borders
    MKN_KUL_FN_SIG_RET bool is_computed(std::string const& key) const;
    void compute(std::size_t ln, std::string const& key);
    MKN_KUL_FN_SIG_RET auto computed(auto& patch,
                                     std::string const& key); // the hybrid model's grids

    std::vector<int> const& domainBox() const override { return domainBox_; }
    std::vector<double> const& cellWidth() const override { return cellWidth_; }
    std::size_t interporder() const override { return interp_order; }

    MKN_KUL_FN_SIG_RET std::string to_str() const override;

    MKN_KUL_FN_SIG_RET auto& hierarchy() const { return hierarchy_; }
    MKN_KUL_FN_SIG_RET auto& level(std::size_t const ln = 0) { return data_(ln); }
    MKN_KUL_FN_SIG_RET std::size_t n_levels() const { return levels_.size(); }
    // level ln's substeps' dts in the last advance (tests: their count follows its stable dt)
    MKN_KUL_FN_SIG_RET std::vector<double> substep_dts(std::size_t const ln) const
    {
        return ln < substep_dts_.size() ? substep_dts_[ln] : std::vector<double>{};
    }

    // a field resource's values on the domain nodes of every local patch of level ln, patch by
    // patch
    MKN_KUL_FN_SIG_RET std::vector<double> domain_values(std::string const& name,
                                                         std::size_t const ln = 0);

    // domain particles of every population of every local patch of level ln
    MKN_KUL_FN_SIG_RET std::size_t particle_count(std::size_t const ln = 0)
    {
        std::size_t n = 0;
        per_patch_(ln, [&](auto& patch) {
            for (auto& pop : patch.ions)
                n += real_particles_(pop.domainParticles());
        });
        return n;
    }

    // an array's particles: SoAVXChunked's without its gaps (weight 0)
    MKN_KUL_FN_SIG_RET static std::size_t real_particles_(auto const& particles)
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

    MKN_KUL_FN_SIG_RET int rank() const { return rank_; }
    MKN_KUL_FN_SIG_RET int ranks() const { return n_ranks_; }

    // over every rank (collective)
    MKN_KUL_FN_SIG_RET std::size_t total_particle_count(std::size_t const ln = 0)
    {
        auto const per_rank = particle_counts_per_rank(ln);
        return std::accumulate(per_rank.begin(), per_rank.end(), std::size_t{0});
    }

    // particle_count of every rank, on every rank (collective)
    MKN_KUL_FN_SIG_RET std::vector<std::size_t> particle_counts_per_rank(std::size_t const ln = 0)
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
    MKN_KUL_FN_SIG_RET std::vector<double> all_domain_values(std::string const& name,
                                                             std::size_t const ln = 0);
    // every rank's domain values of level ln, one per point (a point patches share once), in
    //  point order: the same for any decomposition (collective)
    MKN_KUL_FN_SIG_RET std::vector<double> level_values(std::string const& name,
                                                        std::size_t ln = 0);
    // level_values with where each value is: the points' indices in the level's AMR index space
    //  (dimension per point, the same order), and per direction whether the field is dual there
    //  (a point at (index + 1/2) cells) or primal (at index cells): what a plot needs (collective)
    struct LevelField
    {
        std::vector<int> points;
        std::vector<double> values;
        std::array<bool, dimension> dual{};
    };
    MKN_KUL_FN_SIG_RET LevelField level_field(std::string const& name, std::size_t ln = 0);

    // as PHARE's Simulator::summary, on rank 0 ("" elsewhere, collective): per level
    //  lvl:<n> (<patches>,<cells>,<patch ghost cells>,<level ghost cells>) parts(<particles>),
    //  then tot:<particles>; ghost cells within field_ghost_width of a patch, patch ghosts where
    //  a neighbour (or a periodic image) has them
    MKN_KUL_FN_SIG_RET std::string summary();

    // whistler waves need dt ~ dx^2 (PHARE's getMaxFinerLevelDt); PLAMN_TIME_RATIO overrides it
    //  for experiments on the fine levels' ghosts' time interpolation
    MKN_KUL_FN_SIG_RET int time_ratio() const
    {
        static int const forced = mkn::kul::env::get_as("PLAMN_TIME_RATIO", 0);
        return forced > 0 ? forced : hierarchy_.ratio * hierarchy_.ratio;
    }

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
        // bumped whenever the level's geometry is (re)built: what is kept per decomposition
        //  (boxings, device box lists, launch lists) is rebuilt when it differs
        std::uint64_t generation = 0;
    };

    // damn::amr::advance's hooks
    struct Hooks
    {
        Simulator& self;
        void advance(int ln, damn::amr::Step const& step) { self.advance_level_(ln, step); }
        // the dt level ln asks for after a step (damn::amr::TimeRefinement splits on it)
        double next_dt(int ln, damn::amr::Step const& step) { return self.next_dt_(ln, step); }
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

    auto const& computed_for_(std::string const& key) const;

    void setup_diagnostics_();
    // only >= 0: that level only, the others written empty (PHARE's dump_level)
    void dump_vtkhdf_(diagnostic::Diagnostic& diag, double time, int only = -1);
    // only >= 0: that level only (PHARE's dump_level: the time holds that level)
    void dump_phareh5_(diagnostic::Diagnostic& diag, double time, int only = -1);
    void dump_(diagnostic::Diagnostic& diag, double time, int only = -1); // the format's
    // PHARE's fine_dump_lvl_max: at the end of level ln's substeps that are not a level 0
    //  step's end (the finest after advancing, coarser ones after their synchronization)
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

    void setup_restarts_();
    void write_restart_(std::string const& dir);
    // the resources a restart stores of a level of the model's
    MKN_KUL_FN_SIG_RET static auto restart_names_(auto const& model)
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
    MKN_KUL_FN_SIG_RET auto& data_(std::size_t const ln)
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

    // the calling thread's, a cache line per patch: pool threads write their patches' own
    struct alignas(64) PatchBusy
    {
        double seconds;
    };

    // fn touches only its patch and never python; serial while scope timing (phlop's timer
    //  wants one thread a rank): doc/simulator/simulator.md
    void per_patch_threaded_(std::size_t const ln, auto&& fn)
    {
        using Pool    = damn::core::ThreadPool;
        auto& ld      = data_(ln);
        using Patch_t = solver::HybridModel<opts>::Patch_t;
        auto const n  = ld.data.size();
        if constexpr (!Patch_t::per_patch_mover)
            return per_patch_(ln, fn);
        if (Pool::n_pools < 2 or n < 2 or damn::mpi::scope_timing_active())
            return per_patch_(ln, fn);

        static thread_local std::vector<PatchBusy> busy;
        busy.assign(decomposition_.timing ? n : 0, PatchBusy{0.});
        Pool::INSTANCE().patch_exec(
            n, [](std::size_t const i) { return i; },
            [&](std::size_t const i) {
                auto const t0 = std::chrono::steady_clock::now();
                fn(*ld.data[i]);
                if (!busy.empty())
                    busy[i].seconds
                        = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                              .count();
            });
        for (auto const i : mkn::kul::range(busy.size()))
            decomposition_.busy_of(ln, ld.patches[i].box) += busy[i].seconds;
    }

    // tiled fields: damn's schedules work on leased untiled temporaries: doc/simulator/simulator.md
    //  a level's box copies collected over its patches, run at once (one launch on the GPU)
    static void lease_(auto& ld, std::vector<std::string> const& names, amr::Lease const how)
    {
        if (ld.data.empty())
            return;
        lease_level_(ld, *ld.data[0], names, how);
    }
    static void release_(auto& ld)
    {
        if (ld.data.empty())
            return;
        release_level_(ld, *ld.data[0]);
    }
    static void lease_level_(auto& ld, TiledFieldsPatch auto& first,
                             std::vector<std::string> const& names, amr::Lease const how)
    {
        auto& copies = std::decay_t<decltype(first)>::Untiled_t::copies();
        for (auto& patch : ld.data)
            simulator::for_each_field_grid(*patch, [&](std::string const& name, auto& grid) {
                if (std::find(names.begin(), names.end(), name) != names.end())
                    patch->untiled.lease(name, grid, patch->layout, how, copies);
            });
        copies.run();
    }
    static void lease_level_(auto&, auto&, std::vector<std::string> const&, amr::Lease) {}
    static void release_level_(auto& ld, TiledFieldsPatch auto& first)
    {
        auto& copies = std::decay_t<decltype(first)>::Untiled_t::copies();
        for (auto& patch : ld.data)
            patch->untiled.release(copies);
        copies.run();
    }
    static void release_level_(auto&, auto&) {}
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
        // a level's release, any model's (a refine leases levels of two patch types): at most
        //  four levels' per operation, held in place
        struct Release
        {
            void (*fn)(void*);
            void* ld;
        };
        std::array<Release, 4> releases{};
        std::size_t n = 0;
        ~Leases()
        {
            for (std::size_t i = 0; i < n; ++i)
                releases[i].fn(releases[i].ld);
        }
        template<typename LD>
        void add(LD& ld, std::vector<std::string> const& names, amr::Lease const how)
        {
            if (n == releases.size())
                throw std::logic_error("plamn Simulator::Leases: more than four leases");
            lease_(ld, names, how);
            releases[n++] = {[](void* l) { release_(*static_cast<LD*>(l)); }, std::addressof(ld)};
        }
    };
    template<typename How>
    static amr::Lease constexpr lease_of_()
    {
        if constexpr (std::is_same_v<How, std::decay_t<decltype(damn::amr::sum_borders)>>)
            return amr::Lease::sum;
        return amr::Lease::fill;
    }

    // body(pairs, access, ld) for key's resources on level ln, leased while it runs
    void with_copy_(std::size_t const ln, auto const how, std::string const& key, auto&& body)
    {
        auto const& resources = resources_.at(key);
        models_.visit(ln, [&](auto& model) {
            auto& ld = model.levels[static_cast<int>(ln)];
            static thread_local std::vector<std::string> dst, src; // read by the leases only
            dst.clear(), src.clear();
            for (auto const& r : resources)
                dst.push_back(r.dst), src.push_back(r.src);
            Leases leases;
            leases.add(ld, dst, lease_of_<std::decay_t<decltype(how)>>()); // first: dst wins
            leases.add(ld, src, amr::Lease::read);
            body(resources, access_(model, ld, key, how), ld);
        });
    }

    // a hybrid model's particle arrays, by the route's pairs (the plan) and accessors
    template<amr::ParticleRoute Copies>
    void with_copy_(std::size_t const ln, auto const how, Copies const& copies, auto&& body)
    {
        auto& ld = models_.hybrid.levels[static_cast<int>(ln)];
        static thread_local std::vector<std::string> dst, src; // read by the leases only
        dst.clear(), src.clear();
        for (auto const& r : copies.pairs)
            dst.push_back(r.dst), src.push_back(r.src);
        Leases leases;
        leases.add(ld, dst, lease_of_<std::decay_t<decltype(how)>>()); // first: dst wins
        leases.add(ld, src, amr::Lease::read);
        body(copies.pairs, copies.access, ld);
    }

    // copies what (a resource key, or a particle route) between the patches of level ln
    void copy_(std::size_t const ln, auto const how, auto const& what)
    {
        auto& l = levels_[ln];
        with_copy_(ln, how, what, [&](auto const& pairs, auto const& access, auto& ld) {
            if (n_ranks_ > 1)
                damn::mpi::copy(how, l.neighbours, pairs, access, ld);
            else
                damn::amr::copy(how, l.neighbours, pairs, access, ld);
        });
    }

    // fn(patch) on every patch of level ln, then the copy of what, overlapped per the work mode
    //  (PLAMN_WORK_MODE): doc/simulator/simulator.md
    enum class WorkMode : std::uint8_t { whole_patch, distance_first_overlaps };
    static WorkMode work_mode_()
    {
        static WorkMode const mode = [] {
            auto const* env = std::getenv("PLAMN_WORK_MODE");
            std::string const name{env ? env : "whole_patch"};
            if (name == "whole_patch")
                return WorkMode::whole_patch;
            if (name == "distance_first_overlaps")
                return WorkMode::distance_first_overlaps;
            throw std::runtime_error("plamn: PLAMN_WORK_MODE " + name
                                     + ": whole_patch or distance_first_overlaps");
        }();
        return mode;
    }

    void per_patch_then_copy_(std::size_t const ln, auto&& fn, auto const how, auto const& what)
    {
        using Pool    = damn::core::ThreadPool;
        using Patch_t = solver::HybridModel<opts>::Patch_t;
        auto& ld      = data_(ln);
        auto const n  = ld.data.size();
        if constexpr (Patch_t::per_patch_mover)
            if (work_mode_() == WorkMode::distance_first_overlaps and n_ranks_ > 1
                and Pool::n_pools > 1 and n > 1 and !damn::mpi::scope_timing_active())
            {
                static thread_local std::vector<PatchBusy> busy;
                busy.assign(decomposition_.timing ? n : 0, PatchBusy{0.});
                with_copy_(ln, how, what, [&](auto const& pairs, auto const& access, auto& cld) {
                    damn::mpi::Copy c{how, levels_[ln].neighbours, pairs, access, cld};
                    auto const order = c.remote_first();
                    Pool::INSTANCE().patch_exec(
                        n, [](std::size_t const i) { return i; },
                        [&](std::size_t const i) {
                            auto const t0 = std::chrono::steady_clock::now();
                            fn(*ld.data[i]);
                            if (!busy.empty())
                                busy[i].seconds = std::chrono::duration<double>(
                                                      std::chrono::steady_clock::now() - t0)
                                                      .count();
                        },
                        Pool::PatchHooks{.order   = order,
                                         .on_done = [&](std::size_t const i) { c.done(i); },
                                         .idle    = [&] { c.poll(); }});
                    c.finish();
                });
                for (auto const i : mkn::kul::range(busy.size()))
                    decomposition_.busy_of(ln, ld.patches[i].box) += busy[i].seconds;
                return;
            }
        per_patch_threaded_(ln, fn);
        copy_(ln, how, what);
    }

    // untiled field copies by accessors, resolved once per model; tiled patches (leased
    //  temporaries) and particles by name
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

    // from level ln - 1 into level ln: a hybrid model's particle arrays, by the route's
    //  accessors (both levels are hybrid: particles are split from hybrid levels only)
    template<amr::ParticleRoute Refines>
    void refine_(std::size_t const ln, auto const& op, damn::amr::LevelRegion const region,
                 Refines const& refines)
    {
        auto& l = levels_[ln];
        with_level_pair_(ln, [&](auto& ld, auto& coarse) {
            Leases leases;
            lease_level_pairs_(leases, ld, coarse, refines.pairs, /*refine=*/true);
            auto const by = [&](auto const& access) {
                if (n_ranks_ > 1)
                    damn::mpi::refine(op, region, *l.to_coarser, refines.pairs, access, coarse, ld);
                else
                    damn::amr::refine(op, region, *l.to_coarser, refines.pairs, access, coarse, ld);
            };
            by(particle_access_(ld, coarse, refines));
        });
    }
    template<typename FineLD, typename CoarseLD, typename Refines>
        requires(std::is_same_v<patch_of_t<FineLD>, typename Refines::Patch_t>
                 and std::is_same_v<patch_of_t<CoarseLD>, typename Refines::Patch_t>)
    static auto const& particle_access_(FineLD&, CoarseLD&, Refines const& refines)
    {
        return refines.access;
    }
    static auto const& particle_access_(auto&, auto&, auto const&) { return damn::amr::by_name; }

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
            static thread_local std::vector<std::string> fine, coarser, all; // read by add only
            fine.clear(), coarser.clear();
            for (auto const& p : pairs)
            {
                fine.push_back(p.fine);
                coarser.push_back(p.coarse);
                if (!p.coarse_new.empty())
                    coarser.push_back(p.coarse_new);
            }
            all.assign(fine.begin(), fine.end());
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

    // untiled fields' accessors, resolved once per models and pairs; null: by name (particle
    //  arrays, tiled patches, or no patch of either level here yet)
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
            bool constexpr fine_mhd   = simulator::coupling::is_mhd_patch_v<F>;
            bool constexpr coarse_mhd = simulator::coupling::is_mhd_patch_v<C>;
            // looked up by reference: the pairs copied into a key once, when first resolved
            auto it = level_access_cache_.find(
                std::tuple<bool, bool, std::vector<damn::amr::LevelPair> const&>{
                    fine_mhd, coarse_mhd, pairs});
            if (it == level_access_cache_.end())
            {
                if (damn::amr::size(fine) == 0 or damn::amr::size(coarse) == 0)
                    return static_cast<Access_t const*>(nullptr);
                it = level_access_cache_
                         .emplace(
                             LevelAccessKey{fine_mhd, coarse_mhd, pairs},
                             amr::level_field_access<C::field_ghost_width, F::field_ghost_width>(
                                 *coarse.data[0], *fine.data[0], pairs))
                         .first;
            }
            auto const& cached = std::any_cast<Cached_t const&>(it->second);
            return cached ? &*cached : static_cast<Access_t const*>(nullptr);
        }
    }
    using LevelAccessKey = std::tuple<bool, bool, std::vector<damn::amr::LevelPair>>;
    std::map<LevelAccessKey, std::any, std::less<>> level_access_cache_; // level_access_'s

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
    MKN_KUL_FN_SIG_RET bool should_rebalance_();
    void rebalance_(double t); // level 0 onto new owners, then each finer level

    // doc/simulator/simulator_decomposition.md
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
    MKN_KUL_FN_SIG_RET static int decomposition_reach_();
    MKN_KUL_FN_SIG_RET static bool busy_push_();
    MKN_KUL_FN_SIG_RET static std::array<int, 2 * dimension> d_key_(Box_t const& box);
    void record_decomposition_(std::string const& event, double t, bool always = false);

    void register_resources_();
    // the MHD states' (conservative) and E's resources: MHDState_<s>, MHD_E, MHD_B (state 0's B)
    void register_mhd_resources_();
    MKN_KUL_FN_SIG_RET static std::string mhd_state_key(std::size_t const s)
    {
        return "MHDState_" + std::to_string(s);
    }
    // level ln's patches from its boxes (that level's cells): chopped into the largest patch
    //  size, partitioned along the Hilbert curve by the particles they will hold
    MKN_KUL_FN_SIG_RET damn::amr::Level<dimension> level_of_(int ln,
                                                             std::vector<Box_t> const& boxes);
    // damn::amr::decompose's settings: this run's topology, patch size, spill, its weights (the
    //  particles a patch will hold, or holds; MHD levels: cells), `partition`
    MKN_KUL_FN_SIG_RET damn::amr::Decompose<dimension> decompose_in_(std::string const& partition);
    void make_level_(int ln, std::vector<Box_t> const& boxes);
    // after a regrid / rebalance, finest down to lowest: from their populations, coarsened from
    //  the finer level
    void rebuild_moments_(int lowest);
    void allocate_level_(int ln); // the patch data of hierarchy_'s level ln, linked
    void link_level_(int ln);     // neighbours, map to the coarser level, level ghost cells
    void link_hybrid_level_(int ln);
    void link_mhd_level_(int ln);
    // neighbours reaching reach cells, the map to the coarser level coarser_reach, the level ghost
    //  cells of each local patch (gw: its field ghosts)
    void link_geometry_(int ln, int reach, int coarser_reach, int gw);
    // a patch of level ln (its cells): its particle ghost box among the level's other patches
    MKN_KUL_FN_SIG_RET auto layout_of_(int ln, Box_t const& box) const;
    MKN_KUL_FN_SIG_RET auto make_patch_(auto const& model, int ln, Box_t const& box) const;
    // the boxes of level ln + 1 from level ln's tagged cells (collective)
    MKN_KUL_FN_SIG_RET std::vector<Box_t> cluster_(int ln);
    MKN_KUL_FN_SIG_RET std::vector<std::vector<Box_t>> pretag_();
    // the levels finer than from re-clustered (tagging) at t; always: re-made even where their
    //  boxes did not change (SAMRAI's)
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
    // -DDAMN_SUPER_DEBUG only: doc/simulator/simulator_sums.md
    void sums_(std::size_t ln, char const* stage, double t);
    void debug_regrid_(int ln, double t);
    // a rebalance moves data: what must survive it (rebalance_state_), checked after it
    struct RebalanceState
    {
        std::vector<std::vector<double>> values; // per level and quantity, its domain values
        std::vector<double> particles;           // per level and set: count, sums of w, v, x
    };
    MKN_KUL_FN_SIG_RET RebalanceState rebalance_state_();
    void check_rebalance_(RebalanceState const& before, double t);
    // regrid_level_'s data, the level's model's: where the old level was, moved; the new cells
    //  (outside old_cells) refined from the coarser level, then the ghosts
    void regrid_hybrid_data_(int ln, auto const& map, std::vector<Box_t> const& old_cells, double t,
                             bool moved);
    // a hybrid level's resources whose ghosts a rebalance moves (every field grid, the old level
    //  ghost particles), from this rank's patches (a rank without any has no transfers)
    MKN_KUL_FN_SIG_RET damn::amr::RegridGhosts moved_ghosts_(int ln);
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
    MKN_KUL_FN_SIG_RET double alpha_(std::size_t const ln, double const t) const
    {
        auto const& l = levels_[ln];
        // no coarser step in progress (initialization, regrid): the coarser level's data now
        return l.coarse_dt > 0 ? (t - l.coarse_t) / l.coarse_dt : 1;
    }

    // a level ghost particle lives in one patch, its cell's owner (Surroundings): the others'
    //  copies of a split dropped, old (initialization, regrid, restart) or new (coarse step)

    // what the models' solvers use of a level
    struct LevelServices
    {
        Simulator& self;
        std::size_t ln;

        void sums(char const* stage, double const t) { self.sums_(ln, stage, t); } // SUPER_DEBUG

        MKN_KUL_FN_SIG_RET std::size_t number() const { return ln; }
        MKN_KUL_FN_SIG_RET std::uint64_t generation() const { return self.levels_[ln].generation; }
        MKN_KUL_FN_SIG_RET auto& data() { return self.data_(ln); }
        void per_patch(auto&& fn) { self.per_patch_(ln, fn); }
        // fn touching only its patch, never python: on damn's pools (per_patch_threaded_)
        void per_patch_threaded(auto&& fn) { self.per_patch_threaded_(ln, fn); }
        void copy(auto const how, std::string const& key) { self.copy_(ln, how, key); }
        void copy(auto const how, amr::ParticleRoute auto const& copies)
        {
            self.copy_(ln, how, copies);
        }
        // fn(patch) on every patch, then the copy: overlapped per the work mode
        void per_patch_then_copy(auto&& fn, auto const how, auto const& what)
        {
            self.per_patch_then_copy_(ln, fn, how, what);
        }
        void fill_magnetic_ghosts(std::string const& key, double const t)
        {
            self.fill_magnetic_ghosts_(ln, key, t);
        }
        void fill_electric_ghosts(std::string const& key, double const t)
        {
            self.fill_electric_ghosts_(ln, key, t);
        }
        MKN_KUL_FN_SIG_RET double alpha(double const t) const { return self.alpha_(ln, t); }
        void check_set(std::string const& key, std::string const& where)
        {
            self.check_set_(ln, key, where);
        }
        MKN_KUL_FN_SIG_RET auto mesh_size() const
        {
            return damn::amr::mesh_size(self.hierarchy_, static_cast<int>(ln));
        }

        // with the coarser level (ln > 0)
        MKN_KUL_FN_SIG_RET LevelServices coarser() const { return {self, ln - 1}; }
        MKN_KUL_FN_SIG_RET double coarse_dt() const { return self.levels_[ln].coarse_dt; }
        MKN_KUL_FN_SIG_RET int time_ratio() const { return self.time_ratio(); }
        // its last step's share of its coarser level's step (flux sums' weight)
        MKN_KUL_FN_SIG_RET double substep_weight() const { return self.substep_weight_(ln); }
        void refine(auto const& op, damn::amr::LevelRegion const region,
                    std::vector<damn::amr::LevelPair> const& pairs, double const alpha = 0)
        {
            self.refine_(ln, op, region, pairs, alpha);
        }
        void refine(auto const& op, damn::amr::LevelRegion const region,
                    amr::ParticleRoute auto const& refines)
        {
            self.refine_(ln, op, region, refines);
        }
        void coarsen(auto const& op, std::vector<damn::amr::LevelPair> const& pairs)
        {
            self.coarsen_(ln, op, pairs);
        }
    };
    MKN_KUL_FN_SIG_RET LevelServices level_services_(std::size_t const ln) { return {*this, ln}; }

    // what the MHD model's advance uses of a level
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

        // with the coarser level
        MKN_KUL_FN_SIG_RET std::size_t number() const { return ln; }
        MKN_KUL_FN_SIG_RET MHDLevel coarser() const { return {self, ln - 1}; }
        void coarsen(auto const& op, std::vector<damn::amr::LevelPair> const& pairs)
        {
            self.coarsen_(ln, op, pairs);
        }
        void copy(auto const how, std::string const& key) { self.copy_(ln, how, key); }
        MKN_KUL_FN_SIG_RET double old_time() const { return self.levels_[ln].old_t; }
        MKN_KUL_FN_SIG_RET int time_ratio() const { return self.time_ratio(); }
        // its last step's share of its coarser level's step (flux sums' weight)
        MKN_KUL_FN_SIG_RET double substep_weight() const { return self.substep_weight_(ln); }
    };
    // patch ghosts, then the physical boundaries'
    void fill_mhd_state_ghosts_(std::size_t ln, std::size_t s, double t);
    void fill_mhd_electric_ghosts_(std::size_t ln, double t);
    // fn(patch, its physical boundary regions) for every local patch of MHD level ln
    void mhd_physical_boundaries_(std::size_t ln, auto&& fn);

    // damn::amr::advance's hooks
    void advance_level_(int ln, damn::amr::Step const& step);
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
    // normal: plamn's; high: PHARE's, for comparisons: doc/simulator/simulator.md
    enum class RegridMode : std::uint8_t { normal, high };
    RegridMode regrid_mode_ = RegridMode::normal;
    // refinement="tagging": every level's boxes found from B alone before any particle is made
    //  (simulation/AMR/refinement/tagging/pretag, or PLAMN_PRETAG=1)
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
        // how a level's patches get their ranks, plamn's (PHARE comparisons keep "levels"):
        //  doc/simulator/simulator.md
        std::string partition = "levels";
        double spill          = -1;
        int near_band         = -1; // -1: the default
    } lb_;
    std::optional<amr::CoarsestRebalanceCheck> rebalance_check_;
    int rank_    = 0;
    int n_ranks_ = 1;
    damn::amr::Topology topology_;


    std::array<int, dimension> patch_size_{};
    // per level, without largest_patch_size: a patch is at most the level's domain, as PHARE's
    //  default (level 0's cells on a refined level cut its boxes in 2^dim per level)
    std::vector<std::array<int, dimension>> level_size_;
    // cells per direction of the blocks level 0 is cut on when rebalanced
    static constexpr int rebalance_block = 8;
    std::vector<int> domainBox_;
    std::vector<double> cellWidth_;
    Timestepper timestepper_; // each step's dt, the time after it
    // the adaptive dt: level 0's stable dt (each level steps at its own: next_dt_)
    double stable_dt_(Timestepper::CFL const& cfl);
    // level ln's stable dt at t (hybrid: whistler, MHD: fast + Hall, diffusive), all ranks; the
    //  adaptive run stops (non-finite speed, Timestepper's substep / shrink limits) throw
    double level_stable_dt_(std::size_t ln, double t);
    // this rank's bound of level ln (-1: a non-finite speed)
    double level_bound_(std::size_t ln, Timestepper::CFL const& cfl);
    // damn::amr::TimeRefinement's next_dt: constant, the level's dt of the table (dt /
    //  time_ratio^ln); adaptive, level 0 its step's, a finer level min(its coarser step, stable)
    double next_dt_(int ln, damn::amr::Step const& step);
    // level ln's last step's share of its coarser level's step
    MKN_KUL_FN_SIG_RET double substep_weight_(std::size_t ln) const;
    // per level: its last step (damn's), its dt (constant: the table; adaptive: its last stable)
    std::vector<damn::amr::Step> level_step_;
    std::vector<std::vector<double>> substep_dts_; // per level, in the last advance
    std::vector<double> level_dts_;
    double first_level0_dt_ = -1; // adaptive: the run's first level 0 dt (shrink limit)
    // the substeps' split (SAMRAI's), kept across steps; made at the first advance
    std::optional<damn::amr::TimeRefinement<dimension>> stepping_;
    // a restart's split state, for stepping_ when it is made
    std::optional<damn::amr::TimeRefinementState> restored_stepping_;
    // a restart's timestepper state: resumed (same mode and dt) or not (afresh from the restart)
    std::optional<Timestepper::State> restored_timestepper_;
    void start_timestepper_()
    {
        if (restored_timestepper_)
            timestepper_.resume(*restored_timestepper_, startTime_);
        else
            timestepper_.start_at(startTime_);
        restored_timestepper_.reset();
    }
    // a start or a restart (its dt or mode may differ): the levels' dts and the split afresh
    void restart_stepping_()
    {
        level_step_.clear();
        level_dts_.clear();
        first_level0_dt_ = -1;
        stepping_.reset();
    }
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
