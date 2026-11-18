#ifndef PLAMN_SIMULATOR_SIMULATOR_HPP
#define PLAMN_SIMULATOR_SIMULATOR_HPP

// Simulator: PHARE's simulator shell (dict parsing, times, initialize/advance) over damn
//
// hybrid only. Level 0 is the domain chopped into patches, partitioned over MPI's ranks by the
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
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/inter_level.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/level_map.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/amr/topology.hpp"
#include "damn/hdf5/damn_hdf5.hpp"
#include "damn/core/vector.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/mpi/copy.hpp"
#include "damn/mpi/damn_mpi.hpp"
#include "damn/mpi/inter_level.hpp"
#include "damn/core/def/constants.hpp"
#include "plamn/diagnostic/vtkhdf.hpp"
#include "plamn/simulator/hybrid_patch_view.hpp"
#include "plamn/solver/hybrid/advance.hpp"
#include "plamn/solver/hybrid/computed.hpp"
#include "plamn/solver/hybrid/initialize.hpp"
#include "plamn/solver/hybrid/level_steps.hpp"
#include "plamn/solver/hybrid/model.hpp"
#include "plamn/solver/hybrid/patches.hpp"
#include "plamn/solver/hybrid/restarts.hpp"
#include "plamn/solver/stacks.hpp"
#include "plamn/amr/load_balancing.hpp"
#include "plamn/core/def.hpp"
#include "plamn/core/numerics/interpolator/interpolating.hpp"
#include "plamn/core/numerics/ohm/ohm.hpp"
#include "plamn/initializer/data_provider.hpp"
#include "plamn/simulator/hybrid_patch.hpp"
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

template<auto opts>
class Simulator : public ISimulator
{
public:
    auto static constexpr options = opts;

    std::size_t static constexpr dimension     = opts.dimension;
    std::size_t static constexpr interp_order  = opts.interp_order;
    std::size_t static constexpr nbRefinedPart = opts.nbRefinedPart;

    using Hybrid_t     = solver::HybridStack<opts>;
    using Patch_t      = Hybrid_t::Patch_t;
    using Grid_t       = Patch_t::Grid_t;
    using Layout_t     = Patch_t::Layout_t;
    using Hierarchy_t  = damn::amr::Hierarchy<dimension>;
    using LevelData_t  = Hybrid_t::LevelData_t;
    using Neighbours_t = damn::amr::Neighbours<dimension>;
    using LevelMap_t   = damn::amr::LevelMap<dimension>;
    using Resources_t  = std::map<std::string, std::vector<damn::amr::ResourcePair>>;
    using Box_t        = damn::core::Box<int, dimension>;
    using Interpolator = core::Interpolating<dimension, interp_order>;
    using Splitter_t   = Hybrid_t::Splitter_t;

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

    // computed quantities by key (simulator_computed.hpp): compute is collective when the
    //  quantity sums borders
    NO_DISCARD bool is_computed(std::string const& key) const;
    void compute(std::size_t ln, std::string const& key);
    NO_DISCARD std::vector<Grid_t*> computed(Patch_t& patch, std::string const& key);

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
        void synchronize(int ln, double t) { self.synchronize_(ln, t); }
        void after_step(int ln, double t, bool last) { self.after_step_(ln, t, last); }
    };

    void setup_();

    // computed quantities, the models' (simulator_computed.hpp)
    using Computed_t = solver::HybridModel<opts>::Computed_t;
    Computed_t const& computed_for_(std::string const& key) const;

    // diagnostics, see simulator_diagnostics.hpp
    void setup_diagnostics_();
    // only >= 0: that level only, the others written empty (PHARE's dump_level)
    void dump_vtkhdf_(diagnostic::Diagnostic& diag, double time, int only = -1);
    // only >= 0: that level only (PHARE's dump_level: the time holds that level)
    void dump_phareh5_(diagnostic::Diagnostic& diag, double time, int only = -1);
    void dump_(diagnostic::Diagnostic& diag, double time, int only = -1); // the format's
    // a particle diagnostic's population index (/ions/pop/<name>/domain or levelGhost)
    NO_DISCARD std::size_t particle_population_(std::string const& quantity) const;
    // PHARE's fine_dump_lvl_max: every diagnostic, level ln only, at the end of its substeps
    //  that are not a level 0 step's end (the finest after advancing, coarser ones after their
    //  synchronization)
    void fine_dump_(std::size_t ln, double time);
    using View_t = decltype(simulator::view(std::declval<Patch_t&>()));
    struct Diagnostics
    {
        using Float = std::conditional_t<PLAMN_DIAG_DOUBLES, double, float>;
        std::string dir;
        bool phareh5   = false;    // else pharevtkhdf
        bool overwrite = false;    // pyphare's mode "overwrite": truncated at the first dump only
        int fine_dump_lvl_max = 0; // PHARE's: levels 1 to it dumped at their substeps, 0: none
        std::vector<diagnostic::Diagnostic> list;
        std::map<std::string, diagnostic::Accessor<View_t>> quantities;
#if DAMN_HAS_HIGHFIVE
        std::map<std::string, std::unique_ptr<damn::hdf5::h5::HighFiveFile>> files; // open ones
#endif
        std::set<std::string> created;        // the files made (or truncated) so far
        damn::MinimizingVector<Float> buffer; // a level's rows on this rank
    } diagnostics_;

    // restarts, see simulator_restarts.hpp
    void setup_restarts_();
    void write_restart_(std::string const& dir);
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

    // the hybrid model: its levels' patch data
    solver::HybridModel<opts> hybrid_;
    // level ln's patch data (made if new)
    NO_DISCARD LevelData_t& data_(std::size_t const ln)
    {
        return hybrid_.levels[static_cast<int>(ln)];
    }
    // the levels from n on gone, with their data; the levels to n made (no data yet)
    void resize_levels_(std::size_t const n)
    {
        levels_.resize(n);
        hybrid_.truncate(n);
    }

    // one patch's work, for every local patch of level ln
    void per_patch_(std::size_t const ln, auto&& fn)
    {
        for (auto& patch : data_(ln).data)
            fn(*patch);
    }

    // copies key's resources between the patches of level ln, see register_resources_
    void copy_(std::size_t const ln, auto const how, std::string const& key)
    {
        auto& l = levels_[ln];
        auto& ld = data_(ln);
        if (n_ranks_ > 1)
            damn::mpi::copy(how, l.neighbours, resources_.at(key), ld);
        else
            damn::amr::copy(how, l.neighbours, resources_.at(key), ld);
    }

    // from level ln - 1 into level ln
    void refine_(std::size_t const ln, auto const& op, damn::amr::LevelRegion const region,
                 std::vector<damn::amr::LevelPair> const& pairs, double const alpha = 0)
    {
        auto& l = levels_[ln];
        auto& ld = data_(ln);
        if (n_ranks_ > 1)
            damn::mpi::refine(op, region, *l.to_coarser, pairs, data_(ln - 1), ld, alpha);
        else
            damn::amr::refine(op, region, *l.to_coarser, pairs, data_(ln - 1), ld, alpha);
    }

    // from level ln onto level ln - 1
    void coarsen_(std::size_t const ln, auto const& op,
                  std::vector<damn::amr::LevelPair> const& pairs)
    {
        auto& l = levels_[ln];
        auto& ld = data_(ln);
        if (n_ranks_ > 1)
            damn::mpi::coarsen(op, *l.to_coarser, pairs, ld, data_(ln - 1));
        else
            damn::amr::coarsen(op, *l.to_coarser, pairs, ld, data_(ln - 1));
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

    void register_resources_();
    // level ln's patches from its boxes (that level's cells): chopped into the largest patch
    //  size, partitioned along the Hilbert curve by the particles they will hold
    NO_DISCARD damn::amr::Level<dimension> level_of_(int ln, std::vector<Box_t> const& boxes);
    void make_level_(int ln, std::vector<Box_t> const& boxes);
    void allocate_level_(int ln); // the patch data of hierarchy_'s level ln, linked
    void link_level_(int ln);     // neighbours, map to the coarser level, level ghost cells
    // a patch of level ln (its cells): its particle ghost box among the level's other patches
    NO_DISCARD Layout_t layout_of_(int ln, Box_t const& box) const;
    NO_DISCARD auto make_patch_(int ln, Box_t const& box) const; // the model's
    // the boxes of level ln + 1 from level ln's tagged cells (collective)
    NO_DISCARD std::vector<Box_t> cluster_(int ln);
    NO_DISCARD std::vector<std::vector<Box_t>> pretag_();
    // the levels finer than from re-clustered (tagging) at t; always: re-made even where their
    //  boxes did not change (SAMRAI's, see RegridMode)
    void regrid_(double t, int from = 0, bool always = false);
    // regrid_mode high: the finer levels of ln regridded at SAMRAI's regrid points, after a step
    //  of ln to t (last: ln's last substep of its coarser level's step)
    void after_step_(int ln, double t, bool last);
    // keep_finer: the finer levels stay (each to be regridded onto its own boxes in turn)
    void regrid_level_(int ln, std::vector<Box_t> const& boxes, double t, bool keep_finer = false);

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

        NO_DISCARD std::size_t number() const { return ln; }
        NO_DISCARD auto& data() { return self.data_(ln); }
        void per_patch(auto&& fn) { self.per_patch_(ln, fn); }
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
    std::optional<amr::Tagging> tagging_;
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
#include "plamn/simulator/simulator_diagnostics.hpp" // IWYU pragma: export
#include "plamn/simulator/simulator_setup.hpp"       // IWYU pragma: export

#endif // PLAMN_SIMULATOR_SIMULATOR_HPP
