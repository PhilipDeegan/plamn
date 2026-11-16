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
// split). advance() is damn::amr::advance (time refinement): each level takes PHARE's SolverPPC
// step (predictor1, average, move ions (domain only), predictor2, average, move ions (all),
// corrector), finer levels ratio^2 steps per coarser one (dt ~ dx^2), as PHARE's
// MultiPhysicsIntegrator and HybridHybridMessengerStrategy:
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
template<auto opts>
class Simulator : public ISimulator
{
public:
    auto static constexpr options = opts;

    std::size_t static constexpr dimension     = opts.dimension;
    std::size_t static constexpr interp_order  = opts.interp_order;
    std::size_t static constexpr nbRefinedPart = opts.nbRefinedPart;

    using Patch_t      = simulator::HybridPatch<opts>;
    using Grid_t       = Patch_t::Grid_t;
    using Layout_t     = Patch_t::Layout_t;
    using Hierarchy_t  = damn::amr::Hierarchy<dimension>;
    using LevelData_t  = damn::amr::LevelData<dimension, std::unique_ptr<Patch_t>>;
    using Neighbours_t = damn::amr::Neighbours<dimension>;
    using LevelMap_t   = damn::amr::LevelMap<dimension>;
    using Resources_t  = std::map<std::string, std::vector<damn::amr::ResourcePair>>;
    using Box_t        = damn::core::Box<int, dimension>;
    using Interpolator = core::Interpolating<dimension, interp_order>;
    using Splitter_t
        = amr::Splitter<damn::core::DimConst<dimension>, damn::core::InterpConst<interp_order>,
                        amr::RefinedParticlesConst<nbRefinedPart>>;

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
    NO_DISCARD auto& level(std::size_t const ln = 0) { return levels_.at(ln).data; }
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
                n += pop.domainParticles().size();
        });
        return n;
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
    // one level's data and how it moves
    struct Level
    {
        LevelData_t data;
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

    // computed quantities, see simulator_computed.hpp
    struct Computed
    {
        std::vector<std::pair<std::string, core::HybridQuantity::Scalar>> grids;
        std::function<void(Patch_t&, std::vector<Grid_t*> const&)> local;
        std::string border_sum; // a resources_ key, empty: none
        std::function<void(Patch_t&, std::vector<Grid_t*> const&)> after;
    };
    std::map<std::string, Computed> computed_;
    void setup_computed_();
    Computed const& computed_for_(std::string const& key) const;

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
    NO_DISCARD std::vector<std::string> restart_names_() const;
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

    // one patch's work, for every local patch of level ln
    void per_patch_(std::size_t const ln, auto&& fn)
    {
        for (auto& patch : levels_[ln].data.data)
            fn(*patch);
    }

    // copies key's resources between the patches of level ln, see register_resources_
    void copy_(std::size_t const ln, auto const how, std::string const& key)
    {
        auto& l = levels_[ln];
        if (n_ranks_ > 1)
            damn::mpi::copy(how, l.neighbours, resources_.at(key), l.data);
        else
            damn::amr::copy(how, l.neighbours, resources_.at(key), l.data);
    }

    // from level ln - 1 into level ln
    void refine_(std::size_t const ln, auto const& op, damn::amr::LevelRegion const region,
                 std::vector<damn::amr::LevelPair> const& pairs, double const alpha = 0)
    {
        auto& l = levels_[ln];
        if (n_ranks_ > 1)
            damn::mpi::refine(op, region, *l.to_coarser, pairs, levels_[ln - 1].data, l.data,
                              alpha);
        else
            damn::amr::refine(op, region, *l.to_coarser, pairs, levels_[ln - 1].data, l.data,
                              alpha);
    }

    // from level ln onto level ln - 1
    void coarsen_(std::size_t const ln, auto const& op,
                  std::vector<damn::amr::LevelPair> const& pairs)
    {
        auto& l = levels_[ln];
        if (n_ranks_ > 1)
            damn::mpi::coarsen(op, *l.to_coarser, pairs, l.data, levels_[ln - 1].data);
        else
            damn::amr::coarsen(op, *l.to_coarser, pairs, l.data, levels_[ln - 1].data);
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
    typename Patch_t::Surroundings surroundings_(int ln, Box_t const& box) const;
    // the boxes of level ln + 1 from level ln's tagged cells (collective)
    NO_DISCARD std::vector<Box_t> cluster_(int ln);
    // the levels finer than from re-clustered (tagging) at t; always: re-made even where their
    //  boxes did not change (SAMRAI's, see RegridMode)
    void regrid_(double t, int from = 0, bool always = false);
    // regrid_mode high: the finer levels of ln regridded at SAMRAI's regrid points, after a step
    //  of ln to t (last: ln's last substep of its coarser level's step)
    void after_step_(int ln, double t, bool last);
    void regrid_level_(int ln, std::vector<Box_t> const& boxes, double t);
    void initialize_root_();
    void initialize_refined_(std::size_t ln);
    void refined_moments_(std::size_t ln); // a refined level's moments and J from its particles

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

    void compute_moments_(std::size_t ln);
    void fill_ion_borders_(std::size_t ln);
    void deposit_level_ghosts_(std::size_t ln, double t);
    // a level ghost particle lives in one patch, its cell's owner (Surroundings): the others'
    //  copies of a split dropped, old (initialization, regrid, restart) or new (coarse step)
    void own_level_ghosts_(std::size_t ln, bool new_ghosts);
    void move_ions_(std::size_t ln, double t, double dt, core::UpdaterMode mode);
    template<auto electromag>
    void ampere_ohm_(std::size_t ln);
    void average_(std::size_t ln, double t, double dt);
    void predictor1_(std::size_t ln, double t, double dt);
    void predictor2_(std::size_t ln, double t, double dt);
    void corrector_(std::size_t ln, double t, double dt);

    // damn::amr::advance's hooks, see damn/amr/time_refinement.hpp
    void advance_level_(int ln, double t, double dt);
    void prepare_step_(int ln, double t);
    void first_step_(int ln, double t, double coarse_t, double coarse_dt);
    void last_step_(int ln);
    void synchronize_(int ln, double t);
    void accumulate_flux_sum_(std::size_t ln, double coef);

    initializer::Dict dict_;
    Hierarchy_t hierarchy_;
    std::vector<Level> levels_;
    Resources_t resources_; // what moves together, by name
    std::vector<std::string> population_names_;
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

    std::optional<core::OhmInfo> ohm_info_; // const members: set once, in setup_
    typename Patch_t::ParticlesSynchronisedMover_t mover_{};

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
