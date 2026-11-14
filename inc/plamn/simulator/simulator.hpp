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

#include "damn/amr/box_data.hpp"
#include "damn/amr/copy.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/inter_level.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/level_map.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/amr/partition.hpp"
#include "damn/amr/time_refinement.hpp"
#include "damn/amr/topology.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/logger.hpp"
#include "damn/core/math/array.hpp"
#include "damn/mpi/copy.hpp"
#include "damn/mpi/damn_mpi.hpp"
#include "damn/mpi/inter_level.hpp"
#include "damn/mpi/scope_timing.hpp"
#include "damn/mpi/topology.hpp"
#include "damn/core/def/constants.hpp"

#include "plamn/amr/coarsen.hpp"
#include "plamn/amr/refine.hpp"
#include "plamn/core/data/field/field_algorithm.hpp"
#include "plamn/core/def.hpp"
#include "plamn/core/models/electromag_initializer.hpp"
#include "plamn/core/numerics/ampere/ampere_transformer.hpp"
#include "plamn/core/numerics/faraday/faraday_transformer.hpp"
#include "plamn/core/numerics/interpolator/interpolating.hpp"
#include "plamn/core/numerics/moments/moments.hpp"
#include "plamn/core/numerics/ohm/ohm.hpp"
#include "plamn/core/numerics/ohm/ohm_transformer.hpp"
#include "plamn/core/numerics/particles_synchronised_mover/particles_synchronised_mover_def.hpp"
#include "plamn/initializer/data_provider.hpp"
#include "plamn/simulator/hybrid_patch.hpp"
#include "plamn/simulator/simulator_def.hpp"
#include "plamn/amr/resources.hpp"
#include "plamn/amr/split/split.hpp" // IWYU pragma: keep (the Splitter specializations)
#include "plamn/amr/split/splitter.hpp"

#include "dict.hpp"
#include "phlop/timing/scope.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <format>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>
#include <initializer_list>

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
    };

    void setup_();

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

    void register_resources_();
    void make_level_(int ln, std::vector<Box_t> const& boxes);
    void initialize_root_();
    void initialize_refined_(std::size_t ln);

    // level ghost fills (patch ghosts, then level ghosts from the coarser level at time t)
    void fill_magnetic_ghosts_(std::size_t ln, std::string const& key, double t);
    void fill_electric_ghosts_(std::size_t ln, std::string const& key, double t);
    // B over the level ghost cells of every local patch made divergence free
    void touch_up_level_ghosts_(std::size_t ln, std::string const& key);
    NO_DISCARD double alpha_(std::size_t const ln, double const t) const
    {
        auto const& l = levels_[ln];
        return l.coarse_dt > 0 ? (t - l.coarse_t) / l.coarse_dt : 0;
    }

    void compute_moments_(std::size_t ln);
    void fill_ion_borders_(std::size_t ln);
    void deposit_level_ghosts_(std::size_t ln, double t);
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
    int rank_    = 0;
    int n_ranks_ = 1;
    damn::amr::Topology topology_;

    std::optional<core::OhmInfo> ohm_info_; // const members: set once, in setup_
    typename Patch_t::ParticlesSynchronisedMover_t mover_{};

    std::array<int, dimension> patch_size_{};
    std::vector<int> domainBox_;
    std::vector<double> cellWidth_;
    double dt_          = 0;
    int timeStepNbr_    = 0;
    double startTime_   = 0;
    double finalTime_   = 0;
    double currentTime_ = 0;
    bool isInitialized_ = false;
    bool timing_        = false; // this simulator started the scope timers
};


namespace detail
{
    // the pairs of a vector's components between levels: coarse (and coarse_new, if not
    //  empty) on the coarser level, fine on the finer
    NO_DISCARD inline std::vector<damn::amr::LevelPair>
    vector_pairs(std::string const& coarse, std::string const& fine,
                 std::string const& coarse_new = "")
    {
        std::vector<damn::amr::LevelPair> pairs;
        for (auto const* c : {"_x", "_y", "_z"})
            pairs.push_back({coarse + c, fine + c, coarse_new.empty() ? "" : coarse_new + c});
        return pairs;
    }
} // namespace detail


//-----------------------------------------------------------------------------
//                           setup
//-----------------------------------------------------------------------------

template<auto opts>
void Simulator<opts>::setup_()
{
    auto const& sim = dict_["simulation"];

    // the ranks, if MPI is running (pyplamn starts it with pyphare's startMPI)
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

    auto const models = sim["models"].template to<std::vector<std::string>>();
    if (models != std::vector<std::string>{"HybridModel"})
        throw std::runtime_error("plamn Simulator: only HybridModel for now");

    dt_          = cppdict::get_value(dict_, "simulation/time_step/value", 0.);
    timeStepNbr_ = cppdict::get_value(dict_, "simulation/time_step_nbr", 0);
    finalTime_   = sim["final_time"].template to<double>();
    if (cppdict::get_value(dict_, "simulation/time_step/mode", TimeStepType::constant)
        != TimeStepType::constant)
        throw std::runtime_error("plamn Simulator: constant time step only for now");

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
    if (amr.contains("largest_patch_size"))
    {
        auto const largest = amr["largest_patch_size"].template to<std::vector<int>>();
        for (std::size_t i = 0; i < dimension; ++i)
            patch_size_[i] = largest[i];
    }

    // finer levels: the user's refinement boxes (cells of the coarser level), refined
    hierarchy_.max_levels = cppdict::get_value(dict_, "simulation/AMR/max_nbr_levels", 1);
    if (hierarchy_.max_levels > 1)
    {
        if (!amr.contains("refinement") or !amr["refinement"].contains("boxes"))
            throw std::runtime_error("plamn Simulator: refinement boxes only for now (no tagging)");
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

    ohm_info_.emplace(core::OhmInfo::FROM(sim["algo"]["ohm"]));

    register_resources_();
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

    auto const& ions    = dict_["simulation"]["ions"];
    auto const nbr_pops = ions["nbrPopulations"].template to<std::size_t>();
    for (std::size_t i = 0; i < nbr_pops; ++i)
    {
        auto const name = ions["pop" + std::to_string(i)]["name"].template to<std::string>();
        population_names_.push_back(name);

        // every population's moments in one exchange, PHAREHUB/PHARE#1079
        add("IonPopBorderSum", components(name + "_flux"));
        add("IonPopBorderSum",
            damn::amr::same_names({name + "_density", name + "_charge_density"}));
        // the same, as ghost fills (after a synchronization)
        add("IonPopMoments", components(name + "_flux"));
        add("IonPopMoments", damn::amr::same_names({name + "_density", name + "_charge_density"}));

        add("IonGhostParticles", {{name + "_patch_ghost", name + "_domain"}});
    }
}


template<auto opts>
std::vector<double> Simulator<opts>::box_weights_(int const ln, std::vector<Box_t> const& boxes)
{
    auto const& ions    = dict_["simulation"]["ions"];
    auto const nbr_pops = ions["nbrPopulations"].template to<std::size_t>();

    using Initializers = Patch_t::Types::ParticleInitializerFactory_t;
    std::vector<decltype(Initializers::create(ions["pop0"]["particle_initializer"]))> initializers;
    for (std::size_t i = 0; i < nbr_pops; ++i)
    {
        auto const& pop = ions["pop" + std::to_string(i)];
        initializers.push_back(Initializers::create(pop["particle_initializer"],
                                                    pop["name"].template to<std::string>()));
    }

    std::vector<double> weights(boxes.size(), 0.);
    for (std::size_t k = static_cast<std::size_t>(rank_); k < boxes.size();
         k += static_cast<std::size_t>(n_ranks_))
    {
        Layout_t const layout{boxes[k], damn::amr::mesh_size(hierarchy_, ln),
                              damn::amr::position(hierarchy_, ln, boxes[k].lower), ln};
        double w = static_cast<double>(boxes[k].size()); // field work, a particle's worth a cell
        for (auto const& initializer : initializers)
            w += static_cast<double>(initializer->nbrParticles(layout));
        weights[k] = w;
    }
    if (n_ranks_ > 1)
        MPI_Allreduce(MPI_IN_PLACE, weights.data(), static_cast<int>(weights.size()), MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
    return weights;
}

template<auto opts>
std::vector<double> Simulator<opts>::domain_values(std::string const& name, std::size_t const ln)
{
    auto& ld = levels_.at(ln).data;
    std::vector<double> values;
    for (std::size_t i = 0; i < damn::amr::size(ld); ++i)
    {
        auto const& cells = ld.patches[i].box;
        bool const found  = damn::amr::with_resource(ld.data[i], name, [&](auto& r) {
            if constexpr (damn::amr::BoxData<std::decay_t<decltype(r)>, dimension>)
                damn::core::for_each_point(domain_box(r, cells), [&](auto const& p) {
                    values.push_back(value(r, cells, p));
                });
        });
        if (!found)
            throw std::runtime_error("plamn Simulator: no resource " + name);
    }
    return values;
}

template<auto opts>
std::vector<double> Simulator<opts>::all_domain_values(std::string const& name,
                                                       std::size_t const ln)
{
    auto const mine = domain_values(name, ln);
    if (n_ranks_ == 1)
        return mine;

    auto& ld = levels_.at(ln).data;
    // every rank's values and per patch sizes (a rank may have no patch)
    std::vector<int> sizes; // of my patches, in my level order
    for (std::size_t i = 0; i < damn::amr::size(ld); ++i)
        damn::amr::with_resource(ld.data[i], name, [&](auto& r) {
            if constexpr (damn::amr::BoxData<std::decay_t<decltype(r)>, dimension>)
                sizes.push_back(static_cast<int>(domain_box(r, ld.patches[i].box).size()));
        });

    auto const gather = [&]<typename T>(std::vector<T> const& v, MPI_Datatype const type) {
        int const n = static_cast<int>(v.size());
        std::vector<int> counts(n_ranks_), displs(n_ranks_);
        MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        std::partial_sum(counts.begin(), counts.end() - 1, displs.begin() + 1);
        std::vector<T> all(static_cast<std::size_t>(displs.back() + counts.back()));
        MPI_Allgatherv(v.data(), n, type, all.data(), counts.data(), displs.data(), type,
                       MPI_COMM_WORLD);
        return std::make_pair(all, displs);
    };
    auto const [values, value_at]   = gather(mine, MPI_DOUBLE);
    auto const [all_sizes, size_at] = gather(sizes, MPI_INT);

    // a rank's patches come in level order: walk the level, taking each patch's values from its
    //  rank's next ones
    std::vector<std::size_t> next_value(n_ranks_), next_size(n_ranks_);
    for (int r = 0; r < n_ranks_; ++r)
    {
        next_value[r] = static_cast<std::size_t>(value_at[r]);
        next_size[r]  = static_cast<std::size_t>(size_at[r]);
    }
    std::vector<double> ordered;
    ordered.reserve(values.size());
    for (auto const& patch : hierarchy_.levels[ln].patches)
    {
        auto const size = static_cast<std::size_t>(all_sizes[next_size[patch.rank]++]);
        auto const from = values.begin() + static_cast<std::ptrdiff_t>(next_value[patch.rank]);
        ordered.insert(ordered.end(), from, from + static_cast<std::ptrdiff_t>(size));
        next_value[patch.rank] += size;
    }
    return ordered;
}


//-----------------------------------------------------------------------------
//                           initialize
//-----------------------------------------------------------------------------

// level ln from its boxes (that level's cells): chopped into the largest patch size,
// partitioned along the Hilbert curve by the particles they will hold, allocated, with their
// neighbours and the map to the coarser level
template<auto opts>
void Simulator<opts>::make_level_(int const ln, std::vector<Box_t> const& boxes)
{
    // refined levels: patches of whole coarser cells, the size a multiple of the ratio
    auto size = patch_size_;
    if (ln > 0)
        for (auto& s : size)
            s = std::max(hierarchy_.ratio, s / hierarchy_.ratio * hierarchy_.ratio);
    std::vector<Box_t> patches;
    for (auto const& box : boxes)
        for (auto const& b : damn::amr::chop(box, size))
            patches.push_back(b);
    damn::amr::set_level(
        hierarchy_,
        damn::amr::make_level(ln, patches,
                              damn::amr::partition(patches, box_weights_(ln, patches), n_ranks_)));

    auto const& sim = dict_["simulation"];
    auto const pgw    = Patch_t::particle_ghost_width;
    auto constexpr gw = Patch_t::field_ghost_width;
    levels_.resize(static_cast<std::size_t>(ln) + 1);
    auto& l = levels_[ln];
    l.data  = damn::amr::allocate(hierarchy_.levels[ln], rank_, [&](auto const& p) {
        Layout_t const layout{p.box, damn::amr::mesh_size(hierarchy_, ln),
                              damn::amr::position(hierarchy_, ln, p.box.lower), ln};
        auto const ghost = damn::core::grow(p.box, damn::core::ConstArray<int, dimension>(pgw));
        return std::make_unique<Patch_t>(layout, sim,
                                         damn::amr::covered_by_level(hierarchy_, ln, ghost));
    });

    // the level's neighbours, as far as any resource reaches outside its patch: field ghosts
    // plus primal nodes, or particle ghosts
    auto constexpr reach = std::max<int>(gw + 1, pgw);
    l.neighbours         = damn::amr::make_neighbours(hierarchy_, ln, rank_, topology_, reach);
    if (ln == 0)
        return;

    // to the coarser level: as far as a refined field ghost's stencil, or a split particle's
    //  parent, reaches
    l.to_coarser = damn::amr::make_level_map(hierarchy_, ln - 1, rank_, topology_,
                                             reach + 2 * amr::splitBoxGrowth<Splitter_t>() + 4);
    for (auto const& p : l.data.patches)
    {
        auto& cells = l.level_ghost_cells.emplace_back();
        cells.push_back(damn::core::grow(p.box, damn::core::ConstArray<int, dimension>(gw)));
        for (auto const& covered : damn::amr::covered_by_level(hierarchy_, ln, cells.front()))
        {
            std::vector<Box_t> rest;
            for (auto const& c : cells)
                for (auto const& r : c.remove(covered))
                    rest.push_back(r);
            cells = std::move(rest);
        }
    }
}

template<auto opts>
void Simulator<opts>::initialize()
{
    DAMN_LOG_SCOPE(1, "Simulator::initialize");
    if (isInitialized_)
        throw std::runtime_error("plamn Simulator already initialized");

    make_level_(0, {hierarchy_.domain});
    initialize_root_();
    for (std::size_t ln = 1; ln <= refinement_boxes_.size(); ++ln)
    {
        make_level_(static_cast<int>(ln), refinement_boxes_[ln - 1]);
        initialize_refined_(ln);
    }
    // every level's E and B at t0, for the finer levels' time interpolation
    for (std::size_t ln = 0; ln < levels_.size(); ++ln)
        prepare_step_(static_cast<int>(ln), startTime_);

    currentTime_   = startTime_;
    isInitialized_ = true;
}

// PHARE's HybridLevelInitializer, root level: the user's functions
template<auto opts>
void Simulator<opts>::initialize_root_()
{
    auto const& sim = dict_["simulation"];
    per_patch_(0, [&](auto& patch) {
        core::ElectromagInitializer<dimension>{sim["electromag"]}(patch.em, patch.layout);
        for (auto& pop : patch.ions)
        {
            auto initializer = Patch_t::Types::ParticleInitializerFactory_t::create(
                pop.particleInitializerInfo(), pop.name());
            initializer->loadParticles(pop.domainParticles(), patch.layout);
        }
    });
    copy_(0, damn::amr::fill_ghosts, "EM_B");
    copy_(0, damn::amr::fill_ghosts, "EM_E");

    // moments from the particles, completed across patch borders
    per_patch_(0, [&](auto& patch) {
        core::resetMoments(patch.ions);
        core::depositParticles(patch.ions, patch.layout, Interpolator{}, core::DomainDeposit{});
    });
    fill_ion_borders_(0);
    compute_moments_(0);

    // J from B, then E from Ohm's law
    ampere_ohm_<&Patch_t::em>(0);
    copy_(0, damn::amr::fill_ghosts, "EM_E");
}

// PHARE's HybridLevelInitializer, a refined level: B, E and the particles from the coarser
// level, the level ghost particles split from it, then the moments and J
template<auto opts>
void Simulator<opts>::initialize_refined_(std::size_t const ln)
{
    using damn::amr::LevelRegion;
    refine_(ln, amr::FieldRefiner{}, LevelRegion::Interior, detail::vector_pairs("EM_B", "EM_B"));
    {
        auto& l = levels_[ln];
        for (std::size_t i = 0; i < damn::amr::size(l.data); ++i)
        {
            auto& patch = *l.data.data[i];
            auto bx     = amr::field_resource<Patch_t::field_ghost_width>(patch.emB[0]);
            auto by     = amr::field_resource<Patch_t::field_ghost_width>(patch.emB[1]);
            auto bz     = amr::field_resource<Patch_t::field_ghost_width>(patch.emB[2]);
            amr::divergence_free_touch_up(std::array{&bx, &by, &bz}, l.data.patches[i].box,
                                          l.data.patches[i].box,
                                          damn::amr::mesh_size(hierarchy_, static_cast<int>(ln)));
        }
    }
    refine_(ln, amr::FieldRefiner{}, LevelRegion::Interior, detail::vector_pairs("EM_E", "EM_E"));
    fill_magnetic_ghosts_(ln, "EM_B", startTime_);
    fill_electric_ghosts_(ln, "EM_E", startTime_);

    for (auto const& name : population_names_)
    {
        refine_(ln, amr::ParticleSplitter<Splitter_t>{}, LevelRegion::Interior,
                {{name + "_domain", name + "_domain", ""}});
        refine_(ln, amr::ParticleSplitter<Splitter_t>{}, LevelRegion::LevelGhosts,
                {{name + "_domain", name + "_level_ghost_old", ""}});
    }
    per_patch_(ln, [](auto& patch) {
        for (auto& pop : patch.ions)
            pop.levelGhostParticles() = pop.levelGhostParticlesOld();
    });

    per_patch_(ln, [&](auto& patch) {
        core::resetMoments(patch.ions);
        core::depositParticles(patch.ions, patch.layout, Interpolator{}, core::DomainDeposit{});
        core::depositParticles(patch.ions, patch.layout, Interpolator{}, core::LevelGhostDeposit{});
    });
    fill_ion_borders_(ln);
    compute_moments_(ln);

    // J from B; E was refined
    per_patch_(ln,
               [&](auto& patch) { core::AmpereTransformer{}(patch.layout, patch.em.B, patch.J); });
}


//-----------------------------------------------------------------------------
//                           ghosts
//-----------------------------------------------------------------------------

template<auto opts>
void Simulator<opts>::fill_magnetic_ghosts_(std::size_t const ln, std::string const& key,
                                            double const t)
{
    copy_(ln, damn::amr::fill_ghosts, key);
    if (ln == 0)
        return;
    refine_(ln, amr::FieldRefiner{}, damn::amr::LevelRegion::LevelGhosts,
            detail::vector_pairs("EM_B_old", key, "EM_B"), alpha_(ln, t));
    touch_up_level_ghosts_(ln, key);
}

template<auto opts>
void Simulator<opts>::fill_electric_ghosts_(std::size_t const ln, std::string const& key,
                                            double const t)
{
    copy_(ln, damn::amr::fill_ghosts, key);
    if (ln == 0)
        return;
    refine_(ln, amr::FieldRefiner{}, damn::amr::LevelRegion::LevelGhosts,
            detail::vector_pairs("EM_E_old", key, "EM_E"), alpha_(ln, t));
}

template<auto opts>
void Simulator<opts>::touch_up_level_ghosts_(std::size_t const ln, std::string const& key)
{
    auto constexpr gw = Patch_t::field_ghost_width;
    auto& l           = levels_[ln];
    auto const dx     = damn::amr::mesh_size(hierarchy_, static_cast<int>(ln));
    for (std::size_t i = 0; i < damn::amr::size(l.data); ++i)
    {
        std::vector<amr::FieldResource<typename Patch_t::Grid_t, gw>> B;
        for (auto const* c : {"_x", "_y", "_z"})
            damn::amr::with_resource(l.data.data[i], key + c, [&](auto& r) {
                if constexpr (std::is_same_v<std::decay_t<decltype(r)>,
                                             amr::FieldResource<typename Patch_t::Grid_t, gw>>)
                    B.push_back(r);
            });
        for (auto const& region : l.level_ghost_cells[i])
            amr::divergence_free_touch_up(std::array{&B[0], &B[1], &B[2]}, l.data.patches[i].box,
                                          region, dx);
    }
}


//-----------------------------------------------------------------------------
//                           advance: SolverPPC per level, time refined
//-----------------------------------------------------------------------------

template<auto opts>
double Simulator<opts>::advance()
{
    DAMN_LOG_SCOPE(1, "Simulator::advance");
    if (!isInitialized_)
        throw std::runtime_error("plamn Simulator: initialize() first");

    Hooks hooks{*this};
    damn::amr::advance(hierarchy_, hooks, currentTime_, dt_, time_ratio());
    currentTime_ += dt_;
    return dt_;
}

template<auto opts>
void Simulator<opts>::advance_level_(int const level_number, double const t, double const dt)
{
    auto const ln = static_cast<std::size_t>(level_number);
    predictor1_(ln, t, dt);
    average_(ln, t, dt);
    move_ions_(ln, t, dt, core::UpdaterMode::domain_only);
    predictor2_(ln, t, dt);
    average_(ln, t, dt);
    move_ions_(ln, t, dt, core::UpdaterMode::all);
    corrector_(ln, t, dt);

    // the finest level's share of the coarser level's reflux, per substep: the time average of
    //  E over them (coarser finer levels accumulate theirs when synchronized)
    if (ln > 0 and ln + 1 == levels_.size())
        accumulate_flux_sum_(ln, 1. / time_ratio());
}

// E and B at the start of the step, for the finer level's time interpolation and the reflux
template<auto opts>
void Simulator<opts>::prepare_step_(int const ln, double)
{
    per_patch_(static_cast<std::size_t>(ln), [](auto& patch) {
        for (std::size_t i = 0; i < 3; ++i)
        {
            patch.oldB[i].copyData(patch.emB[i]);
            patch.oldE[i].copyData(patch.emE[i]);
        }
    });
}

// the coarser level is at coarse_t + coarse_dt: the level ghost particles of then, split from
// its domain particles
template<auto opts>
void Simulator<opts>::first_step_(int const level_number, double, double const coarse_t,
                                  double const coarse_dt)
{
    auto const ln = static_cast<std::size_t>(level_number);
    auto& l       = levels_[ln];
    l.coarse_t    = coarse_t;
    l.coarse_dt   = coarse_dt;
    per_patch_(ln, [](auto& patch) {
        for (auto& pop : patch.ions)
            pop.levelGhostParticlesNew().clear();
        for (auto& e : patch.fluxSumE)
            e.zero();
    });
    for (auto const& name : population_names_)
        refine_(ln, amr::ParticleSplitter<Splitter_t>{}, damn::amr::LevelRegion::LevelGhosts,
                {{name + "_domain", name + "_level_ghost_new", ""}});
}

template<auto opts>
void Simulator<opts>::last_step_(int const ln)
{
    per_patch_(static_cast<std::size_t>(ln), [](auto& patch) {
        for (auto& pop : patch.ions)
        {
            std::swap(pop.levelGhostParticlesNew(), pop.levelGhostParticlesOld());
            pop.levelGhostParticlesNew().clear();
            pop.levelGhostParticles() = pop.levelGhostParticlesOld();
        }
    });
}

// level ln onto ln - 1, both at t: E and the moments coarsened, then the coarser level's B
// recomputed from the fine time average of E (reflux), then its ghosts refilled
template<auto opts>
void Simulator<opts>::synchronize_(int const level_number, double const t)
{
    DAMN_LOG_SCOPE(1, "Simulator::synchronize_");
    auto const ln     = static_cast<std::size_t>(level_number);
    auto const coarse = ln - 1;

    coarsen_(ln, amr::ElectricFieldCoarsener{}, detail::vector_pairs("EM_E", "EM_E"));
    std::vector<damn::amr::LevelPair> moments;
    for (auto const& name : population_names_)
    {
        for (auto const& p : detail::vector_pairs(name + "_flux", name + "_flux"))
            moments.push_back(p);
        moments.push_back({name + "_density", name + "_density", ""});
        moments.push_back({name + "_charge_density", name + "_charge_density", ""});
    }
    coarsen_(ln, amr::MomentsCoarsener{}, moments);

    // reflux: the coarser Eavg under the fine level is the fine one's time average
    coarsen_(ln, amr::ElectricFieldCoarsener{}, detail::vector_pairs("EMAvg_E", "EM_fluxSum_E"));
    copy_(coarse, damn::amr::fill_ghosts, "EMAvg_E");
    auto const dt = levels_[ln].coarse_dt;
    per_patch_(coarse, [&](auto& patch) {
        core::FaradayTransformer{}(patch.layout, patch.Bold, patch.emAvg.E, patch.em.B, dt);
    });
    fill_magnetic_ghosts_(coarse, "EM_B", t);
    if (coarse > 0)
        accumulate_flux_sum_(coarse, 1. / time_ratio());

    // ghosts after the coarsening
    fill_electric_ghosts_(coarse, "EM_E", t);
    copy_(coarse, damn::amr::fill_ghosts, "IonPopMoments");
    compute_moments_(coarse);
}

template<auto opts>
void Simulator<opts>::accumulate_flux_sum_(std::size_t const ln, double const coef)
{
    per_patch_(ln, [&](auto& patch) {
        for (std::size_t i = 0; i < 3; ++i)
        {
            auto* sum         = patch.fluxSumE[i].data();
            auto const* avg   = patch.avgE[i].data();
            auto const values = patch.fluxSumE[i].size();
            for (std::size_t k = 0; k < values; ++k)
                sum[k] += coef * avg[k];
        }
    });
}

// the ions' totals from the populations' (complete) moments
template<auto opts>
void Simulator<opts>::compute_moments_(std::size_t const ln)
{
    DAMN_LOG_SCOPE(1, "Simulator::compute_moments_");
    per_patch_(ln, [](auto& patch) { patch.ions.update(); });
}

template<auto opts>
void Simulator<opts>::fill_ion_borders_(std::size_t const ln)
{
    // the first exchange after the push: waits here are other ranks' longer pushes
    DAMN_LOG_SCOPE(1, "Simulator::fill_ion_borders_");
    copy_(ln, damn::amr::sum_borders, "IonPopBorderSum");
}

// the level border's moments: the level ghost particles split from the coarser level at both
// ends of its step, weighted linearly in time (PHARE's fillIonPopMomentGhosts)
template<auto opts>
void Simulator<opts>::deposit_level_ghosts_(std::size_t const ln, double const t)
{
    if (ln == 0)
        return;
    auto alpha = alpha_(ln, t);
    if (alpha < -1e-10 or alpha > 1 + 1e-10)
        throw std::runtime_error("plamn Simulator: level ghost time interpolation coefficient "
                                 + std::to_string(alpha));
    alpha = std::clamp(alpha, 0., 1.);
    per_patch_(ln, [&](auto& patch) {
        Interpolator interpolate;
        for (auto& pop : patch.ions)
        {
            interpolate(pop.levelGhostParticlesOld(), pop.particleDensity(), pop.flux(),
                        patch.layout, 1. - alpha);
            interpolate(pop.levelGhostParticlesNew(), pop.particleDensity(), pop.flux(),
                        patch.layout, alpha);
        }
    });
}

// J from B (Ampere, on the physical box + 1: what Ohm reads, so J is never exchanged), the
// electrons from the ions and J, then E from Ohm's law; electromag: the HybridPatch member
// holding the B used and the E computed, the model's (em) or the predicted one (emPred)
template<auto opts>
template<auto electromag>
void Simulator<opts>::ampere_ohm_(std::size_t const ln)
{
    // one loop each, so each has its own scope (time, counters)
    {
        DAMN_LOG_SCOPE(1, "Simulator::ampere");
        per_patch_(ln, [&](auto& patch) {
            core::AmpereTransformer{}(patch.layout, (patch.*electromag).B, patch.J);
        });
    }
    {
        DAMN_LOG_SCOPE(1, "Simulator::electrons");
        per_patch_(ln, [&](auto& patch) { patch.electrons().update(patch.layout); });
    }
    DAMN_LOG_SCOPE(1, "Simulator::ohm");
    per_patch_(ln, [&](auto& patch) {
        auto& em = patch.*electromag;
        core::OhmTransformer{*ohm_info_}(patch.layout, patch.electrons().density(),
                                         patch.electrons().velocity(), patch.electrons().pressure(),
                                         em.B, patch.J, em.E);
    });
}

template<auto opts>
void Simulator<opts>::average_(std::size_t const ln, double const t, double const dt)
{
    DAMN_LOG_SCOPE(1, "Simulator::average_");
    per_patch_(ln, [](auto& patch) {
        for (std::size_t i = 0; i < 3; ++i)
        {
            core::average(patch.em.B[i], patch.emPred.B[i], patch.emAvg.B[i]);
            core::average(patch.em.E[i], patch.emPred.E[i], patch.emAvg.E[i]);
        }
    });
    fill_electric_ghosts_(ln, "EMAvg_E", t + dt / 2);
}

template<auto opts>
void Simulator<opts>::move_ions_(std::size_t const ln, double const t, double const dt,
                                 core::UpdaterMode const mode)
{
    DAMN_LOG_SCOPE(1, "Simulator::move_ions");
    {
        DAMN_LOG_SCOPE(1, "Simulator::move_ions::push"); // and deposit
        per_patch_(ln, [&](auto& patch) {
            mover_.updatePopulations(patch.ions, patch.emAvg, patch.boxing, dt, mode);
        });
    }
    deposit_level_ghosts_(ln, t + dt);
    fill_ion_borders_(ln);

    if (mode != core::UpdaterMode::domain_only)
    {
        // particles that left their patch go to the patch owning their cell
        copy_(ln, damn::amr::gather_leaving, "IonGhostParticles");
        per_patch_(ln, [](auto& patch) {
            for (auto& pop : patch.ions)
                pop.patchGhostParticles().clear();
        });
    }
    compute_moments_(ln);
}

template<auto opts>
void Simulator<opts>::predictor1_(std::size_t const ln, double const t, double const dt)
{
    DAMN_LOG_SCOPE(1, "Simulator::predictor1_");
    {
        DAMN_LOG_SCOPE(1, "Simulator::faraday");
        per_patch_(ln, [&](auto& patch) {
            core::FaradayTransformer{}(patch.layout, patch.em.B, patch.em.E, patch.emPred.B, dt);
        });
    }
    fill_magnetic_ghosts_(ln, "EMPred_B", t + dt);
    ampere_ohm_<&Patch_t::emPred>(ln);
}

template<auto opts>
void Simulator<opts>::predictor2_(std::size_t const ln, double const t, double const dt)
{
    DAMN_LOG_SCOPE(1, "Simulator::predictor2_");
    {
        DAMN_LOG_SCOPE(1, "Simulator::faraday");
        per_patch_(ln, [&](auto& patch) {
            core::FaradayTransformer{}(patch.layout, patch.em.B, patch.emAvg.E, patch.emPred.B, dt);
        });
    }
    fill_magnetic_ghosts_(ln, "EMPred_B", t + dt);
    ampere_ohm_<&Patch_t::emPred>(ln);
}

template<auto opts>
void Simulator<opts>::corrector_(std::size_t const ln, double const t, double const dt)
{
    DAMN_LOG_SCOPE(1, "Simulator::corrector_");
    {
        DAMN_LOG_SCOPE(1, "Simulator::faraday");
        per_patch_(ln, [&](auto& patch) {
            core::FaradayTransformer{}(patch.layout, patch.em.B, patch.emAvg.E, patch.em.B, dt);
        });
    }
    fill_magnetic_ghosts_(ln, "EM_B", t + dt);
    ampere_ohm_<&Patch_t::em>(ln);
    fill_electric_ghosts_(ln, "EM_E", t + dt);
}

template<auto opts>
Simulator<opts>::~Simulator()
{
    if (!timing_)
        return;
    int mpi_on = 0, mpi_done = 0;
    MPI_Initialized(&mpi_on);
    MPI_Finalized(&mpi_done);
    if (!mpi_on or mpi_done)
        return;         // too late to write
    std::error_code ec; // pyphare makes it on rank 0, maybe not yet
    std::filesystem::create_directories(std::filesystem::path{timings_prefix}.parent_path(), ec);
    damn::mpi::stop_scope_timing();
}

template<auto opts>
std::string Simulator<opts>::summary()
{
    auto constexpr gw = Patch_t::field_ghost_width;

    std::stringstream ss;
    unsigned long long total = 0;
    for (std::size_t ln = 0; ln < levels_.size(); ++ln)
    {
        auto& l = levels_[ln];
        // patches, cells, patch ghost cells, level ghost cells, particles
        std::array<unsigned long long, 5> mine{}, all{};
        for (std::size_t i = 0; i < damn::amr::size(l.data); ++i)
        {
            auto const& box   = l.data.patches[i].box;
            auto const& links = l.neighbours.of[i];
            ++mine[0];
            mine[1] += static_cast<unsigned long long>(box.size());
            damn::core::for_each_point(damn::core::grow(box, gw), [&](auto const& p) {
                if (damn::core::isIn(p, box))
                    return;
                bool const neighbour = std::any_of(links.begin(), links.end(), [&](auto const& k) {
                    return damn::core::isIn(p, damn::core::shift(k.patch.box, k.shift));
                });
                ++mine[neighbour ? 2 : 3];
            });
        }
        mine[4] = particle_count(ln);

        all = mine;
        if (n_ranks_ > 1)
            MPI_Reduce(mine.data(), all.data(), static_cast<int>(mine.size()),
                       MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        ss << "lvl:" << ln << " (" << all[0] << "," << all[1] << "," << all[2] << "," << all[3]
           << ") parts(" << all[4] << "), \n";
        total += all[4];
    }
    if (rank_ != 0)
        return "";
    ss << "tot:" << total;
    return ss.str();
}

template<auto opts>
std::string Simulator<opts>::to_str() const
{
    std::stringstream ss;
    ss << "PLAMN SIMULATOR\n";
    ss << "------------------------------------\n";
    ss << "interpolation order  : " << interp_order << "\n";
    ss << "dimension            : " << dimension << "\n";
    ss << "time step            : " << dt_ << "\n";
    ss << "number of time steps : " << timeStepNbr_ << "\n";
    ss << "levels               : " << levels_.size() << "\n";
    return ss.str();
}

} // namespace plamn

#endif // PLAMN_SIMULATOR_SIMULATOR_HPP
