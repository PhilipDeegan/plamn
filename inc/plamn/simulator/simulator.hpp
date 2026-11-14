#ifndef PLAMN_SIMULATOR_SIMULATOR_HPP
#define PLAMN_SIMULATOR_SIMULATOR_HPP

// Simulator: PHARE's simulator shell (dict parsing, times, initialize/advance) over damn
//
// for now: hybrid only, one level, one rank. The level is a damn::amr::Hierarchy with a
// uniform level 0, its data a LevelData of HybridPatch, communication damn::amr schedules.
// initialize() follows PHARE's HybridLevelInitializer for the root level, advance() its
// SolverPPC: predictor1, average, move ions (domain only), predictor2, average, move ions
// (all), corrector

#include "damn/core/logger.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/move.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/amr/topology.hpp"
#include "damn/amr/box_data.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/math/array.hpp"

#include "plamn/core/def.hpp"
#include "plamn/core/numerics/ohm/ohm.hpp"
#include "plamn/core/models/electromag_initializer.hpp"
#include "plamn/core/numerics/ohm/ohm_transformer.hpp"
#include "plamn/core/numerics/ampere/ampere_transformer.hpp"
#include "plamn/core/numerics/faraday/faraday_transformer.hpp"
#include "plamn/core/numerics/moments/moments.hpp"
#include "plamn/core/numerics/interpolator/interpolating.hpp"
#include "plamn/core/data/field/field_algorithm.hpp"
#include "plamn/simulator/simulator_def.hpp"
#include "plamn/simulator/hybrid_patch.hpp"
#include "plamn/initializer/data_provider.hpp"
#include "plamn/core/numerics/particles_synchronised_mover/particles_synchronised_mover_def.hpp"

#include "dict.hpp"
#include "phlop/timing/scope.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <optional>
#include <cstddef>
#include <stdexcept>
#include <format>
#include <type_traits>

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
    using Resources_t  = std::map<std::string, std::vector<damn::amr::ResourcePair>>;
    using Box_t        = damn::core::Box<int, dimension>;
    using Interpolator = core::Interpolating<dimension, interp_order>;

    // dict: the whole input, copied: patches keep references into it
    explicit Simulator(initializer::Dict const& dict)
        : dict_{dict}
    {
        setup_();
    }

    Simulator(Simulator const&)            = delete;
    Simulator& operator=(Simulator const&) = delete;

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
    NO_DISCARD auto& level() { return level_; }

    // a field resource's values on the domain nodes of every local patch, patch by patch
    NO_DISCARD std::vector<double> domain_values(std::string const& name)
    {
        std::vector<double> values;
        for (std::size_t i = 0; i < damn::amr::size(level_); ++i)
        {
            auto const& cells = level_.patches[i].box;
            bool const found  = damn::amr::with_resource(level_.data[i], name, [&](auto& r) {
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

    // domain particles of every population of every local patch
    NO_DISCARD std::size_t particle_count()
    {
        std::size_t n = 0;
        per_patch_([&](auto& patch) {
            for (auto& pop : patch.ions)
                n += pop.domainParticles().size();
        });
        return n;
    }

private:
    void setup_();

    // one patch's work, for every local patch
    void per_patch_(auto&& fn)
    {
        for (auto& patch : level_.data)
            fn(*patch);
    }

    // moves key's resources between the level's patches, see register_resources_
    void move_(auto const how, std::string const& key)
    {
        damn::amr::move(how, neighbours_, resources_.at(key), level_);
    }

    void register_resources_();
    void compute_moments_();
    void fill_ion_borders_();
    void move_ions_(double dt, core::UpdaterMode mode);
    template<auto electromag>
    void ampere_ohm_();
    void average_();
    void predictor1_(double dt);
    void predictor2_(double dt);
    void corrector_(double dt);

    initializer::Dict dict_;
    Hierarchy_t hierarchy_;
    LevelData_t level_;
    Neighbours_t neighbours_;
    Resources_t resources_; // what moves together, by name
    int rank_ = 0;

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
};


//-----------------------------------------------------------------------------
//                           setup
//-----------------------------------------------------------------------------

template<auto opts>
void Simulator<opts>::setup_()
{
    auto const& sim = dict_["simulation"];

    auto const models = sim["models"].template to<std::vector<std::string>>();
    if (models != std::vector<std::string>{"HybridModel"})
        throw std::runtime_error("plamn Simulator: only HybridModel for now");
    if (sim["AMR"].contains("max_nbr_levels")
        and sim["AMR"]["max_nbr_levels"].template to<int>() > 1)
        throw std::runtime_error("plamn Simulator: one level only for now");

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
    if (sim["AMR"].contains("largest_patch_size"))
    {
        auto const largest = sim["AMR"]["largest_patch_size"].template to<std::vector<int>>();
        for (std::size_t i = 0; i < dimension; ++i)
            patch_size_[i] = largest[i];
    }
    hierarchy_.max_levels = 1;

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

        // every population's moments in one exchange, PHAREHUB/PHARE#1079
        add("IonPopBorderSum", components(name + "_flux"));
        add("IonPopBorderSum",
            damn::amr::same_names({name + "_density", name + "_charge_density"}));

        add("IonGhostParticles", {{name + "_patch_ghost", name + "_domain"}});
    }
}


//-----------------------------------------------------------------------------
//                           initialize
//-----------------------------------------------------------------------------

// PHARE's HybridLevelInitializer, root level
template<auto opts>
void Simulator<opts>::initialize()
{
    DAMN_LOG_SCOPE(1, "Simulator::initialize");
    if (isInitialized_)
        throw std::runtime_error("plamn Simulator already initialized");

    auto const& sim = dict_["simulation"];
    damn::amr::set_level(hierarchy_, damn::amr::uniform_level_0(hierarchy_, patch_size_));

    auto const pgw = Patch_t::particle_ghost_width;
    level_         = damn::amr::allocate(hierarchy_.levels[0], rank_, [&](auto const& p) {
        Layout_t const layout{p.box, damn::amr::mesh_size(hierarchy_, 0),
                              damn::amr::position(hierarchy_, 0, p.box.lower), 0};
        auto const ghost = damn::core::grow(p.box, damn::core::ConstArray<int, dimension>(pgw));
        return std::make_unique<Patch_t>(layout, sim,
                                         damn::amr::covered_by_level(hierarchy_, 0, ghost));
    });

    // the level's neighbours, as far as any resource reaches outside its patch: field ghosts
    // plus primal nodes, or particle ghosts. One rank, one memory domain for now
    auto constexpr reach = std::max<int>(Patch_t::field_ghost_width + 1, pgw);
    neighbours_ = damn::amr::make_neighbours(hierarchy_, 0, rank_, damn::amr::single_node(), reach);

    // the model: fields and particles from the user's functions
    per_patch_([&](auto& patch) {
        core::ElectromagInitializer<dimension>{sim["electromag"]}(patch.em, patch.layout);
        for (auto& pop : patch.ions)
        {
            auto initializer = Patch_t::Types::ParticleInitializerFactory_t::create(
                pop.particleInitializerInfo(), pop.name());
            initializer->loadParticles(pop.domainParticles(), patch.layout);
        }
    });
    move_(damn::amr::fill_ghosts, "EM_B");
    move_(damn::amr::fill_ghosts, "EM_E");

    // moments from the particles, completed across patch borders
    per_patch_([&](auto& patch) {
        core::resetMoments(patch.ions);
        core::depositParticles(patch.ions, patch.layout, Interpolator{}, core::DomainDeposit{});
    });
    fill_ion_borders_();
    compute_moments_();

    // J from B, then E from Ohm's law
    ampere_ohm_<&Patch_t::em>();
    move_(damn::amr::fill_ghosts, "EM_E");

    currentTime_   = startTime_;
    isInitialized_ = true;
}


//-----------------------------------------------------------------------------
//                           advance: SolverPPC
//-----------------------------------------------------------------------------

template<auto opts>
double Simulator<opts>::advance()
{
    DAMN_LOG_SCOPE(1, "Simulator::advance");
    if (!isInitialized_)
        throw std::runtime_error("plamn Simulator: initialize() first");

    auto const dt = dt_;
    predictor1_(dt);
    average_();
    move_ions_(dt, core::UpdaterMode::domain_only);
    predictor2_(dt);
    average_();
    move_ions_(dt, core::UpdaterMode::all);
    corrector_(dt);

    currentTime_ += dt;
    return dt;
}

// the ions' totals from the populations' (complete) moments
template<auto opts>
void Simulator<opts>::compute_moments_()
{
    per_patch_([](auto& patch) { patch.ions.update(); });
}

template<auto opts>
void Simulator<opts>::fill_ion_borders_()
{
    move_(damn::amr::sum_borders, "IonPopBorderSum");
}

// J from B (Ampere, on the physical box + 1: what Ohm reads, so J is never exchanged), the
// electrons from the ions and J, then E from Ohm's law; electromag: the HybridPatch member
// holding the B used and the E computed, the model's (em) or the predicted one (emPred)
template<auto opts>
template<auto electromag>
void Simulator<opts>::ampere_ohm_()
{
    per_patch_([&](auto& patch) {
        auto& em = patch.*electromag;
        core::AmpereTransformer{}(patch.layout, em.B, patch.J);
        patch.electrons().update(patch.layout);
        core::OhmTransformer{*ohm_info_}(patch.layout, patch.electrons().density(),
                                         patch.electrons().velocity(), patch.electrons().pressure(),
                                         em.B, patch.J, em.E);
    });
}

template<auto opts>
void Simulator<opts>::average_()
{
    per_patch_([](auto& patch) {
        for (std::size_t i = 0; i < 3; ++i)
        {
            core::average(patch.em.B[i], patch.emPred.B[i], patch.emAvg.B[i]);
            core::average(patch.em.E[i], patch.emPred.E[i], patch.emAvg.E[i]);
        }
    });
    move_(damn::amr::fill_ghosts, "EMAvg_E");
}

template<auto opts>
void Simulator<opts>::move_ions_(double const dt, core::UpdaterMode const mode)
{
    DAMN_LOG_SCOPE(1, "Simulator::move_ions");
    per_patch_([&](auto& patch) {
        mover_.updatePopulations(patch.ions, patch.emAvg, patch.boxing, dt, mode);
    });
    fill_ion_borders_();

    if (mode != core::UpdaterMode::domain_only)
    {
        // particles that left their patch go to the patch owning their cell
        move_(damn::amr::gather_leaving, "IonGhostParticles");
        per_patch_([](auto& patch) {
            for (auto& pop : patch.ions)
                pop.patchGhostParticles().clear();
        });
    }
    compute_moments_();
}

template<auto opts>
void Simulator<opts>::predictor1_(double const dt)
{
    per_patch_([&](auto& patch) {
        core::FaradayTransformer{}(patch.layout, patch.em.B, patch.em.E, patch.emPred.B, dt);
    });
    move_(damn::amr::fill_ghosts, "EMPred_B");
    ampere_ohm_<&Patch_t::emPred>();
}

template<auto opts>
void Simulator<opts>::predictor2_(double const dt)
{
    per_patch_([&](auto& patch) {
        core::FaradayTransformer{}(patch.layout, patch.em.B, patch.emAvg.E, patch.emPred.B, dt);
    });
    move_(damn::amr::fill_ghosts, "EMPred_B");
    ampere_ohm_<&Patch_t::emPred>();
}

template<auto opts>
void Simulator<opts>::corrector_(double const dt)
{
    per_patch_([&](auto& patch) {
        core::FaradayTransformer{}(patch.layout, patch.em.B, patch.emAvg.E, patch.em.B, dt);
    });
    move_(damn::amr::fill_ghosts, "EM_B");
    ampere_ohm_<&Patch_t::em>();
    move_(damn::amr::fill_ghosts, "EM_E");
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
    return ss.str();
}

} // namespace plamn

#endif // PLAMN_SIMULATOR_SIMULATOR_HPP
