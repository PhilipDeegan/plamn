#ifndef PLAMN_SOLVER_MHD_MODEL_HPP
#define PLAMN_SOLVER_MHD_MODEL_HPP

// the MHD model of a simulation: its patch data per level number, for the levels it is active
//  on, and what its levels' advance needs that is not the levels' data
//  (pyphare's MHDModel input: simulation/algo/fv_method, simulation/algo/time_integrator_type,
//  simulation/grid/domain_boundaries): the heat capacity ratio, the time integration scheme
//  (Euler, TVDRK2, TVDRK3, SSPRK4_5: a run time choice), resistivity and hyper resistivity, Hall,
//  and the physical boundaries' conditions. The reconstruction (and its slope limiter) is the
//  simulation's compile time options'. Empty if the build does not enable it

#include "damn/amr/level_data.hpp"
#include "damn/core/def.hpp"

#include "plamn/core/boundary/boundary.hpp"
#include "plamn/core/models/options/mhd_options_def.hpp"
#include "plamn/core/models/quantities/mhd_quantities.hpp"
#include "plamn/diagnostic/quantities.hpp"
#include "plamn/core/numerics/reconstructions/linear.hpp"
#include "plamn/initializer/data_provider.hpp"
#include "plamn/simulator/mhd_patch.hpp"
#include "plamn/solver/model_def.hpp"

#include "dict.hpp"

#include <cstddef>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace plamn::solver
{
namespace mhd
{
    // the reconstruction of the simulation's options
    template<auto opts>
    auto constexpr reconstruction()
    {
        using enum MHDOpts::SlopeLimiterType;
        static_assert(opts.reconstruction_type == MHDOpts::ReconstructionType::Linear,
                      "plamn: only the linear MHD reconstruction is ported");
        static_assert(opts.riemann_solver_type == MHDOpts::RiemannSolverType::Rusanov,
                      "plamn: only the Rusanov Riemann solver is ported");
        if constexpr (opts.slope_limiter_type == MinMod)
            return core::LinearReconstruction<core::MinModLimiter>{};
        else
            return core::LinearReconstruction<core::VanLeerLimiter>{};
    }

    template<auto opts>
    using Reconstruction_t = decltype(reconstruction<opts>());

    // the states (the advanced one and the stages) a scheme uses
    NO_DISCARD inline std::size_t n_states(MHDOpts::TimeIntegratorType const scheme)
    {
        switch (scheme)
        {
            case MHDOpts::TimeIntegratorType::Euler: return 1;
            case MHDOpts::TimeIntegratorType::TVDRK2: return 2;
            case MHDOpts::TimeIntegratorType::TVDRK3: return 3;
            case MHDOpts::TimeIntegratorType::SSPRK4_5: return 5;
            default: throw std::runtime_error("plamn: no MHD time integration scheme");
        }
    }
} // namespace mhd

template<auto opts, bool enabled = has_mhd_v<opts>>
struct MHDModel
{
};

template<auto opts>
struct MHDModel<opts, true>
{
    using Reconstruction = mhd::Reconstruction_t<opts>;
    using Quantity       = core::MHDQuantity;
    using Patch_t        = simulator::MHDPatch<opts>;
    using LevelData_t    = damn::amr::LevelData<opts.dimension, std::unique_ptr<Patch_t>>;

    std::map<int, LevelData_t> levels; // the levels the model is active on
    std::map<std::string, diagnostic::Accessor<Patch_t>> quantities{}; // diagnostics' fields

    double gamma                       = 5. / 3.;
    MHDOpts::TimeIntegratorType scheme = MHDOpts::TimeIntegratorType::TVDRK2;
    double resistivity                 = 0.;
    double hyper_resistivity           = 0.;
    bool hall                          = false;
    core::BoundaryManager<Quantity> boundaries{};

    NO_DISCARD std::size_t n_states() const { return mhd::n_states(scheme); }

    // the levels from n on gone
    void truncate(std::size_t const n)
    {
        levels.erase(levels.lower_bound(static_cast<int>(n)), levels.end());
    }

    // from the input's simulation dict
    NO_DISCARD static MHDModel from(initializer::Dict const& sim)
    {
        auto const& fv = sim["algo"]["fv_method"];
        MHDModel m;
        m.gamma             = fv["heat_capacity_ratio"].template to<double>();
        m.resistivity       = cppdict::get_value(fv, "resistivity", 0.);
        m.hyper_resistivity = cppdict::get_value(fv, "hyper_resistivity", 0.);
        m.hall              = cppdict::get_value(fv, "hall", false);
        m.scheme            = cppdict::get_value(sim["algo"], "time_integrator_type",
                                                 MHDOpts::TimeIntegratorType::MHDOff);
        if (m.scheme == MHDOpts::TimeIntegratorType::MHDOff)
            throw std::runtime_error("plamn: the MHD model needs a time integration scheme");
        if (m.hall)
            throw std::runtime_error("plamn: Hall MHD is not ported yet");
        auto const& grid = sim["grid"];
        core::validate_physical_boundaries<opts.dimension>(grid);
        if (grid.contains("domain_boundaries"))
            m.boundaries = core::BoundaryManager<Quantity>{
                grid["domain_boundaries"],
                {Quantity::Scalar::rho, Quantity::Scalar::Etot},
                {Quantity::Vector::rhoV, Quantity::Vector::B, Quantity::Vector::E},
                m.gamma};
        return m;
    }
};

} // namespace plamn::solver

#endif // PLAMN_SOLVER_MHD_MODEL_HPP
