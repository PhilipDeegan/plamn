// plamn_cpp_<id>: one simulator build permutation, as pyphare's cpp_<id> modules, from the
//  PLAMN_SIM_* definitions ./mkn.py sets: doc/src/python3/plamn_sim.md

#include "plamn/simulator/simulator.hpp"

#include "plamn_dict.hpp"
#include "plamn_data_wrangler.hpp"

#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unique_ptr.h>
#include <memory>

#if !defined(PLAMN_SIM_DIM)
#define PLAMN_SIM_DIM 1
#endif
#if !defined(PLAMN_SIM_INTERP)
#define PLAMN_SIM_INTERP 1
#endif
#if !defined(PLAMN_SIM_REFINED)
#define PLAMN_SIM_REFINED plamn::defaultNbrRefinedParts(PLAMN_SIM_DIM, PLAMN_SIM_INTERP)
#endif
#if !defined(PLAMN_SIM_LAYOUT)
#define PLAMN_SIM_LAYOUT AoSMapped
#endif
#if !defined(PLAMN_SIM_ALLOCATOR)
#define PLAMN_SIM_ALLOCATOR CPU
#endif
#if !defined(PLAMN_SIM_RECONSTRUCTION)
#define PLAMN_SIM_RECONSTRUCTION MHDOff
#endif
#if !defined(PLAMN_SIM_LIMITER)
#define PLAMN_SIM_LIMITER MHDOff
#endif
#if !defined(PLAMN_SIM_RIEMANN)
#define PLAMN_SIM_RIEMANN MHDOff
#endif
#if !defined(PLAMN_SIM_HALL)
#define PLAMN_SIM_HALL false
#endif
#if !defined(PLAMN_SIM_MODULE)
#define PLAMN_SIM_MODULE plamn_cpp_1_1_2_AoSMapped_CPU
#endif

namespace
{
auto constexpr opts = plamn::SimOpts{
    .dimension           = PLAMN_SIM_DIM,
    .interp_order        = PLAMN_SIM_INTERP,
    .nbRefinedPart       = PLAMN_SIM_REFINED,
    .layout_mode         = plamn::core::LayoutMode::PLAMN_SIM_LAYOUT,
    .alloc_mode          = damn::AllocatorMode::PLAMN_SIM_ALLOCATOR,
    .reconstruction_type = plamn::MHDOpts::ReconstructionType::PLAMN_SIM_RECONSTRUCTION,
    .slope_limiter_type  = plamn::MHDOpts::SlopeLimiterType::PLAMN_SIM_LIMITER,
    .riemann_solver_type = plamn::MHDOpts::RiemannSolverType::PLAMN_SIM_RIEMANN,
    .Hall                = PLAMN_SIM_HALL};
static_assert(opts.mhd_axes_consistent(),
              "plamn: MHD reconstruction, limiter, riemann: all or none");
using Simulator = plamn::Simulator<opts>;

// particles, the hybrid levels' data: a template, so a build without hybrid never instantiates
//  them
template<typename Sim>
void bind_hybrid(nb::class_<Sim>& simulator, nb::module_& m)
{
    if constexpr (plamn::solver::has_hybrid_v<Sim::options>)
    {
        simulator.def("particle_count", &Sim::particle_count, nb::arg("level") = 0)
            .def("total_particle_count", &Sim::total_particle_count, nb::arg("level") = 0)
            .def("particle_counts_per_rank", &Sim::particle_counts_per_rank, nb::arg("level") = 0)
            .def("summary", &Sim::summary)
            .def(
                "patch_count",
                [](Sim& self, std::size_t const level) {
                    return damn::amr::size(self.level(level));
                },
                nb::arg("level") = 0);
        plamn::python3::declare_data_wrangler<Sim>(m);
    }
}
} // namespace


NB_MODULE(PLAMN_SIM_MODULE, m)
{
    auto simulator = nb::class_<Simulator>(m, "Simulator");
    simulator.def("initialize", &Simulator::initialize)
        .def("advance", &Simulator::advance)
        .def("startTime", &Simulator::startTime)
        .def("currentTime", &Simulator::currentTime)
        .def("endTime", &Simulator::endTime)
        .def("timeStep", &Simulator::timeStep)
        .def("to_str", &Simulator::to_str)
        .def("domain_box", &Simulator::domainBox)
        .def("cell_width", &Simulator::cellWidth)
        .def("flush_diagnostics", &Simulator::flush_diagnostics)
        .def("dump_diagnostics", &Simulator::dump_diagnostics, nb::arg("timestamp"),
             nb::arg("timestep"))
        .def("dump_restarts", &Simulator::dump_restarts, nb::arg("timestamp"), nb::arg("timestep"))
        // read back, for tests
        .def("domain_values", &Simulator::domain_values, nb::arg("name"), nb::arg("level") = 0)
        .def("n_levels", &Simulator::n_levels)
        .def("substep_dts", &Simulator::substep_dts, nb::arg("level"))
        // over ranks (collective)
        .def("all_domain_values", &Simulator::all_domain_values, nb::arg("name"),
             nb::arg("level") = 0)
        .def("level_values", &Simulator::level_values, nb::arg("name"), nb::arg("level") = 0)
        .def(
            "level_field", // (points: dimension ints per point, values, dual per direction)
            [](Simulator& self, std::string const& name, std::size_t const level) {
                auto f = self.level_field(name, level);
                return std::make_tuple(std::move(f.points), std::move(f.values),
                                       std::vector<bool>(f.dual.begin(), f.dual.end()));
            },
            nb::arg("name"), nb::arg("level") = 0)
        .def("rank", &Simulator::rank)
        .def(
            "all_patch_count",
            [](Simulator& self, std::size_t const level) {
                return self.hierarchy().levels.at(level).patches.size();
            },
            nb::arg("level") = 0)
        .def(
            "all_patch_boxes", // every patch of the level, (lower, upper) in its cells
            [](Simulator& self, std::size_t const level) {
                std::vector<std::pair<std::vector<int>, std::vector<int>>> boxes;
                for (auto const& p : self.hierarchy().levels.at(level).patches)
                    boxes.emplace_back(std::vector<int>(p.box.lower.begin(), p.box.lower.end()),
                                       std::vector<int>(p.box.upper.begin(), p.box.upper.end()));
                return boxes;
            },
            nb::arg("level") = 0)
        .def("ranks", &Simulator::ranks)
        .def_prop_ro_static("dims", [](nb::handle) { return Simulator::dimension; })
        .def_prop_ro_static("interp_order", [](nb::handle) { return Simulator::interp_order; })
        .def_prop_ro_static("refined_particle_nbr",
                            [](nb::handle) { return Simulator::nbRefinedPart; });

    bind_hybrid(simulator, m);

    m.def("make_simulator",
          [](plamn::python3::Dict const& dict) { return std::make_unique<Simulator>(dict.dict); });
}
