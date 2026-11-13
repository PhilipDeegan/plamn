// plamn_cpp_<id>: one simulator build permutation, as pyphare's cpp_<id> modules
//
//   make_simulator(dict) -> Simulator, dict: the plamn_etc.Dict pyphare filled, handed over
//   by pyphare as the "hierarchy" (pyplamn.etc.make_hierarchy returns it)
//
// the permutation comes from PLAMN_SIM_DIM, PLAMN_SIM_INTERP (defaults 1, 1)

#include "plamn_dict.hpp"
#include "plamn/simulator/simulator.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unique_ptr.h>

#include <memory>

#if !defined(PLAMN_SIM_DIM)
#define PLAMN_SIM_DIM 1
#endif
#if !defined(PLAMN_SIM_INTERP)
#define PLAMN_SIM_INTERP 1
#endif
#if !defined(PLAMN_SIM_MODULE)
#define PLAMN_SIM_MODULE plamn_cpp_1_1_2_AoSMapped_CPU
#endif

namespace
{
auto constexpr opts = plamn::SimOpts{PLAMN_SIM_DIM, PLAMN_SIM_INTERP};
using Simulator     = plamn::Simulator<opts>;
} // namespace


NB_MODULE(PLAMN_SIM_MODULE, m)
{
    nb::class_<Simulator>(m, "Simulator")
        .def("initialize", &Simulator::initialize)
        .def("advance", &Simulator::advance)
        .def("startTime", &Simulator::startTime)
        .def("currentTime", &Simulator::currentTime)
        .def("endTime", &Simulator::endTime)
        .def("timeStep", &Simulator::timeStep)
        .def("to_str", &Simulator::to_str)
        .def("summary", &Simulator::to_str)
        .def("domain_box", &Simulator::domainBox)
        .def("cell_width", &Simulator::cellWidth)
        .def("dump_diagnostics", &Simulator::dump_diagnostics, nb::arg("timestamp"),
             nb::arg("timestep"))
        .def("dump_restarts", &Simulator::dump_restarts, nb::arg("timestamp"), nb::arg("timestep"))
        // read back, for tests
        .def("domain_values", &Simulator::domain_values)
        .def("particle_count", &Simulator::particle_count)
        .def("patch_count", [](Simulator& self) { return damn::amr::size(self.level()); })
        .def_prop_ro_static("dims", [](nb::handle) { return Simulator::dimension; })
        .def_prop_ro_static("interp_order", [](nb::handle) { return Simulator::interp_order; })
        .def_prop_ro_static("refined_particle_nbr",
                            [](nb::handle) { return Simulator::nbRefinedPart; });

    m.def("make_simulator",
          [](plamn::python3::Dict const& dict) { return std::make_unique<Simulator>(dict.dict); });
}
