// plamn_etc: the C++ side pyphare needs before a simulator exists
//
//   Dict:  the simulation input, owned by a python object. pyphare's DictPopulator calls
//          its add_* methods (pyplamn makes dict_populator.dictator() return it)
//   enums: the integer values pyphare writes with add_enum_int
//
// user init functions are python callables kept in C++ (InitFunction<dim>): called with
// zero-copy numpy views of the C++ coordinates, their returned arrays copied into C++

#include "plamn_dict.hpp"
#include "plamn/initializer/data_provider.hpp"
#include "plamn/simulator/simulator_def.hpp"
#include "plamn/core/numerics/ohm/ohm.hpp"
#include "plamn/core/boundary/boundary_defs.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "damn/core/def/damn_config.hpp"
#include "damn/core/memory/span.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <string>
#include <vector>
#include <cstddef>
#include <optional>
#include <stdexcept>

namespace nb = nanobind;



NB_MODULE(plamn_etc, m)
{
    using namespace plamn::python3;
    using V = std::vector<double>;

    nb::class_<Dict>(m, "Dict")
        .def(nb::init<>())
        .def("add_size_t", &Dict::add<std::size_t>)
        .def("add_optional_size_t", &Dict::add<std::optional<std::size_t>>)
        .def("add_bool", &Dict::add<bool>)
        .def("add_int", &Dict::add<int>)
        .def("add_vector_int", &Dict::add<std::vector<int>>)
        .def("add_double", &Dict::add<double>)
        .def("add_string", &Dict::add<std::string>)
        .def("add_vector_string", &Dict::add<std::vector<std::string>>)
        .def("add_array_as_vector",
             [](Dict& self, std::string const& path, InArray const& array) {
                 self.add(path, V(array.data(), array.data() + array.shape(0)));
             })
        .def("addInitFunction1D",
             [](Dict& self, std::string const& path, nb::callable fn) {
                 self.add(path, init_function<1>(std::move(fn)));
             })
        .def("addInitFunction2D",
             [](Dict& self, std::string const& path, nb::callable fn) {
                 self.add(path, init_function<2>(std::move(fn)));
             })
        .def("addInitFunction3D",
             [](Dict& self, std::string const& path, nb::callable fn) {
                 self.add(path, init_function<3>(std::move(fn)));
             })
        .def("stop", &Dict::clear)
        .def("clear", &Dict::clear)
        // read back, for tests
        .def("has", &Dict::has)
        .def("paths", &Dict::paths)
        .def("get_int", &Dict::get<int>)
        .def("get_size_t", &Dict::get<std::size_t>)
        .def("get_bool", &Dict::get<bool>)
        .def("get_double", &Dict::get<double>)
        .def("get_string", &Dict::get<std::string>)
        .def("get_vector_int", &Dict::get<std::vector<int>>)
        .def("get_vector_double", &Dict::get<std::vector<double>>)
        .def("get_vector_string", &Dict::get<std::vector<std::string>>)
        .def("eval_init_function_1d", &Dict::eval<1, V>)
        .def("eval_init_function_2d", &Dict::eval<2, V, V>)
        .def("eval_init_function_3d", &Dict::eval<3, V, V, V>);

    nb::enum_<plamn::core::LayoutMode>(m, "LayoutMode")
        .value("AoS", plamn::core::LayoutMode::AoS)
        .value("AoSMapped", plamn::core::LayoutMode::AoSMapped)
        .value("AoSCMTS", plamn::core::LayoutMode::AoSCMTS)
        .value("SoA", plamn::core::LayoutMode::SoA)
        .value("AoSTS", plamn::core::LayoutMode::AoSTS)
        .value("AoSPC", plamn::core::LayoutMode::AoSPC)
        .value("SoAVX", plamn::core::LayoutMode::SoAVX)
        .value("SoATS", plamn::core::LayoutMode::SoATS)
        .value("SoAVXTS", plamn::core::LayoutMode::SoAVXTS)
        .value("SoAPC", plamn::core::LayoutMode::SoAPC)
        .value("AoSPCTS", plamn::core::LayoutMode::AoSPCTS)
        .value("SoAVXChunked", plamn::core::LayoutMode::SoAVXChunked);

    nb::enum_<damn::AllocatorMode>(m, "AllocatorMode")
        .value("CPU", damn::AllocatorMode::CPU)
        .value("GPU_UNIFIED", damn::AllocatorMode::GPU_UNIFIED);

    nb::enum_<plamn::core::HyperMode>(m, "HyperMode")
        .value("constant", plamn::core::HyperMode::constant)
        .value("spatial", plamn::core::HyperMode::spatial);

    nb::enum_<plamn::core::BoundaryType>(m, "BoundaryType")
        .value("none", plamn::core::BoundaryType::None)
        .value("reflective", plamn::core::BoundaryType::Reflective)
        .value("super_magnetofast_inflow", plamn::core::BoundaryType::SuperMagnetofastInflow)
        .value("open", plamn::core::BoundaryType::Open);

    nb::enum_<plamn::TimeStepType>(m, "TimeStepType")
        .value("constant", plamn::TimeStepType::constant)
        .value("adaptive", plamn::TimeStepType::adaptive);
}
