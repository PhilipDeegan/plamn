// plamn_etc: the C++ side pyphare needs before a simulator exists
//
//   Dict:  the simulation input, owned by a python object. pyphare's DictPopulator calls
//          its add_* methods (pyplamn makes dict_populator.dictator() return it)
//   enums: the integer values pyphare writes with add_enum_int
//
// user init functions are python callables kept in C++ (InitFunction<dim>): called with
// zero-copy numpy views of the C++ coordinates, their returned arrays copied into C++

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

namespace plamn::python3
{
using initializer::InitFunction;

using CArray = nb::ndarray<nb::numpy, double const, nb::ndim<1>>;
using InArray = nb::ndarray<double const, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// x as a numpy array over x's memory; x outlives the call
inline CArray view(damn::core::Span<double const> const& x)
{
    nb::capsule owner(x.data(), [](void*) noexcept {});
    return CArray{x.data(), {static_cast<std::size_t>(x.size())}, owner};
}

// a python callable as InitFunction<dim>
template<std::size_t dim>
InitFunction<dim> init_function(nb::callable fn)
{
    return [fn = std::move(fn)](auto const&... xyz) {
        static_assert(sizeof...(xyz) == dim);
        nb::gil_scoped_acquire gil;
        auto const values = nb::cast<InArray>(fn(view(xyz)...));
        return std::shared_ptr<damn::core::Span<double>>{
            std::make_shared<damn::core::VectorSpan<double>>(
                std::vector<double>(values.data(), values.data() + values.shape(0)))};
    };
}


// the input dict, held by python
struct Dict
{
    initializer::Dict dict;

    template<typename T>
    void add(std::string const& path, T value)
    {
        cppdict::add(path, std::move(value), dict);
    }

    // the node at path, throws if missing
    initializer::Dict const& at(std::string const& path) const
    {
        initializer::Dict const* node = &dict;
        for (auto const& key : cppdict::detail::split_string(path))
        {
            if (!node->contains(key))
                throw std::runtime_error("plamn Dict: no " + path);
            node = &(*node)[key];
        }
        return *node;
    }

    template<typename T>
    T get(std::string const& path) const
    { // cppdict's to<T>() is non-const only, it does not modify
        return const_cast<initializer::Dict&>(at(path)).template to<T>();
    }

    bool has(std::string const& path) const
    {
        try
        {
            (void)at(path);
            return true;
        }
        catch (std::runtime_error const&)
        {
            return false;
        }
    }

    // calls the init function at path, for tests
    template<std::size_t dim, typename... X>
    std::vector<double> eval(std::string const& path, X const&... x) const
    {
        auto const fn  = get<InitFunction<dim>>(path);
        auto const out = fn(damn::core::Span<double const>{x.data(), x.size()}...);
        return {out->data(), out->data() + out->size()};
    }

    // every leaf's full path, sorted
    std::vector<std::string> paths() const
    {
        std::vector<std::string> out;
        auto const walk = [&](auto const& self, initializer::Dict const& node,
                              std::string const& prefix) -> void {
            if (!node.isNode())
            {
                if (!prefix.empty()) // an empty dict is an empty root leaf
                    out.push_back(prefix);
                return;
            }
            for (auto const& [key, child] : std::get<initializer::Dict::node_t>(node.data))
                self(self, *child, prefix.empty() ? key : prefix + "/" + key);
        };
        walk(walk, dict, "");
        return out;
    }

    void clear() { dict = initializer::Dict{}; } // drops held python functions
};

} // namespace plamn::python3


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
