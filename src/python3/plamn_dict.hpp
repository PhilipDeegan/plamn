#ifndef PLAMN_PYTHON3_DICT_HPP
#define PLAMN_PYTHON3_DICT_HPP

// the input dict, owned by a python object (plamn_etc.Dict), and user init functions:
// python callables kept as InitFunction<dim>, called with zero-copy numpy views of the
// C++ coordinates, their returned arrays copied into C++

#include "plamn/initializer/data_provider.hpp"
#include "damn/core/memory/span.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <memory>
#include <string>
#include <vector>
#include <cstddef>
#include <stdexcept>

namespace nb = nanobind;

namespace plamn::python3
{
using initializer::InitFunction;

using CArray  = nb::ndarray<nb::numpy, double const, nb::ndim<1>>;
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

#endif // PLAMN_PYTHON3_DICT_HPP
