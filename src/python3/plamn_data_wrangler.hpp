#ifndef PLAMN_PYTHON3_PLAMN_DATA_WRANGLER_HPP
#define PLAMN_PYTHON3_PLAMN_DATA_WRANGLER_HPP

// Live, zero-copy access to a running simulator's data from python, the API of PHARE's
//  DataWrangler (PHAREHUB/PHARE#1400, #1171), so pyphare's hierarchy_from_sim works unchanged:
//
//   DataWrangler(simulator, hierarchy)   hierarchy: unused, plamn's is the simulator's own
//     getNumberOfLevels()
//     getHybridPatchLevel(lvl) -> HybridPatchLevel
//     sync(patch_datas)                  one rank: the input
//   HybridPatchLevel: getB(c), getE(c), getNi(), getN(pop), getVi(c), getFlux(c, pop),
//     getParticles(pop): one PatchData per local patch
//   PatchData: patchID, origin, lower, upper (AMR cells), nGhosts (field or particle ghost
//     width), data
//
// data is a writable numpy view on the patch's grid, particles the patch's domain particle
//  array, by reference: only valid while the simulator and its patches exist

#include "plamn/core/data/vecfield/vecfield_component.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace plamn::python3
{
namespace nb = nanobind;

template<std::size_t dim>
using FieldView = nb::ndarray<nb::numpy, double, nb::c_contig>;

template<typename Data, std::size_t dim>
struct PatchData
{
    std::string patchID;
    std::array<double, dim> origin{};
    std::array<int, dim> lower{}, upper{};
    std::size_t nGhosts = 0;
    Data data{};
};

// a numpy view on a field's data, no owner: lives as long as the field's grid
template<std::size_t dim>
FieldView<dim> field_view(auto& field)
{
    std::array<std::size_t, dim> shape{};
    for (std::size_t i = 0; i < dim; ++i)
        shape[i] = field.shape()[i];
    return FieldView<dim>(field.data(), dim, shape.data(), nb::handle());
}

template<typename Simulator>
class HybridPatchLevel
{
    auto static constexpr dim = Simulator::dimension;
    using Patch_t             = Simulator::Patch_t;
    using Particles_t         = Patch_t::Particles_t;

public:
    using FieldData     = PatchData<FieldView<dim>, dim>;
    using ParticlesData = PatchData<Particles_t*, dim>;

    HybridPatchLevel(Simulator& sim, std::size_t const lvl)
        : sim_{sim}
    {
        if (lvl != 0)
            throw std::runtime_error("plamn DataWrangler: one level only, no level "
                                     + std::to_string(lvl));
    }

    auto getB(std::string const& c)
    {
        return vector_(c, [](auto& p) -> auto& { return p.em.B; });
    }
    auto getE(std::string const& c)
    {
        return vector_(c, [](auto& p) -> auto& { return p.em.E; });
    }
    auto getVi(std::string const& c)
    {
        return vector_(c, [](auto& p) -> auto& { return p.ions.velocity(); });
    }
    auto getNi()
    {
        return fields_([](auto& p) -> auto& { return p.ions.chargeDensity(); });
    }
    auto getN(std::string const& pop)
    {
        return fields_([&](auto& p) -> auto& { return population_(p, pop).chargeDensity(); });
    }
    auto getFlux(std::string const& c, std::string const& pop)
    {
        return vector_(c, [&](auto& p) -> auto& { return population_(p, pop).flux(); });
    }

    auto getParticles(std::string const& pop)
    {
        std::vector<ParticlesData> out;
        for_each_patch_([&](auto const& patch, auto& data) {
            set_(out.emplace_back(), patch, data);
            out.back().data    = &population_(data, pop).domainParticles();
            out.back().nGhosts = Patch_t::particle_ghost_width;
        });
        return out;
    }

private:
    static auto& population_(Patch_t& patch, std::string const& name)
    {
        for (auto& pop : patch.ions)
            if (pop.name() == name)
                return pop;
        throw std::runtime_error("plamn DataWrangler: no population named " + name);
    }

    void for_each_patch_(auto&& fn)
    {
        auto& level = sim_.level();
        for (std::size_t i = 0; i < level.patches.size(); ++i)
            fn(level.patches[i], *level.data[i]);
    }

    static void set_(auto& pd, auto const& patch, Patch_t const& data)
    {
        pd.patchID = std::to_string(patch.rank) + "#" + std::to_string(patch.id);
        for (std::size_t i = 0; i < dim; ++i)
        {
            pd.origin[i] = data.layout.origin[i];
            pd.lower[i]  = data.layout.cells.lower[i];
            pd.upper[i]  = data.layout.cells.upper[i];
        }
        pd.nGhosts = Patch_t::field_ghost_width;
    }

    auto fields_(auto&& get)
    {
        std::vector<FieldData> out;
        for_each_patch_([&](auto const& patch, auto& data) {
            set_(out.emplace_back(), patch, data);
            out.back().data = field_view<dim>(get(data));
        });
        return out;
    }

    auto vector_(std::string const& c, auto&& get)
    {
        auto const component = core::Components::componentMap().at(c);
        return fields_([&](auto& data) -> auto& { return get(data)(component); });
    }

    Simulator& sim_;
};


template<typename Simulator>
class DataWrangler
{
    auto static constexpr dim = Simulator::dimension;

public:
    using Level_t = HybridPatchLevel<Simulator>;

    DataWrangler(Simulator& sim, nb::handle /*hierarchy*/)
        : sim_{sim}
    {
    }

    std::size_t getNumberOfLevels() const { return 1; }
    auto getHybridPatchLevel(std::size_t const lvl) { return Level_t{sim_, lvl}; }

    // one rank: everything is already on rank 0
    auto sync(std::vector<typename Level_t::FieldData> const& in) { return in; }

private:
    Simulator& sim_;
};


template<typename Data, std::size_t dim>
void declare_patch_data(nb::module_& m, char const* name)
{
    using PD   = PatchData<Data, dim>;
    auto array = [](auto& a, nb::handle self) {
        return nb::ndarray<nb::numpy, std::decay_t<decltype(a[0])>>(a.data(), {dim}, self);
    };
    nb::class_<PD>(m, name)
        .def_ro("patchID", &PD::patchID)
        .def_prop_ro("origin",
                     [=](nb::handle_t<PD> self) { return array(nb::cast<PD&>(self).origin, self); })
        .def_prop_ro("lower",
                     [=](nb::handle_t<PD> self) { return array(nb::cast<PD&>(self).lower, self); })
        .def_prop_ro("upper",
                     [=](nb::handle_t<PD> self) { return array(nb::cast<PD&>(self).upper, self); })
        .def_ro("nGhosts", &PD::nGhosts)
        .def_prop_ro("data", [](PD& self) { return self.data; }, nb::rv_policy::reference);
}


template<typename Simulator>
void declare_data_wrangler(nb::module_& m)
{
    auto constexpr dim = Simulator::dimension;
    using DW           = DataWrangler<Simulator>;
    using Level        = DW::Level_t;
    using Particles_t  = Simulator::Patch_t::Particles_t;
    using Particle_t   = std::decay_t<decltype(std::declval<Particles_t&>()[0])>;

    nb::class_<Particle_t>(m, "Particle")
        .def_prop_ro("weight", [](Particle_t const& p) { return p.weight(); })
        .def_prop_ro("iCell", [](Particle_t const& p) { return p.iCell(); })
        .def_prop_ro("delta", [](Particle_t const& p) { return p.delta(); })
        .def_prop_ro("v", [](Particle_t const& p) { return p.v(); });

    nb::class_<Particles_t>(m, "ParticleArray")
        .def("size", [](Particles_t& self) { return self.size(); })
        .def("__len__", [](Particles_t& self) { return self.size(); })
        .def(
            "__getitem__",
            [](Particles_t& self, std::size_t const i) -> auto& {
                if (i >= self.size())
                    throw nb::index_error();
                return self[i];
            },
            nb::rv_policy::reference_internal);

    declare_patch_data<FieldView<dim>, dim>(m, "PatchDataField");
    declare_patch_data<Particles_t*, dim>(m, "PatchDataParticles");

    nb::class_<Level>(m, "HybridPatchLevel")
        .def("getB", &Level::getB)
        .def("getE", &Level::getE)
        .def("getVi", &Level::getVi)
        .def("getNi", &Level::getNi)
        .def("getN", &Level::getN)
        .def("getFlux", &Level::getFlux)
        .def("getParticles", &Level::getParticles);

    nb::class_<DW>(m, "DataWrangler")
        .def(nb::init<Simulator&, nb::handle>(), nb::keep_alive<1, 2>())
        .def("getNumberOfLevels", &DW::getNumberOfLevels)
        .def("getHybridPatchLevel", &DW::getHybridPatchLevel)
        .def("sync", &DW::sync);
}

} // namespace plamn::python3

#endif /* PLAMN_PYTHON3_PLAMN_DATA_WRANGLER_HPP */
