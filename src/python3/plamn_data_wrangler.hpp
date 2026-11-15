#ifndef PLAMN_PYTHON3_PLAMN_DATA_WRANGLER_HPP
#define PLAMN_PYTHON3_PLAMN_DATA_WRANGLER_HPP

// Live, zero-copy access to a running simulator's data from python, the API of PHARE's
//  DataWrangler (PHAREHUB/PHARE#1400, #1171), so pyphare's hierarchy_from_sim works unchanged:
//
//   DataWrangler(simulator, hierarchy)   hierarchy: unused, plamn's is the simulator's own
//     getNumberOfLevels()
//     getHybridPatchLevel(lvl) -> HybridPatchLevel
//     sync(patch_datas)                  every rank's patch datas on rank 0 (copies), none on
//                                        others; one rank: the input
//   HybridPatchLevel: getB(c), getE(c), getNi(), getN(pop), getVi(c), getFlux(c, pop),
//     getMassDensity(), getChargeFlux(c), getDensity(pop) (particle density; getN: charge
//     density), getParticles(pop): one PatchData per local patch; every quantity written as a
//     diagnostic has one
//   PatchData: patchID, origin, lower, upper (AMR cells), nGhosts (field or particle ghost
//     width), data
//
// data is a writable numpy view on the patch's grid, particles the patch's domain particle
//  array, by reference: only valid while the simulator and its patches exist

#include "damn/mpi/damn_mpi.hpp"

#include "plamn/core/data/particles/particle_components.hpp"
#include "plamn/core/data/vecfield/vecfield_component.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
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
        , lvl_{lvl}
    {
        if (lvl >= sim.n_levels())
            throw std::runtime_error("plamn DataWrangler: no level " + std::to_string(lvl));
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
    auto getMassDensity()
    {
        return fields_([](auto& p) -> auto& { return p.ions.massDensity(); });
    }
    auto getChargeFlux(std::string const& c)
    {
        return vector_(c, [](auto& p) -> auto& { return p.ions.chargeFlux(); });
    }
    auto getDensity(std::string const& pop)
    {
        return fields_([&](auto& p) -> auto& { return population_(p, pop).particleDensity(); });
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
        auto& level = sim_.level(lvl_);
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
    std::size_t lvl_;
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

    std::size_t getNumberOfLevels() const { return sim_.n_levels(); }
    auto getHybridPatchLevel(std::size_t const lvl) { return Level_t{sim_, lvl}; }

    // every rank's patch datas gathered on rank 0, as owned copies (e.g. to plot the whole
    //  domain), in rank order; other ranks get none. One rank: the input
    std::vector<typename Level_t::FieldData>
    sync(std::vector<typename Level_t::FieldData> const& in) const;

private:
    Simulator& sim_;
};


template<typename Simulator>
std::vector<typename DataWrangler<Simulator>::Level_t::FieldData>
DataWrangler<Simulator>::sync(std::vector<typename Level_t::FieldData> const& in) const
{
    using FieldData = typename Level_t::FieldData;
    if (sim_.ranks() == 1)
        return in;

    auto const put = [](std::vector<std::byte>& out, void const* data, std::size_t const bytes) {
        auto const at = out.size();
        out.resize(at + bytes);
        std::memcpy(out.data() + at, data, bytes);
    };
    auto const get = [](std::byte const*& at, void* data, std::size_t const bytes) {
        std::memcpy(data, at, bytes);
        at += bytes;
    };

    // per patch: id length, id, origin, lower, upper, ghosts, shape, values
    std::vector<std::byte> mine;
    for (auto const& pd : in)
    {
        std::uint64_t const id_size = pd.patchID.size();
        put(mine, &id_size, sizeof(id_size));
        put(mine, pd.patchID.data(), id_size);
        put(mine, pd.origin.data(), sizeof(pd.origin));
        put(mine, pd.lower.data(), sizeof(pd.lower));
        put(mine, pd.upper.data(), sizeof(pd.upper));
        std::uint64_t const ghosts = pd.nGhosts;
        put(mine, &ghosts, sizeof(ghosts));
        std::array<std::uint64_t, dim> shape{};
        std::size_t n = 1;
        for (std::size_t i = 0; i < dim; ++i)
            n *= (shape[i] = pd.data.shape(i));
        put(mine, shape.data(), sizeof(shape));
        put(mine, pd.data.data(), n * sizeof(double));
    }

    int const bytes = static_cast<int>(mine.size());
    std::vector<int> counts(sim_.ranks()), displs(sim_.ranks());
    MPI_Gather(&bytes, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<std::byte> all;
    if (sim_.rank() == 0)
    {
        for (std::size_t r = 1; r < counts.size(); ++r)
            displs[r] = displs[r - 1] + counts[r - 1];
        all.resize(static_cast<std::size_t>(displs.back() + counts.back()));
    }
    MPI_Gatherv(mine.data(), bytes, MPI_BYTE, all.data(), counts.data(), displs.data(), MPI_BYTE, 0,
                MPI_COMM_WORLD);
    if (sim_.rank() != 0)
        return {};

    std::vector<FieldData> out;
    std::byte const* at        = all.data();
    std::byte const* const end = all.data() + all.size();
    while (at < end)
    {
        auto& pd              = out.emplace_back();
        std::uint64_t id_size = 0;
        get(at, &id_size, sizeof(id_size));
        pd.patchID.assign(reinterpret_cast<char const*>(at), id_size);
        at += id_size;
        get(at, pd.origin.data(), sizeof(pd.origin));
        get(at, pd.lower.data(), sizeof(pd.lower));
        get(at, pd.upper.data(), sizeof(pd.upper));
        std::uint64_t ghosts = 0;
        get(at, &ghosts, sizeof(ghosts));
        pd.nGhosts = ghosts;
        std::array<std::uint64_t, dim> shape{};
        get(at, shape.data(), sizeof(shape));
        std::array<std::size_t, dim> sizes{};
        std::size_t n = 1;
        for (std::size_t i = 0; i < dim; ++i)
            n *= (sizes[i] = shape[i]);
        auto* values = new double[n];
        get(at, values, n * sizeof(double));
        nb::capsule owner(values, [](void* p) noexcept { delete[] static_cast<double*>(p); });
        pd.data = FieldView<dim>(values, dim, sizes.data(), owner);
    }
    return out;
}


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
    using Particle_t   = plamn::core::Particle<dim>;

    nb::class_<Particle_t>(m, "Particle")
        .def_prop_ro("weight", [](Particle_t const& p) { return p.weight(); })
        .def_prop_ro("iCell", [](Particle_t const& p) { return p.iCell(); })
        .def_prop_ro("delta", [](Particle_t const& p) { return p.delta(); })
        .def_prop_ro("v", [](Particle_t const& p) { return p.v(); });

    auto particles = nb::class_<Particles_t>(m, "ParticleArray")
                         .def("size", [](Particles_t& self) { return self.size(); })
                         .def("__len__", [](Particles_t& self) { return self.size(); });

    if constexpr (plamn::core::HasComponents<Particles_t>)
    {
        // a copy of particle i, whatever the layout (SoA / SoAVX hold no Particle to refer to)
        particles.def("__getitem__", [](Particles_t& self, std::size_t const i) {
            if (i >= self.size())
                throw nb::index_error();
            auto const c = plamn::core::components(std::as_const(self));
            Particle_t p;
            p.weight() = c.weight[i];
            for (std::size_t d = 0; d < dim; ++d)
            {
                p.iCell()[d] = c.iCell[d][i];
                p.delta()[d] = c.delta[d][i];
            }
            for (std::size_t d = 0; d < 3; ++d)
                p.v()[d] = c.v[d][i];
            return p;
        });

        // {name: numpy view} of every component (weight, iCell_x.., delta_x.., v_x..), live and
        //  writable, in the array's own memory: valid as long as the array is
        particles.def("components", [](nb::handle const h) {
            auto& self = nb::cast<Particles_t&>(h);
            nb::dict out;
            plamn::core::components(self).for_each([&](auto const& name, auto const& span) {
                using T = std::remove_cv_t<typename std::decay_t<decltype(span)>::value_type>;
                std::size_t const shape[1]    = {span.size()};
                std::int64_t const strides[1] = {static_cast<std::int64_t>(span.stride_values())};
                out[name.c_str()]
                    = nb::ndarray<nb::numpy, T, nb::ndim<1>>(span.data(), 1, shape, h, strides);
            });
            return out;
        });
    }
    else
        particles.def(
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
        .def("getMassDensity", &Level::getMassDensity)
        .def("getChargeFlux", &Level::getChargeFlux)
        .def("getDensity", &Level::getDensity)
        .def("getParticles", &Level::getParticles);

    nb::class_<DW>(m, "DataWrangler")
        .def(nb::init<Simulator&, nb::handle>(), nb::keep_alive<1, 2>())
        .def("getNumberOfLevels", &DW::getNumberOfLevels)
        .def("getHybridPatchLevel", &DW::getHybridPatchLevel)
        .def("sync", &DW::sync);
}

} // namespace plamn::python3

#endif /* PLAMN_PYTHON3_PLAMN_DATA_WRANGLER_HPP */
