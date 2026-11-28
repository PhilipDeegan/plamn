#ifndef PLAMN_PYTHON3_PLAMN_DATA_WRANGLER_HPP
#define PLAMN_PYTHON3_PLAMN_DATA_WRANGLER_HPP

// live, zero-copy access to a running simulator's data from python, the API of pyphare's live
//  DataWrangler (pyphare's hierarchy_from_sim unchanged): doc/src/python3/plamn_data_wrangler.md
// data and particles are views, by reference: only valid while the simulator and its patches exist

#include "plamn/core/data/particles/particle_iteration.hpp"
#include "damn/mpi/damn_mpi.hpp"

#include "plamn/core/data/particles/particle_components.hpp"
#include "plamn/core/data/vecfield/vecfield_component.hpp"
#include "plamn/solver/hybrid/model.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <array>
#include <map>
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

// a tiled particle array's components(): {name: numpy array} of every component, as the
//  columns' (weight, iCell_x.., delta_x.., v_x..): copies, tile by tile (no contiguous columns)
template<typename Particles_t, std::size_t dim = Particles_t::dimension>
void def_tiled_components(auto& particles)
    requires core::TiledParticles<Particles_t>
{
    particles.def("components", [](Particles_t const& self) {
        std::array<char const*, 3> constexpr xyz{"x", "y", "z"};
        std::vector<std::vector<double>> real(1 + dim + 3);
        std::vector<std::vector<int>> cells(dim);
        core::for_each_particle(self, [&](auto const& p) {
            real[0].push_back(p.weight());
            for (std::size_t d = 0; d < dim; ++d)
            {
                cells[d].push_back(p.iCell()[d]);
                real[1 + d].push_back(p.delta()[d]);
            }
            for (std::size_t d = 0; d < 3; ++d)
                real[1 + dim + d].push_back(p.v()[d]);
        });
        auto const array = [](auto& v) {
            using T          = typename std::decay_t<decltype(v)>::value_type;
            auto* owned      = new std::vector<T>(std::move(v));
            nb::capsule free = nb::capsule(
                owned, [](void* q) noexcept { delete static_cast<std::vector<T>*>(q); });
            std::size_t const shape[1] = {owned->size()};
            return nb::ndarray<nb::numpy, T, nb::ndim<1>>(owned->data(), 1, shape, free);
        };
        nb::dict out;
        out["weight"] = array(real[0]);
        for (std::size_t d = 0; d < dim; ++d)
        {
            out[(std::string{"iCell_"} + xyz[d]).c_str()] = array(cells[d]);
            out[(std::string{"delta_"} + xyz[d]).c_str()] = array(real[1 + d]);
        }
        for (std::size_t d = 0; d < 3; ++d)
            out[(std::string{"v_"} + xyz[d]).c_str()] = array(real[1 + dim + d]);
        return out;
    });
}
template<typename Particles_t>
void def_tiled_components(auto&)
    requires(!core::TiledParticles<Particles_t>)
{
}


template<std::size_t dim>
using FieldView = nb::ndarray<nb::numpy, double, nb::c_contig>;

// a tiled particle array's tile's own array
template<typename Particles_t>
using tile_particles_t = std::decay_t<decltype(std::declval<Particles_t&>()()[0]())>;

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

// hidden as the module is (-fvisibility=hidden): its member templates take the bindings' lambdas
template<typename Simulator>
class [[gnu::visibility("hidden")]] HybridPatchLevel
{
    auto static constexpr dim = Simulator::dimension;
    using Patch_t             = plamn::solver::HybridModel<Simulator::options>::Patch_t;
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
    auto getMomentumTensor(std::string const& c, std::string const& pop)
    {
        static std::map<std::string, std::size_t> const ijk{{"xx", 0}, {"xy", 1}, {"xz", 2},
                                                            {"yy", 3}, {"yz", 4}, {"zz", 5}};
        auto const key       = pop.empty() ? std::string{"/ions/momentum_tensor"}
                                           : "/ions/pop/" + pop + "/momentum_tensor";
        auto const component = ijk.at(c);
        sim_.compute(lvl_, key);
        return fields_([&](auto& data) -> auto& { return *sim_.computed(data, key)[component]; });
    }
    auto getDensity(std::string const& pop)
    {
        return fields_([&](auto& p) -> auto& { return population_(p, pop).particleDensity(); });
    }

    // a population's domain particles: a patch data per patch, a tiled array's one per tile (tiles
    //  count as patches: each its own array, live, as an untiled patch's)
    auto getParticles(std::string const& pop)
        requires(!core::TiledParticles<Particles_t>)
    {
        std::vector<ParticlesData> out;
        for_each_patch_([&](auto const& patch, auto& data) {
            set_(out.emplace_back(), patch, data);
            out.back().data    = &population_(data, pop).domainParticles();
            out.back().nGhosts = Patch_t::particle_ghost_width;
        });
        return out;
    }
    auto getParticles(std::string const& pop)
        requires core::TiledParticles<Particles_t>
    {
        std::vector<PatchData<tile_particles_t<Particles_t>*, dim>> out;
        for_each_patch_([&](auto const& patch, auto& data) {
            auto& tiles = population_(data, pop).domainParticles()();
            for (std::size_t t = 0; t < tiles.size(); ++t)
            {
                auto& tile = tiles[t];
                auto& pd   = out.emplace_back();
                pd.patchID = id_(patch) + "_t" + std::to_string(t);
                for (std::size_t i = 0; i < dim; ++i)
                {
                    pd.lower[i] = tile.lower[i];
                    pd.upper[i] = tile.upper[i];
                    pd.origin[i]
                        = data.layout.origin[i]
                          + (tile.lower[i] - data.layout.cells.lower[i]) * data.layout.dx[i];
                }
                pd.data    = &tile();
                pd.nGhosts = Patch_t::particle_ghost_width;
            }
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

    static std::string id_(auto const& patch)
    {
        return std::to_string(patch.rank) + "#" + std::to_string(patch.id);
    }
    static void set_(auto& pd, std::string const& id, auto const& layout)
    {
        pd.patchID = id;
        for (std::size_t i = 0; i < dim; ++i)
        {
            pd.origin[i] = layout.origin[i];
            pd.lower[i]  = layout.cells.lower[i];
            pd.upper[i]  = layout.cells.upper[i];
        }
        pd.nGhosts = Patch_t::field_ghost_width;
    }
    static void set_(auto& pd, auto const& patch, Patch_t const& data)
    {
        set_(pd, id_(patch), data.layout);
    }

    // a field's patch datas: an untiled field one, a tiled one one per tile (tiles count as
    //  patches: each its own contiguous data, layout and ghosts; their domains do not overlap)
    static void add_field_(std::vector<FieldData>& out, std::string const& id, auto const& layout,
                           auto& field)
        requires(!core::TiledField<decltype(field)>)
    {
        set_(out.emplace_back(), id, layout);
        out.back().data = field_view<dim>(field);
    }
    static void add_field_(std::vector<FieldData>& out, std::string const& id, auto const&,
                           core::TiledField auto& field)
    {
        for (std::size_t t = 0; t < field().size(); ++t)
        {
            auto& tile = field()[t];
            set_(out.emplace_back(), id + "_t" + std::to_string(t), tile.layout());
            out.back().data = field_view<dim>(tile());
        }
    }

    auto fields_(auto&& get)
    {
        std::vector<FieldData> out;
        for_each_patch_([&](auto const& patch, auto& data) {
            add_field_(out, id_(patch), data.layout, get(data));
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


// a particle array's python class: size, __len__, __getitem__ and components() (live views of
//  the columns where it has them, copies tile by tile for a tiled array)
template<typename Particles_t, std::size_t dim>
void declare_particle_array(nb::module_& m, char const* name)
{
    using Particle_t = plamn::core::Particle<dim>;
    auto particles   = nb::class_<Particles_t>(m, name)
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
    {
        using At_t = decltype(core::particle_at(std::declval<Particles_t&>(), std::size_t{}));
        // a reference into the array, or a proxy by value (SoAVX's zip particle, thrust builds):
        //  nanobind's policies are tag objects of their own types, one registration per case
        auto const at = [](Particles_t& self, std::size_t const i) -> At_t {
            if (i >= self.size())
                throw nb::index_error();
            return core::particle_at(self, i); // tiled: through the tiles
        };
        if constexpr (std::is_reference_v<At_t>)
            particles.def("__getitem__", at, nb::rv_policy::reference_internal);
        else
            particles.def("__getitem__", at, nb::rv_policy::move);
        def_tiled_components<Particles_t>(particles);
    }
}

// a tiled array's tiles' own arrays and their patch datas (getParticles: a patch data per tile)
template<typename Particles_t, std::size_t dim>
void declare_tile_particles(nb::module_& m)
    requires core::TiledParticles<Particles_t>
{
    using Tile_t = tile_particles_t<Particles_t>;
    declare_particle_array<Tile_t, dim>(m, "ParticleArrayTile");
    declare_patch_data<Tile_t*, dim>(m, "PatchDataParticleTile");
}
template<typename Particles_t, std::size_t dim>
void declare_tile_particles(nb::module_&)
    requires(!core::TiledParticles<Particles_t>)
{
}


template<typename Simulator>
void declare_data_wrangler(nb::module_& m)
{
    auto constexpr dim = Simulator::dimension;
    using DW           = DataWrangler<Simulator>;
    using Level        = DW::Level_t;
    using Particles_t  = plamn::solver::HybridModel<Simulator::options>::Patch_t::Particles_t;
    using Particle_t   = plamn::core::Particle<dim>;

    nb::class_<Particle_t>(m, "Particle")
        .def_prop_ro("weight", [](Particle_t const& p) { return p.weight(); })
        .def_prop_ro("iCell", [](Particle_t const& p) { return p.iCell(); })
        .def_prop_ro("delta", [](Particle_t const& p) { return p.delta(); })
        .def_prop_ro("v", [](Particle_t const& p) { return p.v(); });

    declare_particle_array<Particles_t, dim>(m, "ParticleArray");
    declare_tile_particles<Particles_t, dim>(m);
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
        .def("getMomentumTensor", &Level::getMomentumTensor, nb::arg("c"), nb::arg("pop") = "")
        .def("getParticles", // per patch, or per tile (tiled layouts): one of two overloads
             [](Level& level, std::string const& pop) { return level.getParticles(pop); });

    nb::class_<DW>(m, "DataWrangler")
        .def(nb::init<Simulator&, nb::handle>(), nb::keep_alive<1, 2>())
        .def("getNumberOfLevels", &DW::getNumberOfLevels)
        .def("getHybridPatchLevel", &DW::getHybridPatchLevel)
        .def("sync", &DW::sync);
}

} // namespace plamn::python3

#endif /* PLAMN_PYTHON3_PLAMN_DATA_WRANGLER_HPP */
