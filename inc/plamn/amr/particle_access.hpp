#ifndef PLAMN_AMR_PARTICLE_ACCESS_HPP
#define PLAMN_AMR_PARTICLE_ACCESS_HPP

// damn's particle copies and refines reach a patch's arrays through accessors, not by name,
//  each array by role (a wrong one does not compile): doc/amr/particle_access.md

#include "plamn/amr/resources.hpp"
#include "damn/amr/copy.hpp"      // IWYU pragma: keep (ResourcePair: a member's element type)
#include "damn/amr/level_map.hpp" // IWYU pragma: keep (LevelPair: a member's element type)

#include <cstddef>
#include <string>
#include <utility>
#include <vector>
#include <format>

namespace damn::amr
{
struct ResourcePair;
} // namespace damn::amr

namespace plamn::amr
{
template<typename Patch>
using PopulationData_t = Patch::PopulationData;

template<typename Patch>
using PopulationArray_t = Patch::Particles_t PopulationData_t<Patch>::*;

template<typename Patch, PopulationArray_t<Patch> array>
concept ParticleSourceArray
    = array == &PopulationData_t<Patch>::domain or array == &PopulationData_t<Patch>::patch_ghost;

template<typename Patch, PopulationArray_t<Patch> array>
concept ParticleDestinationArray = array == &PopulationData_t<Patch>::domain
                                   or array == &PopulationData_t<Patch>::level_ghost_old
                                   or array == &PopulationData_t<Patch>::level_ghost_new;

// the resource names of the arrays (for_each_resource's), for damn's plans; none for
//  level_ghost
template<typename Patch, PopulationArray_t<Patch> array>
    requires(array == &PopulationData_t<Patch>::domain)
std::string array_name(std::string const& pop)
{
    return pop + "_domain";
}
template<typename Patch, PopulationArray_t<Patch> array>
    requires(array == &PopulationData_t<Patch>::patch_ghost)
std::string array_name(std::string const& pop)
{
    return pop + "_patch_ghost";
}
template<typename Patch, PopulationArray_t<Patch> array>
    requires(array == &PopulationData_t<Patch>::level_ghost_old)
std::string array_name(std::string const& pop)
{
    return pop + "_level_ghost_old";
}
template<typename Patch, PopulationArray_t<Patch> array>
    requires(array == &PopulationData_t<Patch>::level_ghost_new)
std::string array_name(std::string const& pop)
{
    return pop + "_level_ghost_new";
}

// data: a LevelData's patch data (a patch behind a unique_ptr)
template<typename Patch, PopulationArray_t<Patch> array>
    requires ParticleSourceArray<Patch, array>
struct ParticleSource
{
    std::size_t pop;
    auto operator()(auto const& data) const
    {
        return particle_resource<Patch::particle_ghost_width>(
            std::as_const(data->populations[pop].get()->*array));
    }
};

template<typename Patch, PopulationArray_t<Patch> array>
    requires ParticleDestinationArray<Patch, array>
struct ParticleDestination
{
    std::size_t pop;
    auto operator()(auto& data) const
    {
        return particle_resource<Patch::particle_ghost_width>(data->populations[pop].get()->*array);
    }
};

template<typename Patch, PopulationArray_t<Patch> src_array, PopulationArray_t<Patch> dst_array>
struct ParticleCopyAccess
{
    ParticleSource<Patch, src_array> src_;
    ParticleDestination<Patch, dst_array> dst_;

    auto src(auto& data) const { return src_(data); }
    auto dst(auto& data) const { return dst_(data); }
};

template<typename Patch, PopulationArray_t<Patch> coarse_array, PopulationArray_t<Patch> fine_array>
struct ParticleRefineAccess
{
    ParticleSource<Patch, coarse_array> coarse_;
    ParticleDestination<Patch, fine_array> fine_;

    auto coarse(auto& data) const { return coarse_(data); }
    auto coarse_new(auto& data) const { return coarse_(data); } // no time interpolation
    auto fine(auto& data) const { return fine_(data); }
};

template<typename Patch, PopulationArray_t<Patch> src, PopulationArray_t<Patch> dst>
struct ParticleCopies
{
    using Patch_t = Patch;
    std::vector<damn::amr::ResourcePair> pairs;
    std::vector<ParticleCopyAccess<Patch, src, dst>> access;

    explicit ParticleCopies(std::vector<std::string> const& populations)
    {
        for (std::size_t i = 0; i < populations.size(); ++i)
        {
            pairs.push_back(
                {array_name<Patch, src>(populations[i]), array_name<Patch, dst>(populations[i])});
            access.push_back({{i}, {i}});
        }
    }
};

template<typename Patch, PopulationArray_t<Patch> coarse, PopulationArray_t<Patch> fine>
struct ParticleRefines
{
    using Patch_t = Patch;
    std::vector<damn::amr::LevelPair> pairs;
    std::vector<ParticleRefineAccess<Patch, coarse, fine>> access;

    explicit ParticleRefines(std::vector<std::string> const& populations)
    {
        for (std::size_t i = 0; i < populations.size(); ++i)
        {
            pairs.push_back({array_name<Patch, coarse>(populations[i]),
                             array_name<Patch, fine>(populations[i]), ""});
            access.push_back({{i}, {i}});
        }
    }
};

// a schedule's pairs (names: damn's plans) with their accessors, for a Patch
template<typename T>
concept ParticleRoute = requires(T const& t) {
    typename T::Patch_t;
    t.pairs;
    t.access;
};

// a hybrid model's particle schedules, by role
template<typename Patch>
struct ParticleRoutes
{
    using PD = PopulationData_t<Patch>;
    static_assert(!ParticleSourceArray<Patch, &PD::level_ghost>
                  and !ParticleDestinationArray<Patch, &PD::level_ghost>);
    static_assert(!ParticleSourceArray<Patch, &PD::level_ghost_old>
                  and !ParticleSourceArray<Patch, &PD::level_ghost_new>);
    static_assert(!ParticleDestinationArray<Patch, &PD::patch_ghost>);

    // patch ghosts (left their patch) to the domain of the patch owning their cell
    ParticleCopies<Patch, &PD::patch_ghost, &PD::domain> gather_leaving;
    // a coarser level's domain split into a finer level's domain (init, regrid: interior)
    ParticleRefines<Patch, &PD::domain, &PD::domain> interior_splits;
    // into its level ghosts: at init / regrid / restart (old), a coarse step's end (new)
    ParticleRefines<Patch, &PD::domain, &PD::level_ghost_old> old_ghost_splits;
    ParticleRefines<Patch, &PD::domain, &PD::level_ghost_new> new_ghost_splits;

    explicit ParticleRoutes(std::vector<std::string> const& populations)
        : gather_leaving{populations}
        , interior_splits{populations}
        , old_ghost_splits{populations}
        , new_ghost_splits{populations}
    {
    }
};

} // namespace plamn::amr

#endif // PLAMN_AMR_PARTICLE_ACCESS_HPP
