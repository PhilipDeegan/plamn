#ifndef PLAMN_CORE_DATA_PARTICLES_PARTICLES_DEF_HPP
#define PLAMN_CORE_DATA_PARTICLES_PARTICLES_DEF_HPP

#include "mkn/kul/std/array.hpp"
#include "mkn/kul/for.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/def/damn_config.hpp"

#include "plamn/core/def.hpp"
#include "plamn/core/data/particles/particle.hpp"

#include <array>
#include <algorithm>
#include <tuple>
#include <cstdint>
#include <type_traits>
#include <cstddef>
#include <iterator>
#include <utility>
#include <variant>

namespace damn::core
{
template<typename Type, std::size_t dim>
class Point;
} // namespace damn::core

namespace plamn::core
{

// a particle is in a box if its cell is
template<typename Particle>
MKN_KUL_FN_SIG_RET auto DAMN_ALL isIn(Particle const& particle,
                                      damn::core::Box<int, Particle::dimension> const& box)
    -> decltype(isIn(particle.iCell(), box), bool())
{
    return isIn(particle.iCell(), box);
}



enum class LayoutMode : std::uint16_t {
    AoS = 0,
    AoSMapped,    // 1
    AoSCMTS,      // 2
    SoA,          // 3 - internal only on CPU: HDF5/restart write buffer + python particle-splitting
                  //     interop (zero-copy span over numpy arrays)
    AoSTS,        // 4 - tiled AoS (MHD only builds: tiled fields without particles)
    AoSPC,        // 5
    SoAVX,        // 6
    SoATS,        // 7
    SoAVXTS,      // 8
    SoAPC,        // 9
    AoSPCTS,      // 10
    SoAVXChunked, // 11 - SoAVX sorted by cell, per cell offsets (as AoSMapped: AoS + CellMap)
    // 12 - AoS sorted by cell, per cell offsets, pushed staged as SoAVX blocks
    AoS4SoAVXChunked,
};

bool constexpr is_tiled(LayoutMode lm)
{
    using enum LayoutMode;
    return mkn::kul::any_in(lm, AoSCMTS, AoSTS, SoATS, SoAVXTS, AoSPCTS);
}

// a tiled particle array, whatever its cv/ref: what the tile/untiled overloads branch on
template<typename P>
concept TiledParticles = is_tiled(std::decay_t<P>::layout_mode);

// flat SoAVX columns, with or without per cell offsets
bool constexpr is_soavx(LayoutMode lm)
{
    using enum LayoutMode;
    return mkn::kul::any_in(lm, SoAVX, SoAVXChunked);
}

// sorted by cell in per cell chunks, gap slots (weight 0) between them (particles_chunked.hpp)
bool constexpr is_chunked(LayoutMode lm)
{
    using enum LayoutMode;
    return mkn::kul::any_in(lm, SoAVXChunked, AoS4SoAVXChunked);
}

// for a forwarding constructor: a self-type first argument (e.g. a non-const This&) excluded,
//  else it is an exact match beating the copy constructor's This const&
template<typename This, typename Super, typename... Args>
bool consteval self_excluding_constructible()
{
    using Tup = std::tuple<Args...>;

    bool constexpr base = std::is_constructible_v<Super, Args&&...>;
    if constexpr (std::tuple_size_v<Tup> > 0)
    {
        bool constexpr isself
            = std::is_same_v<std::decay_t<std::tuple_element_t<0, Tup>>, This>;
        return !isself and base;
    }
    else
        return base;
}

enum class StorageMode : std::uint16_t { ARRAY = 0, VECTOR, SPAN };

enum class ParticleType : std::uint16_t { Domain = 0, Ghost, PatchGhost, LevelGhost, All };


struct ParticlesOptions
{
    std::size_t dim          = 1;
    LayoutMode layout_mode   = LayoutMode::AoSMapped;
    StorageMode storage_mode = StorageMode::VECTOR;
    damn::AllocatorMode alloc_mode = damn::AllocatorMode::CPU;
    bool _const_             = 0; // sometimes needed

    auto constexpr with_layout(LayoutMode const lm) const
    {
        auto copy        = *this;
        copy.layout_mode = lm;
        return copy;
    }
    auto constexpr with_storage(StorageMode const sm) const
    {
        auto copy         = *this;
        copy.storage_mode = sm;
        return copy;
    }
    auto constexpr with_alloc(damn::AllocatorMode const am) const
    {
        auto copy       = *this;
        copy.alloc_mode = am;
        return copy;
    }
};



template<std::size_t dim>
struct ParticleDefaults
{
    using Particle_t = Particle<dim>;
};


// carries where a particle came from when registering a move with move_check;
// per-cell layouts read icell (old AMR cell)
template<std::size_t dim>
struct ParticleTracker
{
    std::array<int, dim> icell{};
};

// tiled layouts additionally read tile_cell (old local tile cell)
template<std::size_t dim>
struct TiledParticleTracker
{
    std::array<int, dim> icell{};
    std::array<std::uint32_t, dim> tile_cell{};
};

// level ghost moves resolve differently: ghost cells are clamp-owned by border tiles,
// leaving the ghost box means deletion — distinct type so move_check can dispatch
template<std::size_t dim>
struct TiledLevelGhostParticleTracker
{
    std::array<int, dim> icell{};
    std::array<std::uint32_t, dim> tile_cell{};
};


// the tracker kind is a compile-time function of the layout and the particle type
template<LayoutMode layout_mode, ParticleType particle_type, std::size_t dim>
using ParticleTracker_t = std::conditional_t<
    is_tiled(layout_mode),
    std::conditional_t<particle_type == ParticleType::LevelGhost,
                       TiledLevelGhostParticleTracker<dim>, TiledParticleTracker<dim>>,
    ParticleTracker<dim>>;

template<LayoutMode layout_mode, ParticleType particle_type, std::size_t dim>
auto make_particle_tracker(auto&&... args) DAMN_ALL
{
    return ParticleTracker_t<layout_mode, particle_type, dim>{args...};
}


template<typename R = std::uint32_t, std::size_t dim>
auto as_local_cell(std::array<int, dim> const& lower,
                   std::array<int, dim> const& icell) DAMN_ALL
{
    return mkn::kul::array_minus<R>(icell, lower);
}
template<std::size_t dim>
auto as_local_cell(damn::core::Box<int, dim> const& box, std::array<int, dim> const& icell) DAMN_ALL
{
    return as_local_cell(box.lower.toArray(), icell);
}
template<std::size_t dim>
auto as_local_cell(damn::core::Box<int, dim> const& box, damn::core::Point<int, dim> const& icell) DAMN_ALL
{
    return as_local_cell(box.lower.toArray(), icell.toArray());
}

template<typename I0, typename I1> // support const vs non-const iterators
auto it_dist(I0&& i0, I1&& i1)
{
    auto const& begin = i0;
    auto const& pos   = i1;
    auto d            = std::distance(begin, pos);
    DAMN_ASSERT(d < 1e18); // should never happen // eg 2635249153387078728
    return d;
}


// !!! TODO(cell-jump-limit): TEMPORARY, REMOVE WITH THE CELL JUMP LIMIT !!!
// cells to map around the particle ghost box, so the selectors (cell map partitions) see every
// particle a push can produce. A half push moves a particle up to 2 cells (Boris throws past
// that): level ghosts are selected after each half push, up to 2 cells past the ghost box;
// domain particles after both, up to 4 cells past the domain, which is the ghost box shrunk by
// the particle ghost width.
// the cell jump limit makes this irrelevant: domain particles throw before leaving the ghost box,
// level ghosts out of it are dropped by a predicate before interpolation, not through the cell map.
// Then remove this and the margin in make_particles below (grep "cell-jump-limit").
MKN_KUL_FN_SIG_RET int constexpr particleMapMargin(int const particleGhostWidth)
{
    return std::max(2, 4 - particleGhostWidth);
}

template<typename Particles_t, typename T, std::size_t D>
Particles_t make_particles(damn::core::Box<T, D> const& box, std::size_t const ghost_cells)
{
    static_assert(Particles_t::storage_mode == StorageMode::VECTOR);

    using enum LayoutMode;
    if constexpr (mkn::kul::any_in(Particles_t::layout_mode, AoSPC, SoAPC))
        return Particles_t{box, ghost_cells};
    else if constexpr (is_tiled(Particles_t::layout_mode))
        return Particles_t{box, ghost_cells};
    else if constexpr (mkn::kul::any_in(Particles_t::layout_mode, AoSMapped, SoAVXChunked,
                                        AoS4SoAVXChunked))
    {
        // !!! TODO(cell-jump-limit): TEMPORARY margin, back to grow(box, ghost_cells) !!!
        int const ghosts = static_cast<int>(ghost_cells);
        return Particles_t{damn::core::grow(box, ghosts + particleMapMargin(ghosts))};
    }
    else
        return Particles_t{};
}

template<typename Particles_t, typename Layout_t>
Particles_t make_particles(Layout_t const& layout)
{
    return make_particles<Particles_t>(layout.cells, Layout_t::options.particle_ghost_width);
}



} // namespace plamn::core


#endif /*PLAMN_CORE_DATA_PARTICLES_PARTICLES_DEF_HPP*/
