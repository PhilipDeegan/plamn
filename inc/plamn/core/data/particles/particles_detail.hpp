#ifndef PLAMN_CORE_DATA_PARTICLES_PARTICLES_DETAIL_HPP
#define PLAMN_CORE_DATA_PARTICLES_PARTICLES_DETAIL_HPP

#include "damn/core/sys/strings.hpp"
#include "damn/core/meta/logic.hpp"

#include "plamn/core/def.hpp"
#include "plamn/core/data/particles/particles_def.hpp"

#include <string_view>
#include <type_traits>
#include <cstddef>
#include <cstdint>
// Mutators
#include "plamn/core/data/particles/arrays/particles_pc.hpp"
#include "plamn/core/data/particles/arrays/particles_ts.hpp"
#include "plamn/core/data/particles/arrays/particles_pc_ts.hpp"
#include "plamn/core/data/particles/arrays/particles_cm_ts.hpp"
// Impls
#include "plamn/core/data/particles/arrays/particles_aos.hpp"
#include "plamn/core/data/particles/arrays/particles_soa.hpp"
#include "plamn/core/data/particles/arrays/particles_soavx.hpp"
#include "plamn/core/data/particles/arrays/particles_soavx_chunked.hpp"
#include "magic_enum/magic_enum.hpp"
#include "magic_enum/magic_enum_switch.hpp"  // IWYU pragma: keep
#include "magic_enum/magic_enum_utility.hpp" // IWYU pragma: keep

namespace damn
{
enum class AllocatorMode : std::uint16_t;
} // namespace damn


namespace plamn::core
{
template<auto opts>
class Particles;


// what the AoS impls are templated on: layout/storage independent, so every layout over
// AoS of the same dim/alloc shares one impl type
struct ParticlesTypeOptions
{
    std::size_t dim;
    damn::AllocatorMode alloc_mode;
};


namespace detail
{
    // void == layout has no impl for this storage mode
    template<auto opts, typename Vector, typename Span>
    using by_storage = std::conditional_t<opts.storage_mode == StorageMode::VECTOR, Vector,
                                          std::conditional_t<opts.storage_mode == StorageMode::SPAN,
                                                             Span, void>>;

    // tiled / per-cell layouts: a container of Particles of an inner layout
    template<auto opts, LayoutMode inner, template<typename> typename Wrap,
             template<typename> typename Vector, template<typename> typename Span,
             typename Inner = Particles<opts.with_layout(inner)>>
    using nested = std::conditional_t<opts.storage_mode == StorageMode::ARRAY, void,
                                      Wrap<by_storage<opts, Vector<Inner>, Span<Inner>>>>;

    template<typename T>
    auto constexpr static id = std::type_identity<T>{};

    template<auto opts, auto o>
    consteval auto resolve()
    {
        using enum LayoutMode;
        auto constexpr lm = opts.layout_mode;
        auto constexpr d  = opts.dim;
        auto constexpr am = opts.alloc_mode;

        if constexpr (lm == AoS)
            return id<by_storage<opts, AoSParticles<AoSVector, o>, AoSParticles<AoSSpan, o>>>;
        else if constexpr (lm == AoSMapped)
            return id<by_storage<opts, AoSMappedParticles<AoSMappedVector, o>,
                                 AoSMappedParticles<AoSMappedSpan, o>>>;
        else if constexpr (lm == SoA)
            return id<by_storage<opts, SoAParticles<SoAVector<d, am>>,
                                 SoAParticles<SoASpan<d, am, opts._const_>>>>;
        else if constexpr (lm == SoAVX)
            return id<by_storage<opts, SoAVXParticles<SoAVXVector<d, am>>,
                                 SoAVXParticles<SoAVXSpan<d, am>>>>;
        else if constexpr (lm == SoAVXChunked)
            return id<by_storage<opts, SoAVXChunkedParticles<SoAVXChunkedVector<d, am>>,
                                 SoAVXChunkedParticles<SoAVXChunkedSpan<d, am>>>>;
        else if constexpr (lm == AoSPC)
            return id<nested<opts, AoS, PerCellParticles, PerCellVector, PerCellSpan>>;
        else if constexpr (lm == SoAPC)
            return id<nested<opts, SoA, PerCellParticles, PerCellVector, PerCellSpan>>;
        else if constexpr (lm == AoSTS)
            return id<nested<opts, AoS, TileSetParticles, TileSetVector, TileSetSpan>>;
        else if constexpr (lm == SoATS)
            return id<nested<opts, SoA, TileSetParticles, TileSetVector, TileSetSpan>>;
        else if constexpr (lm == SoAVXTS)
            return id<nested<opts, SoAVX, TileSetParticles, TileSetVector, TileSetSpan>>;
        else if constexpr (lm == AoSCMTS)
            return id<nested<opts, AoSMapped, MappedTileSetParticles, MappedTileSetVector,
                             MappedTileSetSpan>>;
        else if constexpr (lm == AoSPCTS)
            return id<nested<opts, AoSPC, PCTileSetParticles, PCTileSetVector, PCTileSetSpan>>;
        else
            return id<void>;
    }

} // namespace detail


template<auto opts, auto o = ParticlesTypeOptions{opts.dim, opts.alloc_mode}>
class ParticlesResolver
{
public:
    using value_type = decltype(detail::resolve<opts, o>())::type;
    static_assert(!std::is_void_v<value_type>, "storage mode not supported for this layout");

    struct strings;

    auto static constexpr type_id = strings::type_id;
};

template<auto opts, auto o>
struct ParticlesResolver<opts, o>::strings
{
    std::string_view static constexpr alloc_mode   = magic_enum::enum_name(opts.alloc_mode);
    std::string_view static constexpr layout_mode  = magic_enum::enum_name(opts.layout_mode);
    std::string_view static constexpr storage_mode = magic_enum::enum_name(opts.storage_mode);
    std::string_view static constexpr cma          = ",";
    std::string_view static constexpr _dim         = damn::core::to_string_view_v<std::size_t, opts.dim>;

    auto static constexpr type_id
        = damn::core::join_string_views_v<_dim, cma, layout_mode, cma, alloc_mode, cma, storage_mode, cma>;
};


template<auto opts>
using ResolvedParticles_t = ParticlesResolver<opts>::value_type;



// visit every particle: tiles and cells are walked into, flat layouts iterated directly
template<typename Particles_t>
void per_particle(Particles_t& particles, auto const& fn)
{
    using enum LayoutMode;
    if constexpr (is_tiled(Particles_t::layout_mode))
        for (auto& tile : particles())
            per_particle(tile(), fn);
    else if constexpr (damn::core::any_in(Particles_t::layout_mode, AoSPC, SoAPC))
        for (auto& cell : particles())
            per_particle(cell, fn);
    else
        for (auto& p : particles)
            fn(p);
}


} // namespace plamn::core


#endif /*PLAMN_CORE_DATA_PARTICLES_PARTICLES_DETAIL_HPP*/
