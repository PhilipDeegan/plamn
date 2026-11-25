#ifndef MOMENTS_HPP
#define MOMENTS_HPP

#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/data/field/field_tiles.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"

#include <algorithm>
#include <type_traits>
#include <utility>

namespace plamn
{
namespace core
{
    // whole fields, or only tile tile_idx of tiled fields
    template<typename Ions>
    void resetMoments(Ions& ions, auto const&... tile_idx)
    {
        for (auto& pop : ions)
        {
            maybe_tile_at(pop.particleDensity(), tile_idx...).zero();
            maybe_tile_at(pop.flux(), tile_idx...).zero();
            maybe_tile_at(pop.fluxE(), tile_idx...).zero();
        }
    }


    struct DomainDeposit
    {
    };

    struct LevelGhostDeposit
    {
    };


    // whole patch, or only into tile tile_idx of tiled moments
    template<typename DepositTag>
    void depositParticles(auto& ions, auto& layout, auto interpolate, DepositTag,
                          auto const&... tile_idx)
    {
        static_assert(std::is_same_v<DepositTag, DomainDeposit>
                      or std::is_same_v<DepositTag, LevelGhostDeposit>);

        auto constexpr type = std::is_same_v<DepositTag, DomainDeposit> ? ParticleType::Domain
                                                                        : ParticleType::LevelGhost;
        for (auto& pop : ions)
        {
            auto& particleDensity = pop.particleDensity();
            auto& flux            = pop.flux();
            auto& particles       = [&]() -> auto& {
                if constexpr (type == ParticleType::Domain)
                    return pop.domainParticles();
                else
                    return pop.levelGhostParticlesOld();
            }();

            if constexpr (sizeof...(tile_idx) > 0)
                interpolate.template particleToMesh_tile<type>(tile_idx..., particles,
                                                               particleDensity, flux);
            else
            {
                interpolate.template particleToMesh<type>(particles, layout, particleDensity, flux);
                interpolate.template fluxE_to_mesh<type>(particles, layout, pop.fluxE());
            }
        }
    }

    // a population's fluxE after the deposits and border sums: the E components on the moment
    //  nodes (deposits_fluxE leaves them) are the flux; other layouts: nothing (Ohm reads the
    //  moment nodes' charge flux)
    template<typename Pop>
        requires(deposits_fluxE(
            std::decay_t<decltype(std::declval<Pop&>().domainParticles())>::layout_mode))
    void complete_fluxE(Pop& pop, auto const& layout)
    {
        using Layout_t  = std::decay_t<decltype(layout)>;
        using Scalar    = HybridQuantity::Scalar;
        auto const copy = [&]<auto quantity>(auto& dst, auto const& src) {
            if constexpr (on_moment_nodes<Layout_t::dimension, quantity>())
                std::copy(src.data(), src.data() + src.size(), dst.data());
        };
        copy.template operator()<Scalar::Ex>(pop.fluxE()[0], pop.flux()[0]);
        copy.template operator()<Scalar::Ey>(pop.fluxE()[1], pop.flux()[1]);
        copy.template operator()<Scalar::Ez>(pop.fluxE()[2], pop.flux()[2]);
    }
    template<typename Pop>
        requires(not deposits_fluxE(
            std::decay_t<decltype(std::declval<Pop&>().domainParticles())>::layout_mode))
    void complete_fluxE(Pop&, auto const&)
    {
    }

} // namespace core
} // namespace plamn



#endif // MOMENTS_HPP
