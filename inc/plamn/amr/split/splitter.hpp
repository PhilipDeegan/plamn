#ifndef PLAMN_AMR_SPLIT_SPLITTER_HPP
#define PLAMN_AMR_SPLIT_SPLITTER_HPP

#include "damn/core/def/constants.hpp"
#include "damn/core/def/damn_config.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace damn::core
{
template<typename Type, std::size_t dim>
class Point;
} // namespace damn::core


namespace plamn::amr
{
// the number of particles a coarse one splits into, as damn::core::DimConst
template<std::size_t Constant>
class RefinedParticlesConst : public damn::core::StrongIntegralConstant<Constant>
{
};

int constexpr refinementRatio = 2;

// see meta_utilities.hpp for list of declared permutations.

/** \brief A SplitPattern identifies where refined particles will be placed
 * relative to a coarse particle and their weight.
 *
 * A pattern is a geometric description of the placement of refined particles
 * around the coarse particle. All particles refined with a specific pattern
 * are placed at the same distance (delta) from the coarse particle with the same
 * weight.
 *
 * The concrete splitting of a coarse particle can be the result of a single pattern
 * or the combination of several patterns.
 *
 */
template<typename _dimension, typename _nbRefinedPart>
struct SplitPattern
{
    constexpr SplitPattern(float weight)
        : weight_{weight}
    {
    }

    float weight_;
    std::array<damn::core::Point<float, _dimension{}()>, _nbRefinedPart{}()> deltas_{};
};


/** \brief PatternDispatcher is the way to concretely place all refined particles
 * for a combination of patterns given at compile time.
 */
template<typename... Patterns>
class PatternDispatcher
{
public:
    constexpr PatternDispatcher(Patterns&&... _patterns)
        : patterns{_patterns...}
        , nbRefinedParts{(_patterns.deltas_.size() + ...)}
    {
    }



    template<typename FineiCellDelta, typename ParticleIt, typename Particles>
    inline void operator()(FineiCellDelta const& fine, ParticleIt const& coarseParticle,
                           Particles& refinedParticles, size_t idx = 0) const DAMN_ALL
    {
        dispatch(fine, coarseParticle, refinedParticles, idx);
    }

    // largest offset, in fine cells along any axis, of a refined particle from its parent
    constexpr float maxDelta() const
    {
        float max             = 0;
        auto const patternMax = [&](auto const& pattern) {
            for (auto const& delta : pattern.deltas_)
                for (std::size_t i = 0; i < delta.size(); ++i)
                    max = std::max(max, delta[i] < 0 ? -delta[i] : delta[i]);
        };
        std::apply([&](auto const&... pattern) { (patternMax(pattern), ...); }, patterns);
        return max;
    }

    std::tuple<Patterns...> patterns{};
    std::size_t nbRefinedParts{0};

private:
    template<typename FineiCellDelta, typename ParticleIt, typename Particles>
    void dispatch(FineiCellDelta const& fine, ParticleIt const& particle, //
                  Particles& particles, std::size_t idx) const DAMN_ALL
    {
        using Weight_t = std::decay_t<decltype(particle.weight())>;
        using Delta_t  = std::decay_t<decltype(particle.delta()[0])>;

        constexpr auto dimension = ParticleIt::dimension;
        constexpr auto refRatio  = refinementRatio;
        constexpr std::array power{refRatio, refRatio * refRatio, refRatio * refRatio * refRatio};

        assert(particles.size() >= idx + nbRefinedParts);

        auto split = [&](auto& fineParticle, auto& pattern, auto const& rpIndex) {
            fineParticle.weight()
                = particle.weight() * static_cast<Weight_t>(pattern.weight_) * power[dimension - 1];
            fineParticle.iCell() = fine.iCell();
            fineParticle.delta() = fine.delta();
            fineParticle.v()     = particle.v();

            for (size_t iDim = 0; iDim < dimension; iDim++)
            {
                fineParticle.delta()[iDim] += static_cast<Delta_t>(pattern.deltas_[rpIndex][iDim]);
                Delta_t integra = std::floor(fineParticle.delta()[iDim]);
                fineParticle.delta()[iDim] -= integra;
                fineParticle.iCell()[iDim] += static_cast<std::int32_t>(integra);
            }
        };

        // particles: AoS, begin() + i a particle (e.g. split_buffer's)
        auto const each = [&](auto const& pattern) {
            for (std::size_t rpIndex = 0; rpIndex < pattern.deltas_.size(); rpIndex++)
            {
                auto fineParticle = particles.begin() + idx++;
                split(*fineParticle, pattern, rpIndex);
            }
        };
        std::apply([&](auto const&... pattern) { (each(pattern), ...); }, patterns);
    }
};


template<typename _dimension, typename _interp_order, typename _nbRefinedPart>
struct ASplitter
{
    static constexpr auto dimension     = _dimension{}();
    static constexpr auto interp_order  = _interp_order{}();
    static constexpr auto nbRefinedPart = _nbRefinedPart{}();

    static constexpr int maxCellDistanceFromSplit()
    {
        constexpr int particleSize = interp_order + 1;
        return (particleSize + 1) / 2; // ceil(particleSize / 2), std::ceil is not constexpr
    }

    constexpr ASplitter() {}
};


/** \brief this class is just the template declaration of the Splitter class
 * Only its concrete specializations defined in split_1d, split_2d, split_3d
 * are to be used therefore the constructor here is deleted to prevent instantiation.
 */
template<typename dimension, typename interp_order, typename nbRefinedPart>
class Splitter : public ASplitter<dimension, interp_order, nbRefinedPart>
{
    Splitter() = delete; // Unspecialized template class, never to be instantiated
};

template<typename dim>
struct BlackPattern : SplitPattern<dim, RefinedParticlesConst<1>>
{
    using Super = SplitPattern<dim, RefinedParticlesConst<1>>;
    constexpr BlackPattern(float const weight)
        : Super{weight}
    {
    }
};

template<typename dim>
struct PinkPattern
{
};
template<typename dim>
struct PurplePattern
{
};
template<typename dim>
struct BrownPattern
{
};
template<typename dim>
struct LimePattern
{
};
template<typename dim>
struct WhitePattern
{
};


// fine cells by which a destination box must grow to hold every parent that can put a child in it
template<typename Splitter>
constexpr int splitBoxGrowth()
{
    constexpr float maxDelta  = Splitter{}.maxDelta();
    constexpr int maxDeltaInt = static_cast<int>(maxDelta);
    constexpr int ceilDelta   = maxDelta > maxDeltaInt ? maxDeltaInt + 1 : maxDeltaInt;
    return std::max(Splitter::maxCellDistanceFromSplit(), ceilDelta);
}

} // namespace plamn::amr

#endif // endif PLAMN_AMR_SPLIT_SPLITTER_HPP
