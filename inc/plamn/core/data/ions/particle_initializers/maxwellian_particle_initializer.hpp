#ifndef PLAMN_FLUID_PARTICLE_INITIALIZER_HPP
#define PLAMN_FLUID_PARTICLE_INITIALIZER_HPP

#include "damn/core/math/random.hpp"
#include "mkn/kul/for.hpp"
#include "damn/core/def.hpp"
#include "damn/core/def/damn_config.hpp"
#include "mkn/kul/std/array.hpp"
#include "damn/core/memory/span.hpp"
#include "plamn/core/data/grid/coordinates.hpp"
#include "plamn/core/data/ions/particle_initializers/maxwellian_def.hpp"
#include "plamn/core/data/particles/particle.hpp"
#include "plamn/core/data/particles/particles.hpp"
#include "plamn/core/data/particles/appending/particles_appending.hpp"
#include "plamn/core/data/ions/particle_initializers/particle_initializer.hpp"
#include "plamn/initializer/data_provider.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/models/quantities/quantity.hpp"
#include "mkn/kul/random.hpp"

#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <cstdint>
#include <numeric>
#include <functional>
#include <stdexcept>
#include <array>
#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>

namespace plamn::core
{
enum class Basis { Magnetic, Cartesian };


void maxwellianVelocity(std::array<double, 3> const& V, std::array<double, 3> const& Vth,
                        auto& generator, std::array<double, 3>& partVelocity)
{
    std::normal_distribution<> maxwellX(V[0], Vth[0]);
    std::normal_distribution<> maxwellY(V[1], Vth[1]);
    std::normal_distribution<> maxwellZ(V[2], Vth[2]);

    partVelocity[0] = maxwellX(generator);
    partVelocity[1] = maxwellY(generator);
    partVelocity[2] = maxwellZ(generator);
}


/** @brief a MaxwellianParticleInitializer is a ParticleInitializer that loads particles from a
 * local Maxwellian distribution given density, bulk velocity and thermal velocity profiles.
 *
 * Each cell draws its particles from its own generator, keyed by (seed, population name, level,
 * global AMR cell index), so a seeded load does not depend on the patch tiling, the number of
 * MPI ranks or the load balancing. Without a seed, the key base is drawn from std::random_device.
 * Seeded values are reproducible for a given standard library only, since the std distributions
 * used to draw positions and velocities are implementation-defined.
 */
template<typename Particles, typename Layout>
class MaxwellianParticleInitializer : public ParticleInitializer<Particles, Layout>
{
public:
    static constexpr auto dimension = Layout::dimension;
    using InputFunction             = plamn::initializer::InitFunction<dimension>;

    MaxwellianParticleInitializer(
        InputFunction density, std::array<InputFunction, 3> const& bulkVelocity,
        std::array<InputFunction, 3> const& thermalVelocity,
        std::uint32_t const& nbrParticlesPerCell, std::optional<std::size_t> seed = {},
        Basis const basis                                 = Basis::Cartesian,
        std::array<InputFunction, 3> const& magneticField = {nullptr, nullptr, nullptr},
        double const densityCutOff = 1e-5, std::string const& populationName = "",
        double const over_alloc_factor = .05, double const densityBackground = 0.)
        : density_{density}
        , bulkVelocity_{bulkVelocity}
        , thermalVelocity_{thermalVelocity}
        , magneticField_{magneticField}
        , densityCutOff_{densityCutOff}
        , nbrParticlePerCell_{nbrParticlesPerCell}
        , basis_{basis}
        , rngSeed_{seed}
        , populationKey_{mkn::kul::fnv1a64(populationName)}
        , over_alloc_factor_{over_alloc_factor}
        , densityBackground_{densityBackground}
    {
        if (!std::isfinite(over_alloc_factor_) or over_alloc_factor_ < 0)
            throw std::runtime_error("over_alloc_factor must be finite and non-negative");
    }


    /**
     * @brief load particles in a Particles in a domain defined by the given layout
     */
    void loadParticles(Particles& particles, Layout const& layout) const override;

    // nbrParticlePerCell for each cell whose density is at least the cutoff, as loadParticles;
    //  a background at least the cutoff populates every cell, nothing to sample
    NO_DISCARD std::size_t nbrParticles(Layout const& layout) const override;


    virtual ~MaxwellianParticleInitializer() = default;


    NO_DISCARD static std::uint64_t baseSeed(std::optional<std::size_t> const& seed)
    {
        if (seed.has_value())
            return *seed;
        std::random_device randSeed;
        return (std::uint64_t{randSeed()} << 32) ^ std::uint64_t{randSeed()};
    }

private:
    //! number of cells on the outer layer of the box, where incoming particles first land
    //! a box 2 cells or less wide in some direction has no interior, all its cells count
    NO_DISCARD static std::size_t surface_cell_count_(auto const& box)
    {
        std::size_t all = 1, interior = 1;
        for (auto const n : box.shape())
        {
            all *= n;
            interior *= n > 2 ? n - 2 : 0;
        }
        return all - interior;
    }
    InputFunction density_;
    std::array<InputFunction, 3> bulkVelocity_;
    std::array<InputFunction, 3> thermalVelocity_;
    std::array<InputFunction, 3> magneticField_;

    double densityCutOff_ = 1e-5;
    // the constant part of the density, added to the density function (which then holds what
    //  varies, never taking the density below the cutoff where the background is above it): a
    //  background at least the cutoff populates every cell
    double densityBackground_ = 0.;
    std::uint32_t nbrParticlePerCell_;
    Basis basis_;
    std::optional<std::size_t> rngSeed_;
    std::uint64_t populationKey_;
    double over_alloc_factor_ = .05;
};


class MaxwellianInitFunctions
{
public:
    template<typename FunctionArray, typename... Coords>
    MaxwellianInitFunctions(auto& density, FunctionArray& bulkVelocity,
                            FunctionArray& thermalVelocity, FunctionArray& magneticField,
                            auto const& basis, Coords const&... coords)
        : _n{density(coords...)}

    {
        // std::unique_lock<std::mutex> lock(ThreadPool::mutex(0)); // PYTHON == DEADLOCK

        static_assert(sizeof...(coords) <= 3, "can only provide up to 3 coordinates");
        for (std::uint32_t i = 0; i < 3; i++)
        {
            _V[i]   = bulkVelocity[i](coords...);
            _Vth[i] = thermalVelocity[i](coords...);
        }
        if (basis == Basis::Magnetic)
            for (std::uint32_t i = 0; i < 3; i++)
                _B[i] = magneticField[i](coords...);
    }

    NO_DISCARD std::array<double const*, 3> B() const { return ptrs(_B); }

    NO_DISCARD auto operator()() const { return std::make_tuple(_n->data(), ptrs(_V), ptrs(_Vth)); }

private:
    NO_DISCARD std::array<double const*, 3>
    ptrs(std::array<std::shared_ptr<damn::core::Span<double>>, 3> const& v) const
    {
        return {v[0]->data(), v[1]->data(), v[2]->data()};
    }

    std::shared_ptr<damn::core::Span<double>> const _n, _ppc;
    std::array<std::shared_ptr<damn::core::Span<double>>, 3> _B, _V, _Vth;
};




template<typename Particles_t, typename Layout>
std::size_t
MaxwellianParticleInitializer<Particles_t, Layout>::nbrParticles(Layout const& layout) const
{
    if (densityBackground_ >= densityCutOff_)
        return layout.cells.size() * nbrParticlePerCell_;

    auto const cells = coordinate_vectors(
        layout.dx, layout.cells, mkn::kul::ConstArray<QtyCentering, dimension>(QtyCentering::dual));
    auto const n = std::apply([&](auto const&... c) { return density_(c...); }, cells.spans());

    std::size_t populated = 0;
    for (std::size_t i = 0; i < n->size(); ++i)
        populated += (n->data()[i] + densityBackground_ >= densityCutOff_);
    return populated * nbrParticlePerCell_;
}


template<typename Particles_t, typename Layout>
void MaxwellianParticleInitializer<Particles_t, Layout>::loadParticles(Particles_t& particles,
                                                                       Layout const& layout) const
{
    auto const deltas = [](auto& pos, auto& gen) {
        return mkn::kul::for_N_make_array<dimension>([&](auto) { return pos(gen); });
    };


    auto const cells = coordinate_vectors(
        layout.dx, layout.cells, mkn::kul::ConstArray<QtyCentering, dimension>(QtyCentering::dual));

    auto const fns = std::make_from_tuple<MaxwellianInitFunctions>(std::tuple_cat(
        std::forward_as_tuple(density_, bulkVelocity_, thermalVelocity_, magneticField_, basis_),
        cells.spans()));

    auto const [density, V, Vth] = fns();
    auto const seed              = baseSeed(rngSeed_);

    // the density: the function's values plus the background
    std::vector<double> with_background;
    double const* n = density;
    if (densityBackground_ != 0.)
    {
        with_background.assign(density, density + cells.size());
        for (auto& v : with_background)
            v += densityBackground_;
        n = with_background.data();
    }

    // GPU_UNIFIED SoAVXChunked: filled on the host in a CPU array, then one bulk copy per
    //  column (append_columns), instead of a host write into managed memory per particle
    bool constexpr static staged = Particles_t::layout_mode == LayoutMode::SoAVXChunked
                                   and Particles_t::alloc_mode == damn::AllocatorMode::GPU_UNIFIED;
    using Staging_t              = std::conditional_t<
        staged, Particles<Particles_t::options.with_alloc(damn::AllocatorMode::CPU)>, Particles_t>;
    auto staging = [&]() {
        if constexpr (staged)
            return Staging_t{particles.box()};
        else
            return std::ref(particles);
    }();
    Staging_t& out = staging;

    if constexpr (is_tiled(Particles_t::layout_mode))
    { // per tile storage: reserve where the particles will land
        core::reserve(particles, layout.cells, nbrParticlePerCell_);
    }
    else
    { // flat storage: cells above the density cutoff, plus room for incoming particles
        auto const expected_size = std::accumulate(
            n, n + cells.size(), std::size_t{0}, [&](auto const& sum, auto const& density_value) {
                return (density_value >= densityCutOff_) ? sum + nbrParticlePerCell_ : sum;
            });

        auto const incoming_estimate
            = (surface_cell_count_(layout.cells) * (nbrParticlePerCell_ * over_alloc_factor_));
        if constexpr (requires { out.reserve(expected_size); }) // not per cell (AoSPCTS)
            out.reserve(expected_size + incoming_estimate);
    }

    for (std::size_t flatCellIdx = 0; flatCellIdx < cells.size(); ++flatCellIdx)
    {
        if (n[flatCellIdx] < densityCutOff_)
            continue;

        auto const cellWeight = n[flatCellIdx] / nbrParticlePerCell_;
        auto const iCell      = cells.points[flatCellIdx].toArray();
        mkn::kul::SplitMix64 randGen{
            damn::core::cell_key(seed, populationKey_, layout.level, iCell)};
        ParticleDeltaDistribution<double> deltaDistrib;
        std::array<double, 3> particleVelocity;
        std::array<std::array<double, 3>, 3> basis{}; // set and read only for Basis::Magnetic

        if (basis_ == Basis::Magnetic)
        {
            auto const B = fns.B();
            localMagneticBasis({B[0][flatCellIdx], B[1][flatCellIdx], B[2][flatCellIdx]}, basis);
        }

        for (std::uint32_t ipart = 0; ipart < nbrParticlePerCell_; ++ipart)
        {
            maxwellianVelocity({V[0][flatCellIdx], V[1][flatCellIdx], V[2][flatCellIdx]},
                               {Vth[0][flatCellIdx], Vth[1][flatCellIdx], Vth[2][flatCellIdx]}, //
                               randGen, particleVelocity);

            if (basis_ == Basis::Magnetic)
                particleVelocity = basisTransform(basis, particleVelocity);

            out.emplace_back(cellWeight, iCell, deltas(deltaDistrib, randGen), particleVelocity);
        }
    }

    if constexpr (staged)
        particles.append_columns(out);
    particles.template on_appended<ParticleType::Domain>();
}

} // namespace plamn::core


#endif
