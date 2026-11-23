#ifndef PLAMN_CORE_DATA_PARTICLES_PARTICLES_AOS4SOAVX_CHUNKED_HPP
#define PLAMN_CORE_DATA_PARTICLES_PARTICLES_AOS4SOAVX_CHUNKED_HPP

// AoS4SoAVXChunked: AoS sorted by cell, with per cell chunks (particles_chunked.hpp: the chunks'
//  bookkeeping, as SoAVXChunked's), no CellMap: a cell's particles are one contiguous run, staged
//  into SoAVX blocks for the push. The particle structures' part: a slot moved or written whole,
//  sort() scattering the particles into a second array, swapped in. Host memory only for now

#include "damn/core/vector.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/def/damn_config.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/data/particles/arrays/particles_aos.hpp"
#include "plamn/core/data/particles/arrays/particles_chunked.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

namespace plamn::core
{

template<auto alloc_mode>
struct ChunkContainers
{
    template<typename T>
    using type = typename damn::Vector<T, alloc_mode, 1>::vector_t;
};


template<auto opts>
class AoS4SoAVXChunkedSpan : public AoSSpan<opts>
{
    using Super = AoSSpan<opts>;

public:
    using box_t = damn::core::Box<int, opts.dim>;

    AoS4SoAVXChunkedSpan() = default;

    template<typename Particles_t>
    AoS4SoAVXChunkedSpan(Particles_t& particles)
        : Super{particles}
        , box_{particles.box()}
        , starts_{particles.starts_data()}
        , sizes_{particles.sizes_data()}
        , sorted_{particles.sorted()}
    {
    }

    auto& box() const DAMN_ALL { return box_; }
    auto starts_data() const DAMN_ALL { return starts_; }
    auto sizes_data() const DAMN_ALL { return sizes_; }
    bool sorted() const DAMN_ALL { return sorted_; }

    box_t box_;
    chunk_offset_t const* starts_ = nullptr;
    chunk_offset_t const* sizes_  = nullptr;
    bool sorted_                  = false;
};


template<auto opts>
class AoS4SoAVXChunkedVector
    : public AoSVector<opts>,
      public ChunkState<opts.dim, ChunkContainers<opts.alloc_mode>::template type>
{
    using Super  = AoSVector<opts>;
    using Chunks = ChunkState<opts.dim, ChunkContainers<opts.alloc_mode>::template type>;

public:
    static_assert(opts.alloc_mode == damn::AllocatorMode::CPU, "AoS4SoAVXChunked: host only");
    using box_t = damn::core::Box<int, opts.dim>;

    AoS4SoAVXChunkedVector(box_t const& box = {}, std::size_t const size = 0)
        : Super(size)
        , Chunks{box}
    {
    }

    bool sorted() const { return this->sorted_ and this->sorted_size_ == this->size(); }

    Super scratch_; // the particles sort() scatters into, then swaps in
};


template<template<auto> typename Super_, auto opts>
class AoS4SoAVXChunkedParticles
    : public AoSParticles<Super_, opts>,
      public ChunkedOps<AoS4SoAVXChunkedParticles<Super_, opts>, opts.dim, opts.alloc_mode>
{
    using Super = AoSParticles<Super_, opts>;
    using This  = AoS4SoAVXChunkedParticles;
    using Ops   = ChunkedOps<This, opts.dim, opts.alloc_mode>;
    friend Ops;

public:
    auto static constexpr dimension    = Super::dimension;
    auto static constexpr alloc_mode   = Super::alloc_mode;
    auto static constexpr layout_mode  = LayoutMode::AoS4SoAVXChunked;
    auto static constexpr storage_mode = Super::storage_mode;

    using Particle_t = Super::Particle_t;
    using box_t      = damn::core::Box<int, dimension>;
    using Ops::nbr_particles_in;
    using Super::particles_;
    using Super::size;

    template<typename... Args>
    AoS4SoAVXChunkedParticles(Args&&... args)
        requires(self_excluding_constructible<This, Super, Args...>())
        : Super{std::forward<Args>(args)...}
    {
    }

    AoS4SoAVXChunkedParticles(AoS4SoAVXChunkedParticles const&)            = default;
    AoS4SoAVXChunkedParticles(AoS4SoAVXChunkedParticles&&)                 = default;
    AoS4SoAVXChunkedParticles& operator=(AoS4SoAVXChunkedParticles const&) = default;
    AoS4SoAVXChunkedParticles& operator=(AoS4SoAVXChunkedParticles&&)      = default;

    auto copy(std::size_t const i) const { return particles_[i]; }

    // any mutation but the chunks' own leaves them stale until the next sort()
    void resize(std::size_t const s)
        requires(storage_mode == StorageMode::VECTOR)
    {
        this->unsort();
        Super::resize(s);
    }

    void clear()
        requires(storage_mode == StorageMode::VECTOR)
    {
        this->unsort();
        Super::clear();
    }

    template<typename P>
    void push_back(P const& particle)
        requires(storage_mode == StorageMode::VECTOR)
    {
        this->unsort();
        Super::push_back(as_particle_(particle));
    }

    template<typename... Args>
    auto& emplace_back(Args&&... args)
        requires(storage_mode == StorageMode::VECTOR)
    {
        this->unsort();
        return Super::emplace_back(std::forward<Args>(args)...);
    }

    template<typename Src>
    void append(Src const& src, std::size_t const start, std::size_t const siz)
        requires(storage_mode == StorageMode::VECTOR)
    {
        this->unsort();
        Super::append(src, start, siz);
    }

    auto erase(auto a, auto b)
        requires(storage_mode == StorageMode::VECTOR)
    {
        this->unsort();
        return Super::erase(a, b);
    }

    // the chunks are kept eagerly: rebuilt after every change through these hooks (appends, the
    //  post push sync), queries assert sorted()
    template<auto type>
    void on_appended()
        requires(storage_mode == StorageMode::VECTOR)
    {
        this->sort();
    }

    template<auto type>
    void on_moved(auto&&... /*stream*/)
        requires(storage_mode == StorageMode::VECTOR)
    {
        this->sort();
    }

private:
    // a particle from anything with weight()/iCell()/delta()/v()
    static Particle_t as_particle_(auto const& p)
    {
        if constexpr (std::is_same_v<std::decay_t<decltype(p)>, Particle_t>)
            return p;
        else
        {
            Particle_t q;
            q.weight() = p.weight();
            for (std::size_t d = 0; d < dimension; ++d)
            {
                q.iCell()[d] = p.iCell()[d];
                q.delta()[d] = p.delta()[d];
            }
            for (std::size_t d = 0; d < 3; ++d)
                q.v()[d] = p.v()[d];
            return q;
        }
    }

    // ChunkedOps' storage hooks (particles_chunked.hpp): whole particles
    double weight_at_(std::size_t const i) const { return particles_[i].weight(); }
    void set_weight_(std::size_t const i, double const w) { particles_[i].weight() = w; }
    std::array<int, dimension> cell_at_(std::size_t const i) const
    {
        std::array<int, dimension> cell;
        for (std::size_t d = 0; d < dimension; ++d)
            cell[d] = particles_[i].iCell()[d];
        return cell;
    }

    void move_slot_(std::size_t const to, std::size_t const from)
    {
        particles_[to] = particles_[from];
    }
    void write_slot_(std::size_t const i, auto const& p) { particles_[i] = as_particle_(p); }
    void write_filler_(std::size_t const i, std::array<int, dimension> const& cell)
    {
        Particle_t filler; // weight 0, at rest
        filler.weight() = 0;
        for (std::size_t d = 0; d < dimension; ++d)
        {
            filler.iCell()[d] = cell[d];
            filler.delta()[d] = 0;
        }
        for (std::size_t d = 0; d < 3; ++d)
            filler.v()[d] = 0;
        particles_[i] = filler;
    }

    auto slot_weights_() const
    {
        auto const* ps = particles_.data();
        return [=] DAMN_ALL(std::size_t const i) { return ps[i].weight(); };
    }
    auto slot_cells_() const
    {
        auto const* ps = particles_.data();
        return [=] DAMN_ALL(std::size_t const i) {
            std::array<int, dimension> cell;
            for (std::size_t d = 0; d < dimension; ++d)
                cell[d] = ps[i].iCell()[d];
            return cell;
        };
    }

    // the particles scattered into the scratch array's slots to(i), swapped in
    void scatter_sorted_(auto const& to, std::size_t const n, std::size_t const slots_total)
    {
        auto& scratch = this->scratch_;
        scratch.resize(slots_total);
        auto const* src = particles_.data();
        auto* dst       = scratch.data();
        for (std::size_t i = 0; i < n; ++i)
            if (auto const slot = to(i); slot != std::numeric_limits<std::size_t>::max())
                dst[slot] = src[i];
        std::swap(static_cast<AoSVector<opts>&>(*this), scratch);
    }
};


template<auto opts>
using AoS4SoAVXChunkedVectorParticles = AoS4SoAVXChunkedParticles<AoS4SoAVXChunkedVector, opts>;

} // namespace plamn::core

#endif /* PLAMN_CORE_DATA_PARTICLES_PARTICLES_AOS4SOAVX_CHUNKED_HPP */
