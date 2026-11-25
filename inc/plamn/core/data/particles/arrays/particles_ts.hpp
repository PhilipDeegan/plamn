#ifndef PLAMN_CORE_DATA_PARTICLES_PARTICLES_TILE_SET_HPP
#define PLAMN_CORE_DATA_PARTICLES_PARTICLES_TILE_SET_HPP

#include "damn/core/math/sort.hpp"
#include "damn/core/def/damn_config.hpp"
#include "damn/core/def/detail/thrust.hpp" // IWYU pragma: keep
#include "damn/core/operators.hpp"
#include "damn/core/data/tiles/tile_set.hpp"
#include "damn/core/data/tiles/tile_set_traversal.hpp"
#include "damn/core/memory/span.hpp"
#include "damn/core/exec/kernels.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/data/ndarray/ndarray_vector.hpp"
#include "damn/core/sys/monitoring.hpp"
#include "damn/core/data/ndarray/ndarray_view.hpp"
#include "damn/core/def/detail/mkn_gpu.hpp"
#include "damn/core/logger.hpp"
#include "mkn/kul/std/array.hpp"
#include "mkn/kul/math.hpp"
#include "mkn/kul/for.hpp"
#include "mkn/kul/meta.hpp"
#include "damn/core/vector.hpp"
#include "plamn/core/def.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "mkn/gpu.hpp"

#include <cassert>
#include <tuple>
#include <stdexcept>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace damn::core
{
template<typename Type, std::size_t dim>
class Point;
} // namespace damn::core

namespace plamn::core
{


template<typename Particles>
class ParticlesTile : public damn::core::Box<std::int32_t, Particles::dimension>
{
    auto constexpr static dim = Particles::dimension;
    using This                = ParticlesTile<Particles>;
    using Super               = damn::core::Box<std::int32_t, dim>;

    template<typename P>
    friend class ParticlesTile;


public:
    ParticlesTile(Super const& box, std::size_t const ghost_cells)
        : Super{box}
        , particles{make_particles<Particles>(box, ghost_cells)}
    {
    }

    ParticlesTile(ParticlesTile const&)            = default;
    ParticlesTile(ParticlesTile&&)                 = default;
    ParticlesTile& operator=(ParticlesTile const&) = default;
    ParticlesTile& operator=(ParticlesTile&&)      = default;

    template<typename Ps>
    ParticlesTile(ParticlesTile<Ps>& tile)
        : Super{tile}
        , particles{tile()}
    {
    }


    Super& operator*() DAMN_ALL { return *this; }
    Super const& operator*() const DAMN_ALL { return *this; }

    auto& operator()() DAMN_ALL { return particles; }
    auto& operator()() const DAMN_ALL { return particles; }

    auto& link(std::size_t const idx) { return _links[idx]; }
    auto& links() { return _links; }
    auto& links() const { return _links; }


    template<typename P>
    void reset(ParticlesTile<P>& tile)
    {
        (*this)().reset(tile());
    }

private:
    Particles particles;

    std::array<ParticlesTile*, 7> _links = mkn::kul::ConstArray<ParticlesTile*, 7>(nullptr);
};




template<typename Particles>
class TileSetSpan
{
    friend struct TileSetParticlesService;
    template<typename>
    friend struct CrossTileCopyDAO;
    template<typename>
    friend struct TileSetBlockDAO;

protected:
    using SIZE_T = damn::core::default_span_size_t;

public:
    auto static constexpr alloc_mode = Particles::alloc_mode;
    auto static constexpr dim        = Particles::dimension;

    using This               = TileSetSpan<Particles>;
    using lobox_t            = damn::core::Box<std::uint32_t, dim>;
    using per_tile_particles = Particles;

private:
    using locell_t = std::array<std::uint32_t, dim>;

    template<typename TileSetArray>
    auto resolve(TileSetArray& arr)
    {
        if constexpr (TileSetArray::storage_mode == StorageMode::SPAN)
            return arr.particles_;
        else
            return arr.particles_views_.make_view();
    }

    template<typename TileSetArray>
    auto resolve_gaps(TileSetArray& arr)
    {
        if constexpr (TileSetArray::storage_mode == StorageMode::SPAN)
            return arr.gaps_;
        else
            return *arr.gap_views_;
    }

public:
    auto static constexpr dimension    = dim;
    auto static constexpr storage_mode = StorageMode::SPAN;
    using Particle_t                   = typename ParticleDefaults<dim>::Particle_t;

    template<typename TileSetArray>
    TileSetSpan(TileSetArray& arr)
        : particles_{resolve(arr)}
        , gaps_{resolve_gaps(arr)}
        , gap_idx_{arr.gap_idx_}
        , add_into_{arr.add_into_}
        , cell_size_{arr.cell_size_}
        , cap_{arr.cap_}
        , left_{arr.left_}
        , size_{arr.size()}
        , box_{arr.box_}
        , ghost_box_{arr.ghost_box()}
        , local_ghost_box_{arr.local_box()}
    {
    }

    auto size() const DAMN_ALL { return size_; }
    auto size(std::size_t const& idx) const DAMN_ALL { return particles_.data()[idx].size(); }
    void resize(std::size_t s) { particles_.s = s; }


    template<typename TileSetArray>
    void reset(TileSetArray& arr)
    {
        particles_.reset(arr.particles_);
        size_ = arr.size();
    }

    auto& box() const DAMN_ALL { return box_; }
    auto& ghost_box() const DAMN_ALL { return ghost_box_; }
    auto local_cell(std::array<int, dim> const& icell) const DAMN_ALL
    {
        return as_local_cell(ghost_box_, icell);
    }
    auto local_cell(damn::core::Point<int, dim> const& icell) const DAMN_ALL
    {
        return local_cell(icell.toArray());
    }

    auto& local_box() const DAMN_ALL { return local_ghost_box_; }

    auto local_box(damn::core::Box<int, dim> const& from) const DAMN_ALL
    {
        return lobox_t{local_cell(from.lower), local_cell(from.upper)};
    }

    auto& operator()() DAMN_ALL { return particles_; }
    auto& operator()() const DAMN_ALL { return particles_; }
    auto& operator()(locell_t const& cell) DAMN_ALL { return (*particles_.at(cell))(); }
    auto& operator()(locell_t const& cell) const DAMN_ALL { return (*particles_.at(cell))(); }


    template<auto type, typename... Args>
    void sync(Args&&... args) DAMN_ALL;


    void clear()
    {
        for (auto& tile : particles_)
            tile().clear();
        size_ = 0;
    }


    auto local_tile_cell(std::array<int, dim> const& cell) const DAMN_ALL
    {
        DAMN_ASSERT(particles_.at(local_cell(cell)));
        return local_cell((*particles_.at(local_cell(cell))).lower);
    }

protected:
    void sync_tile_add_new(std::size_t const tidx) DAMN_ALL;
    void sync_tile_rm_left(std::size_t const tidx) DAMN_ALL;


    void static sort(auto from, auto to) DAMN_ALL { damn::core::seq_sort(from, to); }


    damn::core::TileSetView<ParticlesTile<Particles>> particles_;
    damn::core::NdArrayView<dim, damn::core::Span<std::size_t>> gaps_;
    damn::core::NdArrayView<dim, SIZE_T> gap_idx_, add_into_, cell_size_, cap_, left_;

    std::size_t size_;

    damn::core::Box<int, dim> box_, ghost_box_;
    lobox_t local_ghost_box_;

}; // TileSetSpan


template<typename Particles>
class TileSetVector
{
    using This                    = TileSetVector<Particles>;
    bool static constexpr c_order = true;

protected:
    using SIZE_T = damn::core::default_span_size_t;

private:
    template<typename P>
    friend class TileSetSpan;

    friend struct TileSetParticlesService;

    std::uint8_t constexpr static alloc_impl()
    {
        if (Particles::alloc_mode == damn::AllocatorMode::GPU_UNIFIED)
            return 1;
        return 1;
    }

public:
    auto static constexpr dim          = Particles::dimension;
    auto static constexpr alloc_mode   = Particles::alloc_mode;
    auto static constexpr layout_mode  = Particles::layout_mode;
    auto static constexpr storage_mode = StorageMode::VECTOR;
    auto static constexpr dimension    = dim;
    using box_t                        = damn::core::Box<int, dim>;
    using lobox_t                      = damn::core::Box<std::uint32_t, dim>;
    using locell_t                     = std::array<std::uint32_t, dim>;
    using Particle_t                   = ParticleDefaults<dim>::Particle_t;
    using value_type                   = Particle_t;
    using PSpan_t                      = Particles::view_t;
    using per_tile_particles           = Particles;

    template<typename T>
    using vec_helper = damn::Vector<T, alloc_mode, alloc_impl()>;

    template<typename T>
    using nd_array_t = damn::core::NdArrayVector<dim, T, c_order, alloc_mode>;

    using size_t_vector = typename vec_helper<std::size_t>::vector_t;

    using VecTile = ParticlesTile<Particles>;
    using SpnTile = ParticlesTile<PSpan_t>;


    TileSetVector(box_t const& box, auto const ghost_cells)
        : ghost_cells_{ghost_cells}
        , box_{box}
        , ghost_box_{damn::core::grow(box, ghost_cells)}
    {
        cell_size_.zero();
        gap_idx_.zero();
        add_into_.zero();
        left_.zero();
        cap_.zero();

        reset_views();
        damn::core::TileSet<VecTile, alloc_mode>::build_links(particles_);
        damn::core::TileSet<SpnTile, alloc_mode>::build_links(particles_views_);
    }

    TileSetVector(TileSetVector&& that)
        : ghost_cells_{that.ghost_cells_}
        , box_{that.box_}
        , ghost_box_{that.ghost_box_}
        , particles_{std::move(that.particles_)}
        , total_size{that.total_size}
    {
        on_appended(); // without std::swap does not work well
    }

    TileSetVector(TileSetVector const& that)
        : ghost_cells_{that.ghost_cells_}
        , box_{that.box_}
        , ghost_box_{that.ghost_box_}
        , particles_{that.particles_}
        , total_size{that.total_size}
    {
        on_appended();
    }


    // not = default: the defaulted memberwise copy/move leaves particles_views_
    // stale (it isn't rebuilt the way on_appended()/reset_views() does at the end of the
    // copy/move constructors), so reconstruct in place via those constructors instead
    TileSetVector& operator=(TileSetVector&& that)
    {
        if (this == &that)
            return *this;
        this->~TileSetVector();
        new (this) TileSetVector(std::move(that));
        return *this;
    }
    TileSetVector& operator=(TileSetVector const& that)
    {
        if (this == &that)
            return *this;
        this->~TileSetVector();
        new (this) TileSetVector(that);
        return *this;
    }

    auto size() const { return total_size; }
    auto size(std::array<std::uint32_t, dim> const& icell) const { return cell_size_(icell); }
    auto size(std::size_t const& idx) const { return cell_size_.data()[idx]; }

    void _inc(locell_t const& locell)
    {
        ++total_size;
        ++cell_size_(locell);
    }

    template<bool inc_ = true>
    void emplace_back(Particle_t const& p)
    {
        auto const locell = local_cell(p.iCell());
        assert(particles_.at(locell));
        (*particles_.at(locell))().emplace_back(p);
        if constexpr (inc_)
            _inc(locell);
    }

    template<bool inc_ = true>
    void emplace_back(Particles& dst, Particles const& src, std::size_t const& idx)
    {
        dst.emplace_back(src, idx);
    }

    template<typename... Args>
    void emplace_back(double const weight, Args&&... args)
    {
        this->emplace_back(Particle_t{weight, args...});
    }

    template<typename V>
    static auto& get_vec(V& v)
    {
        if constexpr (damn::is_mkn_gpu_vector_v<std::decay_t<V>>)
            DAMN_WITH_MKN_GPU(return mkn::gpu::as_super(v));
        else
            return v;
    }

    void push_back(Particle_t&& p) { emplace_back(p); }
    void push_back(Particle_t const& p) { emplace_back(p); }


    void reset_views()
    {
        damn::core::update_from(reset_particle_views_fn(), particles_views_);
        damn::core::update_from(reset_gap_views_fn(), gap_views_);
    }


    auto reset_gap_views_fn()
    {
        return [&](auto const i) { return damn::core::make_span(*(gaps_.data() + i)); };
    }
    auto reset_particle_views_fn()
    {
        return [&](std::size_t const i) { return SpnTile{particles_[i]}; };
    }



    auto& box() const { return box_; }
    auto& ghost_box() const { return ghost_box_; }

    auto& operator()() { return particles_; }
    auto& operator()() const { return particles_; }
    auto& operator()(locell_t const& cell) { return (*particles_.at(cell))(); }
    // auto& operator()(std::uint32_t const& cell) { return particles_.data() + cell; }
    auto& operator()(locell_t const& cell) const { return (*particles_.at(cell))(); }
    // auto& operator()(std::uint32_t const& cell) const { return particles_.data() + cell; }

    auto& views() { return particles_views_; }
    auto& views() const { return particles_views_; }
    auto& view(locell_t const& cell) { return (*particles_views_.at(cell))(); }

    auto local_cell(std::array<int, dim> const& icell) const
    {
        return as_local_cell(ghost_box_, icell);
    }
    auto local_cell(damn::core::Point<int, dim> const& icell) const { return local_cell(icell.toArray()); }
    auto local_box() const
    {
        return damn::core::box_from_zero_to_upper_minus_one(
            ghost_box_.shape().template toArray<std::uint32_t>());
    }
    auto local_box(damn::core::Box<int, dim> const& from) const
    {
        return lobox_t{local_cell(from.lower), local_cell(from.upper)};
    }

    // refreshes recount/gap-sizing/views after particles were added without going through
    // move_check (a raw append) - see on_moved() for the post-move_check pipeline.
    template<auto type = ParticleType::Domain>
    void on_appended();

    // applies move_check-registered moves: realloc/resize, span-side cross-tile copy, then
    // on_appended() to finalize.
    template<auto type>
    void on_moved(auto&&... args);

    template<auto type>
    void sync_moved(); // realloc ahead of the span-side copy (see sync_check_realloc)

    // GPU_UNIFIED, one block per listed tile: removes the particles pred(particle) selects.
    //  pred runs on the device
    template<typename Pred>
    void erase_if_on_device(std::vector<std::size_t> const& tiles, Pred const pred);

    // GPU_UNIFIED, one block per listed tile: moves the particles pred(particle) selects into
    //  dst, each into the dst tile owning its cell (TileSet::tag_cells_). pred runs on the device
    template<typename Pred>
    void move_if_on_device(This& dst, std::vector<std::size_t> const& tiles, Pred const pred);

    // GPU_UNIFIED: appends n particles from device accessible memory, each into the tile owning
    //  its cell, thread per particle. Every cell must be in the ghost box, as for emplace_back
    template<auto type = ParticleType::Domain>
    void append_on_device(Particle_t const* const particles, std::size_t const n);

    template<auto type>
    void trim();


    void clear()
    {
        for (auto& tile : particles_)
            tile().clear();
        for (auto& tile : particles_views_)
            tile().clear();

        // cell_size_ here is cached per-tile (set in on_appended()), not read back from
        // each tile's own size() - reset it and the rest of the move_check bookkeeping
        // the same way the constructor does, or size(icell) stays stale after clear().
        cell_size_.zero();
        gap_idx_.zero();
        add_into_.zero();
        left_.zero();
        cap_.zero();

        reset_views();
        total_size = 0;
    }


    template<typename V>
    void static resize(V& v, std::size_t const& s, bool const& copy = true)
    {
        damn::resize(v, s, copy);
    }

    template<typename V>
    void static reserve(V& v, std::size_t const& s, bool const& copy = true)
    {
        if (s > v.capacity())
            damn::core::MemoryMonitoring::LOG(__PRETTY_FUNCTION__);

        damn::reserve(v, s, copy);
    }




    auto local_tile_cell(std::array<int, dim> const& cell) const
    {
        return local_cell((*particles_.at(local_cell(cell))).lower);
    }



protected:
    template<auto type>
    void sync_check_realloc();

    auto on_tiles(auto&& fn)
    {
        for (auto& tile : particles_)
            fn(tile);
    }


    std::size_t ghost_cells_;

    damn::core::Box<int, dim> box_, ghost_box_;

    nd_array_t<size_t_vector> gaps_{local_box().shape()};
    nd_array_t<damn::core::Span<std::size_t>> gap_views_{local_box().shape()};

    nd_array_t<SIZE_T> gap_idx_{local_box().shape()};
    nd_array_t<SIZE_T> add_into_{local_box().shape()};
    nd_array_t<SIZE_T> left_{local_box().shape()};
    nd_array_t<SIZE_T> cap_{local_box().shape()};
    nd_array_t<SIZE_T> cell_size_{local_box().shape()};


    // tiles build from amrbox, but `.at()` function maps ghosts box
    damn::core::GhostClampedTileSet<VecTile, alloc_mode> particles_{box_, ghost_cells_, ghost_cells_};
    damn::core::GhostClampedTileSet<SpnTile, alloc_mode> particles_views_
        = damn::core::GhostClampedTileSet<SpnTile, alloc_mode>::make_from(
            [](auto& tile) -> auto& { return tile; }, particles_);

    std::size_t total_size = 0;


}; // TileSetVector<Particles>




template<typename Particles>
template<auto type>
void TileSetVector<Particles>::trim() // change to erase(box)
{
    static_assert(std::is_same_v<decltype(type), ParticleType>);
}



template<typename Particles>
template<auto type>
void TileSetVector<Particles>::sync_check_realloc()
{
    DAMN_LOG_SCOPE(3, "TileSetVector::sync_check_realloc");

    for (std::size_t i = 0; i < particles_.size(); ++i)
    {
        auto const lix       = local_cell(particles_[i].lower);
        auto const& nu       = add_into_(lix);
        auto& real           = particles_[i]();
        auto const& left     = gap_idx_(lix);
        auto const& old_size = real.size();
        cell_size_(lix)      = old_size;
        auto const& new_size = real.size() + nu - left;
        real.reserve(real.size() + nu); // we must add new before removing leavers
        real.resize(new_size);
        cap_(lix) = real.capacity();
        assert(real.size() == cell_size_(lix) + nu - left);
        assert(cap_(lix) >= real.size());
        add_into_(lix) = 0;
    };
}


template<typename Particles>
template<auto type>
void TileSetVector<Particles>::sync_moved()
{
    sync_check_realloc<type>();
    reset_views();

    for (std::size_t i = 0; i < particles_.size(); ++i)
    {
        auto const lix = local_cell(particles_[i].lower);
        particles_views_[i]().resize(cell_size_(lix));
    };
}

template<typename Particles>
template<auto type>
void TileSetVector<Particles>::on_moved(auto&&... args)
{
    sync_moved<type>();                                       // realloc + resize
    TileSetSpan<PSpan_t>{*this}.template sync<type>(args...); // cross-tile copy

    on_appended<type>(); // finalize: recount, size gaps, reset views
}

template<typename Particles>
template<auto type>
void TileSetVector<Particles>::on_appended()
{
    static_assert(mkn::kul::all_are<ParticleType>(type));
    static_assert(type != ParticleType::All);
    DAMN_LOG_SCOPE(3, "TileSetVector::on_appended");

    total_size = 0;
    for (auto const& tile : particles_)
    {
        total_size += tile().size();
        cell_size_(local_cell(tile.lower)) = tile().size();
    }

    on_tiles([&](auto& tile) {
        auto const lix  = local_cell(tile.lower);
        auto const& cs  = cell_size_(lix);
        auto const& cap = tile().capacity();
        auto& gaps      = gaps_(lix);
        if (gaps.size() < cs)
        {
            reserve(gaps, cap, false);
            resize(gaps, cs, false);
        }
        cap_(lix) = cap;
        tile().template on_appended<type>(); // the tile's own, e.g. AoSMapped's cell map
    });

    DAMN_LOG_SCOPE(3, "TileSetVector::sync::reset_views");
    reset_views();
}


template<typename Super_>
struct TileSetParticles : public Super_
{
    using Super              = Super_;
    using This               = TileSetParticles<Super>;
    using Particle_t         = typename Super::Particle_t;
    using per_tile_particles = typename Super::per_tile_particles;

    auto static constexpr alloc_mode   = Super::alloc_mode;
    auto static constexpr dimension    = Super::dimension;
    auto static constexpr storage_mode = Super::storage_mode;
    auto static constexpr size_of_particle() { return sizeof(Particle_t); }

    using Super::local_box;
    using Super::particles_;
    using Super::size;


    // spans own nothing: their copies and moves the defaulted, trivial ones (device kernels
    //  capture them as plain data)
    static constexpr bool is_span = storage_mode == StorageMode::SPAN;

    TileSetParticles(TileSetParticles&& that)
        requires(!is_span)
        : Super{std::forward<Super>(that)}
    {
    }
    TileSetParticles(TileSetParticles&&)
        requires(is_span)
    = default;

    TileSetParticles& operator=(TileSetParticles&&)      = default;
    TileSetParticles(TileSetParticles const&)            = default;
    TileSetParticles& operator=(TileSetParticles const&) = default;

    template<typename... Args>
    TileSetParticles(Args&&... args)
        requires std::is_constructible_v<Super, Args&&...>
    DAMN_ALL : Super{std::forward<Args>(args)...}
    {
    }

    auto data() const DAMN_ALL { return particles_.data(); }
    auto data() DAMN_ALL { return particles_.data(); }


    // registers a particle moving to another tile. Domain particles are only checked while in
    //  the patch box, level ghosts anywhere: they live in their cell's clamp-owner tile
    //  (TileSet::tag_cells_), including another tile's ghost cells. Particles leaving the ghost
    //  box stay where they are, post_move_sync deletes them
    template<auto particle_type>
    auto& move_check(auto const& pt, std::size_t const idx, auto& particle) DAMN_ALL
    {
        using enum ParticleType;
        static_assert(mkn::kul::any_in(particle_type, Domain, LevelGhost));
        auto const& newcell = particle.iCell();
        if constexpr (particle_type == LevelGhost)
            if (!isIn(newcell, Super::ghost_box()))
                return *this;
        if (mkn::kul::array_equals(Super::local_tile_cell(newcell), pt.tile_cell))
            return *this;

        bool constexpr static ATOMIC = true;
        bool constexpr static GPU    = alloc_mode == damn::AllocatorMode::GPU_UNIFIED;
        using Op                     = damn::core::Operators<typename Super::SIZE_T, ATOMIC, GPU>;

        auto& gidx      = Super::gap_idx_(pt.tile_cell);
        auto const nidx = Op{gidx}.increment_return_old();
        auto& gaps      = Super::gaps_(pt.tile_cell);
        assert(nidx < gaps.size());
        gaps[nidx] = idx;

        if (isIn(newcell, Super::ghost_box()))
        {
            auto const nc = Super::local_tile_cell(newcell);
            Op{Super::add_into_(nc)}.increment_return_old();
        }

        return *this;
    }

    auto& evictor() {}



    void print() const {}
    void check() const;


    auto max_size() const
    {
        return mkn::kul::math::max_from(
            this->particles_, [](auto const& v, auto const& i) { return v.data()[i].size(); });
    }

    auto nbr_particles_in(std::array<int, dimension> const arr) const
    {
        auto const& tile = *particles_.at(Super::local_cell(arr));
        return tile().size() / tile.size(); // confusing? :)
    }

    auto nbr_particles_in(damn::core::Box<int, dimension> const box) const
    {
        std::size_t n_particles = 0;

        damn::core::traverse_tiles(this->particles_, box, [&](auto& tile) {
            auto overlap = box * damn::core::grow(tile, this->ghost_cells_);
            assert(overlap);
            damn::core::for_each_point(*overlap, [&](auto const& bix) {
                n_particles += this->cell_size_(this->local_cell(bix));
            });
        });

        return n_particles;
    }


}; // TileSetParticles<Super>


template<typename Particles>
template<auto type, typename... Args>
void TileSetSpan<Particles>::sync(Args&&... args) DAMN_ALL
{
    DAMN_LOG_SCOPE(3, "TileSetSpan::sync(stream)");
    static constexpr auto alloc_mode = Particles::alloc_mode;

    auto& view = *this;

    if constexpr (alloc_mode == damn::AllocatorMode::CPU)
    {
        for (std::size_t tidx = 0; tidx < particles_.size(); ++tidx)
            sync_tile_add_new(tidx);

        for (std::size_t tidx = 0; tidx < particles_.size(); ++tidx)
            sync_tile_rm_left(tidx);
    }
    else if (alloc_mode == damn::AllocatorMode::GPU_UNIFIED)
    {
        if (!DAMN_HAVE_MKN_GPU)
            throw std::runtime_error("no gpu impl");

        // two launches on the stream: every tile's appends must land before any tile swap-pops
        //  its leavers, which can move a just appended particle. __syncthreads between the
        //  passes only orders one block, tiles span several
        DAMN_WITH_MKN_GPU({
            auto const launch = [&](auto&& fn) { // on args' stream, else the default stream
                if constexpr (sizeof...(args) == 0)
                    damn::kernel::launch(particles_.size(), fn);
                else
                {
                    auto const& [stream] = std::forward_as_tuple(args...);
                    damn::kernel::launch(stream, particles_.size(), fn);
                }
            };
            launch([=] DAMN_ALL() mutable { view.sync_tile_add_new(damn::kernel::idx()); });
            launch([=] DAMN_ALL() mutable { view.sync_tile_rm_left(damn::kernel::idx()); });
        })
    }
    else
        throw std::runtime_error(__func__);
}




template<typename Particles>
struct CrossTileCopyDAO
{
    bool constexpr static ATOMIC = true;
    bool constexpr static GPU    = Particles::alloc_mode == damn::AllocatorMode::GPU_UNIFIED;
    using Op                     = damn::core::Operators<damn::core::default_span_size_t, ATOMIC, GPU>;
    using Tile                   = ParticlesTile<typename Particles::per_tile_particles>;

    Particles& ps;
    std::size_t src_tile_idx;
    Tile& tile = ps()[src_tile_idx];

    void copy_in() DAMN_ALL
    {
        auto& real         = tile();
        auto const bix     = ps.local_cell(tile.lower);
        auto const& n_gaps = ps.gap_idx_(bix);
        {
            auto& gaps = ps.gaps_(bix);
            ps.sort(gaps.data(), gaps.data() + n_gaps /*, std::greater<>()*/);
        }
        auto& left       = ps.left_(bix);
        auto const& gaps = ps.gaps_(bix);
        for (std::size_t i = 0; i < n_gaps; ++i)
        {
            auto const& gidx    = gaps[n_gaps - (1 + i)];
            auto const& newcell = ps.local_tile_cell(real.iCell(gidx));
            auto& ntile         = (*ps().at(newcell));
            auto& nparts        = ntile();
            auto const npidx    = next_index(nparts.size(), nparts.size_address());
            DAMN_ASSERT(npidx < ps.cap_(newcell));
            DAMN_ASSERT(&ntile != &tile);
            // ntile owns the new cell: its box for domain particles, a clamp-owned ghost cell
            //  for level ghosts
            DAMN_ASSERT(ps().at(ps.local_cell(real.iCell(gidx))) == &ntile);
            nparts.assign(real, gidx, npidx);
            ++left;
        }
        DAMN_ASSERT(left == n_gaps);
    }

    void rm_left() DAMN_ALL
    {
        auto& real       = tile();
        auto const bix   = ps.local_cell(tile.lower);
        auto const& gaps = ps.gaps_(bix);
        auto& left       = ps.left_(bix);
        auto& gaps_size  = ps.gap_idx_(bix);
        while (left)
        {
            auto const& rsize = real.size();
            auto const& pidx  = gaps[gaps_size - 1];
            if (pidx != rsize - 1)
                real.assign(rsize - 1, pidx);
            real.pop_back();
            --gaps_size;
            --left;
        }
        DAMN_ASSERT(left == 0);
        DAMN_ASSERT(gaps_size == 0);
    }

    auto static next_index(auto npidx, auto const size_address) DAMN_ALL
    {
        while (true)
        {
            auto inc = npidx + 1;
            auto old = Op::compare_and_swap(size_address, npidx, inc);
            if (npidx != old)
            {
                ++npidx;
                continue;
            }
            else
                break;
        }
        return npidx;
    }
};



// block per tile operations on a GPU_UNIFIED tile set view: the block's threads stride the
//  tile's particles, registering gaps (atomics), then one thread swap-pops them
template<typename Span_t>
struct TileSetBlockDAO
{
    bool constexpr static ATOMIC = true;
    bool constexpr static GPU    = Span_t::alloc_mode == damn::AllocatorMode::GPU_UNIFIED;
    using Op                     = damn::core::Operators<damn::core::default_span_size_t, ATOMIC, GPU>;

    static auto tile_lix(Span_t& ps, std::size_t const tidx) DAMN_ALL
    {
        return ps.local_cell(ps()[tidx].lower);
    }

    // registers particle idx of tile lix for removal
    static void gap(Span_t& ps, auto const& lix, std::size_t const idx) DAMN_ALL
    {
        auto const k = Op{ps.gap_idx_(lix)}.increment_return_old();
        DAMN_ASSERT(k < ps.gaps_(lix).size());
        ps.gaps_(lix)[k] = idx;
    }

    // removes the registered particles of tile tidx: descending swap-pop, one thread
    static void remove_gaps(Span_t& ps, std::size_t const tidx) DAMN_ALL
    {
        auto& real   = ps()[tidx]();
        auto const l = tile_lix(ps, tidx);
        auto& n      = ps.gap_idx_(l);
        auto& gaps   = ps.gaps_(l);
        ps.sort(gaps.data(), gaps.data() + n);
        for (auto j = n; j-- > 0;)
        {
            auto const pidx = gaps[j];
            auto const last = real.size() - 1;
            if (pidx != last)
                real.assign(last, pidx);
            real.pop_back();
        }
        n = 0;
    }

    // block threads: registers every particle of tile tidx that pred selects
    template<typename Launcher>
    static void gap_if(Span_t& ps, std::size_t const tidx, auto const& pred) DAMN_ALL
    {
        auto& real   = ps()[tidx]();
        auto const l = tile_lix(ps, tidx);
        for (std::size_t i = Launcher::thread_idx(); i < real.size(); i += Launcher::threads())
            if (pred(real[i]))
                gap(ps, l, i);
    }

    // block threads: counts, per dst tile, the particles of src tile tidx that pred selects
    template<typename Launcher>
    static void count_into(Span_t& src, Span_t& dst, std::size_t const tidx,
                           auto const& pred) DAMN_ALL
    {
        auto& real = src()[tidx]();
        for (std::size_t i = Launcher::thread_idx(); i < real.size(); i += Launcher::threads())
            if (pred(real[i]))
                Op{dst.add_into_(dst.local_tile_cell(real[i].iCell()))}.increment_return_old();
    }

    // counts particle p against the dst tile owning its cell
    static void count_one(Span_t& dst, auto const& p) DAMN_ALL
    {
        DAMN_ASSERT(isIn(p, dst.ghost_box()));
        Op{dst.add_into_(dst.local_tile_cell(p.iCell()))}.increment_return_old();
    }

    // appends particle p to the dst tile owning its cell, capacity reserved from count_one
    static void append_one(Span_t& dst, auto const& p) DAMN_ALL
    {
        auto& nparts = (*dst().at(dst.local_cell(p.iCell())))();
        auto const npidx
            = CrossTileCopyDAO<Span_t>::next_index(nparts.size(), nparts.size_address());
        nparts[npidx] = p;
    }

    // block threads: appends every particle of src tile tidx that pred selects to the dst tile
    //  owning its cell (capacity reserved from count_into) and registers it for removal
    template<typename Launcher>
    static void move_if(Span_t& src, Span_t& dst, std::size_t const tidx,
                        auto const& pred) DAMN_ALL
    {
        auto& real   = src()[tidx]();
        auto const l = tile_lix(src, tidx);
        for (std::size_t i = Launcher::thread_idx(); i < real.size(); i += Launcher::threads())
            if (pred(real[i]))
            {
                auto& nparts = (*dst().at(dst.local_cell(real[i].iCell())))();
                auto const npidx
                    = CrossTileCopyDAO<Span_t>::next_index(nparts.size(), nparts.size_address());
                nparts.assign(real, i, npidx);
                gap(src, l, i);
            }
    }
};


template<typename Particles>
template<typename Pred>
void TileSetVector<Particles>::erase_if_on_device(std::vector<std::size_t> const& tiles,
                                                  Pred const pred)
{
    static_assert(alloc_mode == damn::AllocatorMode::GPU_UNIFIED);
    if (tiles.empty())
        return;

#if DAMN_HAVE_MKN_GPU
    using Launcher = damn::core::gpu::BlockPerLauncher<true>;
    using View_t   = TileSetSpan<PSpan_t>;
    using DAO      = TileSetBlockDAO<View_t>;

    reset_views();
    damn::core::gpu::Vec_t<std::size_t> const tidxs(tiles.begin(), tiles.end());
    auto const* const tp = tidxs.data();
    auto view            = View_t{*this};

    Launcher{tiles.size()}([=] DAMN_ALL() mutable {
        auto const ti = tp[Launcher::block_idx()];
        DAO::template gap_if<Launcher>(view, ti, pred);
        __syncthreads();
        if (Launcher::thread_idx() == 0)
            DAO::remove_gaps(view, ti);
    });

    for (auto const ti : tiles)
        particles_[ti]().resize(particles_views_[ti]().size());
    on_appended<ParticleType::Domain>();
#else
    throw std::runtime_error("TileSetVector::erase_if_on_device needs mkn.gpu");
#endif
}


template<typename Particles>
template<typename Pred>
void TileSetVector<Particles>::move_if_on_device(This& dst, std::vector<std::size_t> const& tiles,
                                                 Pred const pred)
{
    static_assert(alloc_mode == damn::AllocatorMode::GPU_UNIFIED);
    if (tiles.empty())
        return;

#if DAMN_HAVE_MKN_GPU
    using Launcher = damn::core::gpu::BlockPerLauncher<true>;
    using View_t   = TileSetSpan<PSpan_t>;
    using DAO      = TileSetBlockDAO<View_t>;

    damn::core::gpu::Vec_t<std::size_t> const tidxs(tiles.begin(), tiles.end());
    auto const* const tp = tidxs.data();

    reset_views();
    dst.reset_views();
    { // arrivals per dst tile, into dst.add_into_
        auto src_view = View_t{*this};
        auto dst_view = View_t{dst};
        Launcher{tiles.size()}([=] DAMN_ALL() mutable {
            DAO::template count_into<Launcher>(src_view, dst_view, tp[Launcher::block_idx()],
                                               pred);
        });
    }

    // dst vectors grow by their arrivals, dst views stay at the old sizes for the appends
    dst.template sync_moved<ParticleType::Domain>();

    {
        auto src_view = View_t{*this};
        auto dst_view = View_t{dst};
        Launcher{tiles.size()}([=] DAMN_ALL() mutable {
            auto const ti = tp[Launcher::block_idx()];
            DAO::template move_if<Launcher>(src_view, dst_view, ti, pred);
            __syncthreads();
            if (Launcher::thread_idx() == 0)
                DAO::remove_gaps(src_view, ti);
        });
    }

    for (auto const ti : tiles)
        particles_[ti]().resize(particles_views_[ti]().size());
    on_appended<ParticleType::Domain>();
    dst.template on_appended<ParticleType::Domain>();
#else
    throw std::runtime_error("TileSetVector::move_if_on_device needs mkn.gpu");
#endif
}


template<typename Particles>
template<auto type>
void TileSetVector<Particles>::append_on_device(Particle_t const* const particles,
                                                std::size_t const n)
{
    static_assert(alloc_mode == damn::AllocatorMode::GPU_UNIFIED);
    if (n == 0)
        return;

#if DAMN_HAVE_MKN_GPU
    using View_t = TileSetSpan<PSpan_t>;
    using DAO    = TileSetBlockDAO<View_t>;

    reset_views();
    { // arrivals per tile, into add_into_
        auto view = View_t{*this};
        damn::kernel::launch(n, [=] DAMN_ALL() mutable {
            DAO::count_one(view, particles[damn::kernel::idx()]);
        });
    }

    // vectors grow by their arrivals, views stay at the old sizes for the appends
    sync_moved<type>();

    {
        auto view = View_t{*this};
        damn::kernel::launch(n, [=] DAMN_ALL() mutable {
            DAO::append_one(view, particles[damn::kernel::idx()]);
        });
    }

    on_appended<type>();
#else
    throw std::runtime_error("TileSetVector::append_on_device needs mkn.gpu");
#endif
}


template<typename Particles>
void TileSetSpan<Particles>::sync_tile_add_new(std::size_t const tidx) DAMN_ALL
{
    CrossTileCopyDAO<std::decay_t<decltype(*this)>>{*this, tidx}.copy_in();
}


template<typename Particles>
void TileSetSpan<Particles>::sync_tile_rm_left(std::size_t const tidx) DAMN_ALL
{
    CrossTileCopyDAO<std::decay_t<decltype(*this)>>{*this, tidx}.rm_left();
}



template<typename Super>
void TileSetParticles<Super>::check() const
{
    // static constexpr auto alloc_mode  = Super::alloc_mode;
    if constexpr (storage_mode == StorageMode::VECTOR)
    {
        for (auto const& tile : particles_())
        {
            // auto const& tile = *particles_.at(gix);
            // assert(tile().capacity() > tile.size());

            auto const bix   = this->local_cell(tile.lower);
            auto const& gaps = this->gaps_(bix);
            auto& left       = this->left_(bix);
            auto& gaps_size  = this->gap_idx_(bix);

            assert(gaps_size == 0);
            assert(left == 0);
            assert(gaps.size() >= tile().size());
        }
    }
}


} // namespace plamn::core


#endif /* PLAMN_CORE_DATA_PARTICLES_PARTICLES_TILE_SET_HPP */
