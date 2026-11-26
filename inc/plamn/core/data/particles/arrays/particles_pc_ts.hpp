#ifndef PLAMN_CORE_DATA_PARTICLES_PARTICLES_PER_CELL_TILE_SET_HPP
#define PLAMN_CORE_DATA_PARTICLES_PARTICLES_PER_CELL_TILE_SET_HPP

#include "damn/core/operators.hpp"
#include "damn/core/memory/span.hpp"
#include "damn/core/exec/kernels.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/sys/monitoring.hpp"
#include "damn/core/data/tiles/tile_set.hpp"
#include "damn/core/data/ndarray/ndarray_vector.hpp"
#include "damn/core/data/ndarray/ndarray_view.hpp"
#include "damn/core/def/damn_config.hpp"
#include "damn/core/def/detail/mkn_gpu.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/logger.hpp"
#include "mkn/kul/std/array.hpp"
#include "mkn/kul/for.hpp"

#include "plamn/core/def.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/data/particles/particle_translation_tracking.hpp"

#include <cassert>
#include <array>
#include <cstddef>
#include <cstdint>
#include <new>
#include <ostream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

namespace plamn::core
{

template<typename PerCellParticles>
class PCParticlesTile : public damn::core::Box<std::int32_t, PerCellParticles::dimension>
{
    static constexpr auto dim = PerCellParticles::dimension;
    using Super               = damn::core::Box<std::int32_t, dim>;
    using This                = PCParticlesTile<PerCellParticles>;

public:
    static constexpr auto dimension = dim;

    PCParticlesTile(Super const& box, std::size_t const ghost_cells)
        : Super{box}
        , particles{box, ghost_cells}
    {
    }

    PCParticlesTile(PCParticlesTile const&)            = default;
    PCParticlesTile(PCParticlesTile&&)                 = default;
    PCParticlesTile& operator=(PCParticlesTile const&) = default;
    PCParticlesTile& operator=(PCParticlesTile&&)      = default;

    template<typename OtherPerCell>
    PCParticlesTile(PCParticlesTile<OtherPerCell>& other)
        : Super{other}
        , particles{other.particles}
    {
    }

    auto& operator()() DAMN_ALL { return particles; }
    auto& operator()() const DAMN_ALL { return particles; }

    Super& operator*() DAMN_ALL { return *this; }
    Super const& operator*() const DAMN_ALL { return *this; }

    auto& link(std::size_t const idx) { return _links[idx]; }
    auto& links() { return _links; }
    auto& links() const { return _links; }

    // its box, plus for LevelGhost its ghost halo: not deduplicated, adjacent tiles' halos
    //  overlap
    template<ParticleType type>
    void on_reachable_cells(auto&& fn) const DAMN_ALL
    {
        Super const& tbox = *this;
        damn::core::for_each_point(tbox, fn);
        if constexpr (type == ParticleType::LevelGhost)
            damn::core::for_each_point(particles.ghost_box(), [&](auto const& amr) {
                if (!isIn(amr, tbox))
                    fn(amr);
            });
    }

    PerCellParticles particles;

private:
    std::array<This*, 7> _links = mkn::kul::ConstArray<This*, 7>(nullptr);
};




template<typename Particles>
class PCTileSetSpan : public ParticleTranslationTrackerSpan<Particles::dimension>
{
    using Base = ParticleTranslationTrackerSpan<Particles::dimension>;

    template<typename>
    friend struct PCCrossTileCopyDAO;

protected:
    using Base::add_into_;
    using Base::box_;
    using Base::cap_;
    using Base::gap_idx_;
    using Base::gaps_;
    using Base::ghost_box_;
    using Base::left_;
    using Base::size_;

public:
    using SIZE_T                     = typename Base::SIZE_T;
    auto static constexpr alloc_mode = Particles::alloc_mode;
    auto static constexpr dim        = Particles::dimension;

    using This               = PCTileSetSpan<Particles>;
    using lobox_t            = damn::core::Box<std::uint32_t, dim>;
    using per_tile_particles = Particles;
    using Tile_t             = PCParticlesTile<Particles>;

    using Base::box;
    using Base::ghost_box;
    using Base::local_box;
    using Base::local_cell;
    using Base::size;

private:
    using locell_t = std::array<std::uint32_t, dim>;

    template<typename PCTileSetArray>
    auto resolve(PCTileSetArray& arr)
    {
        if constexpr (PCTileSetArray::storage_mode == StorageMode::SPAN)
            return arr.particles_;
        else
            return arr.particles_views_.make_view();
    }

    template<typename PCTileSetArray>
    auto resolve_gaps(PCTileSetArray& arr)
    {
        if constexpr (PCTileSetArray::storage_mode == StorageMode::SPAN)
            return arr.gaps_;
        else
            return *arr.gap_views_;
    }

public:
    auto static constexpr dimension    = dim;
    auto static constexpr storage_mode = StorageMode::SPAN;
    using Particle_t                   = typename ParticleDefaults<dim>::Particle_t;

    template<typename PCTileSetArray>
    PCTileSetSpan(PCTileSetArray& arr);

    auto size(std::size_t const& idx) const DAMN_ALL
    {
        return particles_.data()[idx]().size();
    }
    auto size(locell_t const& icell) const DAMN_ALL { return cell_size_(icell); }

    auto& operator()() DAMN_ALL { return particles_; }
    auto& operator()() const DAMN_ALL { return particles_; }
    auto& operator()(locell_t const& cell) DAMN_ALL { return (*particles_.at(cell))(); }
    auto& operator()(locell_t const& cell) const DAMN_ALL { return (*particles_.at(cell))(); }

    auto local_tile_cell(std::array<int, dim> const& cell) const DAMN_ALL
    {
        DAMN_ASSERT(particles_.at(local_cell(cell)));
        return local_cell((*particles_.at(local_cell(cell))).lower);
    }

    template<auto particle_type>
    auto& move_check(auto const& pt, std::size_t const idx, auto& particle) DAMN_ALL;

    // CPU: tile by tile on the host; GPU_UNIFIED: a kernel thread per tile, on args' stream
    template<auto type, typename... Args>
    void sync(Args&&... args);

    void clear() {} // TODO

protected:
    template<auto type>
    void sync_tile_add_new(std::size_t const tidx) DAMN_ALL;
    template<auto type>
    void sync_tile_rm_left(std::size_t const tidx) DAMN_ALL;

    damn::core::TileSetView<Tile_t> particles_;
    damn::core::NdArrayView<dim, std::size_t> cell_size_;

}; // PCTileSetSpan


template<typename Particles>
class PCTileSetVector
    : public ParticleTranslationTracker<Particles::dimension, Particles::alloc_mode>
{
    using This = PCTileSetVector<Particles>;
    using Base = ParticleTranslationTracker<Particles::dimension, Particles::alloc_mode>;

    template<typename P>
    friend class PCTileSetSpan;

protected:
    using Base::add_into_;
    using Base::box_;
    using Base::cap_;
    using Base::cell_size_;
    using Base::gap_idx_;
    using Base::gap_views_;
    using Base::gaps_;
    using Base::ghost_box_;
    using Base::ghost_cells_;
    using Base::left_;
    using Base::total_size;

public:
    using SIZE_T                       = typename Base::SIZE_T;
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
    using PSpan_t                      = typename Particles::view_t;
    using per_tile_particles           = Particles;
    using Tile_t                       = PCParticlesTile<Particles>;
    using SpnTile                      = PCParticlesTile<PSpan_t>;
    using size_t_vector                = typename Base::size_t_vector;
    using Base::box;
    using Base::ghost_box;
    using Base::local_box;
    using Base::local_cell;
    using Base::zero_bookkeeping;

    PCTileSetVector(box_t const& box, auto const ghost_cells)
        : Base{box, ghost_cells}
    {
        damn::core::TileSet<Tile_t, alloc_mode>::build_links(particles_);
        damn::core::TileSet<SpnTile, alloc_mode>::build_links(particles_views_);
    }

    PCTileSetVector(PCTileSetVector&& that)
        : Base{that.box_, that.ghost_cells_}
        , particles_{std::move(that.particles_)}
    {
        on_appended(); // without std::swap does not work well
    }

    PCTileSetVector(PCTileSetVector const& that)
        : Base{that.box_, that.ghost_cells_}
        , particles_{that.particles_}
    {
        on_appended();
    }

    // not = default: a memberwise copy leaves particles_views_ stale, reconstructed in place
    //  by the constructors (on_appended())
    PCTileSetVector& operator=(PCTileSetVector&& that)
    {
        if (this == &that)
            return *this;
        this->~PCTileSetVector();
        new (this) PCTileSetVector(std::move(that));
        return *this;
    }
    PCTileSetVector& operator=(PCTileSetVector const& that)
    {
        if (this == &that)
            return *this;
        this->~PCTileSetVector();
        new (this) PCTileSetVector(that);
        return *this;
    }

    auto size() const { return total_size; }
    auto size(locell_t const& icell) const { return cell_size_(icell); }
    auto size(std::size_t const& idx) const { return cell_size_.data()[idx]; }

    void emplace_back(Particle_t const& p)
    {
        auto* tile = particles_.at(damn::core::Point<int, dim>{p.iCell()});
        assert(tile);
        (*tile)().emplace_back(p);
        ++total_size;
    }

    template<typename... Args>
    void emplace_back(double const weight, Args&&... args)
    {
        this->emplace_back(Particle_t{weight, args...});
    }

    void push_back(Particle_t const& p) { emplace_back(p); }

    auto& operator()() { return particles_; }
    auto& operator()() const { return particles_; }

    // recount, gap sizing, views: after a raw append, or the end of on_moved()
    template<auto type = ParticleType::Domain>
    void on_appended();

    template<auto type>
    void sync_moved();

    // applies move_check's registered moves: doc/core/data/particles/arrays/particles_pc_ts.md
    template<auto type>
    void on_moved(auto&&... args);

    void reset_views()
    {
        damn::core::update_from([&](std::size_t const i) { return SpnTile{particles_[i]}; }, particles_views_);
        damn::core::update_from([&](std::size_t const i) { return damn::core::make_span(*(gaps_.data() + i)); },
                    gap_views_);
    }

    void clear();

    auto& views() { return particles_views_; }
    auto& views() const { return particles_views_; }

    MKN_KUL_FN_SIG_RET bool is_consistent() const;

protected:
    // cells within ghost_cells_ of the tile's wall: the only ones a particle can enter or leave
    //  the tile from in one step
    void on_tile_wall_cells(auto const& tile, auto&& fn) const;

    // ghost box cells it clamp-owns (TileSet::tag_cells_): where level ghosts live and register
    //  their cross-tile traffic
    void on_tile_owned_ghost_cells(auto const& tile, auto&& fn) const
    {
        damn::core::Box<int, dim> const& tbox = tile;
        damn::core::for_each_point(tile().ghost_box(), [&](auto const& amr) {
            if (!isIn(amr, tbox) and particles_.at(amr) == &tile)
                fn(amr);
        });
    }

    // tiles build from amrbox, but `.at()` function maps ghosts box
    damn::core::GhostClampedTileSet<Tile_t, alloc_mode> particles_{box_, ghost_cells_, ghost_cells_};
    damn::core::GhostClampedTileSet<SpnTile, alloc_mode> particles_views_
        = damn::core::GhostClampedTileSet<SpnTile, alloc_mode>::make_from(
            [](auto& tile) -> auto& { return tile; }, particles_);

}; // PCTileSetVector<Particles>


template<typename Super_>
struct PCTileSetParticles : public Super_
{
    using Super              = Super_;
    using This               = PCTileSetParticles<Super>;
    using Particle_t         = typename Super::Particle_t;
    using per_tile_particles = typename Super::per_tile_particles;

    auto static constexpr alloc_mode   = Super::alloc_mode;
    auto static constexpr dimension    = Super::dimension;
    auto static constexpr storage_mode = Super::storage_mode;
    auto static constexpr size_of_particle() { return sizeof(Particle_t); }

    using Super::size;

    PCTileSetParticles(PCTileSetParticles&&)                 = default;
    PCTileSetParticles& operator=(PCTileSetParticles&&)      = default;
    PCTileSetParticles(PCTileSetParticles const&)            = default;
    PCTileSetParticles& operator=(PCTileSetParticles const&) = default;

    template<typename... Args>
    PCTileSetParticles(Args&&... args)
        requires std::is_constructible_v<Super, Args&&...>
    DAMN_ALL : Super{std::forward<Args>(args)...}
    {
    }

    // no begin() / end(): tiled, enumerate() / per_particle() instead

    auto data() const DAMN_ALL { return static_cast<Particle_t const*>(nullptr); } // TODO
    auto data() DAMN_ALL { return static_cast<Particle_t*>(nullptr); }             // TODO

    // no move_check here: it would shadow PCTileSetSpan's
    auto nbr_particles_in(std::array<int, dimension> const arr) const
    {
        auto const& tile = *this->particles_.at(this->local_cell(arr));
        return tile()(tile().local_cell(arr)).size();
    }

    auto nbr_particles_in(damn::core::Box<int, dimension> const box) const
    {
        std::size_t n_particles = 0;
        damn::core::for_each_point(box,
                       [&](auto const& cell) { n_particles += nbr_particles_in(cell.toArray()); });
        return n_particles;
    }

    auto max_size() const { return std::size_t{0}; /* TODO */ }

    void print() const {}
    void check() const {}


}; // PCTileSetParticles<Super>


template<typename Particles>
struct PCCrossTileCopyDAO
{
    auto static constexpr dim = Particles::dim;
    using Tile                = PCParticlesTile<typename Particles::per_tile_particles>;

    // its box plus (LevelGhost) the ghost cells it clamp-owns: the tile set's bookkeeping is
    //  shared, other halo cells hold nothing
    template<auto type>
    void on_owned_cells(auto&& fn) DAMN_ALL
    {
        damn::core::Box<int, dim> const& tbox = tile;
        tile.template on_reachable_cells<type>([&](auto const& amr) {
            if (isIn(amr, tbox) or ps().at(ps.local_cell(amr)) == &tile)
                fn(amr);
        });
    }

    template<auto type>
    void copy_in() DAMN_ALL;

    // both leaver lists of a cell in one merged rm pass, not two (see sync_rm_left)
    template<auto type>
    void rm_left() DAMN_ALL
    {
        auto& pc = tile();

        on_owned_cells<type>([&](auto const& amr) {
            auto const cix = ps.local_cell(amr);
            pc.sync_rm_left(pc.local_cell(amr), ps.gaps_(cix).data(), ps.gap_idx_(cix),
                            ps.left_(cix));
        });
    }

    Particles& ps;
    std::size_t src_tile_idx;
    Tile& tile = ps()[src_tile_idx];
};


template<typename Particles>
template<typename PCTileSetArray>
PCTileSetSpan<Particles>::PCTileSetSpan(PCTileSetArray& arr)
    : Base{{.gaps            = resolve_gaps(arr),
            .gap_idx         = arr.gap_idx_,
            .add_into        = arr.add_into_,
            .cap             = arr.cap_,
            .left            = arr.left_,
            .size            = arr.size(),
            .box             = arr.box_,
            .ghost_box       = arr.ghost_box(),
            .local_ghost_box = arr.local_box()}}
    , particles_{resolve(arr)}
    , cell_size_{arr.cell_size_}
{
}

template<typename Particles>
template<auto particle_type>
auto& PCTileSetSpan<Particles>::move_check(auto const& pt, std::size_t const idx,
                                           auto& particle) DAMN_ALL
{
    using enum ParticleType;
    static_assert(mkn::kul::any_in(particle_type, Domain, LevelGhost));

    auto const& newcell = particle.iCell();

    if (mkn::kul::array_equals(newcell, pt.icell))
        return *this;

    bool constexpr static ATOMIC = true;
    bool constexpr static GPU    = alloc_mode == damn::AllocatorMode::GPU_UNIFIED;
    using Op                     = damn::core::Operators<SIZE_T, ATOMIC, GPU>;

    // registered against the tile set's cell: the copy (or removal) is the sync's
    auto const leave = [&] DAMN_ALL() {
        auto const old_lcl_cell = local_cell(pt.icell);

        auto& gidx      = gap_idx_(old_lcl_cell);
        auto const nidx = Op{gidx}.increment_return_old();
        auto& gaps      = gaps_(old_lcl_cell);
        assert(nidx < gaps.size());
        gaps[nidx] = idx;
    };

    if constexpr (particle_type == Domain)
    {
        auto& old_tile = *particles_.at(pt.tile_cell);
        if (isIn(newcell, box()) and not isIn(newcell, old_tile))
        {
            leave();
            Op{add_into_(local_cell(newcell))}.increment_return_old();
            return *this;
        }
    }
    else // LevelGhost
    {
        auto& cur_tile = *particles_.at(pt.tile_cell);
        // level ghosts live only in their cell's clamp owner (TileSet::tag_cells_): handed off
        //  to the new cell's owner, even into another tile's domain, even within this ghost box
        if (isIn(newcell, ghost_box()) and particles_.at(local_cell(newcell)) != &cur_tile)
        {
            leave();
            Op{add_into_(local_cell(newcell))}.increment_return_old();
            return *this;
        }

        cur_tile().template move_check<particle_type>(pt, idx, particle);
        return *this;
    }

    // Domain, still this tile's: inside it, or outside the domain (assumed in its ghost box)
    (*particles_.at(pt.tile_cell))().template move_check<particle_type>(pt, idx, particle);

    return *this;
}


template<typename Particles>
template<auto type>
void PCTileSetVector<Particles>::on_appended()
{
    damn::core::MemoryMonitoring::LOG(__PRETTY_FUNCTION__);

    total_size = 0;
    for (auto& tile : particles_)
    {
        tile().template on_appended<type>();
        total_size += tile().size();
    }

    // gaps sized where cross-tile departures register: wall cells, and the clamp-owned ghost
    //  cells whatever the type (the first sync after a fill may run as Domain on level ghosts)
    for (auto& tile : particles_)
    {
        auto const size_cell = [&](auto const& amr_cell) {
            auto const c   = local_cell(amr_cell);
            auto const& cs = tile()(tile().local_cell(amr_cell)).size();
            cell_size_(c)  = cs;
            if (auto& gaps = gaps_(c); gaps.size() < cs)
                gaps.resize(cs);
        };
        on_tile_wall_cells(tile, size_cell);
        on_tile_owned_ghost_cells(tile, size_cell);
    }

    reset_views();
}

template<typename Particles>
template<auto type>
void PCTileSetVector<Particles>::sync_moved()
{
    for (auto& tile : particles_)
    {
        damn::core::Box<int, dim> const& tbox = tile;
        auto& pc                  = tile();

        // shared bookkeeping: exactly one tile, the cell's owner, folds its cross-tile traffic in
        damn::core::for_each_point(pc.ghost_box(), [&](auto const& amr) {
            std::size_t in = 0, out = 0;
            if (isIn(amr, tbox) or particles_.at(amr) == &tile)
            {
                auto const cix = local_cell(amr);
                in             = add_into_(cix);
                out            = gap_idx_(cix);
                add_into_(cix) = 0;
            }
            pc.template sync_moved<type>(pc.local_cell(amr), in, out);
        });
        pc.reset_views(); // per-cell spans start the copy step at the pre-move sizes
    }
    reset_views();
}

template<typename Particles>
template<auto type>
void PCTileSetVector<Particles>::on_moved(auto&&... args)
{
    sync_moved<type>();
    PCTileSetSpan<PSpan_t>{*this}.template sync<type>(args...);

#if DAMN_DEBUG
    auto& vws = views();
    for (std::size_t i = 0; i < particles_.size(); ++i)
        damn::core::for_each_point(particles_[i]().local_box(), [&](auto const& bix) {
            auto const vsize = vws[i]()(bix).size();
            auto const psize = particles_[i]()(bix).size();
            if (vsize != psize)
            {
                DAMN_LOG_LINE_SS("on_moved mismatch: tile " << i << " cell " << bix << " view="
                                                             << vsize << " vector=" << psize);
            }
        });
    for (std::size_t i = 0; i < particles_.size(); ++i)
        damn::core::for_each_point(particles_[i]().local_box(), [&](auto const& bix) {
            assert(vws[i]()(bix).size() == particles_[i]()(bix).size());
        });
#endif

    on_appended<type>();
}

template<typename Particles>
void PCTileSetVector<Particles>::clear()
{
    for (auto& tile : particles_)
        tile().clear(); // resets each tile's own per-cell bookkeeping too
    total_size = 0;

    // the tile set's own cross-tile bookkeeping is separate from the tiles'
    zero_bookkeeping();

    reset_views();
}

template<typename Particles>
MKN_KUL_FN_SIG_RET bool PCTileSetVector<Particles>::is_consistent() const
{
    std::size_t n = 0;
    bool valid    = true;
    for (auto const& tile : particles_)
    {
        auto const& pc = tile();
        damn::core::for_each_point(pc.local_box(), [&](auto const& cell) {
            for (auto const& p : pc(cell.toArray()))
                if (pc.local_cell(p.iCell()) != cell.toArray())
                    valid = false;
        });
        if (!valid)
            return false;
        n += pc.size();
    }
    return n == total_size;
}

template<typename Particles>
void PCTileSetVector<Particles>::on_tile_wall_cells(auto const& tile, auto&& fn) const
{
    damn::core::Box<int, dim> const& tbox = tile;
    auto const w              = static_cast<int>(ghost_cells_);

    bool whole = false;
    for (std::size_t d = 0; d < dim; ++d)
        whole |= tbox.shape()[d] <= 2 * w;

    if (whole) // too small for an interior, every cell is a wall cell
    {
        damn::core::for_each_point(tbox, fn);
        return;
    }
    for (auto const& b : tbox.remove(damn::core::shrink(tbox, w)))
        damn::core::for_each_point(b, fn);
}

template<typename Particles>
template<auto type>
void PCCrossTileCopyDAO<Particles>::copy_in() DAMN_ALL
{
    auto& pc = tile();

    // within-tile movers (off-domain ones into its ghost layer too): the tile's own gap lists
    on_owned_cells<type>([&](auto const& amr) { pc.sync_add_new(pc.local_cell(amr)); });

    // cross-tile leavers: the tile set's gap lists, into the tile owning the new cell
    on_owned_cells<type>([&](auto const& amr) {
        auto const cix     = ps.local_cell(amr);
        auto const& n_gaps = ps.gap_idx_(cix);
        if (!n_gaps)
            return;
        {
            auto& gaps = ps.gaps_(cix);
            pc.sort(gaps.data(), gaps.data() + n_gaps);
        }
        auto& src        = pc(pc.local_cell(amr));
        auto& left       = ps.left_(cix);
        auto const& gaps = ps.gaps_(cix);
        for (std::size_t i = 0; i < n_gaps; ++i)
        {
            auto const& gidx    = gaps[n_gaps - (1 + i)];
            auto const& newcell = src.iCell(gidx);

            if constexpr (type == ParticleType::LevelGhost)
                if (not isIn(newcell, ps.ghost_box()))
                { // left the level ghost box: no destination, rm_left deletes it
                    ++left;
                    continue;
                }

            auto& dst_tile = *ps().at(ps.local_cell(newcell));
            auto& dst_pc   = dst_tile();
            if constexpr (type == ParticleType::Domain)
            { // level ghost dst cells are clamp-owned: outside the dst tile box
                DAMN_ASSERT(not isIn(src[gidx], tile));
                DAMN_ASSERT(isIn(src[gidx], dst_tile));
            }
            [[maybe_unused]] bool const ok
                = dst_pc.append_from(dst_pc.local_cell(newcell), src, gidx);
            DAMN_ASSERT(ok); // capacity is exact-reserved in sync_moved
            ++left;
        }
        DAMN_ASSERT(left == n_gaps);
    });
}


template<typename Particles>
template<auto type, typename... Args>
void PCTileSetSpan<Particles>::sync(Args&&... args)
{
    DAMN_LOG_SCOPE(3, "PCTileSetSpan::sync(stream)");

    if constexpr (alloc_mode == damn::AllocatorMode::CPU)
    {
        for (std::size_t tidx = 0; tidx < particles_.size(); ++tidx)
            sync_tile_add_new<type>(tidx);

        for (std::size_t tidx = 0; tidx < particles_.size(); ++tidx)
            sync_tile_rm_left<type>(tidx);
    }
    else if constexpr (alloc_mode == damn::AllocatorMode::GPU_UNIFIED)
    {
        if (!DAMN_HAVE_MKN_GPU)
            throw std::runtime_error("no gpu impl");

        DAMN_WITH_MKN_GPU({
            auto view         = *this;
            auto const launch = [&](auto&& fn) { // on args' stream, else the default stream
                if constexpr (sizeof...(args) == 0)
                    damn::kernel::launch(particles_.size(), fn);
                else
                {
                    auto const& [stream] = std::forward_as_tuple(args...);
                    damn::kernel::launch(stream, particles_.size(), fn);
                }
            };
            // two launches on the stream: every tile's appends must land before any tile
            //  removes its leavers. __syncthreads between the passes only orders one block,
            //  tiles span several
            launch([=] DAMN_ALL() mutable {
                view.template sync_tile_add_new<type>(damn::kernel::idx());
            });
            launch([=] DAMN_ALL() mutable {
                view.template sync_tile_rm_left<type>(damn::kernel::idx());
            });
        })
    }
    else
        throw std::runtime_error(__func__);
}


template<typename Particles>
template<auto type>
void PCTileSetSpan<Particles>::sync_tile_add_new(std::size_t const tidx) DAMN_ALL
{
    PCCrossTileCopyDAO<std::decay_t<decltype(*this)>>{*this, tidx}.template copy_in<type>();
}


template<typename Particles>
template<auto type>
void PCTileSetSpan<Particles>::sync_tile_rm_left(std::size_t const tidx) DAMN_ALL
{
    PCCrossTileCopyDAO<std::decay_t<decltype(*this)>>{*this, tidx}.template rm_left<type>();
}

} // namespace plamn::core

#endif /* PLAMN_CORE_DATA_PARTICLES_PARTICLES_PER_CELL_TILE_SET_HPP */
