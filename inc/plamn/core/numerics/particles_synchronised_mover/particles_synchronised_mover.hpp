#ifndef PLAMN_CORE_NUMERICS_PARTICLES_SYNCHRONISED_MOVER_HPP
#define PLAMN_CORE_NUMERICS_PARTICLES_SYNCHRONISED_MOVER_HPP

#include "damn/core/def/damn_config.hpp"
#include "damn/core/logger.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/geometry/range.hpp"
#include "damn/core/def/detail/mkn_gpu.hpp"
#include "damn/core/exec/kernels.hpp"
#include "damn/core/exec/thread_pool.hpp"
#include "plamn/core/def.hpp"
#include "plamn/core/errors.hpp"
#include "plamn/core/numerics/pusher/boris.hpp"
#include "plamn/core/numerics/moments/moments.hpp"
#include "plamn/core/numerics/interpolator/interpolator.hpp"
#include "plamn/core/numerics/interpolator/interpolating.hpp"
#include "plamn/core/numerics/boundary_condition/boundary_condition.hpp"
#include "plamn/core/numerics/pusher/boris/detail/multi_boris.hpp"
#include "plamn/core/data/particles/particle_components.hpp"
#include "plamn/core/data/particles/exporting/particles_exporting.hpp"
#include "plamn/core/data/field/field_tiles.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"
#include "plamn/core/data/particles/particles.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "particles_synchronised_mover_def.hpp" // = UpdaterMode
#include "mkn/gpu.hpp"
#include "phlop/timing/scope.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/operators.hpp"
#include "damn/core/vector.hpp"
#include "mkn/kul/for.hpp"
#include "plamn/core/data/particles/arrays/particles_chunked.hpp"

#include <string>
#include <optional>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <tuple>

namespace plamn::core
{

// after a REF move: domain particles leaving the owned domain become patch ghosts or are
// deleted, level ghosts entering it become domain particles. A level ghost lives in one patch
// only (Simulator's own_level_ghosts_): one in a cell another patch of the level covers is
// handed to that patch as a patch ghost, as a domain particle would be (no other patch has a
// copy of it to keep instead, the level ghost duplicate fix); one leaving the ghost box is deleted
namespace detail
{
    template<typename Particles_t>
    void post_move_sync_all(Particles_t& domain, Particles_t& patchGhost, Particles_t& levelGhost,
                            auto const& boxing);
} // namespace detail

template<typename Particles_t>
void post_move_sync(Particles_t& domain, Particles_t& patchGhost, Particles_t& levelGhost,
                    auto const& boxing)
{
    detail::post_move_sync_all(domain, patchGhost, levelGhost, boxing);
}

template<typename Particles_t>
void detail::post_move_sync_all(Particles_t& domain, Particles_t& patchGhost,
                                Particles_t& levelGhost, auto const& boxing)
{
    auto const& nonOwnedBoxes = boxing.nonOwnedBoxes;

    delete_particles_not_in(domain, boxing.nonLevelGhostBox);
    move_in_ghost_layer(patchGhost, domain, boxing.domainBox, boxing.nonLevelGhostBox);
    if (!nonOwnedBoxes.empty())
        for (auto const& box : nonOwnedBoxes)
        {
            move_in_domain(patchGhost, domain, box);
            move_in_domain(patchGhost, levelGhost, box);
        }
    move_in_domain(domain, levelGhost, boxing.domainBox);
    move_in_ghost_layer(patchGhost, levelGhost, boxing.domainBox, boxing.nonLevelGhostBox);
    delete_particles_not_in(levelGhost, boxing.ghostBox);
    delete_particles_not_in(domain, boxing.domainBox);
}


// the fate of a particle after a REF move only depends on its cell, see post_move_sync
template<typename Boxing_t>
struct PostMoveSyncCells
{
    bool owned(auto const& cell) const
    {
        return isIn(cell, boxing.domainBox)
               and (boxing.nonOwnedBoxes.empty() or !isIn(cell, boxing.nonOwnedBoxes));
    }

    // domain particles not owned: non-owned domain cells and the patch ghost layer -> patch
    //  ghost, else deleted
    bool to_patch_ghost(auto const& cell) const
    {
        return isIn(cell, boxing.domainBox) or isIn(cell, boxing.nonLevelGhostBox);
    }

    // level ghosts not owned: still in the level ghost layer (no patch of the level covers the
    //  cell) -> kept, in another patch's cell -> patch ghost (to_patch_ghost), else deleted
    bool keep_level_ghost(auto const& cell) const
    {
        return !isIn(cell, boxing.domainBox) and isIn(cell, boxing.ghostBox)
               and !isIn(damn::core::Point{cell}, boxing.nonLevelGhostBox);
    }

    bool touches_non_owned(auto const& box) const
    {
        for (auto const& nob : boxing.nonOwnedBoxes)
            if (box * nob)
                return true;
        return false;
    }

    Boxing_t const& boxing;
};

template<typename Boxing_t>
PostMoveSyncCells(Boxing_t const&) -> PostMoveSyncCells<Boxing_t>;


// SoAVXChunked (host): the domain's particles that changed cell back into their chunks' gaps
//  (resettle), level ghosts entering the domain into its gaps, no sort while there is room; the
//  same fates as post_move_sync's (PostMoveSyncCells). Not sorted yet, or without slack (a sort
//  each push): post_move_sync's way
template<typename Particles_t>
    requires(Particles_t::layout_mode == LayoutMode::SoAVXChunked
             and Particles_t::alloc_mode == damn::AllocatorMode::CPU)
void post_move_sync(Particles_t& domain, Particles_t& patchGhost, Particles_t& levelGhost,
                    auto const& boxing)
{
    if (!domain.sorted() or !domain.has_slack())
    {
        detail::post_move_sync_all(domain, patchGhost, levelGhost, boxing);
        return;
    }
    PostMoveSyncCells const cells{boxing};
    domain.resettle([&](auto const& cell) { return cells.owned(cell); },
                    [&](auto const& p) {
                        if (cells.to_patch_ghost(p.iCell()))
                            patchGhost.push_back(p);
                    });

    bool appended    = false;
    std::size_t kept = 0;
    for (std::size_t i = 0; i < levelGhost.size(); ++i)
    {
        auto const cell = levelGhost.iCell(i);
        if (cells.owned(cell))
        {
            auto const p = levelGhost.copy(i);
            if (!domain.insert(p))
            {
                domain.push_back(p);
                appended = true;
            }
        }
        else if (cells.keep_level_ghost(cell))
        {
            if (kept != i)
                levelGhost.assign(levelGhost.copy(i), kept);
            ++kept;
        }
        else if (cells.to_patch_ghost(cell))
            patchGhost.push_back(levelGhost.copy(i));
    }
    levelGhost.resize(kept);
    if (appended)
        domain.sort();
}


// a layout's own post push step (is_synced_on_move): the chunks' resettle on the host
template<typename Particles_t>
    requires(Particles_t::layout_mode == LayoutMode::SoAVXChunked
             and Particles_t::alloc_mode == damn::AllocatorMode::CPU)
void synced_on_move(Particles_t& domain, Particles_t& patchGhost, Particles_t& levelGhost,
                    auto const& boxing)
{
    post_move_sync(domain, patchGhost, levelGhost, boxing);
}
namespace detail
{
    // PostMoveSyncCells' fates, for a kernel: the boxing's boxes copied into fixed arrays
    template<std::size_t dim>
    struct DeviceSyncCells
    {
        using Box_t                            = damn::core::Box<int, dim>;
        static constexpr std::size_t max_boxes = 64;

        Box_t domain, ghost;
        std::array<Box_t, max_boxes> non_level_ghost{}, non_owned{};
        std::uint32_t n_non_level_ghost = 0, n_non_owned = 0;

        explicit DeviceSyncCells(auto const& boxing)
            : domain{boxing.domainBox}
            , ghost{boxing.ghostBox}
        {
            auto const copy = [](auto const& from, auto& to, std::uint32_t& n) {
                if (from.size() > max_boxes)
                    throw std::runtime_error("DeviceSyncCells: more boxes than max_boxes");
                for (auto const& b : from)
                    to[n++] = b;
            };
            copy(boxing.nonLevelGhostBox, non_level_ghost, n_non_level_ghost);
            copy(boxing.nonOwnedBoxes, non_owned, n_non_owned);
        }

        static bool in_any(auto const& p, auto const& boxes, std::uint32_t const n) DAMN_ALL
        {
            for (std::uint32_t i = 0; i < n; ++i)
                if (isIn(p, boxes[i]))
                    return true;
            return false;
        }
        bool owned(auto const& p) const DAMN_ALL
        {
            return isIn(p, domain) and !in_any(p, non_owned, n_non_owned);
        }
        bool to_patch_ghost(auto const& p) const DAMN_ALL
        {
            return isIn(p, domain) or in_any(p, non_level_ghost, n_non_level_ghost);
        }
        bool keep_level_ghost(auto const& p) const DAMN_ALL
        {
            return !isIn(p, domain) and isIn(p, ghost)
                   and !in_any(p, non_level_ghost, n_non_level_ghost);
        }
    };

    // a SoAVX array's columns as plain pointers (device copyable: no tuple), from its
    //  components (addresses only, nothing read on the host)
    template<std::size_t dim>
    struct ColumnPointers
    {
        double* weight;
        std::array<int*, dim> iCell;
        std::array<double*, dim> delta;
        std::array<double*, 3> v;

        static ColumnPointers of(auto& particles)
        {
            auto const c = core::components(particles);
            ColumnPointers p{};
            p.weight = &c.weight[0];
            for (std::size_t d = 0; d < dim; ++d)
            {
                p.iCell[d] = &c.iCell[d][0];
                p.delta[d] = &c.delta[d][0];
            }
            for (std::size_t d = 0; d < 3; ++d)
                p.v[d] = &c.v[d][0];
            return p;
        }

        // slot to of this = slot i of from
        void assign(std::size_t const to, ColumnPointers const& from,
                    std::size_t const i) const DAMN_ALL
        {
            weight[to] = from.weight[i];
            for (std::size_t d = 0; d < dim; ++d)
            {
                iCell[d][to] = from.iCell[d][i];
                delta[d][to] = from.delta[d][i];
            }
            for (std::size_t d = 0; d < 3; ++d)
                v[d][to] = from.v[d][i];
        }
    };

    // a slot's fate after a push: stays, moves (to the first or second destination), or deleted
    enum class Fate : chunk_offset_t { stays = 0, first = 1, second = 2, deleted = 3 };

    // src's particles by fate(cell) on the device: those moving appended to first / second (in no
    //  particular order), those moving or deleted weight 0 in src (the following sort drops them)
    template<typename Particles_t>
    void move_by_fate(Particles_t& src, auto const fate, Particles_t* const first,
                      Particles_t* const second)
    {
        static constexpr auto dim = Particles_t::dimension;
        using Op            = damn::core::Operators<chunk_offset_t, /*atomic=*/true, /*GPU=*/true>;
        std::size_t const n = src.size();
        if (n == 0)
            return;
        damn::resize(src.slots_, n, /*copy=*/false);
        damn::resize(src.counts_, 6, /*copy=*/false);
        auto* const fates  = src.slots_.data();
        auto* const counts = src.counts_.data(); // [1], [2]: movers; [4], [5]: cursors
        auto const columns = [](auto& ps) { return ColumnPointers<dim>::of(ps); };
        auto const s       = columns(src);
        auto const* weight = s.weight;
        damn::kernel::launch(6, [=] DAMN_ALL() mutable { counts[damn::kernel::idx()] = 0; });
        damn::kernel::launch(n, [=] DAMN_ALL() mutable {
            auto const i = damn::kernel::idx();
            if (weight[i] == 0) // a gap
            {
                fates[i] = static_cast<chunk_offset_t>(Fate::stays);
                return;
            }
            damn::core::Point<int, dim> cell;
            for (std::size_t d = 0; d < dim; ++d)
                cell[d] = s.iCell[d][i];
            auto const f = static_cast<chunk_offset_t>(fate(cell));
            fates[i]     = f;
            if (f == 1 or f == 2)
                Op{counts[f]}.increment_return_old();
        });
        std::size_t const n_first = counts[1], n_second = counts[2];
        auto const grow = [](Particles_t* const dst, std::size_t const k) {
            std::size_t const offset = dst ? dst->size() : 0;
            if (dst and k)
                dst->resize(offset + k);
            return offset;
        };
        std::size_t const off_first = grow(first, n_first), off_second = grow(second, n_second);
        auto const f1 = columns(first ? *first : src), f2 = columns(second ? *second : src);
        damn::kernel::launch(n, [=] DAMN_ALL() mutable {
            auto const i = damn::kernel::idx();
            auto const f = fates[i];
            if (f == static_cast<chunk_offset_t>(Fate::stays))
                return;
            if (f == 1 or f == 2)
            {
                auto const slot = Op{counts[3 + f]}.increment_return_old();
                auto const to   = (f == 1 ? off_first : off_second) + slot;
                (f == 1 ? f1 : f2).assign(to, s, i);
            }
            s.weight[i] = 0; // moved or deleted: dropped by the sort
        });
    }
} // namespace detail

// on the device (no host access to the particles): one kernel per array classifies its slots by
//  cell (PostMoveSyncCells' rules), the movers appended to their array, the rest left as gaps for
//  the following sort (on_moved) to drop
template<typename Particles_t>
    requires(Particles_t::layout_mode == LayoutMode::SoAVXChunked
             and Particles_t::alloc_mode == damn::AllocatorMode::GPU_UNIFIED)
void synced_on_move(Particles_t& domain, Particles_t& patchGhost, Particles_t& levelGhost,
                    auto const& boxing)
{
    using detail::Fate;
    detail::DeviceSyncCells<Particles_t::dimension> const cells{boxing};
    // the domain's: owned stay, else to the patch ghosts or deleted
    detail::move_by_fate(
        domain,
        [=] DAMN_ALL(auto const& p) {
            return cells.owned(p) ? Fate::stays
                                  : (cells.to_patch_ghost(p) ? Fate::first : Fate::deleted);
        },
        &patchGhost, static_cast<Particles_t*>(nullptr));
    // the level ghosts': owned to the domain, still in the level ghost layer stay, else to the
    //  patch ghosts or deleted
    detail::move_by_fate(
        levelGhost,
        [=] DAMN_ALL(auto const& p) {
            if (cells.owned(p))
                return Fate::second;
            if (cells.keep_level_ghost(p))
                return Fate::stays;
            return cells.to_patch_ghost(p) ? Fate::first : Fate::deleted;
        },
        &patchGhost, &domain);
}

// after a push with spec: a population's arrays placed, by its layout's own step when it syncs
//  on move, else by the generic post_move_sync
template<MoveSpec spec, typename Particles_t>
    requires(is_synced_on_move<spec, Particles_t::layout_mode>())
void post_push_sync(Particles_t& domain, Particles_t& patchGhost, Particles_t& levelGhost,
                    auto const& boxing)
{
    synced_on_move(domain, patchGhost, levelGhost, boxing);
}
template<MoveSpec spec, typename Particles_t>
    requires(!is_synced_on_move<spec, Particles_t::layout_mode>())
void post_push_sync(Particles_t& domain, Particles_t& patchGhost, Particles_t& levelGhost,
                    auto const& boxing)
{
    post_move_sync(domain, patchGhost, levelGhost, boxing);
}


// tiled layouts: a tile only holds particles whose cell it (clamp) owns, and emplace_back
// routes by that same owner, so tile ti of each array is only ever written from tile ti:
// distinct tiles can be synced concurrently, then post_move_sync_finalize once per patch

// AoSCMTS: only the cells of a tile's ghost layer outside the patch domain, and its non-owned
// domain cells, are visited, through the tile's cellmap. Level ghosts only live in patch
// border tiles
template<typename Particles_t>
    requires(Particles_t::layout_mode == LayoutMode::AoSCMTS)
void post_move_sync_tile(Particles_t& domain, Particles_t& patchGhost, Particles_t& levelGhost,
                         auto const& boxing, std::size_t const ti)
{
    PostMoveSyncCells const cells{boxing};
    thread_local std::vector<std::size_t> erased;

    {
        auto& tile        = domain()[ti];
        auto& ps          = tile();
        auto const reach  = damn::core::grow(*tile, domain.ghost_cells()); // the tile's cellmap box
        bool const inside = boxing.domainBox * reach == reach and !cells.touches_non_owned(*tile);
        if (ps.size() and !inside) // else interior tile, every particle is owned
        {
            erased.clear();
            damn::core::for_each_point(reach, [&](auto const& cell) {
                if (cells.owned(cell))
                    return;
                bool const to_patch_ghost = cells.to_patch_ghost(cell);
                for (auto const idx : ps.map(cell))
                {
                    if (to_patch_ghost)
                        patchGhost()[ti]().emplace_back(ps[idx]);
                    erased.push_back(idx);
                }
            });
            domain.erase_in_tile(ti, erased);
        }
    }

    auto& ps = levelGhost()[ti]();
    erased.clear();
    for (std::size_t idx = 0; idx < ps.size(); ++idx)
    {
        auto const& cell = ps[idx].iCell();
        if (cells.owned(cell))
            domain()[ti]().emplace_back(ps[idx]);
        else if (cells.keep_level_ghost(cell))
            continue;
        else if (cells.to_patch_ghost(cell))
            patchGhost()[ti]().emplace_back(ps[idx]);
        erased.push_back(idx);
    }
    levelGhost.erase_in_tile(ti, erased);
}

// sizes and views of the arrays post_move_sync_tile wrote to, once all tiles are synced
template<typename Particles_t>
void post_move_sync_finalize(Particles_t& domain, Particles_t& patchGhost, Particles_t& levelGhost)
{
    domain.on_appended();
    patchGhost.on_appended();
    levelGhost.on_appended();
}

template<typename Particles_t>
    requires(Particles_t::layout_mode == LayoutMode::AoSCMTS)
void post_move_sync(Particles_t& domain, Particles_t& patchGhost, Particles_t& levelGhost,
                    auto const& boxing)
{
    for (std::size_t ti = 0; ti < domain().size(); ++ti)
        post_move_sync_tile(domain, patchGhost, levelGhost, boxing, ti);
    post_move_sync_finalize(domain, patchGhost, levelGhost);
}

// the moments of a patch's domain particles, reset first, to be completed by the border sums: as
//  at level initialization, after mode all's push and the particle exchange, every particle on
//  the patch owning its cell (PHARE's depositDomain after the exchange: leavers deposited once, by
//  their new patch)
template<std::size_t dim, std::size_t interp_order>
void deposit_domain(auto& ions, auto const& layout)
{
    DAMN_LOG_SCOPE(3, "ParticlesSynchronisedMover::depositDomain");
    using Pop_t       = std::decay_t<decltype(*std::begin(ions))>;
    using Particles_t = std::decay_t<decltype(std::declval<Pop_t&>().domainParticles())>;
    resetMoments(ions);
    depositParticles(ions, layout,
                     DepositInterpolating<dim, interp_order, Particles_t::alloc_mode>{},
                     DomainDeposit{}); // tiles too
}

/**
 * @brief ParticlesSynchronisedMover moves/deposits a single patch's ions/electromag, given
 * directly. Mode all moves only (deposit_domain after the exchange), domain_only deposits the
 * pushed copies.
 */
template<typename Particles_t, typename Layout>
class ParticlesSynchronisedMover
{
    static constexpr auto dimension    = Layout::dimension;
    static constexpr auto interp_order = Layout::options.interp_order;
    using ParticleRange                = damn::core::IndexRange<Particles_t>;
    using Selector_t                   = std::function<ParticleRange(ParticleRange&)>;
    using Interpolator                 = plamn::core::Interpolator<dimension, interp_order>;
    using BoundaryCondition            = plamn::core::BoundaryCondition<dimension, interp_order>;
    using Pusher                       = plamn::core::BorisPusher<dimension>;

public:
    using Boxing_t = UpdaterCellMapSelectionBoxing<Selector_t, Layout>;

    MoveSpec constexpr static all_spec{}; // push and sync, the domain deposit after the exchange

    void updatePopulations(auto& ions, auto const& em, Boxing_t const& boxing, double dt,
                           UpdaterMode mode = UpdaterMode::all);

    // its in place push syncs as it goes (updateAll_): no spec without the sync
    template<MoveSpec spec, DepositSpec deposit_spec = DepositSpec{}>
    void update(auto& ions, auto const& em, Boxing_t const& boxing, double dt);

    // mode all moves only: the domain deposit, after the exchange (deposit_domain)
    static void depositDomain(auto& ions, auto const& layout)
    {
        deposit_domain<dimension, interp_order>(ions, layout);
    }

    void reset() { tmp_particles_ = Particles_t{}; }

private:
    void updateAndDepositDomain_(auto& ions, auto const& em, Boxing_t const& boxing);
    void updateAll_(auto& ions, auto const& em, Boxing_t const& boxing);

    Pusher pusher_;
    Interpolator interpolator_;
    std::optional<Particles_t> tmp_particles_;
};


template<typename Particles_t, typename Layout>
void ParticlesSynchronisedMover<Particles_t, Layout>::updatePopulations(auto& ions, auto const& em,
                                                                        Boxing_t const& boxing,
                                                                        double dt, UpdaterMode mode)
{
    if (mode == UpdaterMode::domain_only)
        update<copy_spec>(ions, em, boxing, dt);
    else
        update<all_spec>(ions, em, boxing, dt);
}

template<typename Particles_t, typename Layout>
template<MoveSpec spec, DepositSpec deposit_spec>
void ParticlesSynchronisedMover<Particles_t, Layout>::update(auto& ions, auto const& em,
                                                             Boxing_t const& boxing, double dt)
{
    check_move_spec<spec>();
    static_assert(spec.copy or spec.sync, "AoSMapped: its in place push syncs as it goes");
    DAMN_LOG_SCOPE(3, "ParticlesSynchronisedMover::updatePopulations");

    pusher_.setMeshAndTimeStep(boxing.layout.dx, dt);

    if constexpr (spec.copy)
    {
        resetMoments(ions);
        updateAndDepositDomain_(ions, em, boxing);
    }
    else
    {
        updateAll_(ions, em, boxing);
        if constexpr (spec.deposit)
            depositDomain(ions, boxing.layout);
    }
}


/**
 * @brief ParticlesSynchronisedMover::updateAndDepositDomain_
   evolves moments from time n to n+1 without updating particles, which stay at time n
 */
template<typename Particles_t, typename Layout>
void ParticlesSynchronisedMover<Particles_t, Layout>::updateAndDepositDomain_(
    auto& ions, auto const& em, Boxing_t const& boxing)
{
    DAMN_LOG_SCOPE(3, "ParticlesSynchronisedMover::updateAndDepositDomain_");

    auto const& layout = boxing.layout;

    for (auto& pop : ions)
    {
        tmp_particles_ = pop.domainParticles(); // make local copy
        auto& domain   = *tmp_particles_;

        // first push all domain particles twice
        // accumulate those inNonLevelGhostBox
        auto outRange = damn::core::makeIndexRange(domain);
        auto allowed  = outRange
            = pusher_.move(outRange, outRange, em, pop.charge(), pop.mass(), interpolator_, layout,
                           boxing.noop, boxing.inNonLevelGhostBox);

        interpolator_(allowed, pop.particleDensity(), pop.flux(), layout);

        // level ghosts: pushed while in the ghost box, deposited where they land on the level,
        //  as domain particles (each is held by one patch only)

        auto pushAndAccumulateGhosts = [&](auto const& inputArray) {
            tmp_particles_ = inputArray; // work on local copy

            auto outRange = damn::core::makeIndexRange(*tmp_particles_);

            auto enteredInDomain
                = pusher_.move(outRange, outRange, em, pop.charge(), pop.mass(), interpolator_,
                               layout, boxing.inGhostBox, boxing.inNonLevelGhostBox);

            interpolator_(enteredInDomain, pop.particleDensity(), pop.flux(), layout);
        };

        // !TODO REVISE!
        // After this function is done domain particles overlaping ghost layers of neighbor patches
        // are sent to these neighbor's patchghost particle array.
        // After being pushed, some patch ghost particles may enter the domain. These need to be
        // copied into the domain array so they are transfered to the neighbor patch
        // ghost array and contribute to moments there too.
        // On the contrary level ghost particles entering the domain here do not need to be copied
        // since they contribute to nodes that are not shared with neighbor patches an since
        // level border nodes will receive contributions from levelghost old and new particles

        if (pop.levelGhostParticles().size())
            pushAndAccumulateGhosts(pop.levelGhostParticles());
    }
}


/**
 * @brief ParticlesSynchronisedMover::updateAll_
   evolves moments and particles from time n to n+1
 */
template<typename Particles_t, typename Layout>
void ParticlesSynchronisedMover<Particles_t, Layout>::updateAll_(auto& ions, auto const& em,
                                                                 Boxing_t const& boxing)
{
    DAMN_LOG_SCOPE(1, "ParticlesSynchronisedMover::updateAll_");

    auto const& layout = boxing.layout;

    // push domain particles, erase from array those leaving domain
    // push level ghost particles that are in ghost area (==ghost box without domain)
    // copy ghost particles out of ghost area that are in domain, in particle array
    // finally all particles in non level ghost box are to be interpolated on mesh.
    for (auto& pop : ions)
    {
        auto& domainParticles = pop.domainParticles();
        auto domainPartRange  = damn::core::makeIndexRange(domainParticles);

        auto inDomain = pusher_.move(domainPartRange, domainPartRange, em, pop.charge(), pop.mass(),
                                     interpolator_, layout, boxing.noop, boxing.inOwnedDomain);

        auto now_ghosts
            = damn::core::makeRange(domainParticles, inDomain.iend(), domainParticles.size());
        auto const not_level_ghosts = boxing.inNonLevelGhostBox(now_ghosts);

        // copy out new patch ghosts
        auto& patchGhost = pop.patchGhostParticles();
        patchGhost.reserve(patchGhost.size() + not_level_ghosts.size());
        std::copy(not_level_ghosts.begin(), not_level_ghosts.end(), std::back_inserter(patchGhost));

        DAMN_DEBUG_DO({
            auto const outsideGhostBox = boxing.outsideGhostBox(now_ghosts);
            for (auto const& particle : outsideGhostBox)
            {
                DAMN_LOG_LINE_SS(particle);
                auto const nearbyBox
                    = damn::core::grow(damn::core::Box(particle.iCell(), particle.iCell()), 3);
                for (auto const& xyz : em.E)
                    if (auto const overlap = nearbyBox * qty_ghost_box(layout, xyz))
                        for_each_amr_local(layout, *overlap, [&](auto const& bix, auto const& lix) {
                            DAMN_LOG_LINE_SS(xyz.name() << " at:" << bix << ":" << xyz(lix));
                        });
            }
            if (outsideGhostBox.size())
                throw plamn::core::DictionaryException{}("ID", "Updater::outsideGhostBox");
        })

        domainParticles.erase(now_ghosts); // drop all ghosts

        if (pop.levelGhostParticles().size())
        {
            auto particleRange = damn::core::makeIndexRange(pop.levelGhostParticles());
            auto inGhostLayerRange
                = pusher_.move(particleRange, particleRange, em, pop.charge(), pop.mass(),
                               interpolator_, layout, boxing.inGhostBox, boxing.inGhostLayer);

            auto& particleArray = particleRange.array();
            particleArray.export_particles(domainParticles,
                                           [&](auto const& cell) { return boxing.isOwned(cell); });
            // in another patch's cell: handed to it, as a patch ghost
            particleArray.export_particles(pop.patchGhostParticles(), [&](auto const& cell) {
                return !boxing.isOwned(cell)
                       and (isIn(cell, boxing.domainBox)
                            or isIn(damn::core::Point{cell}, boxing.nonLevelGhostBox));
            });

            particleArray.erase(damn::core::makeRange(particleArray, inGhostLayerRange.iend(),
                                                      particleArray.size()));
        }
    }
}


/**
 * @brief ParallelParticlesSynchronisedMover moves/deposits ions across every patch of a level in
 * parallel, given a ModelAccessor rather than direct ions/electromag references.
 */
template<typename Particles_t, typename Layout>
class ParallelParticlesSynchronisedMover
{
    static constexpr auto dimension    = Layout::dimension;
    static constexpr auto interp_order = Layout::options.interp_order;
    using Interpolator_t               = Interpolator<dimension, interp_order, /*atomic=*/false>;
    using Interpolating_t              = Interpolating<dimension, interp_order, /*atomic=*/false>;

public:
    using Boxing_t = UpdaterSelectionBoxing<Layout>;

    MoveSpec constexpr static all_spec{}; // push and sync, the domain deposit after the exchange

    void updatePopulations(auto& accessor, PatchBoxings auto const& boxings, double const& dt,
                           UpdaterMode mode = UpdaterMode::all);

    template<MoveSpec spec, DepositSpec deposit_spec = DepositSpec{}>
    void update(auto& accessor, PatchBoxings auto const& boxings, double const& dt);

    // mode all moves only: the domain deposit, after the exchange (deposit_domain)
    static void depositDomain(auto& ions, auto const& layout)
    {
        deposit_domain<dimension, interp_order>(ions, layout);
    }

    void reset() {}

protected:
    // per patch steps, shared with mkn_xyz::ParallelParticlesSynchronisedMover (different dispatch)
    static void reset_moments(auto& accessor);
    template<MoveSpec spec>
    static void post_move_sync(auto& accessor, auto const& boxings, std::size_t const i);
    static void deposit(auto& accessor, auto const& boxings, std::size_t const i);

    double dt_ = 0;

private:
    void updateAndDepositDomain_(auto& accessor, PatchBoxings auto const& boxings);
    void move_(auto& accessor, PatchBoxings auto const& boxings);
    template<MoveSpec spec>
    static void sync_(auto& accessor, PatchBoxings auto const& boxings);
    static void deposit_all_(auto& accessor, PatchBoxings auto const& boxings);

    static auto n_tiles_(auto& view)
    {
        auto& ions = std::get<0>(view.args);
        return ions.size() ? tile_count(ions[0].particleDensity()) : std::size_t{0};
    }

    template<MoveSpec spec>
    static void post_move_sync_(auto& view, auto const& boxings);
    static void deposit_tile_(auto& view, std::size_t const t);
};


template<typename Particles_t, typename Layout>
void ParallelParticlesSynchronisedMover<Particles_t, Layout>::reset_moments(auto& accessor)
{
    if constexpr (not is_tiled(Particles_t::layout_mode))
        for (std::size_t i = 0; i < accessor.size(); ++i)
        {
            auto view      = accessor[i];
            auto [ions, _] = view.args;
            resetMoments(ions);
        }
    else
        damn::core::ThreadPool::INSTANCE().tile_exec(
            accessor.size(), [&](auto const i) { return accessor[i]; },
            [](auto& view) { return n_tiles_(view); },
            [](auto& view, auto const t) { resetMoments(std::get<0>(view.args), t); });
}


template<typename Particles_t, typename Layout>
template<MoveSpec spec>
void ParallelParticlesSynchronisedMover<Particles_t, Layout>::post_move_sync_(auto& view,
                                                                              auto const& boxings)
{
    auto& [ions, _]           = view.args;
    auto const& patch_boxings = boxings.at(view.patchID());

    for (auto& pop : ions)
    {
        core::post_push_sync<spec>(pop.domainParticles(), pop.patchGhostParticles(),
                                   pop.levelGhostParticles(), patch_boxings);
        if constexpr (Particles_t::layout_mode == LayoutMode::SoAVXChunked)
        { // offsets stale after the push and the sync
            pop.domainParticles().template on_moved<ParticleType::Domain>();
            pop.patchGhostParticles().template on_moved<ParticleType::PatchGhost>();
            pop.levelGhostParticles().template on_moved<ParticleType::LevelGhost>();
        }
    }
}

template<typename Particles_t, typename Layout>
template<MoveSpec spec>
void ParallelParticlesSynchronisedMover<Particles_t, Layout>::post_move_sync(auto& accessor,
                                                                             auto const& boxings,
                                                                             std::size_t const i)
{
    auto view = accessor[i];
    post_move_sync_<spec>(view, boxings);
}


// only writes the moments of tile t, domain and patch ghosts of a tile in the same task
template<typename Particles_t, typename Layout>
void ParallelParticlesSynchronisedMover<Particles_t, Layout>::deposit_tile_(auto& view,
                                                                            std::size_t const t)
{
    Interpolating_t interp;
    for (auto& pop : std::get<0>(view.args))
    {
        interp.particleToMesh_tile(t, pop.domainParticles(), pop.particleDensity(), pop.flux());
        interp.particleToMesh_tile(t, pop.patchGhostParticles(), pop.particleDensity(), pop.flux());
    }
}

template<typename Particles_t, typename Layout>
void ParallelParticlesSynchronisedMover<Particles_t, Layout>::deposit(auto& accessor,
                                                                      auto const& boxings,
                                                                      std::size_t const i)
{
    auto view            = accessor[i];
    auto [ions, _]       = view.args;
    auto const patch_id  = view.patchID();
    auto const& boxing_i = boxings.at(patch_id);
    Interpolating_t interp;
    for (auto& pop : ions)
    {
        interp.particleToMesh(pop.domainParticles(), boxing_i.layout, pop.particleDensity(),
                              pop.flux());
        interp.particleToMesh(pop.patchGhostParticles(), boxing_i.layout, pop.particleDensity(),
                              pop.flux());
    }
}


template<typename Particles_t, typename Layout>
void ParallelParticlesSynchronisedMover<Particles_t, Layout>::updatePopulations(
    auto& accessor, PatchBoxings auto const& boxings, double const& dt, UpdaterMode mode)
{
    if (mode == UpdaterMode::domain_only)
        update<copy_spec>(accessor, boxings, dt);
    else
        update<all_spec>(accessor, boxings, dt);
}

template<typename Particles_t, typename Layout>
template<MoveSpec spec, DepositSpec deposit_spec>
void ParallelParticlesSynchronisedMover<Particles_t, Layout>::update(
    auto& accessor, PatchBoxings auto const& boxings, double const& dt)
{
    check_move_spec<spec>();
    DAMN_LOG_SCOPE(2, "ParticlesSynchronisedMover::updatePopulations");

    dt_ = dt;
    if constexpr (spec.copy)
    {
        reset_moments(accessor);
        updateAndDepositDomain_(accessor, boxings);
    }
    else
    {
        if (accessor.size() == 0)
            return;
        DAMN_LOG_SCOPE(1, "ParticlesSynchronisedMover::updateAll_");
        move_(accessor, boxings);
        if constexpr (spec.sync)
            sync_<spec>(accessor, boxings);
        if constexpr (spec.deposit)
            deposit_all_(accessor, boxings);
    }
}


template<typename Particles_t, typename Layout>
void ParallelParticlesSynchronisedMover<Particles_t, Layout>::updateAndDepositDomain_(
    auto& accessor, PatchBoxings auto const& boxings)
{
    DAMN_LOG_SCOPE(1, "ParticlesSynchronisedMover::updateAndDepositDomain_");

    if (accessor.size() == 0)
        return;

    using Accessor_t = std::remove_reference_t<decltype(accessor)>;
    MultiBoris<Accessor_t, Interpolator_t>{dt_, accessor}.template move<MultiBorisMode::COPY>(
        boxings);
}


template<typename Particles_t, typename Layout>
void ParallelParticlesSynchronisedMover<Particles_t, Layout>::move_(
    auto& accessor, PatchBoxings auto const& boxings)
{
    using Accessor_t = std::remove_reference_t<decltype(accessor)>;
    DAMN_LOG_SCOPE(1, "ParticlesSynchronisedMover::updateAll_::multi_boris");
    MultiBoris<Accessor_t, Interpolator_t>{dt_, accessor}.move(boxings);
}


// the moments of every patch, reset first, from its domain and patch ghost particles (tiled: a
//  task per tile)
template<typename Particles_t, typename Layout>
void ParallelParticlesSynchronisedMover<Particles_t, Layout>::deposit_all_(
    auto& accessor, PatchBoxings auto const& boxings)
{
    DAMN_LOG_SCOPE(1, "ParticlesSynchronisedMover::updateAll_::deposit");
    reset_moments(accessor);
    auto& tp = damn::core::ThreadPool::INSTANCE();
    if constexpr (is_tiled(Particles_t::layout_mode))
        tp.tile_exec(
            accessor.size(), [&](auto const i) { return accessor[i]; },
            [](auto& view) { return n_tiles_(view); },
            [](auto& view, auto const t) { deposit_tile_(view, t); });
    else
        for (std::size_t i = 0; i < accessor.size(); ++i)
            deposit(accessor, boxings, i);
}


template<typename Particles_t, typename Layout>
template<MoveSpec spec>
void ParallelParticlesSynchronisedMover<Particles_t, Layout>::sync_(
    auto& accessor, PatchBoxings auto const& boxings)
{
    // AoSCMTS: only writes tile t of each population's arrays, see post_move_sync_tile
    auto const post_move_sync_tile = [&](auto& view, auto const t) {
        auto const& patch_boxings = boxings.at(view.patchID());
        for (auto& pop : std::get<0>(view.args))
            core::post_move_sync_tile(pop.domainParticles(), pop.patchGhostParticles(),
                                      pop.levelGhostParticles(), patch_boxings, t);
    };
    auto const post_move_sync_finalize = [](auto& view) {
        for (auto& pop : std::get<0>(view.args))
            core::post_move_sync_finalize(pop.domainParticles(), pop.patchGhostParticles(),
                                          pop.levelGhostParticles());
    };

    {
        auto& tp = damn::core::ThreadPool::INSTANCE();
        {
            DAMN_LOG_SCOPE(1, "ParticlesSynchronisedMover::updateAll_::post_move_sync");
            if constexpr (Particles_t::layout_mode == LayoutMode::AoSCMTS)
            {
                tp.tile_exec(
                    accessor.size(), [&](auto const i) { return accessor[i]; },
                    [](auto& view) { return n_tiles_(view); }, post_move_sync_tile);
                tp.patch_exec(
                    accessor.size(), [&](auto const i) { return accessor[i]; },
                    post_move_sync_finalize);
            }
            else
                tp.patch_exec( // moves particles between tiles
                    accessor.size(), [&](auto const i) { return accessor[i]; },
                    [&](auto& view) { post_move_sync_<spec>(view, boxings); });
        }
    }
}


/**
 * @brief FlatParallelParticlesSynchronisedMover is ParallelParticlesSynchronisedMover for untiled
 * layouts (AoS): untiled fields, so a patch is the unit of work rather than a tile.
 */
template<typename Particles_t, typename Layout>
class FlatParallelParticlesSynchronisedMover
{
    static constexpr auto dimension    = Layout::dimension;
    static constexpr auto interp_order = Layout::options.interp_order;
    using Interpolator_t               = Interpolator<dimension, interp_order, /*atomic=*/false>;
    using Interpolating_t              = Interpolating<dimension, interp_order, /*atomic=*/false>;

public:
    using Boxing_t = UpdaterSelectionBoxing<Layout>;

    MoveSpec constexpr static all_spec{}; // push and sync, the domain deposit after the exchange

    void updatePopulations(auto& accessor, PatchBoxings auto const& boxings, double const& dt,
                           UpdaterMode mode = UpdaterMode::all);

    template<MoveSpec spec, DepositSpec deposit_spec = DepositSpec{}>
    void update(auto& accessor, PatchBoxings auto const& boxings, double const& dt);

    // mode all moves only: the domain deposit, after the exchange (deposit_domain)
    static void depositDomain(auto& ions, auto const& layout)
    {
        deposit_domain<dimension, interp_order>(ions, layout);
    }

    void reset() {}

private:
    static void per_patch(auto& accessor, auto const& fn);
    template<MoveSpec spec>
    static void sync_(auto& accessor, PatchBoxings auto const& boxings);

    double dt_ = 0;
};


template<typename Particles_t, typename Layout>
void FlatParallelParticlesSynchronisedMover<Particles_t, Layout>::per_patch(auto& accessor,
                                                                            auto const& fn)
{
    damn::core::ThreadPool::INSTANCE().patch_exec(
        accessor.size(), [&](auto const i) { return accessor[i]; }, fn);
}


template<typename Particles_t, typename Layout>
void FlatParallelParticlesSynchronisedMover<Particles_t, Layout>::updatePopulations(
    auto& accessor, PatchBoxings auto const& boxings, double const& dt, UpdaterMode mode)
{
    if (mode == UpdaterMode::domain_only)
        update<copy_spec>(accessor, boxings, dt);
    else
        update<all_spec>(accessor, boxings, dt);
}


template<typename Particles_t, typename Layout>
template<MoveSpec spec, DepositSpec deposit_spec>
void FlatParallelParticlesSynchronisedMover<Particles_t, Layout>::update(
    auto& accessor, PatchBoxings auto const& boxings, double const& dt)
{
    check_move_spec<spec>();
    DAMN_LOG_SCOPE(2, "FlatParallelParticlesSynchronisedMover::updatePopulations");

    if (accessor.size() == 0)
        return;

    dt_ = dt;

    using Accessor_t = std::remove_reference_t<decltype(accessor)>;
    MultiBoris<Accessor_t, Interpolator_t> multi_boris{dt_, accessor};

    auto const reset_moments = [&]() {
        DAMN_LOG_SCOPE(2,
                       "FlatParallelParticlesSynchronisedMover::updatePopulations::reset_moments");
        per_patch(accessor, [](auto& view) { resetMoments(std::get<0>(view.args)); });
    };

    if constexpr (spec.copy)
    {
        reset_moments();
        DAMN_LOG_SCOPE(2, "FlatParallelParticlesSynchronisedMover::updatePopulations::move_copy");
        multi_boris.template move<MultiBorisMode::COPY>(boxings);
        return;
    }

    {
        DAMN_LOG_SCOPE(2, "FlatParallelParticlesSynchronisedMover::updatePopulations::move");
        multi_boris.move(boxings);
    }

    if constexpr (spec.sync)
        sync_<spec>(accessor, boxings);

    if constexpr (spec.deposit)
    {
        reset_moments();
        DAMN_LOG_SCOPE(2, "FlatParallelParticlesSynchronisedMover::updatePopulations::deposit");
        per_patch(accessor, [&](auto& view) {
            auto const& layout = boxings.at(view.patchID()).layout;
            Interpolating_t interp;
            for (auto& pop : std::get<0>(view.args))
            {
                interp.particleToMesh(pop.domainParticles(), layout, pop.particleDensity(),
                                      pop.flux());
                interp.particleToMesh(pop.patchGhostParticles(), layout, pop.particleDensity(),
                                      pop.flux());
            }
        });
    }
}


template<typename Particles_t, typename Layout>
template<MoveSpec spec>
void FlatParallelParticlesSynchronisedMover<Particles_t, Layout>::sync_(
    auto& accessor, PatchBoxings auto const& boxings)
{
    DAMN_LOG_SCOPE(2, "FlatParallelParticlesSynchronisedMover::updatePopulations::per_patch");
    per_patch(accessor, [&](auto& view) {
        auto const& boxing = boxings.at(view.patchID());
        for (auto& pop : std::get<0>(view.args))
        {
            {
                DAMN_LOG_SCOPE(2, "FlatParallelParticlesSynchronisedMover::post_move_sync");
                core::post_push_sync<spec>(pop.domainParticles(), pop.patchGhostParticles(),
                                           pop.levelGhostParticles(), boxing);
            }
            if constexpr (Particles_t::layout_mode == LayoutMode::SoAVXChunked)
            {
                DAMN_LOG_SCOPE(2, "FlatParallelParticlesSynchronisedMover::on_moved");
                // chunks stale after the push and the sync, but the domain's (resettled)
                if (!pop.domainParticles().sorted())
                    pop.domainParticles().template on_moved<ParticleType::Domain>();
                pop.patchGhostParticles().template on_moved<ParticleType::PatchGhost>();
                pop.levelGhostParticles().template on_moved<ParticleType::LevelGhost>();
            }
        }
    });
}


} // namespace plamn::core


#if DAMN_HAVE_MKN_GPU

namespace plamn::core::mkn_xyz
{

/**
 * @brief mkn_xyz::ParallelParticlesSynchronisedMover: core::ParallelParticlesSynchronisedMover
 * steps dispatched with mkn.gpu ThreadedStreamLauncher (one host thread + stream per patch), and
 * GPU deposit kernels
 */
template<typename Particles_t, typename Layout>
class ParallelParticlesSynchronisedMover
    : public core::ParallelParticlesSynchronisedMover<Particles_t, Layout>
{
    using Super = core::ParallelParticlesSynchronisedMover<Particles_t, Layout>;

    static constexpr auto dimension    = Layout::dimension;
    static constexpr auto interp_order = Layout::options.interp_order;
    static constexpr bool atomic_interp
        = Particles_t::alloc_mode == damn::AllocatorMode::GPU_UNIFIED;
    using Interpolator_t  = Interpolator<dimension, interp_order, atomic_interp>;
    using Interpolating_t = Interpolating<dimension, interp_order, atomic_interp>;

public:
    using Boxing_t = Super::Boxing_t;

    // damn's synchronous pools (DAMN_THREAD_POOL_SYNC): the host steps on this thread too
    auto constexpr static synchronous = damn::core::ThreadPool::synchronous;

    // push, sync and deposit on the device
    MoveSpec constexpr static all_spec{.deposit = true};

    void updatePopulations(auto& accessor, PatchBoxings auto const& boxings, double const& dt,
                           UpdaterMode mode = UpdaterMode::all);

    template<MoveSpec spec, DepositSpec deposit_spec = DepositSpec{}>
    void update(auto& accessor, PatchBoxings auto const& boxings, double const& dt);

private:
    void updateAndDepositDomain_(auto& accessor, PatchBoxings auto const& boxings);
    template<MoveSpec spec, DepositSpec deposit_spec = DepositSpec{}>
    void updateAndDepositAll_(auto& accessor, PatchBoxings auto const& boxings);

    // untiled GPU_UNIFIED: moments zeroed on each patch's stream, ahead of its push / deposit
    static constexpr bool moments_on_streams
        = Particles_t::alloc_mode == damn::AllocatorMode::GPU_UNIFIED
          and not is_tiled(Particles_t::layout_mode);

    void reset_moments_(auto& accessor, auto& in);
};


template<typename Particles_t, typename Layout>
void ParallelParticlesSynchronisedMover<Particles_t, Layout>::reset_moments_(auto& accessor,
                                                                             auto& in)
{
    if constexpr (moments_on_streams)
    {
        auto zero = [&](auto const i) mutable {
            auto view      = accessor[i];
            auto [ions, _] = view.args;
            auto& stream   = in.streamer.streams[i];
            for (auto& pop : ions)
            {
                auto& density = pop.particleDensity();
                damn::core::gpu::zero_async(density.data(), density.size(), stream);
                for (auto& component : pop.flux())
                    damn::core::gpu::zero_async(component.data(), component.size(), stream);
            }
        };

        if constexpr (synchronous)
            for (std::size_t i = 0; i < accessor.size(); ++i)
                zero(i);
        else
            in.streamer.host(std::move(zero));
    }
}


template<typename Particles_t, typename Layout>
void ParallelParticlesSynchronisedMover<Particles_t, Layout>::updatePopulations(
    auto& accessor, PatchBoxings auto const& boxings, double const& dt, UpdaterMode mode)
{
    if (mode == UpdaterMode::domain_only)
        update<copy_spec>(accessor, boxings, dt);
    else
        update<all_spec>(accessor, boxings, dt);
}

template<typename Particles_t, typename Layout>
template<MoveSpec spec, DepositSpec deposit_spec>
void ParallelParticlesSynchronisedMover<Particles_t, Layout>::update(
    auto& accessor, PatchBoxings auto const& boxings, double const& dt)
{
    check_move_spec<spec>();
    DAMN_LOG_SCOPE(2, "mkn_xyz::ParticlesSynchronisedMover::updatePopulations");

    if constexpr (not moments_on_streams) // else zeroed on each patch's stream, see reset_moments_
        Super::reset_moments(accessor);
    this->dt_ = dt;
    if constexpr (spec.copy)
        updateAndDepositDomain_(accessor, boxings);
    else
        updateAndDepositAll_<spec, deposit_spec>(accessor, boxings);
}


template<typename Particles_t, typename Layout>
void ParallelParticlesSynchronisedMover<Particles_t, Layout>::updateAndDepositDomain_(
    auto& accessor, PatchBoxings auto const& boxings)
{
    DAMN_LOG_SCOPE(1, "mkn_xyz::ParticlesSynchronisedMover::updateAndDepositDomain_");

    if (accessor.size() == 0)
        return;

    using Accessor_t = std::remove_reference_t<decltype(accessor)>;
    MultiBoris<Accessor_t, Interpolator_t> in{this->dt_, accessor};
    reset_moments_(accessor, in);
    in.template move<MultiBorisMode::COPY>(boxings);

    if constexpr (not synchronous)
    {
        in.streamer.join();
        in.streamer.dump_times(detail::timings_dir_str + "/updateAndDepositDomain.txt");
    }
}


template<typename Particles_t, typename Layout>
template<MoveSpec spec, DepositSpec deposit_spec>
void ParallelParticlesSynchronisedMover<Particles_t, Layout>::updateAndDepositAll_(
    auto& accessor, PatchBoxings auto const& boxings)
{
    DAMN_LOG_SCOPE(1, "mkn_xyz::ParticlesSynchronisedMover::updateAndDepositAll_");

    if (accessor.size() == 0)
        return;

    using Accessor_t = std::remove_reference_t<decltype(accessor)>;
    using Tile_vt    = Accessor_t::Model_t::electromag_type::vecfield_type::field_type::value_type;

    MultiBoris<Accessor_t, Interpolator_t> in{this->dt_, accessor};
    reset_moments_(accessor, in);
    in.move(boxings);

    if constexpr (spec.sync)
    {
        auto post_move_sync = [&](auto const i) mutable {
            Super::template post_move_sync<spec>(accessor, boxings, i);
        };

        if constexpr (synchronous)
            for (std::size_t i = 0; i < accessor.size(); ++i)
                post_move_sync(i);
        else
            in.streamer.host(std::move(post_move_sync));
    }

    if constexpr (not spec.deposit)
    {
        // the push (and sync) alone: nothing deposited
    }
    else if constexpr (Particles_t::alloc_mode == damn::AllocatorMode::GPU_UNIFIED
                       and is_tiled(Particles_t::layout_mode))
    {
        auto deposit = [&](auto const i) mutable {
            auto view      = accessor[i];
            auto [ions, _] = view.args;
            for (std::size_t j = 0; j < ions.size(); ++j)
            {
                auto& pop = ions[j];

                if constexpr (Particles_t::layout_mode == LayoutMode::AoSPCTS)
                {
                    // block = (tile, cell), threads stride the cell (see on_tile_cells)
                    auto const deposit_tiles = [&](auto& particles) {
                        using Launcher       = damn::core::gpu::TileCellLauncher<false>;
                        auto const parts     = *particles;
                        auto const max_cells = Launcher::max_cells(parts);
                        if (max_cells == 0)
                            return;

                        std::size_t const threads = damn::kernel::warp_size();
                        auto pdensity             = pop.particleDensity();
                        auto flux                 = *pop.flux();
                        Launcher{parts().size(), max_cells, threads}.stream(
                            in.streamer.streams[i], [=] __device__() mutable {
                                Interpolating_t::template on_tile_cells<Tile_vt, Launcher>(
                                    parts, flux, pdensity, threads);
                            });
                    };
                    deposit_tiles(pop.domainParticles());
                    deposit_tiles(pop.patchGhostParticles());
                }
                else
                {
                    // largest tile field, incl. ghosts
                    auto const n_points = pop.particleDensity().max_tile_size();
                    auto const domain   = *pop.domainParticles();
                    auto pdensity       = pop.particleDensity();
                    auto flux           = *pop.flux();
                    auto const mode     = tile_deposit_mode(
                        n_points, 0, damn::core::gpu::max_dynamic_shared_bytes());
                    with_tile_deposit(mode, [&](auto deposit_c) {
                        using Launcher = damn::core::gpu::ChunkLauncher<false>;
                        Launcher launcher{1, 0};
                        launcher.b.x = damn::kernel::warp_size() * deposit_spec.warps;
                        launcher.g.x = domain().size();
                        launcher.ds  = tile_deposit_bytes(deposit_c(), n_points);
                        launcher.stream(in.streamer.streams[i], [=] __device__() mutable {
                            Interpolating_t::template on_tiles<Tile_vt, decltype(deposit_c)::value>(
                                domain, flux, pdensity);
                        });
                    });
                }
            }
        };

        if constexpr (synchronous)
            for (std::size_t i = 0; i < accessor.size(); ++i)
                deposit(i);
        else
            in.streamer.host(std::move(deposit));
    }
    else
    {
        auto deposit = [&](auto const i) mutable {
            if constexpr (Particles_t::alloc_mode == damn::AllocatorMode::GPU_UNIFIED)
            { // untiled: this updater's atomic interpolating, a device kernel per array
                auto view            = accessor[i];
                auto [ions, _]       = view.args;
                auto const& boxing_i = boxings.at(view.patchID());
                auto& stream         = in.streamer.streams[i];
                Interpolating_t interp;
                for (auto& pop : ions)
                {
                    interp.particleToMesh(pop.domainParticles(), boxing_i.layout,
                                          pop.particleDensity(), pop.flux(), 1., stream);
                    interp.particleToMesh(pop.patchGhostParticles(), boxing_i.layout,
                                          pop.particleDensity(), pop.flux(), 1., stream);
                }
            }
            else
                Super::deposit(accessor, boxings, i);
        };

        if constexpr (synchronous)
            for (std::size_t i = 0; i < accessor.size(); ++i)
                deposit(i);
        else
            in.streamer.host(std::move(deposit));
    }

    if constexpr (not synchronous)
    {
        in.streamer.join();
        in.streamer.dump_times(detail::timings_dir_str + "/updateAndDepositAll.txt");
    }


#if DAMN_DEBUG
    for (std::size_t i = 0; i < accessor.size(); ++i)
    {
        auto view                 = accessor[i];
        auto [ions, _]            = view.args;
        auto const patch_id       = view.patchID();
        auto const& patch_boxings = boxings.at(patch_id);

        for (std::size_t j = 0; j < ions.size(); ++j)
        {
            auto& pop = ions[j];
            particles_domain_is_valid(pop.domainParticles(), patch_boxings.domainBox);
            particles_ghost_is_valid(pop.patchGhostParticles(), patch_boxings.domainBox,
                                     patch_boxings.ghostBox);
            particles_ghost_is_valid(pop.levelGhostParticles(), patch_boxings.domainBox,
                                     patch_boxings.ghostBox);
        }
    }
#endif /* DAMN_DEBUG */
}


} // namespace plamn::core::mkn_xyz

#endif // DAMN_HAVE_MKN_GPU


namespace plamn::core
{
namespace detail
{
    // layouts in unified memory update on the device: mkn_xyz::ParallelParticlesSynchronisedMover,
    // as
    //  test_multi_updater checks them (UpdaterVersion::mkn_parallel), CPU tiled:
    //  ParallelParticlesSynchronisedMover
    template<typename Particles_t, typename Layout>
    using TiledParticlesSynchronisedMoverFor_t =
#if DAMN_HAVE_MKN_GPU
        std::conditional_t<Particles_t::alloc_mode == damn::AllocatorMode::GPU_UNIFIED,
                           mkn_xyz::ParallelParticlesSynchronisedMover<Particles_t, Layout>,
                           ParallelParticlesSynchronisedMover<Particles_t, Layout>>;
#else
        ParallelParticlesSynchronisedMover<Particles_t, Layout>;
#endif
} // namespace detail

// unified memory updates on the device (mkn_xyz::ParallelParticlesSynchronisedMover), tiled or not:
// untiled
//  GPU_UNIFIED (SoAVXChunked) through FlatParallelParticlesSynchronisedMover deposited on the host
template<typename Particles_t, typename Layout>
using ParticlesSynchronisedMoverFor_t = std::conditional_t<
    Particles_t::layout_mode == LayoutMode::AoSMapped,
    ParticlesSynchronisedMover<Particles_t, Layout>,
    std::conditional_t<is_tiled(Particles_t::layout_mode)
                           or Particles_t::alloc_mode == damn::AllocatorMode::GPU_UNIFIED,
                       detail::TiledParticlesSynchronisedMoverFor_t<Particles_t, Layout>,
                       FlatParallelParticlesSynchronisedMover<Particles_t, Layout>>>;

} // namespace plamn::core


#endif // ION_UPDATER_HPP
