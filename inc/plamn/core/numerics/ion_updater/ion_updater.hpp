#ifndef PLAMN_ION_UPDATER_HPP
#define PLAMN_ION_UPDATER_HPP

#include "plamn/core/def.hpp"

#include "damn/core/def/damn_config.hpp"

#include "plamn/core/errors.hpp"
#include "damn/core/logger.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/geometry/range.hpp"
#include "plamn/core/numerics/pusher/boris.hpp"
#include "plamn/core/numerics/moments/moments.hpp"
#include "plamn/core/numerics/interpolator/interpolator.hpp"
#include "plamn/core/numerics/interpolator/interpolating.hpp"
#include "plamn/core/numerics/boundary_condition/boundary_condition.hpp"

#include "plamn/core/numerics/pusher/boris/detail/multi_boris.hpp"
#include "plamn/core/data/particles/exporting/particles_exporting.hpp"

#include "ion_updater_def.hpp" // = UpdaterMode

#include <string>
#include <optional>
#include <unordered_map>


namespace plamn::core
{

// after a REF move: domain particles leaving the owned domain become patch ghosts or are
// deleted, level ghosts entering it become domain particles
template<typename Particles_t>
void post_move_sync(Particles_t& domain, Particles_t& patchGhost,
                    Particles_t& levelGhost, auto const& boxing)
{
    auto const& nonOwnedBoxes = boxing.nonOwnedBoxes;

    delete_particles_not_in(domain, boxing.nonLevelGhostBox);
    move_in_ghost_layer(patchGhost, domain, boxing.domainBox, boxing.nonLevelGhostBox);
    if (!nonOwnedBoxes.empty())
    {
        for (auto const& box : nonOwnedBoxes)
            move_in_domain(patchGhost, domain, box);
        delete_particles_in(levelGhost, nonOwnedBoxes);
    }
    move_in_domain(domain, levelGhost, boxing.domainBox);
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

    // level ghosts not owned: still in the ghost layer -> kept, else (non-owned, gone) deleted
    bool keep_level_ghost(auto const& cell) const
    {
        return !isIn(cell, boxing.domainBox) and isIn(cell, boxing.ghostBox);
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


// tiled layouts: a tile only holds particles whose cell it (clamp) owns, and emplace_back
// routes by that same owner, so tile ti of each array is only ever written from tile ti:
// distinct tiles can be synced concurrently, then post_move_sync_finalize once per patch

// AoSCMTS: only the cells of a tile's ghost layer outside the patch domain, and its non-owned
// domain cells, are visited, through the tile's cellmap. Level ghosts only live in patch
// border tiles
template<typename Particles_t>
    requires(Particles_t::layout_mode == LayoutMode::AoSCMTS)
void post_move_sync_tile(Particles_t& domain, Particles_t& patchGhost,
                         Particles_t& levelGhost, auto const& boxing, std::size_t const ti)
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
        erased.push_back(idx);
    }
    levelGhost.erase_in_tile(ti, erased);
}

// sizes and views of the arrays post_move_sync_tile wrote to, once all tiles are synced
template<typename Particles_t>
void post_move_sync_finalize(Particles_t& domain, Particles_t& patchGhost,
                             Particles_t& levelGhost)
{
    domain.on_appended();
    patchGhost.on_appended();
    levelGhost.on_appended();
}

template<typename Particles_t>
    requires(Particles_t::layout_mode == LayoutMode::AoSCMTS)
void post_move_sync(Particles_t& domain, Particles_t& patchGhost,
                    Particles_t& levelGhost, auto const& boxing)
{
    for (std::size_t ti = 0; ti < domain().size(); ++ti)
        post_move_sync_tile(domain, patchGhost, levelGhost, boxing, ti);
    post_move_sync_finalize(domain, patchGhost, levelGhost);
}

/**
 * @brief IonUpdater moves/deposits a single patch's ions/electromag, given directly.
 */
template<typename Particles_t, typename Layout>
class IonUpdater
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

    void updatePopulations(auto& ions, auto const& em, Boxing_t const& boxing, double dt,
                           UpdaterMode mode = UpdaterMode::all);

    void reset() { tmp_particles_ = Particles_t{}; }

private:
    void updateAndDepositDomain_(auto& ions, auto const& em, Boxing_t const& boxing);
    void updateAndDepositAll_(auto& ions, auto const& em, Boxing_t const& boxing);

    Pusher pusher_;
    Interpolator interpolator_;
    std::optional<Particles_t> tmp_particles_;
};


template<typename Particles_t, typename Layout>
void IonUpdater<Particles_t, Layout>::updatePopulations(auto& ions, auto const& em,
                                                                Boxing_t const& boxing, double dt,
                                                                UpdaterMode mode)
{
    DAMN_LOG_SCOPE(3, "IonUpdater::updatePopulations");

    resetMoments(ions);
    pusher_.setMeshAndTimeStep(boxing.layout.dx, dt);

    if (mode == UpdaterMode::domain_only)
        updateAndDepositDomain_(ions, em, boxing);
    else
        updateAndDepositAll_(ions, em, boxing);
}


/**
 * @brief IonUpdater::updateAndDepositDomain_
   evolves moments from time n to n+1 without updating particles, which stay at time n
 */
template<typename Particles_t, typename Layout>
void IonUpdater<Particles_t, Layout>::updateAndDepositDomain_(auto& ions, auto const& em,
                                                                      Boxing_t const& boxing)
{
    DAMN_LOG_SCOPE(3, "IonUpdater::updateAndDepositDomain_");

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

        // push those in the ghostArea (i.e. stop pushing if they're not out of it)
        // deposit moments on those which leave to go inOwnedDomain

        auto pushAndAccumulateGhosts = [&](auto const& inputArray) {
            tmp_particles_ = inputArray; // work on local copy

            auto outRange = damn::core::makeIndexRange(*tmp_particles_);

            auto enteredInDomain
                = pusher_.move(outRange, outRange, em, pop.charge(), pop.mass(), interpolator_,
                               layout, boxing.inGhostBox, boxing.inOwnedDomain);

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
 * @brief IonUpdater::updateAndDepositAll_
   evolves moments and particles from time n to n+1
 */
template<typename Particles_t, typename Layout>
void IonUpdater<Particles_t, Layout>::updateAndDepositAll_(auto& ions, auto const& em,
                                                                   Boxing_t const& boxing)
{
    DAMN_LOG_SCOPE(1, "IonUpdater::updateAndDepositAll_");

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

        auto now_ghosts = damn::core::makeRange(domainParticles, inDomain.iend(), domainParticles.size());
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
                auto const nearbyBox = damn::core::grow(damn::core::Box(particle.iCell(), particle.iCell()), 3);
                for (auto const& xyz : em.E)
                    if (auto const overlap = nearbyBox * amr_ghost_box(layout, xyz))
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

            particleArray.erase(
                damn::core::makeRange(particleArray, inGhostLayerRange.iend(), particleArray.size()));
        }

        interpolator_( //
            domainParticles, pop.particleDensity(), pop.flux(), layout);
        interpolator_( //
            patchGhost, pop.particleDensity(), pop.flux(), layout);
    }
}


/**
 * @brief ParallelIonUpdater moves/deposits ions across every patch of a level in parallel,
 * given a ModelAccessor rather than direct ions/electromag references.
 */
template<typename Particles_t, typename Layout>
class ParallelIonUpdater
{
    static constexpr auto dimension    = Layout::dimension;
    static constexpr auto interp_order = Layout::options.interp_order;
    using Interpolator_t               = Interpolator<dimension, interp_order, /*atomic=*/false>;
    using Interpolating_t              = Interpolating<dimension, interp_order, /*atomic=*/false>;

public:
    using Boxing_t = UpdaterSelectionBoxing<Layout>;

    auto constexpr static use_main_thread = MultiBorisOptions{}.use_main_thread;

    void updatePopulations(auto& accessor, std::unordered_map<std::string, Boxing_t> const& boxings,
                           double const& dt, UpdaterMode mode = UpdaterMode::all);

    void reset() {}

protected:
    // per patch steps, shared with mkn_xyz::ParallelIonUpdater (different dispatch)
    static void reset_moments(auto& accessor);
    static void post_move_sync(auto& accessor, auto const& boxings, std::size_t const i);
    static void deposit(auto& accessor, auto const& boxings, std::size_t const i);

    double dt_ = 0;

private:
    void updateAndDepositDomain_(auto& accessor,
                                 std::unordered_map<std::string, Boxing_t> const& boxings);
    void updateAndDepositAll_(auto& accessor,
                              std::unordered_map<std::string, Boxing_t> const& boxings);

    static auto n_tiles_(auto& view)
    {
        auto& ions = std::get<0>(view.args);
        return ions.size() ? tile_count(ions[0].particleDensity()) : std::size_t{0};
    }

    static void post_move_sync_(auto& view, auto const& boxings);
    static void deposit_tile_(auto& view, std::size_t const t);
};


template<typename Particles_t, typename Layout>
void ParallelIonUpdater<Particles_t, Layout>::reset_moments(auto& accessor)
{
    if constexpr (use_main_thread or not is_tiled(Particles_t::layout_mode))
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
void ParallelIonUpdater<Particles_t, Layout>::post_move_sync_(auto& view,
                                                                      auto const& boxings)
{
    auto& [ions, _]           = view.args;
    auto const& patch_boxings = boxings.at(view.patchID());

    for (auto& pop : ions)
    {
        core::post_move_sync(pop.domainParticles(), pop.patchGhostParticles(),
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
void ParallelIonUpdater<Particles_t, Layout>::post_move_sync(auto& accessor,
                                                                     auto const& boxings,
                                                                     std::size_t const i)
{
    auto view = accessor[i];
    post_move_sync_(view, boxings);
}


// only writes the moments of tile t, domain and patch ghosts of a tile in the same task
template<typename Particles_t, typename Layout>
void ParallelIonUpdater<Particles_t, Layout>::deposit_tile_(auto& view, std::size_t const t)
{
    Interpolating_t interp;
    for (auto& pop : std::get<0>(view.args))
    {
        interp.particleToMesh_tile(t, pop.domainParticles(), pop.particleDensity(), pop.flux());
        interp.particleToMesh_tile(t, pop.patchGhostParticles(), pop.particleDensity(), pop.flux());
    }
}

template<typename Particles_t, typename Layout>
void ParallelIonUpdater<Particles_t, Layout>::deposit(auto& accessor, auto const& boxings,
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
void ParallelIonUpdater<Particles_t, Layout>::updatePopulations(
    auto& accessor, std::unordered_map<std::string, Boxing_t> const& boxings, double const& dt,
    UpdaterMode mode)
{
    DAMN_LOG_SCOPE(2, "IonUpdater::updatePopulations");

    reset_moments(accessor);
    dt_ = dt;
    if (mode == UpdaterMode::domain_only)
        updateAndDepositDomain_(accessor, boxings);
    else
        updateAndDepositAll_(accessor, boxings);
}


template<typename Particles_t, typename Layout>
void ParallelIonUpdater<Particles_t, Layout>::updateAndDepositDomain_(
    auto& accessor, std::unordered_map<std::string, Boxing_t> const& boxings)
{
    DAMN_LOG_SCOPE(1, "IonUpdater::updateAndDepositDomain_");

    if (accessor.size() == 0)
        return;

    using Accessor_t = std::remove_reference_t<decltype(accessor)>;
    MultiBoris<Accessor_t, Interpolator_t>{dt_, accessor}.template move<MultiBorisMode::COPY>(
        boxings);
}


template<typename Particles_t, typename Layout>
void ParallelIonUpdater<Particles_t, Layout>::updateAndDepositAll_(
    auto& accessor, std::unordered_map<std::string, Boxing_t> const& boxings)
{
    DAMN_LOG_SCOPE(1, "IonUpdater::updateAndDepositAll_");

    if (accessor.size() == 0)
        return;

    using Accessor_t = std::remove_reference_t<decltype(accessor)>;
    {
        DAMN_LOG_SCOPE(1, "IonUpdater::updateAndDepositAll_::multi_boris");
        MultiBoris<Accessor_t, Interpolator_t>{dt_, accessor}.move(boxings);
    }

    static constexpr bool per_tile_deposit
        = damn::core::any_in(Particles_t::layout_mode, LayoutMode::AoSPCTS, LayoutMode::AoSCMTS);

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

    if constexpr (use_main_thread)
    {
        for (std::size_t i = 0; i < accessor.size(); ++i)
            post_move_sync(accessor, boxings, i);
        for (std::size_t i = 0; i < accessor.size(); ++i)
            deposit(accessor, boxings, i);
    }
    else
    {
        auto& tp = damn::core::ThreadPool::INSTANCE();
        {
            DAMN_LOG_SCOPE(1, "IonUpdater::updateAndDepositAll_::post_move_sync");
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
                    [&](auto& view) { post_move_sync_(view, boxings); });
        }
        {
            DAMN_LOG_SCOPE(1, "IonUpdater::updateAndDepositAll_::deposit");
            if constexpr (per_tile_deposit)
                tp.tile_exec(
                    accessor.size(), [&](auto const i) { return accessor[i]; },
                    [](auto& view) { return n_tiles_(view); },
                    [](auto& view, auto const t) { deposit_tile_(view, t); });
            else
            {
                for (std::size_t i = 0; i < accessor.size(); ++i)
                    tp.async([&, i] { deposit(accessor, boxings, i); });
                tp.sync();
            }
        }
    }
}


/**
 * @brief FlatParallelIonUpdater is ParallelIonUpdater for untiled layouts (AoS): untiled
 * fields, so a patch is the unit of work rather than a tile.
 */
template<typename Particles_t, typename Layout>
class FlatParallelIonUpdater
{
    static constexpr auto dimension    = Layout::dimension;
    static constexpr auto interp_order = Layout::options.interp_order;
    using Interpolator_t               = Interpolator<dimension, interp_order, /*atomic=*/false>;

public:
    using Boxing_t = UpdaterSelectionBoxing<Layout>;

    auto constexpr static use_main_thread = MultiBorisOptions{}.use_main_thread;

    void updatePopulations(auto& accessor, std::unordered_map<std::string, Boxing_t> const& boxings,
                           double const& dt, UpdaterMode mode = UpdaterMode::all);

    void reset() {}

private:
    static void per_patch(auto& accessor, auto const& fn);

    double dt_ = 0;
};


template<typename Particles_t, typename Layout>
void FlatParallelIonUpdater<Particles_t, Layout>::per_patch(auto& accessor, auto const& fn)
{
    if constexpr (use_main_thread)
        for (std::size_t i = 0; i < accessor.size(); ++i)
        {
            auto view = accessor[i];
            fn(view);
        }
    else
        damn::core::ThreadPool::INSTANCE().patch_exec(
            accessor.size(), [&](auto const i) { return accessor[i]; }, fn);
}


template<typename Particles_t, typename Layout>
void FlatParallelIonUpdater<Particles_t, Layout>::updatePopulations(
    auto& accessor, std::unordered_map<std::string, Boxing_t> const& boxings, double const& dt,
    UpdaterMode mode)
{
    DAMN_LOG_SCOPE(2, "FlatParallelIonUpdater::updatePopulations");

    if (accessor.size() == 0)
        return;

    dt_ = dt;
    {
        DAMN_LOG_SCOPE(2, "FlatParallelIonUpdater::updatePopulations::reset_moments");
        per_patch(accessor, [](auto& view) { resetMoments(std::get<0>(view.args)); });
    }

    using Accessor_t = std::remove_reference_t<decltype(accessor)>;
    MultiBoris<Accessor_t, Interpolator_t> multi_boris{dt_, accessor};

    if (mode == UpdaterMode::domain_only)
    {
        DAMN_LOG_SCOPE(2, "FlatParallelIonUpdater::updatePopulations::move_copy");
        multi_boris.template move<MultiBorisMode::COPY>(boxings);
        return;
    }

    {
        DAMN_LOG_SCOPE(2, "FlatParallelIonUpdater::updatePopulations::move");
        multi_boris.move(boxings);
    }

    DAMN_LOG_SCOPE(2, "FlatParallelIonUpdater::updatePopulations::per_patch");
    per_patch(accessor, [&](auto& view) {
        auto const& boxing = boxings.at(view.patchID());
        Interpolator_t interp;
        for (auto& pop : std::get<0>(view.args))
        {
            {
                DAMN_LOG_SCOPE(2, "FlatParallelIonUpdater::post_move_sync");
                core::post_move_sync(pop.domainParticles(), pop.patchGhostParticles(),
                                     pop.levelGhostParticles(), boxing);
            }
            if constexpr (Particles_t::layout_mode == LayoutMode::SoAVXChunked)
            {
                DAMN_LOG_SCOPE(2, "FlatParallelIonUpdater::on_moved");
                // offsets stale after the push and the sync
                pop.domainParticles().template on_moved<ParticleType::Domain>();
                pop.patchGhostParticles().template on_moved<ParticleType::PatchGhost>();
                pop.levelGhostParticles().template on_moved<ParticleType::LevelGhost>();
            }
            DAMN_LOG_SCOPE(2, "FlatParallelIonUpdater::deposit");
            interp(pop.domainParticles(), pop.particleDensity(), pop.flux(), view.layout);
            interp(pop.patchGhostParticles(), pop.particleDensity(), pop.flux(), view.layout);
        }
    });
}


} // namespace plamn::core


#if DAMN_HAVE_MKN_GPU

namespace plamn::core::mkn_xyz
{

/**
 * @brief mkn_xyz::ParallelIonUpdater: core::ParallelIonUpdater steps dispatched with mkn.gpu
 * ThreadedStreamLauncher (one host thread + stream per patch), and GPU deposit kernels
 */
template<typename Particles_t, typename Layout>
class ParallelIonUpdater : public core::ParallelIonUpdater<Particles_t, Layout>
{
    using Super = core::ParallelIonUpdater<Particles_t, Layout>;

    static constexpr auto dimension     = Layout::dimension;
    static constexpr auto interp_order  = Layout::options.interp_order;
    static constexpr bool atomic_interp = Particles_t::alloc_mode == damn::AllocatorMode::GPU_UNIFIED;
    using Interpolator_t                = Interpolator<dimension, interp_order, atomic_interp>;
    using Interpolating_t               = Interpolating<dimension, interp_order, atomic_interp>;

public:
    using Boxing_t = Super::Boxing_t;

    auto constexpr static use_main_thread = Super::use_main_thread;

    void updatePopulations(auto& accessor, std::unordered_map<std::string, Boxing_t> const& boxings,
                           double const& dt, UpdaterMode mode = UpdaterMode::all);

private:
    void updateAndDepositDomain_(auto& accessor,
                                 std::unordered_map<std::string, Boxing_t> const& boxings);
    void updateAndDepositAll_(auto& accessor,
                              std::unordered_map<std::string, Boxing_t> const& boxings);

    // untiled GPU_UNIFIED: moments zeroed on each patch's stream, ahead of its push / deposit
    static constexpr bool moments_on_streams
        = Particles_t::alloc_mode == damn::AllocatorMode::GPU_UNIFIED
          and not is_tiled(Particles_t::layout_mode);

    void reset_moments_(auto& accessor, auto& in);
};


template<typename Particles_t, typename Layout>
void ParallelIonUpdater<Particles_t, Layout>::reset_moments_(auto& accessor, auto& in)
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

        if constexpr (use_main_thread)
            for (std::size_t i = 0; i < accessor.size(); ++i)
                zero(i);
        else
            in.streamer.host(std::move(zero));
    }
}


template<typename Particles_t, typename Layout>
void ParallelIonUpdater<Particles_t, Layout>::updatePopulations(
    auto& accessor, std::unordered_map<std::string, Boxing_t> const& boxings, double const& dt,
    UpdaterMode mode)
{
    DAMN_LOG_SCOPE(2, "mkn_xyz::IonUpdater::updatePopulations");

    if constexpr (not moments_on_streams) // else zeroed on each patch's stream, see reset_moments_
        Super::reset_moments(accessor);
    this->dt_ = dt;
    if (mode == UpdaterMode::domain_only)
        updateAndDepositDomain_(accessor, boxings);
    else
        updateAndDepositAll_(accessor, boxings);
}


template<typename Particles_t, typename Layout>
void ParallelIonUpdater<Particles_t, Layout>::updateAndDepositDomain_(
    auto& accessor, std::unordered_map<std::string, Boxing_t> const& boxings)
{
    DAMN_LOG_SCOPE(1, "mkn_xyz::IonUpdater::updateAndDepositDomain_");

    if (accessor.size() == 0)
        return;

    using Accessor_t = std::remove_reference_t<decltype(accessor)>;
    MultiBoris<Accessor_t, Interpolator_t> in{this->dt_, accessor};
    reset_moments_(accessor, in);
    in.template move<MultiBorisMode::COPY>(boxings);

    if constexpr (not use_main_thread)
    {
        in.streamer.join();
        in.streamer.dump_times(detail::timings_dir_str + "/updateAndDepositDomain.txt");
    }
}


template<typename Particles_t, typename Layout>
void ParallelIonUpdater<Particles_t, Layout>::updateAndDepositAll_(
    auto& accessor, std::unordered_map<std::string, Boxing_t> const& boxings)
{
    DAMN_LOG_SCOPE(1, "mkn_xyz::IonUpdater::updateAndDepositAll_");

    if (accessor.size() == 0)
        return;

    using Accessor_t = std::remove_reference_t<decltype(accessor)>;
    using Tile_vt    = Accessor_t::Model_t::electromag_type::vecfield_type::field_type::value_type;

    MultiBoris<Accessor_t, Interpolator_t> in{this->dt_, accessor};
    reset_moments_(accessor, in);
    in.move(boxings);

    auto post_move_sync
        = [&](auto const i) mutable { Super::post_move_sync(accessor, boxings, i); };

    if constexpr (use_main_thread)
        for (std::size_t i = 0; i < accessor.size(); ++i)
            post_move_sync(i);
    else
        in.streamer.host(std::move(post_move_sync));

    if constexpr (Particles_t::alloc_mode == damn::AllocatorMode::GPU_UNIFIED
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
                    auto const mode
                        = tile_deposit_mode(n_points, 0, damn::core::gpu::max_dynamic_shared_bytes());
                    with_tile_deposit(mode, [&](auto deposit_c) {
                        using Launcher = damn::core::gpu::ChunkLauncher<false>;
                        Launcher launcher{1, 0};
                        launcher.b.x = damn::kernel::warp_size();
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

        if constexpr (use_main_thread)
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

        if constexpr (use_main_thread)
            for (std::size_t i = 0; i < accessor.size(); ++i)
                deposit(i);
        else
            in.streamer.host(std::move(deposit));
    }

    if constexpr (not use_main_thread)
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
    // layouts in unified memory update on the device: mkn_xyz::ParallelIonUpdater, as
    //  test_multi_updater checks them (UpdaterVersion::mkn_parallel), CPU tiled: ParallelIonUpdater
    template<typename Particles_t, typename Layout>
    using TiledIonUpdaterFor_t =
#if DAMN_HAVE_MKN_GPU
        std::conditional_t<Particles_t::alloc_mode == damn::AllocatorMode::GPU_UNIFIED,
                           mkn_xyz::ParallelIonUpdater<Particles_t, Layout>,
                           ParallelIonUpdater<Particles_t, Layout>>;
#else
        ParallelIonUpdater<Particles_t, Layout>;
#endif
} // namespace detail

// unified memory updates on the device (mkn_xyz::ParallelIonUpdater), tiled or not: untiled
//  GPU_UNIFIED (SoAVXChunked) through FlatParallelIonUpdater deposited on the host
template<typename Particles_t, typename Layout>
using IonUpdaterFor_t = std::conditional_t<
    Particles_t::layout_mode == LayoutMode::AoSMapped, IonUpdater<Particles_t, Layout>,
    std::conditional_t<is_tiled(Particles_t::layout_mode)
                           or Particles_t::alloc_mode == damn::AllocatorMode::GPU_UNIFIED,
                       detail::TiledIonUpdaterFor_t<Particles_t, Layout>,
                       FlatParallelIonUpdater<Particles_t, Layout>>>;

} // namespace plamn::core


#endif // ION_UPDATER_HPP
