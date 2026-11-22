#ifndef PLAMN_CORE_PUSHER_BORIS_DETAIL_MULTI_BORIS_HPP
#define PLAMN_CORE_PUSHER_BORIS_DETAIL_MULTI_BORIS_HPP

#include "damn/core/logger.hpp"
#include "mkn/kul/throw.hpp"
#include "damn/core/memory/span.hpp"
#include "damn/core/exec/kernels.hpp"
#include "damn/core/exec/thread_pool.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/def/damn_config.hpp"
#include "damn/core/def/detail/mkn_gpu.hpp"
#include "damn/core/geometry/box.hpp"
#include "mkn/kul/for.hpp"
#include "mkn/kul/meta.hpp"
#include "plamn/core/def.hpp"
#include "plamn/core/data/field/field_tiles.hpp"
#include "plamn/core/data/electromag/electromag.hpp"
#include "plamn/core/numerics/pusher/boris/basics.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/data/particles/arrays/particles_soavx.hpp"
#include "plamn/core/data/particles/particles.hpp"
#include "plamn/core/data/tensorfield/tensorfield.hpp"
#include "plamn/core/numerics/interpolator/cell_gather.hpp"
#include "plamn/core/numerics/interpolator/interpolator.hpp"
#include "mkn/gpu.hpp"
#include "plamn/core/data/grid/ghost_width_calculator.hpp"
#include "plamn/core/data/particles/particle.hpp"

#include <cassert>
#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <cmath>
#include <cstdlib>

namespace plamn::core
{

enum class MultiBorisMode : std::uint16_t { REF = 0, COPY };

template<auto particle_type, auto boris_mode, typename Backend_t>
struct MultiBorisFunctors;
template<LayoutMode layout, typename ModelAccessor, typename Interpolator>
struct MultiBorisBackend;


// ── MultiBoris state struct / ModelAccessor ───────────────────────────────────────────
// default dispatch: damn's thread pools (synchronous, on the calling thread, with
//  -DDAMN_THREAD_POOL_SYNC=1)
// see core::mkn_xyz::MultiBoris below for the mkn.gpu ThreadedStreamLauncher dispatch

template<typename ModelAccessor, typename Interpolator>
struct MultiBoris
{
    static constexpr auto dim = ModelAccessor::Layout_t::dimension;
    using Model_t             = ModelAccessor::Model_t;
    using Layout_t            = ModelAccessor::Layout_t;
    using Particles_t         = Model_t::particles_type;
    using Electromag_t        = Model_t::electromag_type;
    using Vecfield_t          = Electromag_t::vecfield_type;
    using Field_t             = Vecfield_t::field_type;
    using Particles_v         = Particles_t::view_t;
    using Backend = MultiBorisBackend<Particles_t::layout_mode, ModelAccessor, Interpolator>;

    MultiBoris(double const dt_, ModelAccessor& _accessor)
        : dt{dt_}
        , accessor{_accessor}
    {
    }

    template<MultiBorisMode mode = MultiBorisMode::REF>
    void move(auto const& boxings);

    double const dt;
    ModelAccessor& accessor;

    auto static mesh(std::array<double, dim> const& ms, double const& ts)
    {
        std::array<double, dim> halfDtOverDl;
        std::transform(std::begin(ms), std::end(ms), std::begin(halfDtOverDl),
                       [ts](double const& x) { return 0.5 * ts / x; });
        return halfDtOverDl;
    }
};


template<typename ModelAccessor, typename Interpolator>
template<MultiBorisMode mode>
void MultiBoris<ModelAccessor, Interpolator>::move(auto const& boxings)
{
    Backend::template move_pooled<mode>(*this, boxings);
}


// ── Per patch steps, shared by every dispatch (core::MultiBoris / mkn_xyz::MultiBoris) ────
// tiles of particles, tiled fields: AoSPCTS (per-cell AoS per tile), AoSCMTS (flat cell-mapped),
// AoSTS/AoSMapped (flat). AoS has its own specialisation below.
// `in` is either MultiBoris state struct, GPU paths require mkn_xyz::MultiBoris (in.streamer)

template<LayoutMode layout, typename ModelAccessor, typename Interpolator>
struct MultiBorisBackend
{
    using MultiBoris_t   = MultiBoris<ModelAccessor, Interpolator>;
    using Layout_t       = MultiBoris_t::Layout_t;
    using Particles_t    = MultiBoris_t::Particles_t;
    using Electromag_t   = MultiBoris_t::Electromag_t;
    using Interpolator_t = Interpolator;

    static constexpr auto dim = Layout_t::dimension;

    // only flat tiles have a push+deposit GPU kernel, AoSPCTS GPU COPY uses move_rest
    static constexpr bool has_gpu_copy
        = mkn::kul::any_in(layout, LayoutMode::AoSTS, LayoutMode::AoSMapped);

    template<auto pt, auto mode>
    using Functors = MultiBorisFunctors<pt, mode, MultiBorisBackend>;

    template<auto type>
    static void sync_particles(auto& particles, auto&... stream)
    {
        particles.template on_moved<type>(stream...);
    }

    template<auto mode>
    static void move_rest(auto& in, auto const i);

    template<auto mode>
    static void move_cpu_copy(auto& in, auto& boxings, auto const i);

    // ── Thread-pooled variants: level 1 (patch -> pool) is dispatched by the caller
    // (move_pooled), which hands us the specific pool already assigned to patch `i`.
    // Here we do level 2: one task per tile, submitted to that SAME pool so all of its
    // threads_per_pool threads work this patch's tiles at once. These must be called
    // from the dispatching thread itself, never from inside a task already running on
    // `pool` — detach_task-then-wait on one's own pool deadlocks.

    template<auto mode>
    static void move_rest_pooled(auto& pool, auto& in, auto const i);

    template<auto mode>
    static void move_cpu_copy_pooled(auto& pool, auto& in, auto& boxings, auto const i);

    // stream: mkn_xyz::MultiBoris passes the patch's stream for the particle array sync
    static void sync_ref(auto& view, auto&... stream);

    // General case: two levels of parallelism. Each patch is assigned one pool
    // (round-robin over whichever is next free), and within that pool every tile of
    // the patch is its own task -- so with N pools of M threads, up to N patches and
    // N*M tiles are in flight simultaneously. Dispatch happens entirely from this
    // (calling) thread so no pool ever detach_task-then-waits on itself.
    template<MultiBorisMode mode>
    static void move_pooled(MultiBoris_t& in, auto const& boxings);

    // GPU only: copies every patch's nonLevelGhostBox into managed memory for the kernels
    static void prepare_gpu_copy(auto& in, auto const& boxings);

    template<auto mode>
    static void move_gpu_copy(auto& in, auto const i);

    // GPU COPY threads per block, blocks per tile follow from the fullest tile
    static constexpr std::size_t gpu_copy_threads = 128;
};


template<LayoutMode layout, typename ModelAccessor, typename Interpolator>
template<auto mode>
void MultiBorisBackend<layout, ModelAccessor, Interpolator>::move_rest(auto& in, auto const i)
{
    DAMN_LOG_SCOPE(3, "MultiBoris::move_rest");

    auto view       = in.accessor[i];
    auto [ions, em] = view.args;

    for (auto& pop : ions)
    {
        auto& domain = pop.domainParticles();
        domain.reset_views();
        (Functors<ParticleType::Domain, mode>{in, view, pop, domain, em})(in, i);

        auto& level_ghost = pop.levelGhostParticles();
        level_ghost.reset_views();
        (Functors<ParticleType::LevelGhost, mode>{in, view, pop, level_ghost, em})(in, i);
    }
}


template<LayoutMode layout, typename ModelAccessor, typename Interpolator>
template<auto mode>
void MultiBorisBackend<layout, ModelAccessor, Interpolator>::move_cpu_copy(auto& in, auto& boxings,
                                                                           auto const i)
{
    DAMN_LOG_SCOPE(3, "MultiBoris::move_cpu_copy");

    auto view       = in.accessor[i];
    auto [ions, em] = view.args;

    auto const per_parts = [&]<auto particle_type>(auto& pop, auto& parts) {
        parts.reset_views();
        Functors<particle_type, mode> fns{in, view, pop, parts, em};
        fns.per_copy_of_cpu_tile(boxings, view, pop);
    };

    for (auto& pop : ions)
    {
        per_parts.template operator()<ParticleType::Domain>(pop, pop.domainParticles());
        per_parts.template operator()<ParticleType::LevelGhost>(pop, pop.levelGhostParticles());
    }
}


template<LayoutMode layout, typename ModelAccessor, typename Interpolator>
template<auto mode>
void MultiBorisBackend<layout, ModelAccessor, Interpolator>::move_rest_pooled(auto& pool, auto& in,
                                                                              auto const i)
{
    auto view       = in.accessor[i];
    auto [ions, em] = view.args;

    for (auto& pop : ions)
    {
        auto& domain = pop.domainParticles();
        domain.reset_views();
        auto& level_ghost = pop.levelGhostParticles();
        level_ghost.reset_views();

        using DomainFn = Functors<ParticleType::Domain, mode>;
        using GhostFn  = Functors<ParticleType::LevelGhost, mode>;

        // shared_ptr keeps the (self-contained: own pps view + electromag copy)
        // Functors alive for every deferred per-tile task below.
        auto domain_fn = std::make_shared<DomainFn>(in, view, pop, domain, em);
        auto ghost_fn  = std::make_shared<GhostFn>(in, view, pop, level_ghost, em);

        auto const n_tiles = domain_fn->pps().size();
        assert(n_tiles == ghost_fn->pps().size());

        for (std::size_t t = 0; t < n_tiles; ++t)
            pool.detach_task(damn::core::ThreadPool::INSTANCE().guard([domain_fn, ghost_fn, t] {
                domain_fn->one_tile(t);
                ghost_fn->one_tile(t);
            }));
    }
}


template<LayoutMode layout, typename ModelAccessor, typename Interpolator>
template<auto mode>
void MultiBorisBackend<layout, ModelAccessor, Interpolator>::move_cpu_copy_pooled(auto& pool,
                                                                                  auto& in,
                                                                                  auto& boxings,
                                                                                  auto const i)
{
    using DomainFn = Functors<ParticleType::Domain, mode>;
    using GhostFn  = Functors<ParticleType::LevelGhost, mode>;

    auto view       = in.accessor[i];
    auto [ions, em] = view.args;

    for (auto& pop : ions)
    {
        auto& domain = pop.domainParticles();
        domain.reset_views();
        auto& level_ghost = pop.levelGhostParticles();
        level_ghost.reset_views();

        auto domain_fn     = std::make_shared<DomainFn>(in, view, pop, domain, em);
        auto ghost_fn      = std::make_shared<GhostFn>(in, view, pop, level_ghost, em);
        auto const n_tiles = domain_fn->pps().size();
        assert(n_tiles == ghost_fn->pps().size());

        // view/pop are cheap handles, safe to capture by copy. Domain and LevelGhost
        // for the SAME tile must run in the SAME task (shared rhoP/flux);
        // different tiles are independent.
        for (std::size_t t = 0; t < n_tiles; ++t)
            pool.detach_task(damn::core::ThreadPool::INSTANCE().guard(
                [domain_fn, ghost_fn, t, view, pop, &boxings]() mutable {
                    domain_fn->one_copy_tile(t, boxings, view, pop);
                    ghost_fn->one_copy_tile(t, boxings, view, pop);
                }));
    }
}


template<LayoutMode layout, typename ModelAccessor, typename Interpolator>
void MultiBorisBackend<layout, ModelAccessor, Interpolator>::sync_ref(auto& view, auto&... stream)
{
    DAMN_LOG_SCOPE(3, "MultiBoris::sync_ref");

    auto [ions, _] = view.args;

    for (auto& pop : ions)
    {
        auto& domain      = pop.domainParticles();
        auto& level_ghost = pop.levelGhostParticles();

        sync_particles<ParticleType::Domain>(domain, stream...);
        sync_particles<ParticleType::LevelGhost>(level_ghost, stream...);
    }
}


template<LayoutMode layout, typename ModelAccessor, typename Interpolator>
template<MultiBorisMode mode>
void MultiBorisBackend<layout, ModelAccessor, Interpolator>::move_pooled(MultiBoris_t& in,
                                                                         auto const& boxings)
{
    static constexpr auto copy   = mode == MultiBorisMode::COPY;
    static constexpr auto is_cpu = Particles_t::alloc_mode == damn::AllocatorMode::CPU;

    auto& TP = damn::core::ThreadPool::INSTANCE();

    { // tile tasks per patch queued on the pools: timed as a whole, per patch only queues
        DAMN_LOG_SCOPE(3, "MultiBoris::move_pooled::push");
        for (std::size_t i = 0; i < in.accessor.size(); ++i)
        {
            auto& pool = TP.get_pool(TP.first_ready_idx());
            if constexpr (copy and is_cpu)
                move_cpu_copy_pooled<mode>(pool, in, boxings, i);
            else
                move_rest_pooled<mode>(pool, in, i);
        }
        TP.sync(); // every patch's tiles, across every pool, are done
    }

    if constexpr (not copy)
    {
        DAMN_LOG_SCOPE(3, "MultiBoris::move_pooled::sync_ref");
        TP.patch_exec(
            in.accessor.size(), [&](auto const i) { return in.accessor[i]; },
            [](auto& view) { sync_ref(view); });
    }
}


// every patch's boxings as managed span sets on in (gpu_nlgb, gpu_non_owned), read by
//  the GPU COPY kernels per patch index: shared by the tiled and flat backends
void prepare_gpu_copy_boxes(auto& in, auto const& boxings)
{
    DAMN_LOG_SCOPE(3, "MultiBoris::prepare_gpu_copy");

    using GpuBoxSpanSet_t = std::remove_cvref_t<decltype(in.gpu_nlgb)>;

    auto const to_gpu = [&](auto const& patch_boxes) {
        std::vector<damn::core::default_span_size_t> sizes;
        sizes.reserve(in.accessor.size());
        for (std::size_t i = 0; i < in.accessor.size(); ++i)
            sizes.push_back(patch_boxes(boxings.at(in.accessor[i].patchID())).size());
        GpuBoxSpanSet_t gpu_boxes{std::move(sizes)};
        std::size_t offset = 0;
        for (std::size_t i = 0; i < in.accessor.size(); ++i)
        {
            auto const& boxes = patch_boxes(boxings.at(in.accessor[i].patchID()));
            std::copy(boxes.begin(), boxes.end(), gpu_boxes.vec.begin() + offset);
            offset += boxes.size();
        }
        return gpu_boxes;
    };

    in.gpu_nlgb      = to_gpu([](auto const& b) -> auto& { return b.nonLevelGhostBox; });
    in.gpu_non_owned = to_gpu([](auto const& b) -> auto& { return b.nonOwnedBoxes; });
}


template<LayoutMode layout, typename ModelAccessor, typename Interpolator>
void MultiBorisBackend<layout, ModelAccessor, Interpolator>::prepare_gpu_copy(auto& in,
                                                                              auto const& boxings)
{
    prepare_gpu_copy_boxes(in, boxings);
}


template<LayoutMode layout, typename ModelAccessor, typename Interpolator>
template<auto mode>
void MultiBorisBackend<layout, ModelAccessor, Interpolator>::move_gpu_copy(
    [[maybe_unused]] auto& in, [[maybe_unused]] auto const i)
{
    DAMN_LOG_SCOPE(3, "MultiBoris::move_gpu_copy");

#if DAMN_HAVE_GPU
    static_assert(has_gpu_copy);
    using Tile_vt_       = Electromag_t::vecfield_type::field_type::value_type;
    using Interpolating_ = Interpolating<dim, Layout_t::options.interp_order,
                                         /*atomic_ops=*/true, Interpolator_t>;

    auto view       = in.accessor[i];
    auto [ions, em] = view.args;

    for (auto& pop : ions)
    {
        auto const coef1     = pop.charge() * (0.5 * in.dt / pop.mass());
        auto const halfdt    = in.mesh(view.layout.dx, in.dt);
        auto const nlgb      = in.gpu_nlgb[i];
        auto const non_owned = in.gpu_non_owned[i]; // its type: no excluded boxes
        auto rhop            = pop.particleDensity();
        auto flux            = *pop.flux();
        auto const n_points  = rhop.max_tile_size(); // largest tile field, incl. ghosts
        auto const extra
            = Interpolating_::template push_deposit_extra_bytes<typename Particles_t::Particle_t>(
                gpu_copy_threads);
        auto const deposit_mode
            = tile_deposit_mode(n_points, extra, damn::core::gpu::max_dynamic_shared_bytes());

        auto const launch = [&](auto parts, auto const filter_box, auto const exclude_boxes) {
            if (parts().size() == 0)
                return;
            // grid x = tile, y = blocks per tile, enough for the fullest tile
            auto const max_tile_particles = [&]() {
                std::size_t max = 0;
                for (auto const& tile : parts())
                    max = std::max<std::size_t>(max, tile().size());
                return max;
            }();
            if (max_tile_particles == 0)
                return;
            using Launcher = damn::core::gpu::BlockPerLauncher<false>;
            // several blocks per tile beat one even with the staged modes' reduction into the
            //  tile (measured: 6.8 ms vs 60 ms, 10 tiles of 100k particles)
            auto const blocks_per_tile = Launcher::blocks_for(max_tile_particles, gpu_copy_threads);
            with_tile_deposit(deposit_mode, [&](auto deposit_c) {
                Launcher launcher{parts().size(), gpu_copy_threads, blocks_per_tile};
                launcher.ds = tile_deposit_bytes(deposit_c(), n_points, extra);
                launcher.stream(in.streamer.streams[i], [=] __device__() mutable {
                    Interpolating_::template on_tiles_push_deposit<Tile_vt_, ParticleType::Domain,
                                                                   Particles_t::alloc_mode,
                                                                   decltype(deposit_c)::value>(
                        parts, em, flux, rhop, filter_box, exclude_boxes, coef1, halfdt);
                });
            });
        };

        // level ghosts as domain particles: each is held by one patch only
        launch(*pop.domainParticles(), nlgb, decltype(non_owned){});
        launch(*pop.levelGhostParticles(), nlgb, decltype(non_owned){});
    }
#else
    throw std::runtime_error("NEEDS GPU IMPL!");
#endif
}


template<typename ModelAccessor, typename Interpolator>
struct MultiBorisBackend<LayoutMode::AoS, ModelAccessor, Interpolator>
{
    using MultiBoris_t   = MultiBoris<ModelAccessor, Interpolator>;
    using Layout_t       = MultiBoris_t::Layout_t;
    using Particles_t    = MultiBoris_t::Particles_t;
    using Interpolator_t = Interpolator;

    static constexpr auto dim        = Layout_t::dimension;
    static constexpr auto alloc_mode = Particles_t::alloc_mode;

    // mkn_xyz::MultiBoris dispatch: untiled, untracked (leavers partitioned after, see
    //  post_move_sync), GPU_UNIFIED SoAVX pushes (REF) and pushes copies + deposits (COPY) on the
    //  device, a thread per particle, global atomics
    static constexpr bool flat = true;
    static constexpr bool has_gpu_copy
        = alloc_mode == damn::AllocatorMode::GPU_UNIFIED and is_soavx(Particles_t::layout_mode);

    static void prepare_gpu_copy(auto& in, auto const& boxings)
    {
        prepare_gpu_copy_boxes(in, boxings);
    }

    template<MultiBorisMode mode>
    static void move_gpu_copy(auto& in, auto const i);

    template<MultiBorisMode mode>
    static void move_cpu_copy(auto& in, auto const& boxings, auto const i)
    {
        auto view = in.accessor[i];
        move_patch<mode>(in, view, boxings);
    }

    template<MultiBorisMode mode>
    static void move_rest(auto& in, auto const& boxings, auto const i);

    static void sync_ref(auto& /*view*/, auto& /*stream*/) {}

    // one particle, host or device: false if a level ghost left the ghost box on the first
    //  half step (it is then not accelerated)
    template<auto type>
    static bool push_particle(auto& particle, auto const& em, auto const& layout,
                              auto const& ghostBox, auto const& halfdt,
                              double const coef1) DAMN_ALL;

    template<MultiBorisMode mode>
    static void move_pooled(MultiBoris_t& in, auto const& boxings);

    template<MultiBorisMode mode>
    static void move_patch(MultiBoris_t& in, auto& view, auto const& boxings);

    template<auto type>
    static bool push(auto& particle, auto&&... args);

    template<auto type>
    static void push_all(auto& particles, auto&&... args);

    template<auto type>
    static void push_copies_and_deposit(auto const& particles, auto& pop, auto&&... args);

    // fn(particle) for each particle: SoAVX stores components in columns, so a Particle<dim>
    //  is loaded per index and, if `store`, written back after fn
    template<bool store>
    static void per_particle(auto& particles, auto&& fn)
    {
        if constexpr (is_soavx(Particles_t::layout_mode))
            for (std::size_t i = 0; i < particles.size(); ++i)
            {
                if (particles.weight(i) == 0) // SoAVXChunked's gaps
                    continue;
                auto particle = particles.copy(i);
                fn(particle);
                if constexpr (store)
                    particles.assign(particle, i);
            }
        else if constexpr (store)
            for (auto& particle : particles)
                fn(particle);
        else
            for (auto particle : particles)
                fn(particle);
    }
};

// SoAVX: untiled and untracked like AoS (leavers are sorted / partitioned out after the
//  push), so the AoS backend, which branches where SoAVX can't be pushed in place
template<typename ModelAccessor, typename Interpolator>
struct MultiBorisBackend<LayoutMode::SoAVX, ModelAccessor, Interpolator>
    : public MultiBorisBackend<LayoutMode::AoS, ModelAccessor, Interpolator>
{
};

// SoAVXChunked: as SoAVX, the offsets are rebuilt after the post push sync (on_moved)
//  CPU: its own patch loop, batched in phases (load + half step; m2p + accelerate + half step;
//  deposit or store back) rather than the flat backend's per particle load, push, deposit (built
//  for a GPU thread per particle). Same particles in the same order: the same results. GPU: the
//  flat backend's
#if defined(MKN_WITH_MKN_AVX)
namespace boris::avx
{
    // the columns of a block's slots: an array's own (SoAVXChunked), or a staging buffer's
    template<std::size_t dim>
    struct Columns
    {
        double* w;
        int* c[dim];
        double* d[dim];
        double* v[3];
    };

    NO_DISCARD auto columns(auto& particles)
    {
        static constexpr auto dim = std::decay_t<decltype(particles)>::dimension;
        Columns<dim> cols;
        cols.w = particles.weight().data();
        for (std::size_t k = 0; k < dim; ++k)
        {
            cols.c[k] = particles.iCell()[k].data();
            cols.d[k] = particles.delta()[k].data();
        }
        for (std::size_t k = 0; k < 3; ++k)
            cols.v[k] = particles.v()[k].data();
        return cols;
    }

    // a staging buffer's (SoAVXArray: AoS particles staged, pushed, written back)
    template<std::size_t dim, std::size_t N>
    NO_DISCARD auto columns(SoAVXArray<dim, N>& staged)
    {
        Columns<dim> cols;
        cols.w = staged.weight_.data();
        for (std::size_t k = 0; k < dim; ++k)
        {
            cols.c[k] = staged.iCell_[k].data();
            cols.d[k] = staged.delta_[k].data();
        }
        for (std::size_t k = 0; k < 3; ++k)
            cols.v[k] = staged.v_[k].data();
        return cols;
    }

    // staged slot i as a particle (the deposit of a staged copy)
    template<std::size_t dim, std::size_t N>
    NO_DISCARD auto particle_at(SoAVXArray<dim, N> const& staged, std::size_t const i)
    {
        Particle<dim> p;
        p.weight() = staged.weight_[i];
        for (std::size_t k = 0; k < dim; ++k)
        {
            p.iCell()[k] = staged.iCell_[k][i];
            p.delta()[k] = staged.delta_[k][i];
        }
        for (std::size_t k = 0; k < 3; ++k)
            p.v()[k] = staged.v_[k][i];
        return p;
    }

    // per cell deposit (COPY, sorted slots, order 1): the lanes ending the push in the cell
    //  (CellDepositV), when the cell is in the non level ghost box
    template<std::size_t dim, std::size_t L>
    struct Window : public CellDepositV<dim, L>
    {
        bool deposits = false;
    };

    // the push of one block of L slots on mkn::avx lanes (MultiBorisBackend<SoAVXChunked>::
    //  batched_avx's): the columns masked loaded and stored (slots past n and gaps load weight 0:
    //  never live, their memory untouched), advance and accelerate on the lanes, the per cell
    //  gather g (CellGatherV, null: m2p per lane); REF: stored back, the crossings (record) to
    //  on_crossed(slot, old cell); COPY: deposited (win: the cell's window, else per lane)
    template<auto type, MultiBorisMode mode, typename Interpolator_t, std::size_t L, bool record,
             typename EM, typename Layout, typename Boxing, typename Halfdt, typename RhoP,
             typename Flux>
    struct Push
    {
        static constexpr auto dim       = Layout::dimension;
        using Vd                        = mkn::avx::Type<double, L>;
        using Mk                        = mkn::avx::Mask<double, L>;
        using Particle_t                = Particle<dim>;
        using Window_t                  = Window<dim, L>;
        static constexpr bool copy      = mode == MultiBorisMode::COPY;
        static constexpr bool int_lanes = L >= 4; // mkn::avx's int32 lanes: 4, 8

        EM const& em;
        Layout const& layout;
        Boxing const& boxing;
        Halfdt const& halfdt;
        double coef1;
        RhoP& rhoP;
        Flux& flux;
        mutable Interpolator_t interp{}; // m2p / p2m per lane: made once, not per block

        // one block of L slots from i0 (n of them real); everything it works on is its own (lane
        //  state, constants, the columns' pointers): it can stay in registers. Inlined into each
        //  call site: g and win known there (null or not), else gcc keeps one copy testing them
        //  (~9% slower on the coupled case)
        [[gnu::always_inline]] void operator()(Columns<dim> const& cols, std::size_t const i0,
                                               std::size_t const n, auto const* g, auto const& cell,
                                               Window_t* win, auto&& on_crossed) const
        {
            // a column of cells as lanes: int32 lanes for a full block (a partial one could read
            //  past the array), else per lane
            auto const load_cells = [](int const* const p, std::size_t const n) {
                if constexpr (int_lanes)
                    if (n == L)
                        return mkn::avx::to_double(mkn::avx::unaligned_load<std::int32_t, L>(p));
                double t[L];
                for (std::size_t l = 0; l < L; ++l)
                    t[l] = l < n ? static_cast<double>(p[l]) : 0.;
                return mkn::avx::unaligned_load<double, L>(t);
            };
            auto const store_cells = [](int* const p, auto const& c) { // full blocks only
                if constexpr (int_lanes)
                    mkn::avx::unaligned_store(p, mkn::avx::to_int32(c));
            };
            auto const bc = [](double const x) DAMN_LAMBDA_ALWAYS_INLINE {
                return Vd{mkn::avx::load<double, L>(x)};
            };
            auto const store = [](double* p, Vd const& v)
                                   DAMN_LAMBDA_ALWAYS_INLINE { mkn::avx::unaligned_store(p, v); };
            auto const not_  = [&](Mk const& m) DAMN_LAMBDA_ALWAYS_INLINE {
                return mkn::avx::gt(mkn::avx::select(m, bc(0.), bc(1.)), bc(.5));
            };

            // the columns' pointers and the constants as locals: through cols or this they would be
            //  reloaded after every store (they might alias)
            double* const Wp = cols.w;
            double* Dp[dim];
            int* Cp[dim];
            double* Vp[3];
            for (std::size_t k = 0; k < dim; ++k)
            {
                Dp[k] = cols.d[k];
                Cp[k] = cols.c[k];
            }
            for (std::size_t k = 0; k < 3; ++k)
                Vp[k] = cols.v[k];
            double const coef1 = this->coef1;
            auto const& em     = this->em;
            auto const& layout = this->layout;
            auto const& boxing = this->boxing;
            auto const& halfdt = this->halfdt;
            auto& rhoP         = this->rhoP;
            auto& flux         = this->flux;
            auto& interp       = this->interp;

            Vd const limit = bc(static_cast<double>(particle_half_push_limit));
            Vd h[dim];
            for (std::size_t k = 0; k < dim; ++k)
                h[k] = bc(halfdt[k]);
            Vd lane_idx;
            {
                double t[L];
                for (std::size_t l = 0; l < L; ++l)
                    t[l] = static_cast<double>(l);
                lane_idx = mkn::avx::unaligned_load<double, L>(t);
            }
            Mk const none_m = mkn::avx::lt(bc(1.), bc(0.));

            Vd w, d[dim], c[dim], v[3], E[3], B[3];

            // boris::advance on the lanes; a lane of m over the limit throws its exception
            auto const advance = [&](Mk const& m) DAMN_LAMBDA_ALWAYS_INLINE {
                for (std::size_t k = 0; k < dim; ++k)
                {
                    Vd const x = d[k] + h[k] * v[k];
                    if (mkn::avx::any(m & mkn::avx::gt(mkn::avx::abs(x), limit))) [[unlikely]]
                    {
                        double xs[L], vs[L], ms[L];
                        store(xs, x);
                        store(vs, v[k]);
                        store(ms, mkn::avx::select(m, bc(1.), bc(0.)));
                        for (std::size_t l = 0; l < L; ++l)
                            if (ms[l] > .5 and std::abs(xs[l]) > particle_half_push_limit)
                                mkn::kul::throw_exception<boris::MoveTwoCellException>(xs[l],
                                                                                       vs[l]);
                    }
                    Vd const f = mkn::avx::floor(x);
                    d[k]       = x - f;
                    c[k]       = f + c[k];
                }
            };

            // the lanes as Particles, for the per lane m2p and deposit
            auto const lane_particles = [&](std::array<Particle_t, L>& ps) {
                double t[L];
                store(t, w);
                for (std::size_t l = 0; l < L; ++l)
                    ps[l].weight() = t[l];
                for (std::size_t k = 0; k < dim; ++k)
                {
                    store(t, c[k]);
                    for (std::size_t l = 0; l < L; ++l)
                        ps[l].iCell()[k] = static_cast<int>(t[l]);
                    store(t, d[k]);
                    for (std::size_t l = 0; l < L; ++l)
                        ps[l].delta()[k] = t[l];
                }
                for (std::size_t k = 0; k < 3; ++k)
                {
                    store(t, v[k]);
                    for (std::size_t l = 0; l < L; ++l)
                        ps[l].v()[k] = t[l];
                }
            };

            Mk const first = mkn::avx::lt(lane_idx, bc(static_cast<double>(n)));
            w              = mkn::avx::mask_load(first, Wp + i0);
            for (std::size_t k = 0; k < dim; ++k)
            {
                d[k] = mkn::avx::mask_load(first, Dp[k] + i0);
                c[k] = load_cells(Cp[k] + i0, n);
            }
            for (std::size_t k = 0; k < 3; ++k)
                v[k] = mkn::avx::mask_load(first, Vp[k] + i0);

            Mk const live = mkn::avx::gt(w, bc(0.)); // past n and gaps: weight 0
            Vd c_loaded[dim];                        // REF domain: crossings recorded
            for (std::size_t k = 0; k < dim; ++k)
                c_loaded[k] = c[k];

            advance(live);

            // the lanes' tests on the lanes: of m, those with their cell in [lo, hi]
            auto const inside = [&](Mk m, auto const& lo, auto const& hi) {
                for (std::size_t k = 0; k < dim; ++k)
                    m = m
                        & not_(mkn::avx::lt(c[k], bc(static_cast<double>(lo[k])))
                               | mkn::avx::gt(c[k], bc(static_cast<double>(hi[k]))));
                return m;
            };
            Mk accel = live;
            if constexpr (type == ParticleType::LevelGhost)
                accel = inside(live, boxing.ghostBox.lower, boxing.ghostBox.upper);
            Mk const off = g ? accel & not_(inside(accel, cell, cell)) : accel;

            std::array<Particle_t, L> ps; // per lane fallbacks only
            if (g)
                g->gather(d, E, B);
            if (mkn::avx::any(off)) // m2p per lane for the lanes off the cell (or all, without g)
            {
                lane_particles(ps);
                double e[3][L], b[3][L], o[L];
                store(o, mkn::avx::select(off, bc(1.), bc(0.)));
                for (std::size_t k = 0; k < 3; ++k)
                {
                    store(e[k], E[k]);
                    store(b[k], B[k]);
                }
                for (std::size_t l = 0; l < L; ++l)
                    if (o[l] > .5)
                    {
                        auto const [pE, pB] = interp.m2p(ps[l], em, layout);
                        for (std::size_t k = 0; k < 3; ++k)
                        {
                            e[k][l] = pE[k];
                            b[k][l] = pB[k];
                        }
                    }
                for (std::size_t k = 0; k < 3; ++k)
                {
                    E[k] = mkn::avx::unaligned_load<double, L>(e[k]);
                    B[k] = mkn::avx::unaligned_load<double, L>(b[k]);
                }
            }

            { // boris::accelerate
                Vd const k1 = bc(coef1), one = bc(1.), two = bc(2.);
                Vd const vx1 = v[0] + k1 * E[0], vy1 = v[1] + k1 * E[1], vz1 = v[2] + k1 * E[2];
                Vd const rx = k1 * B[0], ry = k1 * B[1], rz = k1 * B[2];
                Vd const rx2 = rx * rx, ry2 = ry * ry, rz2 = rz * rz;
                Vd const rxry = rx * ry, rxrz = rx * rz, ryrz = ry * rz;
                Vd const invDet = one / (one + rx2 + ry2 + rz2);
                Vd const mxx = one + rx2 - ry2 - rz2, mxy = two * (rxry + rz),
                         mxz = two * (rxrz - ry);
                Vd const myx = two * (rxry - rz), myy = one + ry2 - rx2 - rz2,
                         myz = two * (ryrz + rx);
                Vd const mzx = two * (rxrz + ry), mzy = two * (ryrz - rx),
                         mzz = one + rz2 - rx2 - ry2;
                Vd const v0  = (mxx * vx1 + mxy * vy1 + mxz * vz1) * invDet + k1 * E[0];
                Vd const v1  = (myx * vx1 + myy * vy1 + myz * vz1) * invDet + k1 * E[1];
                Vd const v2  = (mzx * vx1 + mzy * vy1 + mzz * vz1) * invDet + k1 * E[2];
                v[0]         = mkn::avx::select(accel, v0, v[0]);
                v[1]         = mkn::avx::select(accel, v1, v[1]);
                v[2]         = mkn::avx::select(accel, v2, v[2]);
            }

            { // second half step: the lanes not accelerated stay
                Vd d0[dim], c0[dim];
                for (std::size_t k = 0; k < dim; ++k)
                {
                    d0[k] = d[k];
                    c0[k] = c[k];
                }
                advance(accel);
                for (std::size_t k = 0; k < dim; ++k)
                {
                    d[k] = mkn::avx::select(accel, d[k], d0[k]);
                    c[k] = mkn::avx::select(accel, c[k], c0[k]);
                }
            }

            if constexpr (copy)
            {
                Mk const in_win = (win and win->deposits) ? inside(accel, cell, cell) : none_m;
                if (win and win->deposits)
                    win->add(in_win, w, d, v);
                Mk const rest = accel & not_(in_win); // deposited per lane, if in the box
                if (mkn::avx::any(rest))
                {
                    lane_particles(ps);
                    double r[L];
                    store(r, mkn::avx::select(rest, bc(1.), bc(0.)));
                    for (std::size_t l = 0; l < L; ++l)
                        if (r[l] > .5 and isIn(ps[l].iCell(), boxing.nonLevelGhostBox))
                            interp.particleToMesh(ps[l], rhoP, flux, layout);
                }
            }
            else
            {
                if constexpr (record)
                { // the live lanes whose cell changed
                    Mk crossed = none_m;
                    for (std::size_t k = 0; k < dim; ++k)
                        crossed = crossed | mkn::avx::lt(c[k], c_loaded[k])
                                  | mkn::avx::gt(c[k], c_loaded[k]);
                    crossed = crossed & live;
                    if (mkn::avx::any(crossed))
                    {
                        double x[L], from[dim][L];
                        store(x, mkn::avx::select(crossed, bc(1.), bc(0.)));
                        for (std::size_t k = 0; k < dim; ++k)
                            store(from[k], c_loaded[k]);
                        for (std::size_t l = 0; l < L; ++l)
                            if (x[l] > .5)
                            {
                                std::array<int, dim> cell;
                                for (std::size_t k = 0; k < dim; ++k)
                                    cell[k] = static_cast<int>(from[k][l]);
                                on_crossed(i0 + l, cell);
                            }
                    }
                }
                // a full block of live lanes (a chunk's inside: nearly all) stores its cells as
                //  int32 lanes, else per lane (no masked int32 store)
                bool const all_live = int_lanes and n == L and !mkn::avx::any(not_(live));
                double lv[L];
                if (!all_live)
                    store(lv, mkn::avx::select(live, bc(1.), bc(0.)));
                for (std::size_t k = 0; k < dim; ++k)
                {
                    mkn::avx::mask_store(live, Dp[k] + i0, d[k]);
                    if (all_live)
                        store_cells(Cp[k] + i0, c[k]);
                    else
                    {
                        double t[L];
                        store(t, c[k]);
                        for (std::size_t l = 0; l < n; ++l)
                            if (lv[l] > .5)
                                Cp[k][i0 + l] = static_cast<int>(t[l]);
                    }
                }
                for (std::size_t k = 0; k < 3; ++k)
                    mkn::avx::mask_store(live, Vp[k] + i0, v[k]);
            }
        }
    };

    template<auto type, MultiBorisMode mode, typename Interpolator_t, std::size_t L, bool record>
    NO_DISCARD auto push(auto const& em, auto const& layout, auto const& boxing, auto const& halfdt,
                         double const coef1, auto& rhoP, auto& flux)
    {
        return Push<type, mode, Interpolator_t, L, record, std::decay_t<decltype(em)>,
                    std::decay_t<decltype(layout)>, std::decay_t<decltype(boxing)>,
                    std::decay_t<decltype(halfdt)>, std::decay_t<decltype(rhoP)>,
                    std::decay_t<decltype(flux)>>{em, layout, boxing, halfdt, coef1, rhoP, flux};
    }
} // namespace boris::avx
#endif // MKN_WITH_MKN_AVX

template<typename ModelAccessor, typename Interpolator>
struct MultiBorisBackend<LayoutMode::SoAVXChunked, ModelAccessor, Interpolator>
    : public MultiBorisBackend<LayoutMode::AoS, ModelAccessor, Interpolator>
{
    using Super               = MultiBorisBackend<LayoutMode::AoS, ModelAccessor, Interpolator>;
    using MultiBoris_t        = Super::MultiBoris_t;
    using Particles_t         = Super::Particles_t;
    using Interpolator_t      = Super::Interpolator_t;
    static constexpr auto dim = Super::dim;
    static constexpr bool cpu = Super::alloc_mode == damn::AllocatorMode::CPU;
    using Particle_t          = Particle<dim>;

    template<MultiBorisMode mode>
    static void move_pooled(MultiBoris_t& in, auto const& boxings)
    {
        if constexpr (cpu)
            damn::core::ThreadPool::INSTANCE().patch_exec(
                in.accessor.size(), [&](auto const i) { return in.accessor[i]; },
                [&](auto& view) { move_patch<mode>(in, view, boxings); });
        else
            Super::template move_pooled<mode>(in, boxings);
    }

    template<MultiBorisMode mode>
    static void move_cpu_copy(auto& in, auto const& boxings, auto const i)
    {
        if constexpr (cpu)
        {
            auto view = in.accessor[i];
            move_patch<mode>(in, view, boxings);
        }
        else
            Super::template move_cpu_copy<mode>(in, boxings, i);
    }

    template<MultiBorisMode mode>
    static void move_rest(auto& in, auto const& boxings, auto const i)
    {
        if constexpr (cpu)
        {
            auto view = in.accessor[i];
            move_patch<mode>(in, view, boxings);
        }
        else
            Super::template move_rest<mode>(in, boxings, i);
    }

    template<MultiBorisMode mode>
    static void move_patch(auto& in, auto& view, auto const& boxings)
    {
        DAMN_LOG_SCOPE(3, "MultiBoris::chunked::move_patch");

        auto [ions, em]    = view.args;
        auto const& layout = view.layout;
        auto const& boxing = boxings.at(view.patchID());
        auto const halfdt  = in.mesh(layout.dx, in.dt);

        for (auto& pop : ions)
        {
            double const coef1 = pop.charge() * (0.5 * in.dt / pop.mass());
#if defined(MKN_WITH_MKN_AVX)
            batched_avx<ParticleType::Domain, mode>(pop.domainParticles(), pop, em, layout, boxing,
                                                    halfdt, coef1);
            batched_avx<ParticleType::LevelGhost, mode>(pop.levelGhostParticles(), pop, em, layout,
                                                        boxing, halfdt, coef1);
            continue;
#endif
            batched<ParticleType::Domain, mode>(pop.domainParticles(), pop, em, layout, boxing,
                                                halfdt, coef1);
            batched<ParticleType::LevelGhost, mode>(pop.levelGhostParticles(), pop, em, layout,
                                                    boxing, halfdt, coef1);
        }
    }

    // the push on lanes of the columns: blocks of `lanes` consecutive slots, each component a
    //  [lane] array loaded from (and stored back to) its column, the arithmetic in lane loops
    //  REF: pushed in place (a level ghost leaving the ghost box on the first half step is stored
    //  half pushed, not accelerated, as push_particle: post_move_sync drops it). COPY: copies
    //  pushed, deposited if they end in the non level ghost box, in slot order. Gap slots
    //  (weight 0) are computed with the others and neither stored nor deposited
    static constexpr std::size_t lanes = 8;

    template<auto type, MultiBorisMode mode>
    static void batched(auto& particles, auto& pop, auto const& em, auto const& layout,
                        auto const& boxing, auto const& halfdt, double const coef1)
    {
        static constexpr bool copy  = mode == MultiBorisMode::COPY;
        static constexpr auto limit = static_cast<double>(particle_half_push_limit);

        Interpolator_t interp;
        auto& rhoP = pop.particleDensity();
        auto& flux = pop.flux();

        auto& W = particles.weight();
        auto& C = particles.iCell();
        auto& D = particles.delta();
        auto& V = particles.v();

        double w[lanes], d[dim][lanes], v[3][lanes], E[3][lanes], B[3][lanes];
        int c[dim][lanes];
        bool live[lanes], accel[lanes];

        // delta += halfdt v, the cell crossed into c (boris::advance on the lanes); a lane of
        //  mask moving more than the limit throws boris::advance's exception
        auto const advance = [&](bool const* mask) {
            bool over    = false;
            double bad_x = 0, bad_v = 0; // a lane over the limit, for the exception
            for (std::size_t k = 0; k < dim; ++k)
                for (std::size_t l = 0; l < lanes; ++l)
                {
                    double const x = d[k][l] + halfdt[k] * v[k][l];
                    bool const b   = mask[l] and std::abs(x) > limit;
                    over           = over or b;
                    bad_x          = b ? x : bad_x;
                    bad_v          = b ? v[k][l] : bad_v;
                    double const f = std::floor(x);
                    d[k][l]        = x - f;
                    c[k][l]        = static_cast<int>(f + c[k][l]);
                }
            if (over) [[unlikely]]
                mkn::kul::throw_exception<boris::MoveTwoCellException>(bad_x, bad_v);
        };
        auto const lane_particle = [&](std::size_t const l) {
            Particle_t p;
            p.weight() = w[l];
            for (std::size_t k = 0; k < dim; ++k)
            {
                p.iCell()[k] = c[k][l];
                p.delta()[k] = d[k][l];
            }
            for (std::size_t k = 0; k < 3; ++k)
                p.v()[k] = v[k][l];
            return p;
        };

        // cell: the chunk's cell (AMR) with its stencils in g, for the per cell gather, or null
        auto const block
            = [&](std::size_t const i0, std::size_t const n, auto const* g, auto const& cell) {
                  // always `lanes` lanes (fixed trip counts, the loops vectorise alike): a partial
                  //  block's extra lanes copy its last slot, not live
                  std::size_t slot[lanes];
                  for (std::size_t l = 0; l < lanes; ++l)
                      slot[l] = i0 + (l < n ? l : n - 1);
                  for (std::size_t l = 0; l < lanes; ++l)
                      w[l] = W[slot[l]];
                  for (std::size_t k = 0; k < dim; ++k)
                      for (std::size_t l = 0; l < lanes; ++l)
                      {
                          c[k][l] = C[k][slot[l]];
                          d[k][l] = D[k][slot[l]];
                      }
                  for (std::size_t k = 0; k < 3; ++k)
                      for (std::size_t l = 0; l < lanes; ++l)
                          v[k][l] = V[k][slot[l]];
                  for (std::size_t l = 0; l < lanes; ++l)
                      live[l] = l < n and w[l] != 0;

                  advance(live);

                  for (std::size_t l = 0; l < lanes; ++l)
                  {
                      accel[l] = live[l];
                      if constexpr (type == ParticleType::LevelGhost)
                          accel[l] = accel[l] and isIn(lane_particle(l).iCell(), boxing.ghostBox);
                  }

                  // m2p: from the cell's stencils on the lanes still in the cell at mid step, else
                  // per
                  //  lane (the stencils differ)
                  bool in_cell[lanes];
                  for (std::size_t l = 0; l < lanes; ++l)
                  {
                      in_cell[l] = g != nullptr;
                      if (g)
                          for (std::size_t k = 0; k < dim; ++k)
                              in_cell[l] = in_cell[l] and c[k][l] == cell[k];
                  }
                  if (g)
                      g->gather(d, E, B);
                  for (std::size_t l = 0; l < lanes; ++l)
                  {
                      if (!accel[l] or in_cell[l])
                          continue;
                      auto const [pE, pB] = interp.m2p(lane_particle(l), em, layout);
                      for (std::size_t k = 0; k < 3; ++k)
                      {
                          E[k][l] = pE[k];
                          B[k][l] = pB[k];
                      }
                  }

                  // boris::accelerate on the lanes
                  double v1[3][lanes];
                  for (std::size_t l = 0; l < lanes; ++l)
                  {
                      double const vx1 = v[0][l] + coef1 * E[0][l];
                      double const vy1 = v[1][l] + coef1 * E[1][l];
                      double const vz1 = v[2][l] + coef1 * E[2][l];
                      double const rx = coef1 * B[0][l], ry = coef1 * B[1][l], rz = coef1 * B[2][l];
                      double const rx2 = rx * rx, ry2 = ry * ry, rz2 = rz * rz;
                      double const rxry = rx * ry, rxrz = rx * rz, ryrz = ry * rz;
                      double const invDet = 1. / (1. + rx2 + ry2 + rz2);
                      double const mxx = 1. + rx2 - ry2 - rz2, mxy = 2. * (rxry + rz),
                                   mxz = 2. * (rxrz - ry);
                      double const myx = 2. * (rxry - rz), myy = 1. + ry2 - rx2 - rz2,
                                   myz = 2. * (ryrz + rx);
                      double const mzx = 2. * (rxrz + ry), mzy = 2. * (ryrz - rx),
                                   mzz = 1. + rz2 - rx2 - ry2;
                      v1[0][l] = (mxx * vx1 + mxy * vy1 + mxz * vz1) * invDet + coef1 * E[0][l];
                      v1[1][l] = (myx * vx1 + myy * vy1 + myz * vz1) * invDet + coef1 * E[1][l];
                      v1[2][l] = (mzx * vx1 + mzy * vy1 + mzz * vz1) * invDet + coef1 * E[2][l];
                  }
                  for (std::size_t k = 0; k < 3; ++k)
                      for (std::size_t l = 0; l < lanes; ++l)
                          v[k][l] = accel[l] ? v1[k][l] : v[k][l];

                  // second half step, the lanes not accelerated stay where the first left them
                  int c1[dim][lanes];
                  double d1[dim][lanes];
                  for (std::size_t k = 0; k < dim; ++k)
                      for (std::size_t l = 0; l < lanes; ++l)
                      {
                          c1[k][l] = c[k][l];
                          d1[k][l] = d[k][l];
                      }
                  advance(accel);
                  for (std::size_t k = 0; k < dim; ++k)
                      for (std::size_t l = 0; l < lanes; ++l)
                      {
                          c[k][l] = accel[l] ? c[k][l] : c1[k][l];
                          d[k][l] = accel[l] ? d[k][l] : d1[k][l];
                      }

                  if constexpr (copy)
                  {
                      for (std::size_t l = 0; l < lanes; ++l)
                          if (accel[l])
                          {
                              auto const p = lane_particle(l);
                              if (isIn(p.iCell(), boxing.nonLevelGhostBox))
                                  interp.particleToMesh(p, rhoP, flux, layout);
                          }
                  }
                  else
                  {
                      for (std::size_t k = 0; k < dim; ++k)
                          for (std::size_t l = 0; l < n; ++l)
                              if (live[l])
                              {
                                  C[k][i0 + l] = c[k][l];
                                  D[k][i0 + l] = d[k][l];
                              }
                      for (std::size_t k = 0; k < 3; ++k)
                          for (std::size_t l = 0; l < n; ++l)
                              if (live[l])
                                  V[k][i0 + l] = v[k][l];
                  }
              };

        using Gather_t = CellGather1<dim, lanes>;
        static constexpr bool per_cell
            = type == ParticleType::Domain and Interpolator_t::interp_order == 1;
        Gather_t const* const none = nullptr;
        std::array<int, dim> const no_cell{};

        if constexpr (per_cell)
            if (particles.sorted())
            {
                Gather_t g;
                damn::core::for_each_point(particles.shell(), [&](auto const& cell) {
                    auto const [b, size] = particles.cell_range(cell);
                    if (size == 0)
                        return;
                    g.load(em, layout, cell);
                    std::array<int, dim> amr;
                    for (std::size_t k = 0; k < dim; ++k)
                        amr[k] = cell[k];
                    std::size_t i = b;
                    for (; i + lanes <= b + size; i += lanes)
                        block(i, lanes, &g, amr);
                    if (i < b + size)
                        block(i, b + size - i, &g, amr);
                });
                return;
            }

        auto const size = particles.size();
        std::size_t i   = 0;
        for (; i + lanes <= size; i += lanes)
            block(i, lanes, none, no_cell);
        if (i < size)
            block(i, size - i, none, no_cell);
    }

#if defined(MKN_WITH_MKN_AVX)
    // batched on mkn::avx lanes (builds with -w mkn.avx): the widest double lanes the build has,
    //  the columns masked loaded and stored (lanes past the block's end and gap slots load weight
    //  0: never live, their memory untouched), advance and accelerate on the lanes, the per cell
    //  gather (CellGatherV) for sorted domain arrays; per lane: the cells (int columns), the
    //  level ghost box test, m2p off the cell and the deposit
    static constexpr std::size_t avx_lanes = mkn::avx::Options::N<double>();

    template<auto type, MultiBorisMode mode>
    static void batched_avx(auto& particles, auto& pop, auto const& em, auto const& layout,
                            auto const& boxing, auto const& halfdt, double const coef1)
    {
        static constexpr std::size_t L = avx_lanes;
        static constexpr bool copy     = mode == MultiBorisMode::COPY;
        // the REF push of a domain array records its crossings (SoAVXChunked::resettle)
        static constexpr bool record = !copy and type == ParticleType::Domain;
        if constexpr (record)
            particles.begin_crossers();

        auto& rhoP = pop.particleDensity();
        auto& flux = pop.flux();

        using Window       = boris::avx::Window<dim, L>;
        auto const dep_box = damn::core::grow(
            layout.cells, std::decay_t<decltype(layout)>::options.field_ghost_width);
        auto const reset = [&](Window& win, auto const& cell) {
            win.reset(cell, dep_box);
            win.deposits = isIn(cell, boxing.nonLevelGhostBox);
        };
        auto const flush = [&](Window const& win) { win.flush(rhoP, flux); };

        auto const push = boris::avx::push<type, mode, Interpolator_t, L, record>(
            em, layout, boxing, halfdt, coef1, rhoP, flux);
        auto const crossed
            = [&](std::size_t const i, auto const& from) { particles.add_crosser(i, from); };
        auto const block = [&](std::size_t const i0, std::size_t const n, auto const* g,
                               auto const& cell, Window* win) {
            push(boris::avx::columns(particles), i0, n, g, cell, win, crossed);
        };

        using Gather_t = CellGatherV<dim, L>;
        // sorted arrays (domain, and level ghosts: on_moved / on_appended sort them too)
        static constexpr bool per_cell = Interpolator_t::interp_order == 1;
        Gather_t const* const none     = nullptr;
        std::array<int, dim> const no_cell{};

        if constexpr (per_cell)
            if (particles.sorted())
            {
                Gather_t g;
                Window win;
                damn::core::for_each_point(particles.shell(), [&](auto const& cell) {
                    auto const [b, size] = particles.cell_range(cell);
                    if (size == 0)
                        return;
                    if (!isIn(cell, boxing.ghostBox)) // the stencils would leave the fields:
                    {                                 //  per lane (not accelerated anyway)
                        for (std::size_t i = b; i < b + size; i += L)
                            block(i, std::min(L, b + size - i), none, no_cell, nullptr);
                        return;
                    }
                    g.load(em, layout, cell);
                    std::array<int, dim> amr;
                    for (std::size_t k = 0; k < dim; ++k)
                        amr[k] = cell[k];
                    if constexpr (copy)
                        reset(win, cell);
                    for (std::size_t i = b; i < b + size; i += L)
                        block(i, std::min(L, b + size - i), &g, amr, copy ? &win : nullptr);
                    if constexpr (copy)
                        flush(win);
                });
                return;
            }

        for (std::size_t i = 0; i < particles.size(); i += L)
            block(i, std::min(L, particles.size() - i), none, no_cell, nullptr);
    }
#endif // MKN_WITH_MKN_AVX
};


template<typename ModelAccessor, typename Interpolator>
template<MultiBorisMode mode>
void MultiBorisBackend<LayoutMode::AoS, ModelAccessor, Interpolator>::move_pooled(
    MultiBoris_t& in, auto const& boxings)
{
    damn::core::ThreadPool::INSTANCE().patch_exec(
        in.accessor.size(), [&](auto const i) { return in.accessor[i]; },
        [&](auto& view) { move_patch<mode>(in, view, boxings); });
}


template<typename ModelAccessor, typename Interpolator>
template<MultiBorisMode mode>
void MultiBorisBackend<LayoutMode::AoS, ModelAccessor, Interpolator>::move_patch(
    MultiBoris_t& in, auto& view, auto const& boxings)
{
    DAMN_LOG_SCOPE(3, "MultiBoris::flat::move_patch");

    auto [ions, em]    = view.args;
    auto const& layout = view.layout;
    auto const& boxing = boxings.at(view.patchID());
    auto const halfdt  = in.mesh(layout.dx, in.dt);
    Interpolator_t interp;

    for (auto& pop : ions)
    {
        double const coef1 = pop.charge() * (0.5 * in.dt / pop.mass());
        auto const args    = std::forward_as_tuple(em, layout, boxing, halfdt, coef1, interp);

        auto const per_type = [&]<auto type>(auto& particles) {
            std::apply(
                [&](auto&... as) {
                    if constexpr (mode == MultiBorisMode::COPY)
                        push_copies_and_deposit<type>(particles, pop, as...);
                    else
                        push_all<type>(particles, as...);
                },
                args);
        };

        per_type.template operator()<ParticleType::Domain>(pop.domainParticles());
        per_type.template operator()<ParticleType::LevelGhost>(pop.levelGhostParticles());
    }
}


template<typename ModelAccessor, typename Interpolator>
template<auto type>
bool MultiBorisBackend<LayoutMode::AoS, ModelAccessor, Interpolator>::push(auto& particle,
                                                                           auto&&... args)
{
    auto&& [em, layout, boxing, halfdt, coef1, interp] = std::forward_as_tuple(args...);
    return push_particle<type>(particle, em, layout, boxing.ghostBox, halfdt, coef1);
}


template<typename ModelAccessor, typename Interpolator>
template<auto type>
bool MultiBorisBackend<LayoutMode::AoS, ModelAccessor, Interpolator>::push_particle(
    auto& particle, auto const& em, auto const& layout, auto const& ghostBox, auto const& halfdt,
    double const coef1) DAMN_ALL
{
    particle.iCell() = boris::advance<alloc_mode>(particle, halfdt);

    if constexpr (type == ParticleType::LevelGhost)
        if (!isIn(particle.iCell(), ghostBox))
            return false;

    Interpolator_t interp;
    boris::accelerate(particle, interp.m2p(particle, em, layout), coef1);
    particle.iCell() = boris::advance<alloc_mode>(particle, halfdt);
    return true;
}


// REF push of one patch: GPU_UNIFIED SoAVX on the device, a thread per particle loading it from
//  the columns, pushing and storing it back; else the host loop
template<typename ModelAccessor, typename Interpolator>
template<MultiBorisMode mode>
void MultiBorisBackend<LayoutMode::AoS, ModelAccessor, Interpolator>::move_rest(auto& in,
                                                                                auto const& boxings,
                                                                                auto const i)
{
    DAMN_LOG_SCOPE(3, "MultiBoris::flat::move_rest");

    auto view = in.accessor[i];
    if constexpr (alloc_mode == damn::AllocatorMode::GPU_UNIFIED
                  and is_soavx(Particles_t::layout_mode))
    {
#if DAMN_HAVE_MKN_GPU
        auto [ions, em]     = view.args;
        auto const layout   = view.layout;
        auto const ghostBox = boxings.at(view.patchID()).ghostBox;
        auto const halfdt   = in.mesh(layout.dx, in.dt);

        for (auto& pop : ions)
        {
            double const coef1 = pop.charge() * (0.5 * in.dt / pop.mass());
            auto const push    = [&]<auto type>(auto& particles) {
                auto ps = particles.view();
                if (ps.size() == 0)
                    return;
                damn::kernel::launch(in.streamer.streams[i], ps.size(), [=] DAMN_ALL() mutable {
                    auto const pi = damn::kernel::idx();
                    auto particle = ps.copy(pi);
                    push_particle<type>(particle, em, layout, ghostBox, halfdt, coef1);
                    ps.assign(particle, pi);
                });
            };
            push.template operator()<ParticleType::Domain>(pop.domainParticles());
            push.template operator()<ParticleType::LevelGhost>(pop.levelGhostParticles());
        }
#endif
    }
    else
        move_patch<mode>(in, view, boxings);
}


// COPY of one patch on the device: a thread per particle pushes a copy loaded from the columns,
//  and deposits it (global atomics) if it stays in this patch's deposit boxes, as
//  push_copies_and_deposit: domain particles and level ghosts in the non level ghost boxes
template<typename ModelAccessor, typename Interpolator>
template<MultiBorisMode mode>
void MultiBorisBackend<LayoutMode::AoS, ModelAccessor, Interpolator>::move_gpu_copy(
    [[maybe_unused]] auto& in, [[maybe_unused]] auto const i)
{
    DAMN_LOG_SCOPE(3, "MultiBoris::flat::move_gpu_copy");

#if DAMN_HAVE_MKN_GPU
    static_assert(has_gpu_copy);
    using Deposit_t = core::Interpolator<dim, Layout_t::options.interp_order, /*atomic=*/true>;

    auto view           = in.accessor[i];
    auto [ions, em]     = view.args;
    auto const layout   = view.layout;
    auto const halfdt   = in.mesh(layout.dx, in.dt);
    auto const ghostBox = damn::core::grow(layout.cells, Layout_t::options.particle_ghost_width);
    auto const nlgb     = in.gpu_nlgb[i];

    for (auto& pop : ions)
    {
        double const coef1 = pop.charge() * (0.5 * in.dt / pop.mass());
        auto rhop          = pop.particleDensity();
        auto flux          = *pop.flux();

        auto const launch = [&]<auto type>(auto const& particles) {
            using Span_t = SoAVXConstSpanParticles<dim, alloc_mode>;
            Span_t const ps{particles};
            if (ps.size() == 0)
                return;
            damn::kernel::launch(in.streamer.streams[i], ps.size(), [=] DAMN_ALL() mutable {
                auto particle = ps.copy(damn::kernel::idx());
                if (!push_particle<type>(particle, em, layout, ghostBox, halfdt, coef1))
                    return;

                // level ghosts as domain particles: each is held by one patch only
                if (isIn(particle.iCell(), nlgb))
                    Deposit_t{}.particleToMesh(particle, rhop, flux, layout);
            });
        };

        launch.template operator()<ParticleType::Domain>(pop.domainParticles());
        launch.template operator()<ParticleType::LevelGhost>(pop.levelGhostParticles());
    }
#else
    throw std::runtime_error("NEEDS GPU IMPL!");
#endif
}


template<typename ModelAccessor, typename Interpolator>
template<auto type>
void MultiBorisBackend<LayoutMode::AoS, ModelAccessor, Interpolator>::push_all(auto& particles,
                                                                               auto&&... args)
{
    per_particle</*store=*/true>(particles, [&](auto& particle) { push<type>(particle, args...); });
}


template<typename ModelAccessor, typename Interpolator>
template<auto type>
void MultiBorisBackend<LayoutMode::AoS, ModelAccessor, Interpolator>::push_copies_and_deposit(
    auto const& particles, auto& pop, auto&&... args)
{
    auto&& [em, layout, boxing, halfdt, coef1, interp] = std::forward_as_tuple(args...);

    auto& rhoP = pop.particleDensity();
    auto& flux = pop.flux();

    per_particle</*store=*/false>(particles, [&](auto& particle) { // pushes a copy
        if (!push<type>(particle, args...))
            return;

        // level ghosts as domain particles: each is held by one patch only
        bool const deposits = isIn(particle.iCell(), boxing.nonLevelGhostBox);

        if (deposits)
            interp.particleToMesh(particle, rhoP, flux, layout);
    });
}



// ── Per-tile functors (shared between CPU and GPU, and every dispatch) ─────────────────
// per-tile containers differ per layout (per-cell for AoSPCTS, flat cell-mapped for
// AoSCMTS, flat for AoSTS): each is handled by its own requires(layout) overload

template<auto particle_type, auto boris_mode, typename Backend_t>
struct MultiBorisFunctors
{
    static_assert(mkn::kul::all_are<ParticleType>(particle_type));

    using Layout_t       = Backend_t::Layout_t;
    using Particles_t    = Backend_t::Particles_t;
    using Electromag_t   = Backend_t::Electromag_t;
    using Interpolator_t = Backend_t::Interpolator_t;
    using Particles_v    = Particles_t::view_t;

    using Vecfield_t    = Electromag_t::vecfield_type;
    using Field_t       = Vecfield_t::field_type;
    using Tile_vt       = Field_t::value_type::value_type;
    using VecField_vt   = basic::TensorField<Tile_vt, 1>;
    using Electromag_vt = basic::Electromag<VecField_vt>;

    static constexpr auto dim = Layout_t::dimension;
    using Particle_t          = ParticleDefaults<dim>::Particle_t;

    static constexpr std::size_t copy_batch_size = 1024;
    static_assert(Particles_t::storage_mode == StorageMode::VECTOR);

    MultiBorisFunctors(auto& in, auto& view, auto& pop, auto& parts, auto& em)
        : pps{*parts}
        , electromag{em}
        , coef1{pop.charge() * (0.5 * in.dt / pop.mass())}
        , halfdt{in.mesh(view.layout.dx, in.dt)}
        , particles{parts}
    {
        check_particles(parts);
        check_particles_views(parts);
    }

    void operator()(auto& in, [[maybe_unused]] auto const i)
    {
        // GPU_UNIFIED without streams (core::MultiBoris) falls back to the CPU loop
        if constexpr (Particles_t::alloc_mode == damn::AllocatorMode::GPU_UNIFIED
                      and requires { in.streamer; })
            on_gpu_tiles(in, i);
        else
            on_cpu_tiles();
    }

    void on_gpu_tiles([[maybe_unused]] auto& in, [[maybe_unused]] auto const i)
    {
#if !DAMN_HAVE_GPU
        throw std::runtime_error("NEEDS GPU IMPL!");
#else
        if constexpr (Particles_t::layout_mode == LayoutMode::AoSPCTS)
            on_gpu_tile_cells(in, i);
        else
            on_gpu_flat_tiles(in, i);
#endif
    }

    // AoSPCTS: block = (tile, cell), block threads stride the cell's particles
    void on_gpu_tile_cells([[maybe_unused]] auto& in, [[maybe_unused]] auto const i)
    {
#if DAMN_HAVE_GPU
        using Launcher = damn::core::gpu::TileCellLauncher<false>;

        auto const max_cells = Launcher::max_cells(pps);
        if (max_cells == 0)
            return;

        std::size_t const threads = damn::kernel::warp_size();
        Launcher{pps().size(), max_cells, threads}.stream(
            in.streamer.streams[i], [self = *this, threads] __device__() mutable {
                auto const tile_idx = Launcher::tile_idx();
                auto& tile          = self.pps()[tile_idx];
                auto& parts         = tile();
                auto const& lobox   = parts.local_box();
                if (Launcher::cell_idx() >= lobox.size())
                    return;
                auto const bix = damn::core::point_at(lobox, Launcher::cell_idx());
                self.per_cell(parts.particles_(bix), self.electromag.E[0][tile_idx].layout(),
                              self.em_tile(tile_idx), self.pps.local_cell(tile.lower),
                              Launcher::thread_idx(), threads);
            });
#endif
    }

    // flat tiles: block = tile, warp of threads striding the tile's particles
    void on_gpu_flat_tiles([[maybe_unused]] auto& in, [[maybe_unused]] auto const i)
    {
#if DAMN_HAVE_GPU
        using Launcher = damn::core::gpu::ChunkLauncher<false>;
        Launcher launcher{1, 0};
        launcher.b.x           = damn::kernel::warp_size();
        launcher.g.x           = pps().size();
        auto const tile_picker = [pps = pps] __device__() {
            return std::make_tuple(blockIdx.x, &pps()[blockIdx.x], threadIdx.x,
                                   damn::kernel::warp_size());
        };
        launcher.stream(in.streamer.streams[i],
                        [=, self = *this] __device__() mutable { self.per_tile(tile_picker); });
#endif
    }

    void on_cpu_tiles()
    {
        for (std::size_t tileidx = 0; tileidx < pps().size(); ++tileidx)
            one_tile(tileidx);
    }

    // one tile's worth of work — the unit of dispatch for pool.detach_task() in the
    // thread-pooled path (see MultiBorisBackend::move_rest_pooled).
    void one_tile(std::size_t const tile_idx)
    {
        auto const tile_picker
            = [&]() { return std::make_tuple(tile_idx, &pps()[tile_idx], 0, 1); };
        per_tile(tile_picker);
    }

    static auto tracker(auto&&... args) DAMN_ALL
    {
        return make_particle_tracker<Particles_t::layout_mode, particle_type, dim>(args...);
    }

    // tile_cell names the tile PHYSICALLY holding a particle (for level ghosts this is
    // also their cell's clamp owner)

    void per_tile(auto const& tile_picker)
        requires(Particles_t::layout_mode == LayoutMode::AoSPCTS)
    DAMN_ALL
    {
        auto&& [tile_idx, tileptr, tidx, ws] = tile_picker();
        auto& tile                           = *tileptr;
        auto const& layout                   = electromag.E[0][tile_idx].layout();
        auto const& em                       = em_tile(tile_idx);
        auto& parts                          = tile();
        auto const tile_cell                 = pps.local_cell(tile.lower);
        check_electromag(em); // debug builds: once a tile, not a particle (a whole tile scan)

        // GPU uses on_gpu_tile_cells, one block per cell
        damn::core::for_each_point(parts.local_box(), [&](auto const& bix) {
            per_cell(parts.particles_(bix), layout, em, tile_cell, tidx, ws);
        });
    }

    // flat tiles (AoSCMTS, AoSTS, ...): pid is the index in the tile's array
    void per_tile(auto const& tile_picker)
        requires(Particles_t::layout_mode != LayoutMode::AoSPCTS)
    DAMN_ALL
    {
        auto&& [tile_idx, tileptr, tidx, ws] = tile_picker();
        auto& tile                           = *tileptr;
        auto const& layout                   = electromag.E[0][tile_idx].layout();
        auto const& em                       = em_tile(tile_idx);
        auto& parts                          = tile();
        auto const tile_cell                 = pps.local_cell(tile.lower);
        check_electromag(em); // debug builds: once a tile, not a particle (a whole tile scan)

        auto const each = pps()[tile_idx]().size() / ws;

        auto const one = [&] DAMN_ALL(std::size_t const pidx) {
            per_any_particle(parts, layout, tracker(parts.iCell(pidx), tile_cell), pidx, em);
        };

        std::size_t pid = 0;
        for (; pid < each; ++pid)
            one(pid * ws + tidx);
        if constexpr (Particles_t::alloc_mode == damn::AllocatorMode::GPU_UNIFIED)
            if (static_cast<std::size_t>(tidx) < parts.size() - (ws * each))
                one(pid * ws + tidx);
    }


    // per-cell buckets are tracked even in the tile ghost layer, so any particle whose
    // cell changed needs registering — domain and level ghost alike
    void move_check(auto const& pt, std::size_t const pidx, auto& particle)
        requires(Particles_t::layout_mode == LayoutMode::AoSPCTS)
    DAMN_ALL
    {
        pps.template move_check<particle_type>(pt, pidx, particle);
    }

    // registers on the vector: a cellmap can't be updated through its view
    void move_check(auto const& pt, std::size_t const pidx, auto& particle)
        requires(Particles_t::layout_mode == LayoutMode::AoSCMTS)
    {
        particles.template move_check<particle_type>(pt, pidx, particle);
    }

    // other flat tiles: domain particles staying in the patch box register, level ghosts
    //  always (the array skips those leaving the ghost box)
    void move_check(auto const& pt, std::size_t const pidx, auto& particle)
        requires(not mkn::kul::any_in(Particles_t::layout_mode, LayoutMode::AoSPCTS,
                                      LayoutMode::AoSCMTS))
    DAMN_ALL
    {
        if constexpr (particle_type == ParticleType::Domain)
        {
            if (isIn(particle, pps.box()))
                pps.template move_check<particle_type>(pt, pidx, particle);
        }
        else
            pps.template move_check<particle_type>(pt, pidx, particle);
    }

    // AoSPCTS: one cell of a tile, particles from tidx strided by ws (CPU: 0, 1)
    void per_cell(auto& cell_particles, auto const& layout, auto const& em, auto const& tile_cell,
                  std::size_t const tidx, std::size_t const ws) DAMN_ALL
    {
        for (std::size_t pid = tidx; pid < cell_particles.size(); pid += ws)
            per_particle(cell_particles[pid], layout,
                         tracker(cell_particles[pid].iCell(), tile_cell), pid, em);
    }

    void per_any_particle(auto& particles, auto&&... args) DAMN_ALL
    {
        auto const& pidx = std::get<2>(std::forward_as_tuple(args...));
#if DAMN_HAVE_THRUST
        using enum LayoutMode;
        if constexpr (mkn::kul::any_in(Particles_t::layout_mode, SoA, SoAPC, SoATS))
            per_particle(SoAZipParticle{particles, pidx}, args...);
        else
#endif
            per_particle(particles[pidx], args...);
    }


    void per_particle_still_in_ghost_box(auto&&... args) DAMN_ALL
    {
        static constexpr auto alloc_mode             = Particles_t::alloc_mode;
        auto const& [particle, layout, pt, pidx, em] = std::forward_as_tuple(args...);

        {
            Interpolator_t interp;
            boris::accelerate(particle, interp.m2p(particle, em, layout), coef1);
        }
        particle.iCell() = boris::advance<alloc_mode>(particle, halfdt);

        if constexpr (boris_mode == MultiBorisMode::REF)
            // particles are tracked even in the tile ghost layer, so any particle
            // whose cell changed needs registering — domain and level ghost alike. pt was
            // built in per_tile, before advance() ran.
            move_check(pt, pidx, particle);
    }

    void per_particle(auto&&... args) DAMN_ALL
    {
        static constexpr auto alloc_mode             = Particles_t::alloc_mode;
        auto const& [particle, layout, pt, pidx, em] = std::forward_as_tuple(args...);

        particle.iCell() = boris::advance<alloc_mode>(particle, halfdt);

        if constexpr (particle_type == ParticleType::Domain)
            per_particle_still_in_ghost_box(args...);
        else if constexpr (particle_type == ParticleType::LevelGhost)
        {
            if (isIn(particle, pps.ghost_box()))
                per_particle_still_in_ghost_box(args...);
            else if constexpr (boris_mode == MultiBorisMode::REF)
                // left the ghost box on the first half-step — register the departure or
                // the tile's cellmap goes stale; move_check resolves it as a deletion
                move_check(pt, pidx, particle);
        }
        else
        {
            DAMN_ASSERT(false);
        }
    }

    // one tile's worth of the COPY-mode deposit pass. Domain and LevelGhost write into
    // the SAME tile's rhoP/flux, so callers must run this tile's Domain and
    // LevelGhost passes in one task (never two concurrent tasks for the same tile_idx)
    // — see MultiBorisBackend::move_cpu_copy_pooled. Different tiles are independent.
    void one_copy_tile(std::size_t const tile_idx, auto& boxings, auto& view, auto& pop)
        requires(mkn::kul::any_in(Particles_t::layout_mode, LayoutMode::AoSPCTS,
                                  LayoutMode::AoSCMTS))
    {
        auto const& patch_id      = view.patchID();
        auto const& patch_boxings = boxings.at(patch_id);
        auto const& patch_box     = pps.box();

        auto& tile           = pps()[tile_idx];
        bool const is_border = patch_box * damn::core::grow(tile, 1) != damn::core::grow(tile, 1);
        auto const& layout   = electromag.E[0][tile_idx].layout();
        auto const tile_em   = em_tile(tile_idx);
        auto& rhoP           = pop.particleDensity()[tile_idx];
        auto F               = tile_at(pop.flux(), tile_idx);
        Interpolator_t interp;

        // level ghosts as domain particles (each is held by one patch, and lives only in its
        //  clamp-owner tile: seen once), the tile reduction folds halo contributions into the
        //  neighbours
        auto const deposits = [&](auto const& p) {
            return !is_border || isIn(p.iCell(), patch_boxings.nonLevelGhostBox);
        };

        // batched in phases (copy + half step, m2p + accelerate + half step, deposit) as
        // interleaving m2p and p2m per particle is slower, see updater_realloc's
        // ParticlesSynchronisedMover1::Pusher::move_domain_copy
        std::array<Particle_t, copy_batch_size> batch;
        std::size_t n = 0;

        auto const flush = [&]() {
            std::size_t m = 0;
            for (std::size_t j = 0; j < n; ++j)
            {
                auto& p = batch[j];
                boris::accelerate(p, interp.m2p(p, tile_em, layout), coef1);
                p.iCell() = boris::advance<damn::AllocatorMode::CPU>(p, halfdt);
                if (deposits(p))
                    batch[m++] = p;
            }
            for (std::size_t j = 0; j < m; ++j)
                interp.particleToMesh(batch[j], rhoP(), F, layout);
            n = 0;
        };

        auto const per_array = [&](auto const& ps) {
            for (auto const& p : ps)
            {
                batch[n]         = p;
                batch[n].iCell() = boris::advance<damn::AllocatorMode::CPU>(batch[n], halfdt);

                // as in per_particle: a level ghost leaving the ghost box on the first
                // half step is dropped before m2p, which can't interpolate outside it
                if constexpr (particle_type == ParticleType::LevelGhost)
                    if (!isIn(batch[n], pps.ghost_box()))
                        continue;

                if (++n == copy_batch_size)
                    flush();
            }
        };

        for_each_copy_array(tile(), per_array);
        flush();
    }

    static void for_each_copy_array(auto& tile_particles, auto&& fn)
        requires(Particles_t::layout_mode == LayoutMode::AoSPCTS)
    {
        for (auto& cell : tile_particles()) // all cells, for levelghosts, batch spans cells
            fn(cell);
    }

    static void for_each_copy_array(auto& tile_particles, auto&& fn)
        requires(Particles_t::layout_mode == LayoutMode::AoSCMTS)
    {
        fn(tile_particles); // one flat array per tile
    }

    void one_copy_tile(std::size_t const tile_idx, auto& boxings, auto& view, auto& pop)
        requires(not mkn::kul::any_in(Particles_t::layout_mode, LayoutMode::AoSPCTS,
                                      LayoutMode::AoSCMTS))
    {
        auto const& patch_id      = view.patchID();
        auto const& patch_boxings = boxings.at(patch_id);
        auto const& patch_box     = pps.box();

        auto& tile           = pps()[tile_idx];
        bool const is_border = patch_box * damn::core::grow(tile, 1) != damn::core::grow(tile, 1);
        auto const& layout   = electromag.E[0][tile_idx].layout();
        auto const tile_em   = em_tile(tile_idx);
        auto& rhoP           = pop.particleDensity()[tile_idx];
        auto F               = tile_at(pop.flux(), tile_idx);
        Interpolator_t interp;

        auto const tile_cell = pps.local_cell(tile.lower);

        auto on_tile = [&]<bool border>() {
            for (auto p : tile())
            {
                per_particle(p, layout, tile_cell, 0, tile_em);
                if constexpr (!border)
                    interp.particleToMesh(p, rhoP(), F, layout);
                else
                {
                    if (isIn(p.iCell(), patch_boxings.nonLevelGhostBox))
                        interp.particleToMesh(p, rhoP(), F, layout);
                }
            }
        };

        if (is_border)
            on_tile.template operator()<true>();
        else
            on_tile.template operator()<false>();
    }

    void per_copy_of_cpu_tile(auto& boxings, auto& view, auto& pop)
    {
        for (std::size_t tile_idx = 0; tile_idx < pps().size(); ++tile_idx)
            one_copy_tile(tile_idx, boxings, view, pop);
    }

    auto em_tile(auto const tidx) DAMN_ALL
    {
        return electromag.template as<Electromag_vt>([&] DAMN_ALL(auto const& vf) {
            return mkn::kul::for_N_make_array<3>([&](auto i) { return vf[i][tidx](); });
        });
    }


    Particles_v pps;
    Electromag_t::Super const electromag;
    double const coef1;
    std::array<double, dim> halfdt;
    Particles_t& particles;
};

} // namespace plamn::core


#if DAMN_HAVE_MKN_GPU

namespace plamn::core::mkn_xyz
{

// ── mkn.gpu dispatch: one ThreadedStreamLauncher host thread + stream per patch ────────
// same per patch steps/functors as core::MultiBoris, plus the GPU kernel paths

template<typename ModelAccessor, typename Interpolator>
struct MultiBoris : core::MultiBoris<ModelAccessor, Interpolator>
{
    using Super           = core::MultiBoris<ModelAccessor, Interpolator>;
    using Backend         = Super::Backend;
    using Particles_t     = Super::Particles_t;
    using Box_t           = damn::core::Box<int, Super::dim>;
    using StreamLauncher  = damn::core::gpu::ThreadedStreamLauncher<ModelAccessor>;
    using GpuBoxSpanSet_t = damn::core::SpanSet<Box_t, damn::core::default_span_size_t,
                                                mkn::gpu::ManagedAllocator<Box_t>>;

    // damn's synchronous pools (DAMN_THREAD_POOL_SYNC): the host steps on this thread too
    static constexpr bool synchronous = damn::core::ThreadPool::synchronous;

    MultiBoris(double const dt_, ModelAccessor& _accessor)
        : Super{dt_, _accessor}
    {
    }

    template<MultiBorisMode mode = MultiBorisMode::REF>
    void move(auto const& boxings);

    StreamLauncher streamer{this->accessor, synchronous ? 0u : 1u};
    GpuBoxSpanSet_t gpu_nlgb;
    GpuBoxSpanSet_t gpu_non_owned;
};


template<typename ModelAccessor, typename Interpolator>
template<MultiBorisMode mode>
void MultiBoris<ModelAccessor, Interpolator>::move(auto const& boxings)
{
    static constexpr auto copy     = mode == MultiBorisMode::COPY;
    static constexpr auto is_gpu   = Particles_t::alloc_mode == damn::AllocatorMode::GPU_UNIFIED;
    static constexpr auto gpu_copy = copy and is_gpu and Backend::has_gpu_copy;

    if constexpr (gpu_copy)
        Backend::prepare_gpu_copy(*this, boxings);

    auto move = [&](auto const i) mutable {
        if constexpr (gpu_copy)
            Backend::template move_gpu_copy<mode>(*this, i);
        else if constexpr (copy) // CPU, or GPU_UNIFIED without a GPU copy kernel (AoSPCTS)
        {
            if constexpr (is_gpu) // unified memory: wait for prior work on this patch
                streamer.streams[i].sync();
            Backend::template move_cpu_copy<mode>(*this, boxings, i);
        }
        else if constexpr (requires { Backend::flat; }) // untiled: needs the patch boxings
            Backend::template move_rest<mode>(*this, boxings, i);
        else
            Backend::template move_rest<mode>(*this, i);
    };
    auto sync = [&](auto const i) mutable {
        if constexpr (not copy)
        {
            if constexpr (is_gpu) // unified memory: wait for the patch's kernels
                streamer.streams[i].sync();
            auto view = this->accessor[i];
            Backend::sync_ref(view, streamer.streams[i]);
        }
    };

    if constexpr (synchronous)
    {
        for (std::size_t i = 0; i < this->accessor.size(); ++i)
        {
            move(i);
            sync(i);
        }
    }
    else
    {
        // .host() takes a forwarding reference: passing the named lvalues directly
        // would deduce a reference type and store a dangling ref to this stack frame
        // once move() returns, so move them in to get a real, owned copy instead
        streamer.host(std::move(move));
        streamer.host(std::move(sync));
    }
}


} // namespace plamn::core::mkn_xyz

#endif // DAMN_HAVE_MKN_GPU


#endif
