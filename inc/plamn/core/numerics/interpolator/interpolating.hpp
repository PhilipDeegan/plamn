#ifndef PLAMN_CORE_NUMERICS_INTERPOLATOR_INTERPOLATING_HPP
#define PLAMN_CORE_NUMERICS_INTERPOLATOR_INTERPOLATING_HPP

#include "damn/core/exec/kernels.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/geometry/range.hpp"
#include "damn/core/def/damn_config.hpp"
#include "damn/core/def/detail/mkn_gpu.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/logger.hpp"
#include "mkn/kul/for.hpp"
#include "mkn/kul/meta.hpp"
#include "mkn/kul/env.hpp"

#include "plamn/core/data/electromag/electromag.hpp" // IWYU pragma: keep (basic::Electromag: GPU builds)
#include "plamn/core/data/field/field_tiles.hpp"
#include "plamn/core/data/tensorfield/tensorfield.hpp"
#include "plamn/core/data/particles/particles.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/data/particles/arrays/particles_soavx.hpp"
#include "plamn/core/numerics/pusher/boris/basics.hpp" // IWYU pragma: keep (GPU builds' push)

#include "interpolator.hpp"

#if defined(MKN_WITH_MKN_AVX) // cell_deposit_sorted
#include "cell_gather.hpp"
#endif

#include <cassert>
#include <string>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <type_traits>

namespace plamn::core
{

// visits, for tile tidx, the particles of every cell its level ghost halo reaches, read
// from that cell's clamp-owner tile (TileSet::tag_cells_). Level ghosts are never in the
// patch domain, so only halo cells outside of it are visited: none for interior tiles
template<typename Particles>
void on_reachable_level_ghosts(Particles const& particles, std::size_t const tidx, auto&& fn)
    requires(mkn::kul::any_in(Particles::layout_mode, LayoutMode::AoSPCTS, LayoutMode::AoSCMTS))
{
    auto const& patch_box    = particles.box();
    auto const ghost_cells   = patch_box.lower[0] - particles.ghost_box().lower[0];
    auto const& tile         = particles()[tidx];
    auto const tile_halo     = damn::core::grow(*tile, ghost_cells);
    auto const halo_in_patch = tile_halo * patch_box;
    if (halo_in_patch and *halo_in_patch == tile_halo)
        return; // interior tile

    damn::core::for_each_point(tile_halo, [&](auto const& amr) {
        if (isIn(amr, patch_box))
            return;
        auto const& owner = (*particles().at(amr))();
        if constexpr (Particles::layout_mode == LayoutMode::AoSCMTS)
            for (auto const idx : owner.map(amr))
                fn(damn::core::makeRange(owner, idx, idx + 1));
        else
            fn(owner(owner.local_cell(amr)));
    });
}


// how a per-tile GPU deposit kernel accumulates rhoP + the 3 flux components, all with atomics:
//  shared_all  all four tile fields staged in dynamic shared memory, one pass over the particles
//  shared_one  one field staged at a time, four passes (a pushing kernel pushes again per pass)
//  global      no staging, straight into the tile's fields in global memory
enum class TileDeposit : std::uint8_t { shared_all = 0, shared_one, global };

std::size_t constexpr tile_deposit_fields(TileDeposit const mode) DAMN_ALL
{
    return mode == TileDeposit::shared_all ? 4 : mode == TileDeposit::shared_one ? 1 : 0;
}

// dynamic shared memory of a deposit kernel: staged fields of n_points each, then extra_bytes
std::size_t constexpr tile_deposit_bytes(TileDeposit const mode, std::size_t const n_points,
                                         std::size_t const extra_bytes = 0) DAMN_ALL
{
    return tile_deposit_fields(mode) * n_points * sizeof(double) + extra_bytes;
}

// the fewest passes that fit in max_bytes of dynamic shared memory. For benchmarking,
//  PLAMN_GPU_TILE_DEPOSIT=shared_all|shared_one|global forces a mode (throws if it can't fit)
inline TileDeposit tile_deposit_mode(std::size_t const n_points, std::size_t const extra_bytes,
                                     std::size_t const max_bytes)
{
    using enum TileDeposit;
    static std::optional<TileDeposit> const forced = []() -> std::optional<TileDeposit> {
        auto const env = mkn::kul::env::get("PLAMN_GPU_TILE_DEPOSIT");
        if (!env)
            return std::nullopt;
        if (*env == "shared_all")
            return shared_all;
        if (*env == "shared_one")
            return shared_one;
        if (*env == "global")
            return global;
        throw std::runtime_error("PLAMN_GPU_TILE_DEPOSIT: unknown mode " + *env);
    }();

    auto const fits = [&](TileDeposit const mode) {
        return tile_deposit_bytes(mode, n_points, extra_bytes) <= max_bytes;
    };

    if (forced)
    {
        if (!fits(*forced))
        {
            // printed too: this runs on stream worker threads, where the exception terminates
            std::string const msg
                = "PLAMN_GPU_TILE_DEPOSIT: forced mode needs "
                  + std::to_string(tile_deposit_bytes(*forced, n_points, extra_bytes))
                  + " bytes of shared memory, the maximum is " + std::to_string(max_bytes);
            std::cerr << msg << std::endl;
            throw std::runtime_error(msg);
        }
        return *forced;
    }
    for (auto const mode : {shared_all, shared_one})
        if (fits(mode))
            return mode;
    return global;
}

// calls fn(std::integral_constant<TileDeposit, mode>{}), to make a runtime mode a kernel template
//  argument
template<typename Fn>
void with_tile_deposit(TileDeposit const mode, Fn&& fn)
{
    using enum TileDeposit;
    if (mode == shared_all)
        fn(std::integral_constant<TileDeposit, shared_all>{});
    else if (mode == shared_one)
        fn(std::integral_constant<TileDeposit, shared_one>{});
    else
        fn(std::integral_constant<TileDeposit, global>{});
}


// simple facade to launch e.g. GPU kernels if needed
//  and to not need to modify the interpolator much for gpu specifically

template<std::size_t dim, std::size_t interpOrder, bool atomic_ops = false,
         typename Interpolator_t = Interpolator<dim, interpOrder, atomic_ops>>
class Interpolating
{
public:
    inline void operator()(auto const& particles, auto& rhoP, auto& flux, auto const& layout,
                           double coef = 1.)
    {
        DAMN_LOG_SCOPE(3, "Interpolating::particleToMesh");
        particleToMesh(particles, layout, rhoP, flux, coef);
    }

    // LevelGhost: level ghosts live only in their clamp-owner tile and are deposited before
    // the tile reduction + border sums (fillFluxBorders/fillDensityBorders), so like domain
    // particles each tile deposits only its own, the reduction completes the tile seams
    // GPU_UNIFIED falls back to the CPU path until there's an optimized GPU version
    template<auto type = ParticleType::Domain, typename Particles>
    void particleToMesh(Particles const& particles, auto const& /*layout: the tiles'*/, auto& rhoP,
                        auto& flux, double coef = 1.)
        requires(mkn::kul::any_in(Particles::layout_mode, LayoutMode::AoSPCTS, LayoutMode::AoSCMTS))
    DAMN_ALL
    {
        for (std::size_t tidx = 0; tidx < particles().size(); ++tidx)
            particleToMesh_tile<type>(tidx, particles, rhoP, flux, coef);
    }

    // only writes the moments of tile tidx, distinct tiles can be deposited concurrently
    // AoSTS GPU_UNIFIED: host loop over unified memory like its particleToMesh below, the
    //  device deposit is the updater's (on_tiles / on_tiles_push_deposit)
    template<auto type = ParticleType::Domain, typename Particles>
    void particleToMesh_tile(std::size_t const tidx, Particles const& particles, auto& rhoP,
                             auto& flux, double coef = 1.)
        requires(mkn::kul::any_in(Particles::layout_mode, LayoutMode::AoSPCTS, LayoutMode::AoSCMTS,
                                  LayoutMode::AoSTS))
    {
        static_assert(mkn::kul::any_in(type, ParticleType::Domain, ParticleType::LevelGhost));

        auto [rhop, F]      = tiles_at(tidx, rhoP, flux);
        auto const& rho_lay = tile_layout(rhoP, tidx);
        auto const deposit  = [&](auto const& cell_particles) {
            for (auto const& p : cell_particles)
                interp_.particleToMesh(p, rhop, F, rho_lay, coef);
        };

        auto& tile_particles = particles()[tidx]();
        if constexpr (mkn::kul::any_in(Particles::layout_mode, LayoutMode::AoSCMTS,
                                       LayoutMode::AoSTS))
            deposit(tile_particles);
        else
            damn::core::for_each_point(tile_particles.local_box(),
                                       [&](auto const& bix) { deposit(tile_particles(bix)); });
    }

    template<auto type = ParticleType::Domain, typename Particles>
    void particleToMesh(Particles const& particles, auto const& layout, auto& rhoP, auto& flux,
                        double coef = 1.)
        requires(Particles::layout_mode == LayoutMode::AoSMapped)
    DAMN_ALL
    {
        interp_(particles, rhoP, flux, layout, coef);
    }

    template<auto type = ParticleType::Domain, typename Particles>
    void particleToMesh(Particles const& particles, auto const& /*layout: the tiles'*/, auto& rhoP,
                        auto& flux, double coef = 1.)
        requires(Particles::layout_mode == LayoutMode::AoSTS)
    DAMN_ALL
    {
        // GPU_UNIFIED falls back to the CPU path until there's an optimized GPU version
        for (std::size_t tidx = 0; tidx < particles().size(); ++tidx)
        {
            auto [rhop, F]      = tiles_at(tidx, rhoP, flux);
            auto const& rho_lay = tile_layout(rhoP, tidx);
            for (auto const& p : particles()[tidx]())
                interp_.particleToMesh(p, rhop, F, rho_lay, coef);
        }
    }

    template<auto type = ParticleType::Domain, typename Particles>
    void particleToMesh(Particles const& particles, auto const& layout, auto& rhoP, auto& flux,
                        double coef = 1.)
        requires(Particles::layout_mode == LayoutMode::SoATS)
    DAMN_ALL
    {
        for (auto const& tile : particles())
            for (std::size_t i = 0; i < tile().size(); ++i)
                interp_.particleToMesh(tile()[i], rhoP, flux, layout, coef);
    }

    template<auto type = ParticleType::Domain, typename Particles>
    void particleToMesh(Particles const& particles, auto const& layout, auto& rhoP, auto& flux,
                        double coef = 1.)
        requires(Particles::layout_mode == LayoutMode::AoSPC)
    DAMN_ALL
    {
        auto constexpr static alloc_mode = Particles::alloc_mode;

        if constexpr (alloc_mode == damn::AllocatorMode::CPU)
        {
            damn::core::for_each_point(particles.local_box(), [&](auto const& bix) {
                for (auto const& p : particles(bix))
                    interp_.particleToMesh(p, rhoP, flux, layout, coef);
            });
        }
        // else if (alloc_mode == AllocatorMode::GPU_UNIFIED and impl < 2)
        // {
        //     static_assert(atomic_ops, "GPU must be atomic");
        //     DAMN_WITH_MKN_GPU(
        //         mkn::gpu::GDLauncher{particles.size()}([=] DAMN_ALL() mutable {
        //             auto it = particles[mkn::gpu::idx()];
        //             Interpolator_t{}.particleToMesh(*it, density, flux, layout, coef);
        //         }); //
        //     )
        // }
        else if (alloc_mode == damn::AllocatorMode::GPU_UNIFIED /*and impl == 2*/)
        {
            DAMN_WITH_MKN_GPU({ //
                using lobox_t  = Particles::lobox_t;
                using Launcher = damn::core::gpu::BoxCellNLauncher<lobox_t>;
                auto lobox     = particles.local_box();
                Launcher{lobox, particles.max_size()}(
                    [=] DAMN_ALL() mutable { box_kernel(particles, layout, flux, rhoP, coef); });
            })
        }
        else
            throw std::runtime_error("fail");
    }

    // stream: optional, GPU_UNIFIED SoAVX deposits on it (queued after prior work on it)
    template<auto type = ParticleType::Domain, typename Particles>
    void particleToMesh(Particles const& particles, auto const& layout, auto& rhoP, auto& flux,
                        double coef = 1., auto&... stream)
        requires(mkn::kul::any_in(Particles::layout_mode, LayoutMode::AoS, LayoutMode::SoA,
                                  LayoutMode::SoAVX, LayoutMode::SoAVXChunked,
                                  LayoutMode::AoS4SoAVXChunked))
    DAMN_ALL
    {
        auto constexpr static alloc_mode = Particles::alloc_mode;

        // not atomic (level init, messenger ghosts): on the host, GPU_UNIFIED memory included
        if constexpr (alloc_mode == damn::AllocatorMode::CPU or !atomic_ops)
        {
            if constexpr (is_soavx(Particles::layout_mode))
            {
#if defined(MKN_WITH_MKN_AVX)
                if constexpr (Particles::layout_mode == LayoutMode::SoAVXChunked
                              and alloc_mode == damn::AllocatorMode::CPU and interpOrder == 1)
                    if (particles.sorted() and coef == 1.)
                    {
                        cell_deposit_sorted<mkn::avx::Options::N<double>()>(particles, rhoP, flux,
                                                                            layout);
                        return;
                    }
#endif
                for (auto const& it : particles)
                    if (it.weight() != 0) // SoAVXChunked's gaps
                        interp_.particleToMesh(it, rhoP, flux, layout, coef);
            }
            else if constexpr (Particles::layout_mode == LayoutMode::AoS4SoAVXChunked)
            {
#if defined(MKN_WITH_MKN_AVX)
                if constexpr (interpOrder == 1)
                    if (particles.sorted() and coef == 1.)
                    {
                        cell_deposit_sorted<mkn::avx::Options::N<double>()>(particles, rhoP, flux,
                                                                            layout);
                        return;
                    }
#endif
                for (auto const& p : particles)
                    if (p.weight() != 0) // the chunks' gaps
                        interp_.particleToMesh(p, rhoP, flux, layout, coef);
            }
            else
                interp_(particles, rhoP, flux, layout, coef);
        }
        else if constexpr (alloc_mode == damn::AllocatorMode::GPU_UNIFIED)
        {
            static_assert(atomic_ops, "GPU must be atomic");
            if constexpr (is_soavx(Particles::layout_mode))
            {
#if DAMN_HAVE_MKN_GPU_HW
                if constexpr (Particles::layout_mode == LayoutMode::SoAVXChunked
                              and interpOrder == 1)
                    if (particles.sorted()
                        and particles.size() >= warp_cell_min_ppc() * particles.shell().size())
                    {
                        warp_cell_deposit(particles, layout, rhoP, flux, coef, stream...);
                        return;
                    }
#endif
                // a thread per particle, loaded from the columns, global atomics
                // plain views (basic::, device copyable), written through per thread copies
                DAMN_WITH_MKN_GPU( //
                    SoAVXConstSpanParticles<Particles::dimension, alloc_mode> const ps{
                        particles};                   //
                    auto rho_v  = untiled_view(rhoP); //
                    auto flux_v = untiled_view(flux); //
                    damn::kernel::launch(stream..., ps.size(),
                                         [=] DAMN_ALL() {
                                             auto rho = rho_v;
                                             auto F   = flux_v;
                                             Interpolator_t{}.particleToMesh(
                                                 ps.copy(damn::kernel::idx()), rho, F, layout,
                                                 coef);
                                         }); //
                )
            }
            else
            {
                DAMN_WITH_MKN_GPU( //
                    damn::kernel::launch(
                        particles.size(),
                        [=] DAMN_ALL() mutable {
                            auto particle = particles.begin() + damn::kernel::idx();
                            Interpolator_t{}.particleToMesh(mkn::kul::deref(particle), rhoP, flux,
                                                            layout, coef);
                        }); //
                )
            }
        }
        else
            throw std::runtime_error("fail");
    }

#if DAMN_HAVE_MKN_GPU_HW
    // the stencil's per direction weights, for a kernel laying out its own sums
    struct PrimalStencil : Interpolator_t
    {
        using Interpolator_t::primal_weights_;
    };

    // warp_cell_deposit's particles a shell cell, on average: below it (sparse arrays: patch
    //  ghosts, few particles per cell) the warps mostly idle, a thread per particle is faster
    //  (crossover ~25 a cell, 3D, RTX 5080). PLAMN_GPU_WARP_CELL_MIN_PPC overrides it (tests: 0)
    static std::size_t warp_cell_min_ppc()
    {
        static auto const min
            = mkn::kul::env::get_as("PLAMN_GPU_WARP_CELL_MIN_PPC", std::size_t{16});
        return min;
    }

    // sorted SoAVXChunked, interp 1: a warp per cell of the chunks' shell, whose particles all
    //  deposit on the cell's 2^dim nodes. Each lane sums its share of them in registers (4 fields
    //  x 2^dim values), the warp reduce-scatters those (lane L ends with value L's total): one
    //  global atomic per field and node a cell, not a particle. Why and how:
    //  doc/core/numerics/interpolator/interpolating.md
    template<typename Particles>
    void warp_cell_deposit(Particles const& particles, auto const& layout, auto& rhoP, auto& flux,
                           double const coef, auto&... stream)
        requires(interpOrder == 1)
    {
        using Layout_t                          = std::decay_t<decltype(layout)>;
        std::uint32_t constexpr static n_nodes  = 1u << dim;
        std::uint32_t constexpr static n_values = 4 * n_nodes; // <= 32: one value a lane
        static_assert(n_values <= 32);

        SoAVXConstSpanParticles<dim, Particles::alloc_mode> const ps{particles};
        auto const* const starts = particles.starts_data();
        auto const* const sizes  = particles.sizes_data();
        auto const shell         = particles.shell();
        auto rho_v               = untiled_view(rhoP);
        auto flux_v              = untiled_view(flux);
        auto const origin
            = damn::core::grow(layout.cells, Layout_t::options.field_ghost_width).lower;
        auto const fshape         = rho_v.shape();
        std::size_t const n_cells = shell.size();

        damn::kernel::launch(stream..., n_cells * 32, [=] __device__() {
            auto const i = damn::kernel::idx();
            auto const c = i / 32;
            if (c >= n_cells or sizes[c] == 0) // the whole warp: n_cells * 32 threads
                return;
            auto const lane = static_cast<std::uint32_t>(i % 32);

            damn::core::Box<int, dim> nodes; // the cell's: [cell, cell + 1]
            {
                auto r = c;
                for (std::size_t d = dim; d-- > 0;)
                {
                    auto const extent
                        = static_cast<std::size_t>(shell.upper[d] - shell.lower[d] + 1);
                    nodes.lower[d] = shell.lower[d] + static_cast<int>(r % extent);
                    nodes.upper[d] = nodes.lower[d] + 1;
                    r /= extent;
                }
            }

            // value v: node v / 4 (bit d - 1 - k its offset in direction k), field v % 4; compile
            //  time indices only (a run time index would put v in local memory)
            double v[32] = {};
            for (std::uint32_t j = lane; j < sizes[c]; j += 32)
            {
                auto const p  = ps.copy(starts[c] + j);
                auto const pw = p.weight();
                if (pw == 0) // a gap
                    continue;
                bool own = true; // else moved since the sort (an in place push, no sync yet)
                for (std::size_t d = 0; d < dim; ++d)
                    own = own and p.iCell()[d] == nodes.lower[d];
                if (!own)
                { // on its own, as the thread per particle deposit
                    auto rho = rho_v;
                    auto F   = flux_v;
                    Interpolator_t{}.particleToMesh(p, rho, F, layout, coef);
                    continue;
                }
                PrimalStencil st;
                st.p2m_setup(p, nodes); // starts 0: the cell's nodes
                auto const& pv = p.v();
                mkn::kul::for_N<n_nodes>([&](auto in) {
                    std::uint32_t constexpr n = in();
                    double s                  = pw;
                    mkn::kul::for_N<dim>([&](auto id) {
                        s *= st.primal_weights_[id()][(n >> (dim - 1 - id())) & 1];
                    });
                    v[n * 4 + 0] += s;
                    v[n * 4 + 1] += s * pv[0];
                    v[n * 4 + 2] += s * pv[1];
                    v[n * 4 + 3] += s * pv[2];
                });
            }

            // halving: after the step of h a lane holds h values, those whose index has the lane's
            //  bit h, in the end value lane
            mkn::kul::for_N<5>([&](auto is) {
                std::uint32_t constexpr h = 32u >> (is() + 1);
                bool const upper          = lane & h;
#pragma unroll
                for (std::uint32_t k = 0; k < h; ++k)
                {
                    double const send = upper ? v[k] : v[k + h];
                    double const keep = upper ? v[k + h] : v[k];
                    v[k]              = keep + __shfl_xor_sync(0xffffffff, send, h);
                }
            });
            if (lane >= n_values or v[0] == 0)
                return;
            auto const n   = lane / 4;
            std::size_t at = 0;
            for (std::size_t d = 0; d < dim; ++d)
                at = at * fshape[d]
                     + static_cast<std::size_t>(
                         nodes.lower[d] + static_cast<int>((n >> (dim - 1 - d)) & 1) - origin[d]);
            double* const dst[4]
                = {rho_v.data(), flux_v[0].data(), flux_v[1].data(), flux_v[2].data()};
            atomicAdd(dst[lane % 4] + at, v[0] * coef);
        });
    }
#endif

    // AoSPCTS GPU deposit, launched with gpu::TileCellLauncher: block = (tile, cell), the block's
    // threads stride the cell's particles. Cells share field nodes, so the interpolator must be
    // atomic. Field_t as on_tiles (tiled field value_type)
    template<typename Field_t, typename Launcher>
    static void on_tile_cells(auto& particles, auto& Flux, auto& Rhop, std::size_t const threads,
                              double coef = 1.) DAMN_DEV
    {
#if DAMN_HAVE_MKN_GPU_HW
        static_assert(atomic_ops, "GPU must be atomic");
        using TensorField_t = basic::TensorField<typename Field_t::value_type, 1>;

        auto const tidx   = Launcher::tile_idx();
        auto& cps         = particles()[tidx]();
        auto const& lobox = cps.local_box();
        if (Launcher::cell_idx() >= lobox.size())
            return;

        auto& parts        = cps(damn::core::point_at(lobox, Launcher::cell_idx()));
        auto& rhop_tile    = Rhop[tidx];
        auto flux          = Flux.template as<TensorField_t>([&](auto& c) { return c[tidx](); });
        auto const& layout = rhop_tile.layout();
        auto& pdensity     = rhop_tile();
        for (std::size_t pid = Launcher::thread_idx(); pid < parts.size(); pid += threads)
            Interpolator_t{}.particleToMesh(parts[pid], pdensity, flux, layout, coef);
#endif // DAMN_HAVE_MKN_GPU_HW
    }

    template<bool in_box = false, typename Particles>
    static void box_kernel(Particles& particles, auto& layout, auto& flux, auto& density,
                           double coef = 1.) DAMN_ALL
    {
#if DAMN_HAVE_MKN_GPU_HW
        using lobox_t         = Particles::lobox_t;
        using Launcher        = damn::core::gpu::BoxCellNLauncher<lobox_t>;
        auto const& lobox     = particles.local_box();
        auto const& blockidx  = Launcher::block_idx();
        auto const& threadIdx = Launcher::thread_idx();
        auto& parts           = particles(damn::core::point_at(lobox, blockidx));

        auto const interp = [&]() {
            Interpolator_t{}.particleToMesh(parts[threadIdx], density, flux, layout, coef);
        };

        if (threadIdx < parts.size())
        {
            if constexpr (!in_box)
            {
                interp();
            }
            else
            {
                if (isIn(parts[threadIdx], particles.box()))
                    interp();
            }
        }
#endif // DAMN_HAVE_MKN_GPU_HW
    }

#if 0 // old unused
    static void chunk_kernel_ts(auto& particles, auto& layout, auto& flux,
                                auto& density, double coef = 1.) DAMN_ALL
    {
#if DAMN_HAVE_MKN_GPU_HW

        damn::core::Point<std::uint32_t, 3> const rcell{threadIdx.z, threadIdx.y, threadIdx.x};
        extern __shared__ double data[];

        auto& tile        = *particles().at(rcell + 2);
        auto& parts       = tile();
        auto const each   = parts.size() / 8;
        auto const tile_p = rcell % 2;
        auto const pidx   = tile_p[2] + tile_p[1] * 2 + tile_p[0] * 2 * 2;
        auto const ji     = [&](auto const i) { return i * 8 + pidx; };
        auto doX          = [&]<std::uint8_t IDX = 0>(auto& feeld) {
            auto v = damn::core::make_array_view(&data[0], feeld.shape());
            if (damn::kernel::idx() == 0)
                for (auto& e : v)
                    e = 0;
            Interpolator_t interp;
            std::size_t i = 0, j = ji(i);
            __syncthreads();
            for (; i < each; ++i, j = ji(i))
                interp.p2m_setup(parts[j], layout),
                    interp.template p2m_per_component<IDX>(parts[j], v);
            if (pidx < parts.size() - (8 * each))
                interp.p2m_setup(parts[j], layout),
                    interp.template p2m_per_component<IDX>(parts[j], v);
            __syncthreads();
            if (damn::kernel::idx() == 0)
                for (i = 0; i < v.size(); ++i)
                    feeld.data()[i] = v.data()[i];
        };

        doX(density);
        doX.template operator()<1>(flux[0]);
        doX.template operator()<2>(flux[1]);
        doX.template operator()<3>(flux[2]);


#endif // DAMN_HAVE_MKN_GPU_HW
    }
#endif

    // accumulates into one tile's rhoP + flux (global memory) as deposit_v says (TileDeposit).
    //  each_particle(dep) must call dep(particle) for every particle this thread deposits; every
    //  thread of the block calls it, and it may __syncthreads. With shared_one it runs four times
    template<auto deposit_v>
    static void tile_deposit(auto pdensity, auto flux, auto const& tbox, double const coef,
                             auto&& each_particle) DAMN_DEV
    {
#if DAMN_HAVE_MKN_GPU_HW
        static_assert(atomic_ops, "GPU must be atomic");
        using enum TileDeposit;

        extern __shared__ double data[];
        auto const ziz  = static_cast<int>(mkn::kul::math::product(tbox.shape()));
        auto const tidx = static_cast<int>(threadIdx.x);
        auto const bdim = static_cast<int>(blockDim.x);

        // pdensity/flux are copies of the tile's views (same data), rho/fs are restaged
        auto rho = pdensity;
        auto fs  = flux;

        // staged sums into the tile's fields: several blocks per tile (gridDim.y) add atomically
        auto const flush = [multi = gridDim.y > 1](double& dst, double const v) {
            if (multi)
                atomicAdd(&dst, v);
            else
                dst += v;
        };

        if constexpr (deposit_v == global)
        {
            each_particle([&](auto const& p) {
                Interpolator_t interp;
                interp.particleToMesh(p, rho, fs, tbox, coef);
            });
        }
        else if constexpr (deposit_v == shared_all)
        {
            auto s_rho = damn::core::make_array_view(&data[ziz * 0], pdensity.shape());
            auto s_fx  = damn::core::make_array_view(&data[ziz * 1], flux[0].shape());
            auto s_fy  = damn::core::make_array_view(&data[ziz * 2], flux[1].shape());
            auto s_fz  = damn::core::make_array_view(&data[ziz * 3], flux[2].shape());

            for (int i = tidx; i < ziz * 4; i += bdim)
                data[i] = 0;
            __syncthreads();

            rho.reset(s_rho);
            fs[0].reset(s_fx);
            fs[1].reset(s_fy);
            fs[2].reset(s_fz);
            each_particle([&](auto const& p) {
                Interpolator_t interp;
                interp.particleToMesh(p, rho, fs, tbox, coef);
            });
            __syncthreads();

            for (int i = tidx; i < ziz; i += bdim)
            {
                flush(pdensity.data()[i], s_rho.data()[i]);
                flush(flux[0].data()[i], s_fx.data()[i]);
                flush(flux[1].data()[i], s_fy.data()[i]);
                flush(flux[2].data()[i], s_fz.data()[i]);
            }
        }
        else if constexpr (deposit_v == shared_one)
        {
            mkn::kul::for_N<4>([&](auto ic) {
                auto constexpr c = ic();
                auto& dst        = [&]() -> auto& {
                    if constexpr (c == 0)
                        return pdensity;
                    else
                        return flux[c - 1];
                }();
                auto staged = dst;
                auto s_f    = damn::core::make_array_view(&data[0], dst.shape());

                for (int i = tidx; i < ziz; i += bdim)
                    data[i] = 0;
                __syncthreads();

                staged.reset(s_f);
                each_particle([&](auto const& p) {
                    Interpolator_t interp;
                    interp.p2m_setup(p, tbox);
                    interp.template p2m_per_component<c>(p, staged, coef);
                });
                __syncthreads();

                for (int i = tidx; i < ziz; i += bdim)
                    flush(dst.data()[i], s_f.data()[i]);
                __syncthreads(); // shared memory is reused by the next field
            });
        }
#endif
    }


    template<typename Field_t, auto deposit_v = TileDeposit::shared_all, std::size_t warps = 1>
    static void on_tiles(auto& particles, auto& Flux, auto& Rhop, double coef = 1.) DAMN_DEV
    {
#if DAMN_HAVE_MKN_GPU_HW
        using TensorField_t = basic::TensorField<typename Field_t::value_type, 1>;

        auto& rhop_tile = Rhop[blockIdx.x];
        auto const ft   = Flux.template as<TensorField_t>([&](auto& c) { return c[blockIdx.x](); });
        auto& parts     = particles()[blockIdx.x]();

        // a contiguous share of the tile's particles per group of blockDim.x / warps threads (a
        //  warp each, launched as warp_size() * warps, see DepositSpec): sorted particles keep each
        //  group on its own cells, so groups don't contend for the same staged nodes. From the
        //  launch, not warpSize: correct whatever width the host launched for
        static_assert(warps > 0);
        auto const width = blockDim.x / warps;
        auto const group = threadIdx.x / width;
        auto const lane  = threadIdx.x % width;
        auto const n     = parts.size();
        auto const begin = n * group / warps;
        auto const end   = n * (group + 1) / warps;
        tile_deposit<deposit_v>(rhop_tile(), ft, rhop_tile.ghost_box(), coef, [&](auto&& dep) {
            for (std::size_t i = begin + lane; i < end; i += width)
                dep(parts[i]);
        });
#endif
    }

    // dynamic shared memory on_tiles_push_deposit needs after its staged fields, for block_size
    //  threads per block
    template<typename Particle_t>
    static std::size_t push_deposit_extra_bytes(std::size_t const block_size)
    {
        return 2 * block_size * sizeof(Particle_t) + sizeof(int);
    }

    // Boris push + deposit in one kernel on register copies (originals untouched), the deposited
    //  ones batched through a block buffer in shared memory: doc/core/numerics/interpolator/
    //  interpolating.md
    template<typename Field_t, auto particle_type_v, auto alloc_mode_v,
             auto deposit_v = TileDeposit::shared_all>
    static void on_tiles_push_deposit(auto& particles, auto const& electromag, auto& Flux,
                                      auto& Rhop, auto const& filter_box, auto const& exclude_boxes,
                                      double const coef1, auto const& halfdt,
                                      double const coef = 1.) DAMN_DEV
    {
#if DAMN_HAVE_MKN_GPU_HW
        using TensorField_t = basic::TensorField<typename Field_t::value_type, 1>;
        using Scalar_vt     = typename Field_t::value_type;
        using VecField_vt   = basic::TensorField<Scalar_vt, 1>;
        using Electromag_vt = basic::Electromag<VecField_vt>;
        using Particle_t    = ParticleDefaults<dim>::Particle_t;

        extern __shared__ double data[];

        auto& rhop_tile = Rhop[blockIdx.x];
        auto const ft   = Flux.template as<TensorField_t>([&](auto& c) { return c[blockIdx.x](); });
        auto const tbox = rhop_tile.ghost_box();
        auto const ziz  = mkn::kul::math::product(tbox.shape());
        auto const tidx = threadIdx.x;
        auto const ws   = blockDim.x;

        // block buffer and atomic counter after the staged fields
        auto* const part_buf = reinterpret_cast<Particle_t*>(reinterpret_cast<char*>(&data[0])
                                                             + tile_deposit_fields(deposit_v) * ziz
                                                                   * sizeof(double));
        auto& buf_count      = *reinterpret_cast<int*>(reinterpret_cast<char*>(part_buf)
                                                       + 2 * ws * sizeof(Particle_t));

        auto const em      = electromag.template as<Electromag_vt>([=] DAMN_ALL(auto const& vf) {
            return mkn::kul::for_N_make_array<3>(
                [&vf] DAMN_ALL(auto i) { return vf[i][blockIdx.x](); });
        });
        auto const& layout = electromag.E[0][blockIdx.x].layout();

        auto& parts        = particles()[blockIdx.x]();
        auto const n       = parts.size();
        auto const stride  = static_cast<std::size_t>(ws) * gridDim.y; // a batch over all blocks
        auto const nbatch  = (n + stride - 1) / stride;
        auto const b_start = static_cast<std::size_t>(blockIdx.y) * ws;

        auto const push_and_deposit = [&](auto&& dep) {
            if (tidx == 0)
                buf_count = 0;
            __syncthreads();

            for (std::size_t batch = 0; batch < nbatch; ++batch)
            {
                std::size_t const pidx = batch * stride + b_start + tidx;
                if (pidx < n)
                {
                    Particle_t p = parts[pidx]; // register copy — original untouched

                    p.iCell() = boris::advance<alloc_mode_v>(p, halfdt);
                    {
                        Interpolator_t interp;
                        boris::accelerate(p, interp.m2p(p, em, layout), coef1);
                    }
                    p.iCell() = boris::advance<alloc_mode_v>(p, halfdt);

                    if (isIn(p.iCell(), filter_box) and !isIn(p.iCell(), exclude_boxes))
                    {
                        int const slot = atomicAdd(&buf_count, 1);
                        part_buf[slot] = p;
                    }
                }

                __syncthreads();

                if (buf_count >= static_cast<int>(ws))
                {
                    dep(part_buf[tidx]);

                    // every thread reads buf_count before thread 0 rewrites it
                    int const remaining = buf_count - static_cast<int>(ws);
                    __syncthreads();

                    if (static_cast<int>(tidx) < remaining)
                        part_buf[tidx] = part_buf[tidx + ws];
                    __syncthreads();

                    if (tidx == 0)
                        buf_count = remaining;
                    __syncthreads();
                }
            }

            if (static_cast<int>(tidx) < buf_count)
                dep(part_buf[tidx]);
        };

        tile_deposit<deposit_v>(rhop_tile(), ft, tbox, coef, push_and_deposit);
#endif
    }


    template<typename Field>
    static void on_cpu_tiles(auto& particles, auto& /*layout*/, double coef = 1.) DAMN_ALL
    {
        using TensorField_t = basic::TensorField<typename Field::Super, 1>;

        for (auto& tile : particles())
        {
            auto& parts                                = tile();
            auto const tbox                            = tile.field_box();
            auto const& [density, flux0, flux1, flux2] = tile.fields();
            auto const r0                              = density;

            TensorField_t flux{flux0, flux1, flux2};
            Interpolator_t interp;
            std::size_t pid = 0;

            interp.particleToMesh(parts[pid], density, flux, /*layout,*/ tbox, coef);
        }
    }




    template<typename Layout>
    static void chunk_kernel(auto& particles, Layout& layout, auto& flux, auto& density,
                             double /*coef*/ = 1.) DAMN_DEV
    {
#if DAMN_HAVE_MKN_GPU_HW
        auto constexpr static nghosts = Layout::options.field_ghost_width;
        extern __shared__ double data[];
        auto const& lobox = particles.local_box();
        auto const ziz    = 9 * 9 * 9;

        auto const t_x = threadIdx.x;
        auto const t_y = threadIdx.y;
        auto const t_z = threadIdx.z;

        damn::core::Point<std::uint32_t, 3> const tcell{t_x, t_y, t_z};
        damn::core::Point<std::uint32_t, 3> const locell = tcell + nghosts;

        {
            Interpolator_t interp;
            auto& parts = particles(locell);

            { // rho
                auto const r0 = density.data();
                auto v        = damn::core::make_array_view(&data[0], density.shape());
                density.setData(v.data());
                if (damn::kernel::idx() == 0)
                    for (auto& e : v)
                        e = 0;
                __syncthreads();
                for (std::size_t i = 0; i < parts.size(); ++i)
                {
                    p2m_setup(parts[i], layout);
                    p2m_per_component(parts[i], v);
                }
                __syncthreads();
                density.setData(r0);
                if (damn::kernel::idx() == 0)
                    for (std::size_t i = 0; i < ziz; ++i)
                        density.data()[i] = v.data()[i];
            }

            {
                auto const f0 = flux[0].data();
                auto v        = damn::core::make_array_view(&data[0], flux[0].shape());
                flux[0].setData(v.data());
                if (damn::kernel::idx() == 0)
                    for (auto& e : v)
                        e = 0;
                __syncthreads();
                for (std::size_t i = 0; i < parts.size(); ++i)
                {
                    p2m_setup(parts[i], layout);
                    p2m_per_component(parts[i], flux[0]);
                }
                __syncthreads();
                flux[0].setData(f0);
                if (damn::kernel::idx() == 0)
                    for (std::size_t i = 0; i < ziz; ++i)
                        flux[0].data()[i] = v.data()[i];
            }

            {
                auto const f1 = flux[1].data();
                auto v        = damn::core::make_array_view(&data[0], flux[1].shape());
                flux[1].setData(v.data());
                if (damn::kernel::idx() == 0)
                    for (auto& e : v)
                        e = 0;
                __syncthreads();
                for (std::size_t i = 0; i < parts.size(); ++i)
                {
                    p2m_setup(parts[i], layout);
                    p2m_per_component(parts[i], flux[1]);
                }
                __syncthreads();
                flux[1].setData(f1);
                if (damn::kernel::idx() == 0)
                    for (std::size_t i = 0; i < ziz; ++i)
                        flux[1].data()[i] = v.data()[i];
            }

            {
                auto const f2 = flux[2].data();
                auto v        = damn::core::make_array_view(&data[0], flux[2].shape());
                flux[2].setData(v.data());
                if (damn::kernel::idx() == 0)
                    for (auto& e : v)
                        e = 0;
                __syncthreads();
                for (std::size_t i = 0; i < parts.size(); ++i)
                {
                    p2m_setup(parts[i], layout);
                    p2m_per_component(parts[i], flux[2]);
                }
                __syncthreads();
                flux[2].setData(f2);
                if (damn::kernel::idx() == 0)
                    for (std::size_t i = 0; i < ziz; ++i)
                        flux[2].data()[i] = v.data()[i];
            }
        }


#endif // DAMN_HAVE_MKN_GPU_HW
    }




    Interpolator_t interp_;
};

template<std::size_t dim, std::size_t interpOrder, bool atomic_ops = false>
struct MomentumTensorInterpolating
{
    using Interpolator_t = MomentumTensorInterpolator<dim, interpOrder, atomic_ops>;

    Interpolator_t interp_;

public:
    inline void operator()(auto& particles, auto& momentumTensor, auto const& layout,
                           double mass = 1.)
    {
        assert(momentumTensor.isUsable());
        particleToMesh(particles, momentumTensor, layout, mass);
    }

    template<auto type = ParticleType::Domain, typename Particles_t>
    void particleToMesh(Particles_t& particles, auto& momentumTensor, auto const& layout,
                        double mass = 1.)
        requires(Particles_t::layout_mode == LayoutMode::AoSMapped)
    {
        interp_(particles, momentumTensor, layout, mass);
    }

    // SoAVX columns have no particle reference to iterate: deposited from an AoS buffer
    //  loaded in batches, on the host (diagnostics only)
    template<auto type = ParticleType::Domain, typename Particles_t>
    void particleToMesh(Particles_t& particles, auto& momentumTensor, auto const& layout,
                        double mass = 1.)
        requires(is_soavx(Particles_t::layout_mode))
    {
        static constexpr std::size_t batch = 1024;
        using Buffer_t = Particles<ParticlesOptions{Particles_t::dimension, LayoutMode::AoS}>;

        Buffer_t buffer;
        buffer.reserve(std::min(batch, particles.size()));
        for (std::size_t i = 0; i < particles.size(); ++i)
        {
            buffer.emplace_back(particles.copy(i));
            if (buffer.size() == batch or i + 1 == particles.size())
            {
                interp_(buffer, momentumTensor, layout, mass);
                buffer.clear();
            }
        }
    }

    // unlike Interpolating::particleToMesh, level ghosts are deposited after the tile reduction
    // (fillPopMomTensor), so each tile gathers every level ghost its halo reaches
    template<auto type = ParticleType::Domain, typename Particles_t>
    void particleToMesh(Particles_t& particles, auto& momentumTensor, auto const& /*layout*/,
                        double mass = 1.)
        requires(Particles_t::layout_mode == LayoutMode::AoSPCTS)
    {
        static_assert(mkn::kul::any_in(type, ParticleType::Domain, ParticleType::LevelGhost));

        for (std::size_t tidx = 0; tidx < particles().size(); ++tidx)
        {
            auto mt            = tile_at(momentumTensor, tidx);
            auto const& mt_lay = tile_layout(momentumTensor, tidx);
            auto const deposit
                = [&](auto const& cell_particles) { interp_(cell_particles, mt, mt_lay, mass); };

            if constexpr (type == ParticleType::LevelGhost)
                on_reachable_level_ghosts(particles, tidx, deposit);
            else
            {
                auto& cps = particles()[tidx]();
                damn::core::for_each_point(cps.local_box(),
                                           [&](auto const& bix) { deposit(cps(bix)); });
            }
        }
    }

    template<auto type = ParticleType::Domain, typename Particles_t>
    void particleToMesh(Particles_t& particles, auto& momentumTensor, auto const& /*layout*/,
                        double mass = 1.)
        requires(Particles_t::layout_mode == LayoutMode::AoSCMTS)
    {
        static_assert(mkn::kul::any_in(type, ParticleType::Domain, ParticleType::LevelGhost));

        for (std::size_t tidx = 0; tidx < particles().size(); ++tidx)
        {
            auto mt            = tile_at(momentumTensor, tidx);
            auto const& mt_lay = tile_layout(momentumTensor, tidx);
            auto const deposit
                = [&](auto const& tile_particles) { interp_(tile_particles, mt, mt_lay, mass); };

            if constexpr (type == ParticleType::LevelGhost)
                on_reachable_level_ghosts(particles, tidx, deposit);
            else
                deposit(particles()[tidx]());
        }
    }

    template<auto type = ParticleType::Domain, typename Particles_t>
    void particleToMesh(Particles_t& particles, auto& momentumTensor, auto const& /*layout*/,
                        double mass = 1.)
        requires(Particles_t::layout_mode == LayoutMode::AoSTS)
    {
        for (std::size_t tidx = 0; tidx < particles().size(); ++tidx)
        {
            auto mt = tile_at(momentumTensor, tidx); // interp_ takes it by reference
            interp_(particles()[tidx](), mt, tile_layout(momentumTensor, tidx), mass);
        }
    }
};

// the deposit of particles of alloc_mode: on the device (atomics) for unified memory, else on
//  the host (no atomics)
template<std::size_t dim, std::size_t interpOrder, auto alloc_mode>
using DepositInterpolating
    = Interpolating<dim, interpOrder, alloc_mode == damn::AllocatorMode::GPU_UNIFIED>;

} // namespace plamn::core

#endif /*PLAMN_CORE_NUMERICS_INTERPOLATOR_INTERPOLATING_HPP*/
