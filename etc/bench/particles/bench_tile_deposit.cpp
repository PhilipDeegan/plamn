// the device deposit of one SoAVXChunked GPU_UNIFIED patch's domain particles: untiled, a thread
//  per particle into the patch fields with global atomics (the simulator's), against kernels over
//  tiles of the same, cell sorted, particles: a block per (tile, split of the tile's cells),
//  threads over each cell's chunk, into
//    smem_all  the tile's deposit footprint, all four fields staged in dynamic shared memory
//    smem_one  one field staged at a time, four passes over the particles
//    global    nothing staged: straight into the patch fields, in tile order
//  the staged sums then added into the patch fields atomically (neighbouring footprints overlap).
//  warp_win: a warp per window of 1 or 2 cells a side, in registers: see warp_window and
//  doc/etc/bench/particles/bench_tile_deposit.md.
//  window_rs: warp_win(1)'s per cell sums added into a window of cells' nodes in shared memory
//  (block atomics, a block per window, a warp per cell at a time), each window node then added once
//  into the patch fields. Every variant is checked against the untiled deposit (max relative
//  difference).
//
//   mkn build -p bench_tile_deposit -dKO -g 0 -x res/mkn/clang_cuda [-a -DPLAMN_BENCH_INTERP=2]
//   mkn run -p bench_tile_deposit -x res/mkn/clang_cuda
//
// env: PLAMN_CELLS (12 a side), PLAMN_PPC (100), PLAMN_SEED, PLAMN_REPEATS (5: the fastest
//  printed), PLAMN_TILES (4,6,8,12: tile sizes), PLAMN_SPLITS (1,2,4,8: blocks per tile),
//  PLAMN_BLOCK (256: threads per block, clamped to each kernel's limit)

#include "damn/core/def/damn_config.hpp"
#include "damn/core/exec/kernels.hpp"
#include "plamn/core/data/particles/particles.hpp"
#include "plamn/core/numerics/interpolator/interpolator.hpp"
#include "plamn/simulator/simulator_def.hpp"

#include "etc/bench/particles/bench_particles.hpp"
#include "tst/plamn/core/data/grid/test_layout_fixtures.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef PLAMN_BENCH_INTERP
#define PLAMN_BENCH_INTERP 1
#endif

namespace plamn::core::bench
{
auto static const cells   = mkn::kul::env::get_as("PLAMN_CELLS", std::uint32_t{12});
auto static const ppc     = mkn::kul::env::get_as("PLAMN_PPC", std::size_t{100});
auto static const seed    = mkn::kul::env::get_as("PLAMN_SEED", std::size_t{1067});
auto static const repeats = mkn::kul::env::get_as("PLAMN_REPEATS", std::size_t{5});
auto static const block   = mkn::kul::env::get_as("PLAMN_BLOCK", std::uint32_t{256});

std::vector<std::uint32_t> list_of(char const* const name, std::string const& fallback)
{
    std::vector<std::uint32_t> out;
    std::stringstream ss{mkn::kul::env::get_as(name, fallback)};
    for (std::string item; std::getline(ss, item, ',');)
        out.push_back(static_cast<std::uint32_t>(std::stoul(item)));
    return out;
}

std::uint32_t constexpr ipow(std::uint32_t const base, std::size_t const exp) DAMN_ALL
{
    std::uint32_t r = 1;
    for (std::size_t i = 0; i < exp; ++i)
        r *= base;
    return r;
}

enum class Staging : std::uint8_t { all, one, global };
char const* name_of(Staging const s)
{
    return s == Staging::all ? "smem_all" : s == Staging::one ? "smem_one" : "global";
}

template<std::size_t dim, std::size_t interp>
struct TileDeposit
{
    auto constexpr static alloc    = damn::AllocatorMode::GPU_UNIFIED;
    auto constexpr static ref_opts = SimOpts{.dimension = dim, .interp_order = interp};
    auto constexpr static opts     = SimOpts{.dimension    = dim,
                                             .interp_order = interp,
                                             .layout_mode  = LayoutMode::SoAVXChunked,
                                             .alloc_mode   = alloc};
    using RefTypes                 = PLAMN_Types<ref_opts>::Hybrid;
    using Layout_t                 = PLAMN_Types<opts>::Hybrid::Layout_t;
    using Interpolator_t           = Interpolator<dim, interp, /*atomic=*/true>;
    using Box_t                    = damn::core::Box<int, dim>;
    using Span_t                   = SoAVXConstSpanParticles<dim, alloc>;

    // nodes a particle of a tile's cells deposits on, beyond its cells: interp on either side,
    //  one more above (primal)
    static Box_t footprint(Box_t const& tile) DAMN_ALL
    {
        Box_t fb = tile;
        for (std::size_t d = 0; d < dim; ++d)
            fb.lower[d] -= interp, fb.upper[d] += interp + 1;
        return fb;
    }

    typename RefTypes::Layout_t const ref_layout = *TestLayout<typename RefTypes::Layout_t>{cells};
    Layout_t const layout                        = *TestLayout<Layout_t>{cells};
    UsableIons_t<opts> ions
        = from_ions<opts>(layout, make_ions<ref_opts>(ref_layout, ppc, seed), ppc);
    std::size_t const n_particles = layout.cells.size() * ppc;

    auto& pop() { return ions.populations[0]; }

    void zero()
    {
        auto rho  = untiled_view(pop().particleDensity());
        auto flux = untiled_view(pop().flux());
        damn::kernel::launch(rho.size(), [=] DAMN_ALL() mutable {
            auto const i      = damn::kernel::idx();
            rho.data()[i]     = 0;
            flux[0].data()[i] = 0;
            flux[1].data()[i] = 0;
            flux[2].data()[i] = 0;
        });
        mkn::gpu::sync();
    }

    std::vector<double> values()
    {
        mkn::gpu::sync();
        auto& rho  = pop().particleDensity();
        auto& flux = pop().flux();
        std::vector<double> out(rho.data(), rho.data() + rho.size());
        for (std::size_t c = 0; c < 3; ++c)
            out.insert(out.end(), flux[c].data(), flux[c].data() + flux[c].size());
        return out;
    }

    void untiled()
    {
        Span_t const ps{pop().domainParticles()};
        auto rho_v     = untiled_view(pop().particleDensity());
        auto flux_v    = untiled_view(pop().flux());
        auto const lay = layout;
        damn::kernel::launch(ps.size(), [=] DAMN_ALL() {
            auto rho = rho_v;
            auto F   = flux_v;
            Interpolator_t{}.particleToMesh(ps.copy(damn::kernel::idx()), rho, F, lay, 1.);
        });
        mkn::gpu::sync();
    }

    template<Staging staging>
    void tiled(std::uint32_t const tile, std::uint32_t const splits)
    {
        auto& particles = pop().domainParticles();
        Span_t const ps{particles};
        auto const* const starts = particles.starts_data();
        auto const* const sizes  = particles.sizes_data();
        auto const shell         = particles.shell();
        auto rho_v               = untiled_view(pop().particleDensity());
        auto flux_v              = untiled_view(pop().flux());
        auto const cells_box     = layout.cells;
        // the patch fields' AMR origin: what the layout's deposit indexes from
        auto const patch_box = damn::core::grow(cells_box, Layout_t::options.field_ghost_width);
        auto const origin    = patch_box.lower;
        auto const fshape    = rho_v.shape();

        std::array<int, dim> n_tiles{};
        std::uint32_t n_blocks = 1;
        for (std::size_t d = 0; d < dim; ++d)
        {
            auto const n = (cells_box.upper[d] - cells_box.lower[d] + 1 + tile - 1) / tile;
            n_tiles[d]   = static_cast<int>(n);
            n_blocks *= n;
        }
        std::size_t fp_nodes = 1;
        for (std::size_t d = 0; d < dim; ++d)
            fp_nodes *= tile + 2 * interp + 2;
        std::size_t const n_fields = staging == Staging::all ? 4 : staging == Staging::one ? 1 : 0;

        damn::core::gpu::ChunkLauncher<false> launcher{1, 0};
        launcher.g.x = n_blocks;
        launcher.g.y = splits;
        launcher.ds  = n_fields * fp_nodes * sizeof(double);
        auto kernel  = [=] __device__() mutable {
#if DAMN_HAVE_MKN_GPU_HW
            extern __shared__ double data[];
            // the block's tile
            Box_t tbox;
            {
                std::uint32_t t = blockIdx.x;
                for (std::size_t d = dim; d-- > 0;)
                {
                    auto const c = static_cast<int>(t % n_tiles[d]);
                    t /= n_tiles[d];
                    tbox.lower[d] = cells_box.lower[d] + c * static_cast<int>(tile);
                    tbox.upper[d]
                        = std::min(tbox.lower[d] + static_cast<int>(tile) - 1, cells_box.upper[d]);
                }
            }
            auto const fb      = footprint(tbox);
            auto const fbshape = fb.shape();
            std::uint32_t n_fb = 1, n_cells = 1;
            std::array<std::uint32_t, dim> shape{}, tshape{};
            for (std::size_t d = 0; d < dim; ++d)
            {
                shape[d]  = static_cast<std::uint32_t>(fbshape[d]);
                tshape[d] = static_cast<std::uint32_t>(tbox.upper[d] - tbox.lower[d] + 1);
                n_fb *= shape[d];
                n_cells *= tshape[d];
            }

            // calls dep(particle) on this thread's share of the block's cells' particles
            auto const each = [&](auto&& dep) {
                for (std::uint32_t k = blockIdx.y; k < n_cells; k += gridDim.y)
                {
                    damn::core::Point<int, dim> cell;
                    std::uint32_t r = k;
                    for (std::size_t d = dim; d-- > 0;)
                        cell[d] = tbox.lower[d] + static_cast<int>(r % tshape[d]), r /= tshape[d];
                    auto const c     = Chunks<dim>::flat(shell, cell);
                    auto const start = starts[c];
                    for (std::uint32_t j = threadIdx.x; j < sizes[c]; j += blockDim.x)
                        if (auto const p = ps.copy(start + j); p.weight() != 0)
                            dep(p);
                }
            };
            // staged node i into the patch field at the same AMR node
            auto const flush = [&](double* const dst, double const* const src) {
                for (std::uint32_t i = threadIdx.x; i < n_fb; i += blockDim.x)
                {
                    if (src[i] == 0)
                        continue;
                    std::uint32_t r = i;
                    std::size_t at = 0, stride = 1;
                    for (std::size_t d = dim; d-- > 0;)
                    {
                        auto const node = fb.lower[d] + static_cast<int>(r % shape[d]);
                        r /= shape[d];
                        at += static_cast<std::size_t>(node - origin[d]) * stride;
                        stride *= fshape[d];
                    }
                    atomicAdd(dst + at, src[i]);
                }
            };

            if constexpr (staging == Staging::global)
            {
                auto rho = rho_v;
                auto F   = flux_v;
                each([&](auto const& p) {
                    Interpolator_t{}.particleToMesh(p, rho, F, patch_box, 1.);
                });
            }
            else if constexpr (staging == Staging::all)
            {
                for (std::uint32_t i = threadIdx.x; i < 4 * n_fb; i += blockDim.x)
                    data[i] = 0;
                __syncthreads();
                auto rho   = rho_v;
                auto F     = flux_v;
                auto s_rho = damn::core::make_array_view(&data[0], shape);
                auto s_fx  = damn::core::make_array_view(&data[1 * n_fb], shape);
                auto s_fy  = damn::core::make_array_view(&data[2 * n_fb], shape);
                auto s_fz  = damn::core::make_array_view(&data[3 * n_fb], shape);
                rho.reset(s_rho);
                F[0].reset(s_fx);
                F[1].reset(s_fy);
                F[2].reset(s_fz);
                each([&](auto const& p) { Interpolator_t{}.particleToMesh(p, rho, F, fb, 1.); });
                __syncthreads();
                flush(rho_v.data(), &data[0]);
                for (std::size_t c = 0; c < 3; ++c)
                    flush(flux_v[c].data(), &data[(c + 1) * n_fb]);
            }
            else
            {
                mkn::kul::for_N<4>([&](auto ic) {
                    auto constexpr c = ic();
                    auto dst         = [&]() {
                        if constexpr (c == 0)
                            return rho_v;
                        else
                            return flux_v[c - 1];
                    }();
                    for (std::uint32_t i = threadIdx.x; i < n_fb; i += blockDim.x)
                        data[i] = 0;
                    __syncthreads();
                    auto staged = dst;
                    auto s_f    = damn::core::make_array_view(&data[0], shape);
                    staged.reset(s_f);
                    each([&](auto const& p) {
                        Interpolator_t interp_;
                        interp_.p2m_setup(p, fb);
                        interp_.template p2m_per_component<c>(p, staged, 1.);
                    });
                    __syncthreads();
                    flush(dst.data(), &data[0]);
                    __syncthreads(); // shared memory reused by the next field
                });
            }
#endif
        };
        launcher.b.x = block_for(kernel);

        launcher(kernel);
        mkn::gpu::sync();
    }

    // the simulator's per direction stencil start (local to the box given p2m_setup) and weights
    struct Stencil : Interpolator_t
    {
        using Interpolator_t::primal_startIndex_;
        using Interpolator_t::primal_weights_;
    };

    // a warp per window of W^dim cells, no shared memory. A cell's particles deposit on a fixed set
    //  of (primal) nodes per direction, the union of their stencils: iCell..iCell+1 at interp 1,
    //  iCell-1..iCell+2 at 2 (a particle's 3 of them by its delta) and 3; so does a window, its
    //  footprint. The warp sums the footprint's 4 fields x nodes values in passes of 32 (each its
    //  own read of the window's particles, from L1), each lane over its share of the particles,
    //  every pass value's field and node known at compile time; the pass reduce-scattered over the
    //  warp (doc/etc/bench/particles/bench_tile_deposit.md), lane L adding value L into the patch
    //  field: one global atomic per window, field and footprint node
    template<std::uint32_t W>
    void warp_window()
    {
        auto& particles = pop().domainParticles();
        Span_t const ps{particles};
        auto const* const starts = particles.starts_data();
        auto const* const sizes  = particles.sizes_data();
        auto const shell         = particles.shell();
        auto rho_v               = untiled_view(pop().particleDensity());
        auto flux_v              = untiled_view(pop().flux());
        auto const cells_box     = layout.cells;
        auto const origin = damn::core::grow(cells_box, Layout_t::options.field_ghost_width).lower;
        auto const fshape = rho_v.shape();

        int constexpr static below          = interp == 1 ? 0 : 1; // the footprint past the cells
        int constexpr static above          = interp == 1 ? 1 : 2;
        int constexpr static side           = W + below + above;
        std::uint32_t constexpr static n_fp = ipow(side, dim);
        std::uint32_t constexpr static n_values = 4 * n_fp;
        std::uint32_t constexpr static passes   = (n_values + 31) / 32;
        std::uint32_t constexpr static support  = interp + 1;

        std::array<int, dim> n_win{};
        std::uint32_t n_windows = 1;
        for (std::size_t d = 0; d < dim; ++d)
        {
            auto const extent
                = static_cast<std::uint32_t>(cells_box.upper[d] - cells_box.lower[d] + 1);
            if (extent % W)
                throw std::runtime_error("warp_window: cells not a multiple of the window");
            n_win[d] = static_cast<int>(extent / W);
            n_windows *= extent / W;
        }

        auto kernel = [=] __device__() mutable {
#if DAMN_HAVE_MKN_GPU_HW
            auto const lane = threadIdx.x % 32;
            auto const win  = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
            if (win >= n_windows) // the whole warp
                return;
            Box_t tbox, fb;
            {
                std::uint32_t t = win;
                for (std::size_t d = dim; d-- > 0;)
                {
                    auto const c = static_cast<int>(t % n_win[d]);
                    t /= n_win[d];
                    tbox.lower[d] = cells_box.lower[d] + c * static_cast<int>(W);
                    tbox.upper[d] = tbox.lower[d] + static_cast<int>(W) - 1;
                    fb.lower[d]   = tbox.lower[d] - below;
                    fb.upper[d]   = tbox.upper[d] + above;
                }
            }
            double* const dst[4]
                = {rho_v.data(), flux_v[0].data(), flux_v[1].data(), flux_v[2].data()};

            mkn::kul::for_N<passes>([&](auto ip) {
                std::uint32_t constexpr pass = ip();
                double v[32]                 = {};
                for (std::uint32_t k = 0; k < ipow(W, dim); ++k)
                {
                    damn::core::Point<int, dim> cell;
                    {
                        auto r = k;
                        for (std::size_t d = dim; d-- > 0;)
                            cell[d] = tbox.lower[d] + static_cast<int>(r % W), r /= W;
                    }
                    auto const c = Chunks<dim>::flat(shell, cell);
                    for (std::uint32_t j = lane; j < sizes[c]; j += 32)
                    {
                        auto const p  = ps.copy(starts[c] + j);
                        auto const pw = p.weight();
                        if (pw == 0)
                            continue;
                        Stencil st;
                        st.p2m_setup(p, fb); // starts local to the footprint
                        auto const& pv = p.v();
                        // per direction, the particle's weights at its offset in the footprint
                        //  (its cell's place in the window + its delta's shift), 0 elsewhere:
                        //  compile time indices only (a run time index into a register array
                        //  would put it in local memory)
                        double u[dim][side];
                        mkn::kul::for_N<dim>([&](auto id) {
                            std::size_t constexpr d = id();
                            auto const off          = static_cast<int>(st.primal_startIndex_[d]);
                            mkn::kul::for_N<side>([&](auto in) {
                                int constexpr n = in();
                                double x        = 0;
                                mkn::kul::for_N<side - support + 1>([&](auto io) {
                                    int constexpr o = io();
                                    if constexpr (n - o >= 0 and n - o < static_cast<int>(support))
                                        x = off == o ? st.primal_weights_[d][n - o] : x;
                                });
                                u[d][n] = x;
                            });
                        });
                        mkn::kul::for_N<32>([&](auto ii) {
                            std::uint32_t constexpr idx = pass * 32 + ii();
                            if constexpr (idx < n_values)
                            {
                                std::uint32_t constexpr f = idx / n_fp;
                                std::uint32_t constexpr n = idx % n_fp;
                                double s                  = pw;
                                mkn::kul::for_N<dim>([&](auto id) {
                                    std::size_t constexpr d = id();
                                    s *= u[d][(n / ipow(side, dim - 1 - d)) % side];
                                });
                                if constexpr (f == 0)
                                    v[ii()] += s;
                                else
                                    v[ii()] += s * pv[f - 1];
                            }
                        });
                    }
                }
                // halving: after the step of h a lane holds h values, those whose index has
                //  the lane's bit h; in the end value lane
                mkn::kul::for_N<5>([&](auto is) {
                    std::uint32_t constexpr h = 32u >> (is() + 1);
                    bool const upper          = lane & h;
#pragma unroll
                    for (std::uint32_t i = 0; i < h; ++i)
                    {
                        double const send = upper ? v[i] : v[i + h];
                        double const keep = upper ? v[i + h] : v[i];
                        v[i]              = keep + __shfl_xor_sync(0xffffffff, send, h);
                    }
                });
                auto const idx = pass * 32 + lane;
                if (idx >= n_values or v[0] == 0)
                    return;
                auto const n   = idx % n_fp;
                std::size_t at = 0;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    auto const node
                        = fb.lower[d] + static_cast<int>(n / ipow(side, dim - 1 - d)) % side;
                    at = at * fshape[d] + static_cast<std::size_t>(node - origin[d]);
                }
                atomicAdd(dst[idx / n_fp] + at, v[0]);
            });
#endif
        };
        damn::core::gpu::ChunkLauncher<false> launcher{1, 0};
        launcher.b.x = block_for(kernel);
        launcher.g.x = (n_windows * 32 + launcher.b.x - 1) / launcher.b.x;
        launcher(kernel);
        mkn::gpu::sync();
    }

    // interp 1: a window's nodes are its cells' [lower, upper + 1]
    void window_rs(std::uint32_t const tile)
        requires(interp == 1)
    {
        auto& particles = pop().domainParticles();
        Span_t const ps{particles};
        auto const* const starts = particles.starts_data();
        auto const* const sizes  = particles.sizes_data();
        auto const shell         = particles.shell();
        auto rho_v               = untiled_view(pop().particleDensity());
        auto flux_v              = untiled_view(pop().flux());
        auto const cells_box     = layout.cells;
        auto const origin = damn::core::grow(cells_box, Layout_t::options.field_ghost_width).lower;
        auto const fshape = rho_v.shape();
        std::uint32_t constexpr static n_nodes  = 1u << dim;
        std::uint32_t constexpr static n_values = 4 * n_nodes;

        std::array<int, dim> n_tiles{};
        std::uint32_t n_blocks = 1, wnodes = 1;
        for (std::size_t d = 0; d < dim; ++d)
        {
            auto const n = (cells_box.upper[d] - cells_box.lower[d] + 1 + tile - 1) / tile;
            n_tiles[d]   = static_cast<int>(n);
            n_blocks *= n;
            wnodes *= tile + 1;
        }

        damn::core::gpu::ChunkLauncher<false> launcher{1, 0};
        launcher.g.x = n_blocks;
        launcher.ds  = 4 * wnodes * sizeof(double);
        auto kernel  = [=] __device__() mutable {
#if DAMN_HAVE_MKN_GPU_HW
            extern __shared__ double data[];
            Box_t tbox;
            {
                std::uint32_t t = blockIdx.x;
                for (std::size_t d = dim; d-- > 0;)
                {
                    auto const c = static_cast<int>(t % n_tiles[d]);
                    t /= n_tiles[d];
                    tbox.lower[d] = cells_box.lower[d] + c * static_cast<int>(tile);
                    tbox.upper[d]
                        = std::min(tbox.lower[d] + static_cast<int>(tile) - 1, cells_box.upper[d]);
                }
            }
            std::array<std::uint32_t, dim> tshape{}, wshape{};
            std::uint32_t n_cells = 1, n_w = 1;
            for (std::size_t d = 0; d < dim; ++d)
            {
                tshape[d] = static_cast<std::uint32_t>(tbox.upper[d] - tbox.lower[d] + 1);
                wshape[d] = tshape[d] + 1;
                n_cells *= tshape[d];
                n_w *= wshape[d];
            }
            for (std::uint32_t i = threadIdx.x; i < 4 * n_w; i += blockDim.x)
                data[i] = 0;
            __syncthreads();

            auto const lane = threadIdx.x % 32;
            for (std::uint32_t k = threadIdx.x / 32; k < n_cells; k += blockDim.x / 32)
            {
                damn::core::Point<int, dim> cell;
                {
                    auto r = k;
                    for (std::size_t d = dim; d-- > 0;)
                        cell[d] = tbox.lower[d] + static_cast<int>(r % tshape[d]), r /= tshape[d];
                }
                auto const c       = Chunks<dim>::flat(shell, cell);
                double v[n_values] = {};
                for (std::uint32_t j = lane; j < sizes[c]; j += 32)
                {
                    auto const p = ps.copy(starts[c] + j);
                    auto const w = p.weight();
                    if (w == 0)
                        continue;
                    auto const& dl = p.delta();
                    auto const& pv = p.v();
#pragma unroll
                    for (std::uint32_t n = 0; n < n_nodes; ++n)
                    {
                        double s = w;
                        for (std::size_t d = 0; d < dim; ++d)
                            s *= (n >> (dim - 1 - d)) & 1 ? dl[d] : 1. - dl[d];
                        v[n] += s;
                        v[1 * n_nodes + n] += s * pv[0];
                        v[2 * n_nodes + n] += s * pv[1];
                        v[3 * n_nodes + n] += s * pv[2];
                    }
                }
                mkn::kul::for_N<std::bit_width(n_values) - 1>([&](auto is) {
                    std::uint32_t constexpr h = n_values >> (is() + 1);
                    bool const upper          = lane & h;
#pragma unroll
                    for (std::uint32_t i = 0; i < h; ++i)
                    {
                        double const send = upper ? v[i] : v[i + h];
                        double const keep = upper ? v[i + h] : v[i];
                        v[i]              = keep + __shfl_xor_sync(0xffffffff, send, h);
                    }
                });
                for (std::uint32_t m = n_values; m < 32; m *= 2)
                    v[0] += __shfl_xor_sync(0xffffffff, v[0], m);
                if (lane < n_values and v[0] != 0)
                {
                    auto const n     = lane % n_nodes;
                    std::uint32_t at = 0;
                    for (std::size_t d = 0; d < dim; ++d)
                        at = at * wshape[d]
                             + static_cast<std::uint32_t>(cell[d] - tbox.lower[d]
                                                          + ((n >> (dim - 1 - d)) & 1));
                    atomicAdd(&data[(lane / n_nodes) * n_w + at], v[0]);
                }
            }
            __syncthreads();

            double* const dst[4]
                = {rho_v.data(), flux_v[0].data(), flux_v[1].data(), flux_v[2].data()};
            for (std::uint32_t i = threadIdx.x; i < 4 * n_w; i += blockDim.x)
            {
                if (data[i] == 0)
                    continue;
                std::uint32_t r = i % n_w;
                std::size_t at = 0, stride = 1;
                for (std::size_t d = dim; d-- > 0;)
                {
                    auto const node = tbox.lower[d] + static_cast<int>(r % wshape[d]);
                    r /= wshape[d];
                    at += static_cast<std::size_t>(node - origin[d]) * stride;
                    stride *= fshape[d];
                }
                atomicAdd(dst[i / n_w] + at, data[i]);
            }
#endif
        };
        launcher.b.x = block_for(kernel);

        launcher(kernel);
        mkn::gpu::sync();
    }

    // the block clamped to what kernel's registers allow (a fixed launch past it fails), kept
    //  for the printout
    std::uint32_t used_block = 0;
    std::uint32_t block_for(auto& kernel)
    {
        auto const max    = mkn::gpu::getDeviceKernelMaxThreadsPerBlock<mkn::gpu::Fixed>(kernel);
        return used_block = std::min<std::uint32_t>(block, static_cast<std::uint32_t>(max));
    }

    double time(auto&& fn)
    {
        std::vector<double> ms;
        for (std::size_t r = 0; r < repeats + 1; ++r) // the first untimed: first touches
        {
            zero();
            auto const t0 = std::chrono::steady_clock::now();
            fn();
            auto const t1 = std::chrono::steady_clock::now();
            if (r > 0)
                ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        return *std::min_element(ms.begin(), ms.end());
    }

    static double max_rel_diff(std::vector<double> const& ref, std::vector<double> const& cmp)
    {
        double scale = 0, diff = 0;
        for (auto const v : ref)
            scale = std::max(scale, std::abs(v));
        for (std::size_t i = 0; i < ref.size(); ++i)
            diff = std::max(diff, std::abs(ref[i] - cmp[i]));
        return diff / scale;
    }

    void operator()()
    {
        auto& particles = pop().domainParticles();
        particles.sort();

        std::printf(
            "%uD interp %zu, %u^%u cells, %zu ppc (%zu particles), block %u, smem opt-in %zu B\n",
            static_cast<unsigned>(dim), interp, cells, static_cast<unsigned>(dim), ppc, n_particles,
            block, damn::core::gpu::max_dynamic_shared_bytes());
        std::printf("%-9s %5s %6s %7s %8s %9s %9s %7s %10s %6s\n", "staging", "tile", "splits",
                    "blocks", "smem KB", "ms", "ns/part", "x ref", "max rel", "block");

        auto const ref_ms = time([&] { untiled(); });
        auto const ref    = values();
        std::printf("%-9s %5s %6s %7zu %8s %9.3f %9.3f %7.2f %10s\n", "untiled", "-", "-",
                    (n_particles + 255) / 256, "-", ref_ms, ref_ms * 1e6 / n_particles, 1., "-");

        auto const window = [&]<std::uint32_t W>() {
            if (cells % W)
                return;
            auto const n_windows = static_cast<std::size_t>(std::pow(cells / W, dim));
            auto const ms        = time([&] { warp_window<W>(); });
            std::printf("%-9s %5u %6s %7zu %8s %9.3f %9.3f %7.2f %10.1e %6u\n", "warp_win", W, "-",
                        (n_windows * 32 + used_block - 1) / used_block, "-", ms,
                        ms * 1e6 / n_particles, ms / ref_ms, max_rel_diff(ref, values()),
                        used_block);
        };
        window.template operator()<1>();
        window.template operator()<2>();
        if constexpr (interp == 1)
        {
            for (auto const tile : list_of("PLAMN_TILES", "4,6,8,12"))
            {
                std::size_t wn = 1, n_tiles = 1;
                for (std::size_t d = 0; d < dim; ++d)
                    wn *= tile + 1, n_tiles *= (cells + tile - 1) / tile;
                auto const ms = time([&] { window_rs(tile); });
                std::printf("%-9s %5u %6s %7zu %8.1f %9.3f %9.3f %7.2f %10.1e %6u\n", "window_rs",
                            tile, "-", n_tiles, 4 * wn * sizeof(double) / 1024., ms,
                            ms * 1e6 / n_particles, ms / ref_ms, max_rel_diff(ref, values()),
                            used_block);
            }
        }

        auto const run = [&]<Staging staging>(std::uint32_t const tile,
                                              std::uint32_t const splits) {
            std::size_t fp = 1, n_tiles = 1;
            for (std::size_t d = 0; d < dim; ++d)
                fp *= tile + 2 * interp + 2, n_tiles *= (cells + tile - 1) / tile;
            auto const n_fields = staging == Staging::all ? 4 : staging == Staging::one ? 1 : 0;
            auto const bytes    = n_fields * fp * sizeof(double);
            if (bytes > damn::core::gpu::max_dynamic_shared_bytes())
            {
                std::printf("%-9s %5u %6u %7zu %8.1f  does not fit\n", name_of(staging), tile,
                            splits, n_tiles * splits, bytes / 1024.);
                return;
            }
            auto const ms = time([&] { tiled<staging>(tile, splits); });
            std::printf("%-9s %5u %6u %7zu %8.1f %9.3f %9.3f %7.2f %10.1e %6u\n", name_of(staging),
                        tile, splits, n_tiles * splits, bytes / 1024., ms, ms * 1e6 / n_particles,
                        ms / ref_ms, max_rel_diff(ref, values()), used_block);
        };
        for (auto const tile : list_of("PLAMN_TILES", "4,6,8,12"))
            for (auto const splits : list_of("PLAMN_SPLITS", "1,2,4,8"))
            {
                run.template operator()<Staging::all>(tile, splits);
                run.template operator()<Staging::one>(tile, splits);
                run.template operator()<Staging::global>(tile, splits);
            }
    }
};

} // namespace plamn::core::bench

int main()
{
    plamn::core::bench::TileDeposit<3, PLAMN_BENCH_INTERP>{}();
    return 0;
}
