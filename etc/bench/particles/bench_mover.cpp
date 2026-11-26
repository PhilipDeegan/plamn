// each step of a particles mover timed on its own, per layout (MoveSpec,
//  particles_synchronised_mover_def.hpp): the push in place, + its sync, + the deposit of the
//  pushed particles, all three, and the copy push with its deposit (UpdaterMode::domain_only). The
//  patches are copies of one patch for the load only: no patch ghost exchange between them.
//
//   mkn build -p bench_mover -dKO -g 0 -x res/mkn/mpi [-x res/mkn/clang_cuda] [-a
//   -DPLAMN_BENCH_DIM=2] mkn run -p bench_mover [-x ...]
//
// env: PLAMN_CELLS (20 a side), PLAMN_PPC (100), PLAMN_PATCHES (1), PLAMN_THREADS (1, a pool's
//  threads), PLAMN_REPEATS (3: the fastest and the median printed), PLAMN_SEED, PLAMN_TIMESTEP,
//  PLAMN_WARM (1: each step once untimed first, on the same patches: without the first touches),
//  PLAMN_ONLY (the layouts whose name has it, e.g. GPU);
//  tiles as a run: DAMN_TILING_MAX_TILE_SIZE / DAMN_TILING_MIN_TILE_SIZE, the device deposit
//  PLAMN_GPU_TILE_DEPOSIT

#include "damn/core/exec/thread_pool.hpp" // defaults to 1 thread, setup during static init!
#include "plamn/core/data/particles/particles.hpp"
#include "plamn/core/numerics/particles_synchronised_mover/particles_synchronised_mover.hpp" // IWYU pragma: keep
#include "plamn/core/data/particles/appending/particles_appending.hpp"

#include "plamn/simulator/simulator_def.hpp"

#include "tst/plamn/core/data/grid/test_grid_fixtures.hpp"
#include "tst/plamn/core/data/particles/test_particles.hpp"
#include "tst/plamn/core/data/electromag/test_electromag_fixtures.hpp"
#include "tst/plamn/core/data/ion_population/test_ion_population_fixtures.hpp"
#include "tst/plamn/amr/resources_manager/test_resources_manager_fixtures.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#ifndef PLAMN_BENCH_DIM
#define PLAMN_BENCH_DIM 3
#endif

namespace plamn::core::bench
{
auto static const cells     = mkn::kul::env::get_as("PLAMN_CELLS", std::uint32_t{20});
auto static const ppc       = mkn::kul::env::get_as("PLAMN_PPC", std::size_t{100});
auto static const n_patches = mkn::kul::env::get_as("PLAMN_PATCHES", std::size_t{1});
auto static const n_threads = mkn::kul::env::get_as("PLAMN_THREADS", std::size_t{1});
auto static const repeats   = mkn::kul::env::get_as("PLAMN_REPEATS", std::size_t{3});
auto static const seed      = mkn::kul::env::get_as("PLAMN_SEED", std::size_t{1067});
auto static const dt        = mkn::kul::env::get_as("PLAMN_TIMESTEP", double{.001});
auto static const warm      = mkn::kul::env::get_as("PLAMN_WARM", std::size_t{0});
auto static const only      = mkn::kul::env::get_as("PLAMN_ONLY", std::string{});

bool static const premain = []() {
    DAMN_WITH_MKN_GPU({
        mkn::gpu::setLimitMallocHeapSize(std::uint64_t{500000000});
        // per-thread device stack (default ~1KB) overflows in the MultiBoris push kernel
        mkn::gpu::setLimitStackSize(std::size_t{16384});
    })
    damn::core::ThreadPool::threads_per_pool = n_threads;
    return true;
}();

// clang-format off
MoveSpec constexpr push         {.copy = false, .deposit = false, .sync = false};
MoveSpec constexpr push_sync    {.copy = false, .deposit = false, .sync = true};
MoveSpec constexpr push_deposit {.copy = false, .deposit = true,  .sync = false};
MoveSpec constexpr push_all     {.copy = false, .deposit = true,  .sync = true};
// clang-format on


template<auto opts>
auto make_ions(auto const& layout) // the reference particles, AoSMapped
{
    UsableIons_t<opts> ions{layout, "protons"};
    auto& particles = ions.populations[0].particles;
    particles.domain_particles.reserve(layout.cells.size() * ppc);
    core::add_particles(particles.domain_particles, layout.cells, ppc);
    add_ghost_particles(particles.level_ghost_particles, layout.cells, ppc, 1);
    delta_disperse(particles.domain_particles, seed);
    vary_velocity(particles.domain_particles, -6, 6, seed);
    delta_disperse(particles.level_ghost_particles, seed);
    vary_velocity(particles.level_ghost_particles, -6, 6, seed);
    return ions;
}

template<auto opts>
auto from_ions(auto const& layout, auto const& from)
{
    UsableIons_t<opts> ions{layout, "protons"};
    auto& dst = ions.populations[0].particles;
    auto& src = from.populations[0].particles;
    reserve(dst.domain_particles, layout.cells, ppc);
    append_particles<ParticleType::Domain>(src.domain_particles, dst.domain_particles);
    append_particles<ParticleType::LevelGhost>(src.level_ghost_particles,
                                               dst.level_ghost_particles);
    return ions;
}


template<std::size_t dim, auto layout_mode, auto alloc_mode, std::size_t interp = 1>
struct Case
{
    auto constexpr static ref_opts = SimOpts{.dimension = dim, .interp_order = interp};
    auto constexpr static opts     = SimOpts{.dimension    = dim,
                                             .interp_order = interp,
                                             .layout_mode  = layout_mode,
                                             .alloc_mode   = alloc_mode};

    using RefTypes                  = PLAMN_Types<ref_opts>::Hybrid;
    using Types                     = PLAMN_Types<opts>::Hybrid;
    using Layout_t                  = Types::Layout_t;
    using Ions_t                    = UsableIons_t<opts>;
    using EM_t                      = UsableElectromag<TensorFieldOptions<Types>{}>;
    using Patch_t                   = test::HybridPatch<Ions_t, EM_t>;
    using Patches                   = std::vector<mkn::kul::aggregate_adapter<Patch_t>>;
    using Particles_t               = Patch_t::Model_t::particles_type;
    using Mover_t                   = ParticlesSynchronisedMoverFor_t<Particles_t, Layout_t>;
    bool constexpr static per_patch = layout_mode == LayoutMode::AoSMapped;

    typename RefTypes::Layout_t const ref_layout = *TestLayout<typename RefTypes::Layout_t>{cells};
    Layout_t const layout                        = *TestLayout<Layout_t>{cells};
    UsableIons_t<ref_opts> const ref_ions        = make_ions<ref_opts>(ref_layout);

    Patches make_patches() const
    {
        Patches patches;
        for (std::size_t i = 0; i < n_patches; ++i)
            if constexpr (per_patch) // the reference layout: made as the reference, same seed
                patches.emplace_back(layout, make_ions<opts>(layout));
            else
                patches.emplace_back(layout, from_ions<opts>(layout, ref_ions));
        return patches;
    }

    template<MoveSpec spec, DepositSpec deposit = DepositSpec{}>
    void run(Patches& patches) const
    {
        using Boxing_t = Mover_t::Boxing_t;
        if constexpr (per_patch)
            for (auto& patch : patches)
            {
                auto& [layout_, ions, em, electromag] = patch.model.state;
                Boxing_t const boxing{layout_, {layout_.cells}};
                Mover_t{}.template update<spec, deposit>(ions, electromag, boxing, dt);
            }
        else
        {
            std::unordered_map<std::string, Boxing_t> boxings;
            boxings.try_emplace("patch_id", Boxing_t{layout, {layout.cells}});
            auto const quantities = [&](int i) {
                return std::forward_as_tuple(patches[i].model.state.ions,
                                             patches[i].model.state.electromag);
            };
            auto accessor = test::make_model_level_accessor(patches, quantities);
            Mover_t{}.template update<spec, deposit>(accessor, boxings, dt);
        }
        DAMN_WITH_MKN_GPU(mkn::gpu::sync();)
    }

    template<MoveSpec spec, DepositSpec deposit = DepositSpec{}>
    void time(std::string_view const name, std::string_view const step) const
    {
        if constexpr (per_patch and not spec.copy and not spec.sync)
            return; // AoSMapped: its in place push syncs as it goes
        else if constexpr (layout_mode == LayoutMode::AoSTS and spec.sync)
            return; // a test layout: its sync sorts by hand, slow by design
        else
        {
            std::vector<double> ms;
            for (std::size_t r = 0; r < repeats; ++r)
            {
                auto patches = make_patches(); // not timed
                if (warm) // once untimed on the same patches: first touches (managed memory)
                    run<spec, deposit>(patches);
                auto const t0 = std::chrono::steady_clock::now();
                run<spec, deposit>(patches);
                auto const t1 = std::chrono::steady_clock::now();
                ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            std::sort(ms.begin(), ms.end());
            auto const particles = static_cast<double>(n_patches * layout.cells.size() * ppc);
            std::printf("%-22s %-14s %10.3f %10.3f %8.2f\n", std::string{name}.c_str(),
                        std::string{step}.c_str(), ms[0], ms[ms.size() / 2],
                        ms[0] * 1e6 / particles);
        }
    }

    void operator()(std::string_view const name) const
    {
        if constexpr (is_tiled(layout_mode))
        { // what the deposit stages per tile: its field's largest tile, ghosts included
            auto patches     = make_patches();
            auto const& rho  = patches[0].model.state.ions[0].particleDensity();
            auto const nodes = rho.max_tile_size();
            std::printf("%-22s tiles %zu, largest density tile %u nodes (%.1f a side), %zu "
                        "particles a tile\n",
                        std::string{name}.c_str(), static_cast<std::size_t>(tile_count(rho)),
                        static_cast<unsigned>(nodes), std::cbrt(static_cast<double>(nodes)),
                        layout.cells.size() * ppc / static_cast<std::size_t>(tile_count(rho)));
        }
        time<push>(name, "push");
        time<push_sync>(name, "push+sync");
        time<push_deposit>(name, "push+deposit");
        time<push_all>(name, "push+sync+dep");
        time<copy_spec>(name, "copy+deposit");
        if constexpr (alloc_mode == damn::AllocatorMode::GPU_UNIFIED)
        { // the device deposit after an in place push, 8 warps a tile (DepositSpec)
            time<push_deposit, DepositSpec{.warps = 8}>(name, "push+dep w8");
            time<push_all, DepositSpec{.warps = 8}>(name, "push+sync+dep w8");
        }
    }
};

template<std::size_t dim, auto layout_mode, auto alloc_mode, std::size_t interp = 1>
void run_case(std::string_view const name)
{
    if (!only.empty() and name.find(only) == std::string_view::npos)
        return;
    Case<dim, layout_mode, alloc_mode, interp>{}(name);
}

} // namespace plamn::core::bench


int main()
{
    using namespace plamn::core;
    using namespace plamn::core::bench;
    using enum LayoutMode;
    using enum damn::AllocatorMode;
    auto constexpr dim = std::size_t{PLAMN_BENCH_DIM};

    std::printf("%uD, %u^%u cells, %zu ppc, %zu patch(es), %zu thread(s), %zu repeat(s)\n",
                static_cast<unsigned>(dim), cells, static_cast<unsigned>(dim), ppc, n_patches,
                n_threads, repeats);
    std::printf("%-22s %-14s %10s %10s %8s\n", "layout", "step", "min ms", "median ms", "ns/part");

    run_case<dim, AoSMapped, CPU>("AoSMapped");
    run_case<dim, AoS, CPU>("AoS");
    run_case<dim, AoSTS, CPU>("AoSTS");
    run_case<dim, AoSCMTS, CPU>("AoSCMTS");
    run_case<dim, AoSPCTS, CPU>("AoSPCTS");
    DAMN_WITH_THRUST(run_case<dim, SoAVXChunked, CPU>("SoAVXChunked");)
    DAMN_WITH_GPU(run_case<dim, AoSTS, GPU_UNIFIED>("AoSTS GPU_UNIFIED");)
    DAMN_WITH_GPU(run_case<dim, AoSTS, GPU_UNIFIED, 2>("AoSTS GPU_UNIFIED i2");)
    return 0;
}
