#include "plamn/core/def.hpp"
//  tests/core/numerics/particles_synchronised_mover/test_multi_updater.cpp
//
//  requires
//  - cmake:    -DwithPhlop
//  - cxxflags: -DDAMN_LOG_LEVEL=1
//  - env:      PLAMN_SCOPE_TIMING=1

// USE HIP_VISIBLE_DEVICES OR CUDA_VISIBLE_DEVICES env vars

#include <stdexcept>
#include "damn/core/logger.hpp"

#include "damn/core/sys/monitoring.hpp"
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

#include "gtest/gtest.h"

#include <cmath>
#include <tuple>
#include <algorithm>
#include <cstddef>
#include <cstdlib>

namespace plamn::core
{
// RUNTIME ENV VAR OVERRIDES
auto static const bytes
    = damn::core::get_env_as("PLAMN_GPU_BYTES", std::uint64_t{500000000}); // .5GB
auto static const cells     = damn::core::get_env_as("PLAMN_CELLS", std::uint32_t{20});
auto static const ppc       = damn::core::get_env_as("PLAMN_PPC", std::size_t{33});
auto static const seed      = damn::core::get_env_as("PLAMN_SEED", std::size_t{1067});
auto static const n_patches = damn::core::get_env_as("PLAMN_PATCHES", std::size_t{1});
auto static const dt        = damn::core::get_env_as("PLAMN_TIMESTEP", double{.001});
auto static const shufle    = damn::core::get_env_as("PLAMN_UNSORTED", std::size_t{0});
auto static const do_cmp    = damn::core::get_env_as("PLAMN_COMPARE", std::size_t{1});
auto static const n_threads = damn::core::get_env_as("PLAMN_THREADS", std::size_t{1});
auto static const n_push    = damn::core::get_env_as("PLAMN_PUSHES", std::size_t{1});
auto static const cmp_only  = damn::core::get_env_as("PLAMN_CMP_ONLY", std::size_t{0});
auto static const ref_only  = damn::core::get_env_as("PLAMN_REF_ONLY", std::size_t{0});

bool static const premain = []() {
    assert(!(cmp_only and ref_only) && "Cant have both only");
    DAMN_WITH_MKN_GPU({ //
        mkn::gpu::setLimitMallocHeapSize(bytes);
        // per-thread device stack (default ~1KB) overflows in the MultiBoris push kernel
        mkn::gpu::setLimitStackSize(damn::core::get_env_as("PLAMN_GPU_STACK", std::size_t{16384}));
    })

    DAMN_WITH_PHLOP({
        DAMN_LOG_LINE_SS("compare    : " << do_cmp);
        DAMN_LOG_LINE_SS("cells      : " << cells);
        DAMN_LOG_LINE_SS("ppc        : " << ppc);
        DAMN_LOG_LINE_SS("particles  : " << std::pow(cells, 3) * ppc);
        DAMN_LOG_LINE_SS("n_patches  : " << n_patches);
        DAMN_LOG_LINE_SS("seed       : " << seed);
        DAMN_LOG_LINE_SS("particle MB ≈ " << n_patches * std::pow(cells, 3) * ppc * 76 / 1e6);
        phlop::scope_timer().file_name(".damn_times.0.txt").init();
    })
    damn::core::ThreadPool::threads_per_pool = n_threads;
    return true;
}();

auto& pool = *damn::core::ThreadPool::INSTANCE().thread_pools[0];


// serial       = ParticlesSynchronisedMover, one patch at a time (always used for the ref patches)
// parallel     = ParallelParticlesSynchronisedMover, all patches via BS thread pool
// mkn_parallel = mkn_xyz::ParallelParticlesSynchronisedMover, all patches via mkn.gpu
// ThreadedStreamLauncher
enum class UpdaterVersion : std::uint16_t { serial = 0, parallel, mkn_parallel };

template<UpdaterVersion version, typename Particles_t, typename Layout_t>
auto constexpr updater_impl()
{
    if constexpr (version == UpdaterVersion::serial)
        return static_cast<ParticlesSynchronisedMover<Particles_t, Layout_t>*>(0);
    else if constexpr (version == UpdaterVersion::parallel) // Parallel or FlatParallel (AoS)
        return static_cast<ParticlesSynchronisedMoverFor_t<Particles_t, Layout_t>*>(0);
#if DAMN_HAVE_MKN_GPU
    else if constexpr (version == UpdaterVersion::mkn_parallel)
        return static_cast<mkn_xyz::ParallelParticlesSynchronisedMover<Particles_t, Layout_t>*>(0);
#endif
}

template<UpdaterVersion version, typename Particles_t, typename Layout_t>
using Updater_t = std::decay_t<decltype(*updater_impl<version, Particles_t, Layout_t>())>;


namespace detail::strings
{
    constexpr static std::string_view update  = "update,";
    constexpr static std::string_view compare = "compare,";
    constexpr static std::string_view cma     = ",";
} // namespace detail::strings


template<typename ParticlesSynchronisedMover_t>
void update_serial(UpdaterMode mode, auto& patches)
{
    using Boxing_t = ParticlesSynchronisedMover_t::Boxing_t;

    for (std::size_t i = 0; i < n_push; ++i)
        for (auto& patch : patches)
        {
            auto& [layout, ions, em, electromag] = patch.model.state;

            Boxing_t const boxing{layout, {layout.cells}};
            ParticlesSynchronisedMover_t{}.updatePopulations(ions, electromag, boxing, dt, mode);
            if (mode == UpdaterMode::all) // moves only, the domain deposit follows
                ParticlesSynchronisedMover_t::depositDomain(ions, layout);
            ions.computeChargeDensity();
            ions.computeBulkVelocity();
        }
}

template<typename ParticlesSynchronisedMover_t>
void update_parallel(UpdaterMode mode, auto& patches)
{
    using Boxing_t = ParticlesSynchronisedMover_t::Boxing_t;

    auto const& layout = patches[0].layout;
    std::unordered_map<std::string, Boxing_t> boxings;
    boxings.try_emplace("patch_id", Boxing_t{layout, {layout.cells}});

    auto const quantities = [&](int i) {
        return std::forward_as_tuple(patches[i].model.state.ions,
                                     patches[i].model.state.electromag);
    };
    auto accessor = test::make_model_level_accessor(patches, quantities);

    try
    {
        particles_domain_is_valid(
            patches[0].model.state.ions.populations[0].particles.domain_particles, layout.cells);
        DAMN_LOG_LINE_STR("pre-push domain_particles valid");
    }
    catch (std::exception const& e)
    {
        DAMN_LOG_LINE_SS("pre-push domain_particles INVALID: " << e.what());
    }

    try
    {
        for (std::size_t i = 0; i < n_push; ++i)
        {
            ParticlesSynchronisedMover_t{}.updatePopulations(accessor, boxings, dt, mode);

            for (auto& patch : patches)
            {
                auto& [layout, ions, em, electromag] = patch.model.state;
                if (mode == UpdaterMode::all) // moves only, the domain deposit follows
                    ParticlesSynchronisedMover_t::depositDomain(ions, layout);
                ions.computeChargeDensity();
                ions.computeBulkVelocity();
            }
        }
    }
    catch (std::exception const& e)
    {
        DAMN_LOG_LINE_SS(e.what());
        std::abort();
    }
}

template<UpdaterVersion version, typename Patches>
void update(UpdaterMode mode, Patches& patches)
{
    using Layout_t                     = Patches::value_type::Layout_t;
    using Particles                    = Patches::value_type::Model_t::particles_type;
    using ParticlesSynchronisedMover_t = Updater_t<version, Particles, Layout_t>;
    auto constexpr function_id
        = damn::core::join_string_views_v<detail::strings::update, Particles::type_id,
                                          detail::strings::cma>;
    DAMN_LOG_LINE_STR(function_id);
    DAMN_LOG_SCOPE(1, function_id);

    if constexpr (version == UpdaterVersion::serial)
        update_serial<ParticlesSynchronisedMover_t>(mode, patches);
    else
        update_parallel<ParticlesSynchronisedMover_t>(mode, patches);
}


template<auto opts>
auto make_ref_ions(auto const& layout)
{
    UsableIons_t<opts> ions{layout, "protons"};

    EXPECT_EQ(ions.populations[0].particles.domain_particles.size(), 0ull);

    auto const disperse = [&](auto& particles) {
        delta_disperse(particles.domain_particles, seed);
        vary_velocity(particles.domain_particles, -6, 6, seed);
        if (shufle > 0)
            shuffle(particles.domain_particles, seed);

        delta_disperse(particles.level_ghost_particles, seed);
        vary_velocity(particles.level_ghost_particles, -6, 6, seed);
    };

    auto const particle_box = layout.cells;
    auto add_particles      = [&](auto& particles) {
        particles.domain_particles.reserve(particle_box.size() * ppc);
        core::add_particles(particles.domain_particles, particle_box, ppc);

        add_ghost_particles(particles.level_ghost_particles, particle_box, ppc, 1);
    };

    add_particles(ions.populations[0].particles);
    disperse(ions.populations[0].particles);


    EXPECT_EQ(ions.populations[0].particles.domain_particles.size(), particle_box.size() * ppc);

    return ions;
}


template<auto opts>
auto from_ions(auto const& layout, auto const& from)
{
    UsableIons_t<opts> ions{layout, "protons"};
    EXPECT_EQ(ions.populations[0].particles.domain_particles.size(), 0ull);

    auto _add_particles_from = [&]<auto type>(auto& src, auto& dst) {
        reserve(dst, layout.cells, ppc);
        append_particles<type>(src, dst);
    };

    _add_particles_from.template operator()<ParticleType::Domain>(
        from.populations[0].particles.domain_particles,
        ions.populations[0].particles.domain_particles);

    particles_domain_is_valid(ions.populations[0].particles.domain_particles, layout.cells);

    _add_particles_from.template operator()<ParticleType::LevelGhost>(
        from.populations[0].particles.level_ghost_particles,
        ions.populations[0].particles.level_ghost_particles);

    auto const particle_box = layout.cells;
    EXPECT_EQ(ions.populations[0].particles.domain_particles.size(), particle_box.size() * ppc);
    return ions;
}


template<auto type, typename Layout_t, typename P0, typename P1>
void check_particles(Layout_t const& layout, P0& ref, P1& cmp_, double const atol,
                     std::string const& name)
{
    static_assert(std::is_same_v<decltype(type), ParticleType>);

    using CPU_ref = Particles<ParticlesOptions{P0::dimension, LayoutMode::AoS, StorageMode::VECTOR,
                                               damn::AllocatorMode::CPU}>;

    auto const box = layout.cells;
    // level/patch ghost particles live outside the domain box entirely - the sorter's
    // flat-index cell key isn't defined out there, so sorting with the plain domain box
    // can scatter/misgroup ghost-region particles differently between ref and cmp even
    // when the underlying data matches. Grow the box for non-domain comparisons so the
    // flattener covers every cell these particles can actually occupy.
    auto const sort_box = type == ParticleType::Domain
                              ? box
                              : damn::core::grow(box, Layout_t::options.particle_ghost_width);
    try
    {
        if constexpr (type == ParticleType::Domain)
            particles_domain_is_valid(cmp_, box);
        else
            particles_ghost_is_valid(cmp_, box, sort_box);
        DAMN_LOG_LINE_SS(name << " post-push cmp_ valid size=" << cmp_.size());
    }
    catch (std::exception const& e)
    {
        DAMN_LOG_LINE_SS(name << " post-push cmp_ INVALID: " << e.what());
        EXPECT_TRUE(0 == 1);
    }

    auto cmp = [&]() {
        if constexpr (type == ParticleType::LevelGhost and is_tiled(P1::layout_mode))
            return clamp_owner_particles<CPU_ref>(cmp_);
        else
            return convert_particles<CPU_ref>(cmp_, layout);
    }();
    sort_particles(cmp, sort_box);
    sort_particles(ref, sort_box);

    EXPECT_EQ(ref.size(), cmp.size()) << name;

    // atol is absolute: scale it by the largest |v|, else for |v| > ~4.5 one ULP (8.9e-16 at
    //  |v| in [4, 8)) already exceeds 1e-15 and any rounding difference between two push
    //  implementations (e.g. FMA contraction, seen with hipcc) fails the comparison
    auto const v_max = [&]() {
        double v = 0;
        for (std::size_t i = 0; i < ref.size(); ++i)
            for (std::size_t d = 0; d < 3; ++d)
                v = std::max(v, std::abs(ref[i].v()[d]));
        return v;
    }();
    auto const p_atol = atol * std::max(1., 4 * v_max);

    auto const report = compare_particles(ref, cmp, p_atol);
    if (report)
    {
        DAMN_LOG_LINE_SS("Comparing Particle Arrays OK: " << P0::id() << " vs " << P1::id());
    }
    else
    {
        DAMN_LOG_LINE_SS("Comparing Particle Arrays FAIL: " << P0::id() << " vs " << P1::id());
    }
    DAMN_LOG_LINE_SS("results: " << report.why());
    if (not report and report.idx < ref.size() and report.idx < cmp.size())
    { // how they differ: the first mismatch, the largest differences, how many
        auto const& r = ref[report.idx];
        auto const& c = cmp[report.idx];
        DAMN_LOG_LINE_SS("ref[" << report.idx << "]: " << r);
        DAMN_LOG_LINE_SS("cmp[" << report.idx << "]: " << c);

        double max_dv = 0, max_ddelta = 0;
        std::size_t n_bad = 0;
        for (std::size_t i = 0; i < std::min(ref.size(), cmp.size()); ++i)
        {
            for (std::size_t d = 0; d < 3; ++d)
                max_dv = std::max(max_dv, std::abs(ref[i].v()[d] - cmp[i].v()[d]));
            for (std::size_t d = 0; d < Layout_t::dimension; ++d)
                max_ddelta = std::max(max_ddelta, std::abs(ref[i].delta()[d] - cmp[i].delta()[d]));
            if (not particle_compare(ref[i], cmp[i], i, p_atol))
                ++n_bad;
        }
        DAMN_LOG_LINE_SS("atol " << p_atol << ", max |dv| " << max_dv << ", max |ddelta| "
                                 << max_ddelta << ", " << n_bad << " of " << ref.size()
                                 << " particles differ");
    }

    EXPECT_TRUE(report);
}

template<typename C>
void compare(auto const& layout, auto& ref, C& cmp)
{
    using Particles_t = C::Particles_t;
    auto constexpr function_id
        = damn::core::join_string_views_v<detail::strings::compare, Particles_t::type_id,
                                          detail::strings::cma>;
    DAMN_LOG_SCOPE(1, function_id);

    using enum LayoutMode;
    using enum damn::AllocatorMode;

    double diff = 1e-15;
    if constexpr (Particles_t::alloc_mode == GPU_UNIFIED)
        diff *= 1e3; // atomics no order guaranteed
    else if constexpr (is_tiled(Particles_t::layout_mode))
        diff *= 1e1; // p2m op order diff

    check_particles<ParticleType::Domain>(layout, ref.populations[0].particles.domain_particles,
                                          cmp.populations[0].particles.domain_particles, diff,
                                          "domain");

    check_particles<ParticleType::PatchGhost>(
        layout, ref.populations[0].particles.patch_ghost_particles,
        cmp.populations[0].particles.patch_ghost_particles, diff, "patchghost");

    check_particles<ParticleType::LevelGhost>(
        layout, ref.populations[0].particles.level_ghost_particles,
        cmp.populations[0].particles.level_ghost_particles, diff, "levelghost");

    {
        // zero_ghost_layer(layout, ref.rhoC, cmp.rhoC);
        auto const rhoCport = compare_reduced_fields(ref.rhoC, cmp.rhoC, diff);
        DAMN_LOG_LINE_SS("results: ions rhoC" << rhoCport.why());
        EXPECT_TRUE(rhoCport);

        // zero_ghost_layer(layout, ref.rhoM, cmp.rhoM);
        auto const rhoMport = compare_reduced_fields(ref.rhoM, cmp.rhoM, diff);
        DAMN_LOG_LINE_SS("results: ions rhoM" << rhoMport.why());
        EXPECT_TRUE(rhoMport);
    }

    auto const freport
        = compare_reduced_tensor_fields(ref.populations[0].F, cmp.populations[0].F, diff);
    DAMN_LOG_LINE_SS("results: " << freport.why());
    EXPECT_TRUE(freport);

    auto const rhoCport
        = compare_reduced_fields(ref.populations[0].rhoC, cmp.populations[0].rhoC, diff);
    DAMN_LOG_LINE_SS("results: " << rhoCport.why());
    EXPECT_TRUE(rhoCport);

    auto const rhoPport
        = compare_reduced_fields(ref.populations[0].rhoP, cmp.populations[0].rhoP, diff);
    DAMN_LOG_LINE_SS("results: " << rhoPport.why());
    EXPECT_TRUE(rhoPport);
}


template<std::size_t _dim, auto _layout_mode, auto _alloc_mode, auto _updater_mode,
         auto _updater = UpdaterVersion::parallel>
struct TestParam
{
    static_assert(std::is_same_v<decltype(_layout_mode), LayoutMode>);
    static_assert(std::is_same_v<decltype(_alloc_mode), damn::AllocatorMode>);
    static_assert(std::is_same_v<decltype(_updater), UpdaterVersion>);
    auto constexpr static dim          = _dim;
    auto constexpr static layout_mode  = _layout_mode;
    auto constexpr static alloc_mode   = _alloc_mode;
    auto constexpr static updater_mode = _updater_mode;
    auto constexpr static updater      = _updater;
    auto constexpr static opts
        = SimOpts{.dimension = _dim, .layout_mode = _layout_mode, .alloc_mode = _alloc_mode};
};



template<typename Param>
struct MultiPatchParticlesSynchronisedMoverTest : public ::testing::Test
{
    auto constexpr static dim          = Param::dim;
    auto constexpr static layout_mode  = Param::layout_mode;
    auto constexpr static alloc_mode   = Param::alloc_mode;
    auto constexpr static updater_mode = Param::updater_mode;
    auto constexpr static updater      = Param::updater;

    auto constexpr static ref_opts = SimOpts{.dimension = dim};
    auto constexpr static cmp_opts = Param::opts;

    using RefCoreTypes = PLAMN_Types<ref_opts>::Hybrid;
    using CmpCoreTypes = PLAMN_Types<cmp_opts>::Hybrid;

    using RefLayout_t = RefCoreTypes::Layout_t;
    using CmpLayout_t = CmpCoreTypes::Layout_t;

    using TestRefLayout_t = TestLayout<RefLayout_t>;
    using TestCmpLayout_t = TestLayout<CmpLayout_t>;

    using RefParticles_t = ParticlesAoSMapped<dim>;
    using CmpParticles_t
        = Particles<ParticlesOptions{dim, layout_mode, StorageMode::VECTOR, alloc_mode}>;

    auto constexpr static ref_field_opts = TensorFieldOptions<RefCoreTypes>{};
    auto constexpr static cmp_field_opts = TensorFieldOptions<CmpCoreTypes>{};

    using RefIons_t = UsableIons_t<ref_opts>;
    using RefEM_t   = UsableElectromag<ref_field_opts>;
    using CmpIons_t = UsableIons_t<cmp_opts>;
    using CmpEM_t   = UsableElectromag<cmp_field_opts>;

    RefLayout_t const ref_layout = *TestRefLayout_t{cells};
    CmpLayout_t const cmp_layout = *TestCmpLayout_t{cells};

    MultiPatchParticlesSynchronisedMoverTest() {}

    void run()
    {
        auto& ref = ref_patches.emplace_back(ref_layout, make_ref_ions<ref_opts>(ref_layout));

        if (!ref_only)
            for (std::size_t i = 0; i < n_patches; i++)
                cmp_patches.emplace_back(cmp_layout,
                                         from_ions<cmp_opts>(cmp_layout, ref.model.state.ions));

        if (!cmp_only)
            update<UpdaterVersion::serial>(updater_mode, ref_patches);

        if (!ref_only)
            update<updater>(updater_mode, cmp_patches);

        if (do_cmp)
            for (auto& cmp : cmp_patches)
                compare(cmp_layout, ref_patches[0].model.state.ions, cmp.model.state.ions);
    }



    using RefPatch = test::HybridPatch<RefIons_t, RefEM_t>;
    using CmpPatch = test::HybridPatch<CmpIons_t, CmpEM_t>;

    std::vector<damn::core::aggregate_adapter<RefPatch>> ref_patches{};
    std::vector<damn::core::aggregate_adapter<CmpPatch>> cmp_patches{};
};

// clang-format off
using Permutations_t = testing::Types< // ! notice commas !
    TestParam<1, LayoutMode::AoSPCTS, damn::AllocatorMode::CPU, UpdaterMode::domain_only>
   ,TestParam<1, LayoutMode::AoSPCTS, damn::AllocatorMode::CPU, UpdaterMode::all>
   ,TestParam<2, LayoutMode::AoSPCTS, damn::AllocatorMode::CPU, UpdaterMode::domain_only>
   ,TestParam<2, LayoutMode::AoSPCTS, damn::AllocatorMode::CPU, UpdaterMode::all>
   ,TestParam<3, LayoutMode::AoSPCTS, damn::AllocatorMode::CPU, UpdaterMode::domain_only>
   ,TestParam<3, LayoutMode::AoSPCTS, damn::AllocatorMode::CPU, UpdaterMode::all>
   ,TestParam<1, LayoutMode::AoSCMTS, damn::AllocatorMode::CPU, UpdaterMode::domain_only>
   ,TestParam<1, LayoutMode::AoSCMTS, damn::AllocatorMode::CPU, UpdaterMode::all>
   ,TestParam<2, LayoutMode::AoSCMTS, damn::AllocatorMode::CPU, UpdaterMode::domain_only>
   ,TestParam<2, LayoutMode::AoSCMTS, damn::AllocatorMode::CPU, UpdaterMode::all>
   ,TestParam<3, LayoutMode::AoSCMTS, damn::AllocatorMode::CPU, UpdaterMode::domain_only>
   ,TestParam<3, LayoutMode::AoSCMTS, damn::AllocatorMode::CPU, UpdaterMode::all>

   ,TestParam<1, LayoutMode::AoS, damn::AllocatorMode::CPU, UpdaterMode::domain_only>
   ,TestParam<1, LayoutMode::AoS, damn::AllocatorMode::CPU, UpdaterMode::all>
   ,TestParam<2, LayoutMode::AoS, damn::AllocatorMode::CPU, UpdaterMode::domain_only>
   ,TestParam<2, LayoutMode::AoS, damn::AllocatorMode::CPU, UpdaterMode::all>
   ,TestParam<3, LayoutMode::AoS, damn::AllocatorMode::CPU, UpdaterMode::domain_only>
   ,TestParam<3, LayoutMode::AoS, damn::AllocatorMode::CPU, UpdaterMode::all>

DAMN_WITH_THRUST( // SoAVX* iteration needs thrust; SoAVX (unsorted) off: not a priority, its
                  //  GPU_UNIFIED mkn_parallel cases fail intermittently (moments/particles race)
//   ,TestParam<1, LayoutMode::SoAVX, AllocatorMode::CPU, UpdaterMode::domain_only>
//   ,TestParam<1, LayoutMode::SoAVX, AllocatorMode::CPU, UpdaterMode::all>
//   ,TestParam<2, LayoutMode::SoAVX, AllocatorMode::CPU, UpdaterMode::domain_only>
//   ,TestParam<2, LayoutMode::SoAVX, AllocatorMode::CPU, UpdaterMode::all>
//   ,TestParam<3, LayoutMode::SoAVX, AllocatorMode::CPU, UpdaterMode::domain_only>
//   ,TestParam<3, LayoutMode::SoAVX, AllocatorMode::CPU, UpdaterMode::all>

   ,TestParam<1, LayoutMode::SoAVXChunked, damn::AllocatorMode::CPU, UpdaterMode::domain_only>
   ,TestParam<1, LayoutMode::SoAVXChunked, damn::AllocatorMode::CPU, UpdaterMode::all>
   ,TestParam<2, LayoutMode::SoAVXChunked, damn::AllocatorMode::CPU, UpdaterMode::domain_only>
   ,TestParam<2, LayoutMode::SoAVXChunked, damn::AllocatorMode::CPU, UpdaterMode::all>
   ,TestParam<3, LayoutMode::SoAVXChunked, damn::AllocatorMode::CPU, UpdaterMode::domain_only>
   ,TestParam<3, LayoutMode::SoAVXChunked, damn::AllocatorMode::CPU, UpdaterMode::all>
)

   ,TestParam<1, LayoutMode::AoSTS, damn::AllocatorMode::CPU, UpdaterMode::domain_only>
   ,TestParam<1, LayoutMode::AoSTS, damn::AllocatorMode::CPU, UpdaterMode::all>
   ,TestParam<2, LayoutMode::AoSTS, damn::AllocatorMode::CPU, UpdaterMode::domain_only>
   ,TestParam<2, LayoutMode::AoSTS, damn::AllocatorMode::CPU, UpdaterMode::all>
   ,TestParam<3, LayoutMode::AoSTS, damn::AllocatorMode::CPU, UpdaterMode::domain_only>
   ,TestParam<3, LayoutMode::AoSTS, damn::AllocatorMode::CPU, UpdaterMode::all>

DAMN_WITH_MKN_GPU(
   ,TestParam<1, LayoutMode::AoSPCTS, damn::AllocatorMode::CPU, UpdaterMode::domain_only, UpdaterVersion::mkn_parallel>
   ,TestParam<1, LayoutMode::AoSPCTS, damn::AllocatorMode::CPU, UpdaterMode::all,         UpdaterVersion::mkn_parallel>
   ,TestParam<2, LayoutMode::AoSPCTS, damn::AllocatorMode::CPU, UpdaterMode::domain_only, UpdaterVersion::mkn_parallel>
   ,TestParam<2, LayoutMode::AoSPCTS, damn::AllocatorMode::CPU, UpdaterMode::all,         UpdaterVersion::mkn_parallel>
   ,TestParam<3, LayoutMode::AoSPCTS, damn::AllocatorMode::CPU, UpdaterMode::domain_only, UpdaterVersion::mkn_parallel>
   ,TestParam<3, LayoutMode::AoSPCTS, damn::AllocatorMode::CPU, UpdaterMode::all,         UpdaterVersion::mkn_parallel>
)

DAMN_WITH_GPU(
   // AoSPCTS on the device: off for now. The CUDA simulator (PantheonSim) loses its cell change
   //  registrations, it passes on NVIDIA GPUs: https://github.com/mkn/mkn.gpu/pull/44
// ,TestParam<1, LayoutMode::AoSPCTS, damn::AllocatorMode::GPU_UNIFIED, UpdaterMode::domain_only, UpdaterVersion::mkn_parallel>
// ,TestParam<1, LayoutMode::AoSPCTS, damn::AllocatorMode::GPU_UNIFIED, UpdaterMode::all,         UpdaterVersion::mkn_parallel>
// ,TestParam<2, LayoutMode::AoSPCTS, damn::AllocatorMode::GPU_UNIFIED, UpdaterMode::domain_only, UpdaterVersion::mkn_parallel>
// ,TestParam<2, LayoutMode::AoSPCTS, damn::AllocatorMode::GPU_UNIFIED, UpdaterMode::all,         UpdaterVersion::mkn_parallel>
// ,TestParam<3, LayoutMode::AoSPCTS, damn::AllocatorMode::GPU_UNIFIED, UpdaterMode::domain_only, UpdaterVersion::mkn_parallel>
// ,TestParam<3, LayoutMode::AoSPCTS, damn::AllocatorMode::GPU_UNIFIED, UpdaterMode::all,         UpdaterVersion::mkn_parallel>

   ,TestParam<1, LayoutMode::AoSTS, damn::AllocatorMode::GPU_UNIFIED, UpdaterMode::domain_only, UpdaterVersion::mkn_parallel>
   ,TestParam<1, LayoutMode::AoSTS, damn::AllocatorMode::GPU_UNIFIED, UpdaterMode::all,         UpdaterVersion::mkn_parallel>
   ,TestParam<2, LayoutMode::AoSTS, damn::AllocatorMode::GPU_UNIFIED, UpdaterMode::domain_only, UpdaterVersion::mkn_parallel>
   // 2D/3D AoSTS on the device, moving all: off for now, as AoSPCTS above
// ,TestParam<2, LayoutMode::AoSTS, damn::AllocatorMode::GPU_UNIFIED, UpdaterMode::all,         UpdaterVersion::mkn_parallel>
   ,TestParam<3, LayoutMode::AoSTS, damn::AllocatorMode::GPU_UNIFIED, UpdaterMode::domain_only, UpdaterVersion::mkn_parallel>
// ,TestParam<3, LayoutMode::AoSTS, damn::AllocatorMode::GPU_UNIFIED, UpdaterMode::all,         UpdaterVersion::mkn_parallel>

DAMN_WITH_THRUST( // SoAVX* GPU needs thrust (rocThrust on ROCm)
//   ,TestParam<1, LayoutMode::SoAVX, AllocatorMode::GPU_UNIFIED, UpdaterMode::domain_only, UpdaterVersion::mkn_parallel>
//   ,TestParam<1, LayoutMode::SoAVX, AllocatorMode::GPU_UNIFIED, UpdaterMode::all,         UpdaterVersion::mkn_parallel>
//   ,TestParam<2, LayoutMode::SoAVX, AllocatorMode::GPU_UNIFIED, UpdaterMode::domain_only, UpdaterVersion::mkn_parallel>
//   ,TestParam<2, LayoutMode::SoAVX, AllocatorMode::GPU_UNIFIED, UpdaterMode::all,         UpdaterVersion::mkn_parallel>
//   ,TestParam<3, LayoutMode::SoAVX, AllocatorMode::GPU_UNIFIED, UpdaterMode::domain_only, UpdaterVersion::mkn_parallel>
//   ,TestParam<3, LayoutMode::SoAVX, AllocatorMode::GPU_UNIFIED, UpdaterMode::all,         UpdaterVersion::mkn_parallel>

   ,TestParam<1, LayoutMode::SoAVXChunked, damn::AllocatorMode::GPU_UNIFIED, UpdaterMode::domain_only, UpdaterVersion::mkn_parallel>
   ,TestParam<1, LayoutMode::SoAVXChunked, damn::AllocatorMode::GPU_UNIFIED, UpdaterMode::all,         UpdaterVersion::mkn_parallel>
   ,TestParam<2, LayoutMode::SoAVXChunked, damn::AllocatorMode::GPU_UNIFIED, UpdaterMode::domain_only, UpdaterVersion::mkn_parallel>
   ,TestParam<2, LayoutMode::SoAVXChunked, damn::AllocatorMode::GPU_UNIFIED, UpdaterMode::all,         UpdaterVersion::mkn_parallel>
   ,TestParam<3, LayoutMode::SoAVXChunked, damn::AllocatorMode::GPU_UNIFIED, UpdaterMode::domain_only, UpdaterVersion::mkn_parallel>
   ,TestParam<3, LayoutMode::SoAVXChunked, damn::AllocatorMode::GPU_UNIFIED, UpdaterMode::all,         UpdaterVersion::mkn_parallel>
)
)

>;
// clang-format on

TYPED_TEST_SUITE(MultiPatchParticlesSynchronisedMoverTest, Permutations_t, );

TYPED_TEST(MultiPatchParticlesSynchronisedMoverTest, test_updater)
{
    this->run();
}

template<auto opts> // used by gtest
void PrintTo(Particles<opts> const& arr, std::ostream* os)
{
    *os << arr;
}

} // namespace plamn::core


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    try
    {
        auto r = RUN_ALL_TESTS();
        DAMN_WITH_PHLOP(phlop::scope_timer().shutdown());
        damn::core::MemoryMonitoring::PRINT();
        return r;
    }
    catch (std::runtime_error const& e)
    {
        DAMN_LOG_LINE_SS(e.what());
    }
    return 1;
}
