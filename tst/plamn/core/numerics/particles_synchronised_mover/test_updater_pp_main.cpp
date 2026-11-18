#include "plamn/core/def.hpp"
// tests/core/numerics/particles_synchronised_mover/test_updater_pp_main.cpp
//
//  requires
//  - cmake:    -DwithPhlop
//  - cxxflags: -DDAMN_LOG_LEVEL=1
//  - env:      PLAMN_SCOPE_TIMING=1

#include "plamn/core/data/particles/sorting/particles_sorting.hpp"
#include "damn/core/def/detail/mkn_avx.hpp"
#include <memory>

#define DAMN_UNDEF_ASSERT
// #define DAMN_LOG_LEVEL 3
// #define DAMN_LOG_SCOPE_PRINT 1


#include "mkn/kul/for.hpp"
#include "mkn/kul/string.hpp"
#include "mkn/kul/env.hpp"
#include "plamn/core/numerics/particles_synchronised_mover/particles_synchronised_mover.hpp"
#include "plamn/core/numerics/particles_synchronised_mover/particles_synchronised_mover_pc.hpp"
#include "plamn/core/numerics/particles_synchronised_mover/particles_synchronised_mover_per_particle.hpp"

#include "plamn/core/data/particles/appending/particles_appending.hpp"
#include "plamn/core/data/particles/serializing/particles_serializing.hpp"

#include "plamn/simulator/simulator_def.hpp"


#include "tst/plamn/core/data/electromag/test_electromag_fixtures.hpp"
#include "tst/plamn/core/data/ion_population/test_ion_population_fixtures.hpp"

#include "gtest/gtest.h"

namespace plamn::core
{

template<auto opts> // used by gtest
void PrintTo(Particles<opts> const& arr, std::ostream* os)
{
    // assert(arr.size());
    *os << arr;
}

bool constexpr static WITH_PATCH_GHOST  = false;
bool constexpr static USE_SERIALIZATION = 0;

auto static const bytes   = mkn::kul::env::get_as("PLAMN_GPU_BYTES", std::uint64_t{500000000});
auto static const cells   = mkn::kul::env::get_as("PLAMN_CELLS", std::uint32_t{4});
auto static const ppc     = mkn::kul::env::get_as("PLAMN_PPC", std::size_t{4});
auto static const seed    = mkn::kul::env::get_as("PLAMN_SEED", std::size_t{1012});
auto static const dt      = mkn::kul::env::get_as("PLAMN_TIMESTEP", double{.001});
auto static const shufle  = mkn::kul::env::get_as("PLAMN_UNSORTED", std::size_t{0});
auto static const do_cmp  = mkn::kul::env::get_as("PLAMN_COMPARE", std::size_t{!USE_SERIALIZATION});
auto static const gen_bin = mkn::kul::env::get_as("PLAMN_GENERATE_PARTICLES_BIN", std::size_t{0});

std::string static const aos_particles_bin = "aos_particles.bin";

bool static const premain = []() {
    DAMN_WITH_MKN_GPU(                           //
        mkn::gpu::setLimitMallocHeapSize(bytes); //
    )
    DAMN_WITH_PHLOP( //
        DAMN_LOG_LINE_STR("cells: " << cells); DAMN_LOG_LINE_STR("ppc  : " << ppc);
        DAMN_LOG_LINE_STR("seed : " << seed);

        using namespace plamn; //
        using namespace std::literals;
        if (auto e = mkn::kul::env::get("PLAMN_SCOPE_TIMING", "false"); e == "1" || e == "true")
            phlop::threaded::ScopeTimerMan::INSTANCE()
                .file_name(".damn_times.0.txt")
                // .force_strings()
                // .headers("fn"s, "dim"s, "layout"s, "alloc"s, "storage"s, "time"s)
                .init(); //
    )
    return true;
}();

template<typename ParticlesSynchronisedMover_t>
auto construct_(auto& ions, auto const& em, auto const& layout)
{
    plamn::initializer::Dict dict;
    dict["simulation"]["algo"]["ion_updater"]["pusher"]["name"] = std::string{"modified_boris"};
    return ParticlesSynchronisedMover_t{dict["simulation"]["algo"]["ion_updater"]};
}

template<typename Ions, typename EM, typename Layout_t>
auto get_updater_for(Ions& ions, EM const& em, Layout_t const& layout)
{
    using Particles = typename Ions::particles_type;
    if constexpr (mkn::kul::any_in(Particles::layout_mode, LayoutMode::AoSMapped))
        return construct_<ParticlesSynchronisedMover<Ions, EM, Layout_t>>(ions, em, layout);
    else if constexpr (Particles::layout_mode == LayoutMode::AoSPC)
        return construct_<ParticlesSynchronisedMoverPC<Ions, EM, Layout_t>>(ions, em, layout);
    else
        return construct_<ParticlesSynchronisedMoverPP<Ions, EM, Layout_t>>(ions, em, layout);
}


namespace detail::strings
{
    constexpr static std::string_view update = "update,";
    constexpr static std::string_view cma    = ",";
} // namespace detail::strings


template<typename Layout_t>
void udpate(auto& ions, auto const& em, Layout_t const& layout)
{
    // no timings
    auto updater                       = get_updater_for(ions, em, layout);
    using ParticlesSynchronisedMover_t = std::decay_t<decltype(updater)>;
    using Boxing_t
        = std::decay_t<decltype(*selection_boxing_impl<ParticlesSynchronisedMover_t, Layout_t>())>;

    Boxing_t const boxing{layout,
                          damn::core::grow(layout.cells, Layout_t::options.particle_ghost_width)};
    updater.updatePopulations(ions, em, boxing, dt);
}

template<typename Ions>
void update(Ions& ions, auto const& em, auto const& layout)
{
    using Particles = typename Ions::particles_type;
    auto constexpr function_id
        = mkn::kul::join_string_views_v<detail::strings::update, Particles::type_id,
                                        detail::strings::cma>;
    DAMN_LOG_LINE_STR(function_id);
    DAMN_LOG_SCOPE(1, function_id);


    udpate(ions, em, layout);
}


template<typename Particles_t, typename Layout_t>
auto make_ions(Layout_t const& layout)
{
    auto constexpr static alloc_mode = Particles_t::alloc_mode;
    auto constexpr static dim        = Layout_t::dimension;
    auto constexpr static interp     = Layout_t::options.interp_order;

    auto ions_p = std::make_shared<UsableIons_t<Particles_t, interp>>(layout, "protons");
    auto& ions  = *ions_p;

    if constexpr (USE_SERIALIZATION)
        if (!gen_bin)
            return ions_p; // NO REF!

    EXPECT_EQ(ions.populations[0].particles.domain_particles.size(), 0);

    auto disperse = [](auto& particles) {
        delta_disperse(particles.domain_particles, seed);
        if constexpr (WITH_PATCH_GHOST)
            delta_disperse(particles.patch_ghost_particles, seed);
        if (shufle > 0)
        {
            shuffle(particles.domain_particles, seed);
            if constexpr (WITH_PATCH_GHOST)
                shuffle(particles.patch_ghost_particles, seed);
        }
    };

    auto add_particles = [&](auto& particles) {
        add_particles_in(particles.domain_particles, layout.cells, ppc);
        if constexpr (WITH_PATCH_GHOST)
            add_ghost_particles(particles.patch_ghost_particles, layout.cells, ppc,
                                Layout_t::options.particle_ghost_width);
    };

    add_particles(ions.populations[0].particles);
    disperse(ions.populations[0].particles);

    EXPECT_EQ(ions.populations[0].particles.domain_particles.size(), layout.cells.size() * ppc);

    if constexpr (USE_SERIALIZATION)
        if (gen_bin)
            serialize_particles(aos_particles_bin, ions.populations[0].particles.domain_particles);

    return ions_p;
}



template<typename Particles_t, typename Layout_t, typename Ions>
auto from_ions(Layout_t const& layout, Ions const& from)
{
    using FromParticles_t            = typename Ions::particles_type;
    auto constexpr static alloc_mode = Particles_t::alloc_mode;
    auto constexpr static dim        = Layout_t::dimension;
    auto constexpr static interp     = Layout::options.interp_order;
    auto ions_p = std::make_shared<UsableIons_t<Particles_t, interp>>(layout, "protons");
    auto& ions  = *ions_p;
    EXPECT_EQ(ions.populations[0].particles.domain_particles.size(), 0);

    auto _add_particles_from = [&]<auto type>(auto& src, auto& dst) {
        reserve(dst, layout.cells, ppc);
        append_particles<type>(src, dst);
    };

    if constexpr (USE_SERIALIZATION)
        deserialize_particles<Particles_t, FromParticles_t>(
            aos_particles_bin, ions.populations[0].particles.domain_particles);
    else
        _add_particles_from.template operator()<ParticleType::Domain>(
            from.populations[0].particles.domain_particles,
            ions.populations[0].particles.domain_particles);

    if constexpr (WITH_PATCH_GHOST)
        _add_particles_from.template operator()<ParticleType::Ghost>(
            from.populations[0].particles.patch_ghost_particles,
            ions.populations[0].particles.patch_ghost_particles);

    EXPECT_EQ(ions.populations[0].particles.domain_particles.size(), layout.cells.size() * ppc);
    return ions_p;
}

template<bool quiet = false, typename Ions, typename Layout_t>
auto& evolve(Ions& ions, Layout_t const& layout)
{
    using Particles_t                = typename Ions::particles_type;
    auto constexpr static alloc_mode = Particles_t::alloc_mode;
    auto constexpr static dim        = Layout_t::dimension;
    auto constexpr static interp     = Layout::options.interp_order;

    assert(ions.populations[0].particles.domain_particles.size() > 0);
    auto constexpr static sim_opts
        = SimOpts{.dimension = dim, .interp_order = interp, .alloc_mode = alloc_mode};
    using Hybrid_t                   = PLAMN_Types<sim_opts>::Hybrid;
    auto constexpr static field_opts = TensorFieldOptions<Hybrid_t>{};
    UsableElectromag<field_opts> em{layout};
    if constexpr (quiet)
        udpate(*ions, *em, layout);
    else
        update(*ions, *em, layout);
    assert(ions.populations[0].particles.domain_particles.size() > 0);
    return ions;
}


template<typename P0, typename P1>
void check_particles(auto const& layout, P0& ref, P1& cmp, std::size_t ghosts = 0)
{
    using CPU_ref = Particles<ParticlesOptions{P0::dimension, LayoutMode::AoS, StorageMode::VECTOR,
                                               damn::AllocatorMode::CPU}>;

    auto const box = damn::core::grow(layout.cells, ghosts);

    sort_particles(cmp, box);
    sort_particles(ref, box);

    EXPECT_EQ(ref.size(), cmp.size());

    auto const report = compare_particles(ref, cmp);
    if (!report)
    {
        DAMN_LOG_LINE_STR("Comparing Particle Arrays: " << P0::id() << " vs " << P1::id());
        DAMN_LOG_LINE_STR("results: " << report.why());
    }
    auto const tmp = convert_particles_and_sort<CPU_ref>(cmp, layout);
    EXPECT_EQ(ref, tmp);
    EXPECT_TRUE(report);
}

template<typename C>
void compare(auto const& layout, auto& ref, C& cmp)
{
    if (!do_cmp)
        return;

    using Particles_t = typename C::Particles_t;

    DAMN_LOG_LINE_STR(ref.populations[0].particles.domain_particles.begin().copy());
    DAMN_LOG_LINE_STR(cmp.populations[0].particles.domain_particles.begin().copy());

    check_particles(layout, ref.populations[0].particles.domain_particles,
                    cmp.populations[0].particles.domain_particles);

    if constexpr (WITH_PATCH_GHOST)
        check_particles(layout, ref.populations[0].particles.patch_ghost_particles,
                        cmp.populations[0].particles.patch_ghost_particles);
    using enum LayoutMode;
    using enum damn::AllocatorMode;

    double diff = 1e-13; // 0.0000000000000011102230
    if constexpr (Particles_t::alloc_mode == GPU_UNIFIED)
        diff *= 1e2; // atomics no order guaranteed#
    // else if constexpr (any_in(Particles_t::layout_mode, AoSTS, SoATS))
    //     diff *= 1e1; // p2m op order diff
    // else if constexpr (any_in(Particles_t::layout_mode, SoAVX))
    //     diff *= 1e1; // not sure really


    {
        auto const freport
            = compare_tensor_fields(*ref.populations[0].F, *cmp.populations[0].F, diff);
        // if (!freport)
        DAMN_LOG_LINE_STR("results: " << freport.why());
        EXPECT_TRUE(freport);
    }
    auto const freport = compare_fields(*ref.populations[0].rho, *cmp.populations[0].rho, diff);
    // if (!freport)
    DAMN_LOG_LINE_STR("results: " << freport.why());
    EXPECT_TRUE(freport);
}


template<std::size_t _dim, auto _layout_mode, auto _alloc_mode = plamn::AllocatorMode::CPU>
struct TestParam
{
    static_assert(std::is_same_v<decltype(_layout_mode), LayoutMode>);
    static_assert(std::is_same_v<decltype(_alloc_mode), plamn::AllocatorMode>);
    auto constexpr static dim         = _dim;
    auto constexpr static layout_mode = _layout_mode;
    auto constexpr static alloc_mode  = _alloc_mode;
};

template<typename Ions, std::size_t dim, std::size_t interp>
struct DefaultIons
{
    using Layout_t = TestLayout<typename PLAMN_Types<SimOpts{dim, interp}>::Hybrid::Layout_t>;

    Layout_t const layout{cells};
    std::shared_ptr<Ions> init; // = make_ions<typename Ions::particles_type>(layout);
    std::shared_ptr<Ions> ions; // = make_ions<typename Ions::particles_type>(layout);

    DefaultIons()
    {
        assert(premain);
        init = make_ions<typename Ions::particles_type>(layout);

        DAMN_LOG_LINE_SS("");
        if constexpr (!USE_SERIALIZATION)
        {
            DAMN_LOG_LINE_SS("");
            ions = make_ions<typename Ions::particles_type>(layout);
            evolve<true>(*ions, *layout);
        }
    }
};

template<std::size_t dim, std::size_t interp>
using DefaultIons_t = DefaultIons<UsableIons_t<ParticlesAoSMapped<dim>, interp>, dim, interp>;


// static inline auto refs = std::tuple<DefaultIons_t<1, 1>, int, int>{}; // only 1d
// static inline auto refs = std::tuple<int, DefaultIons_t<2, 1>, int>{};// only 2d
static inline auto refs = std::tuple<int, int, DefaultIons_t<3, 1>>{}; // only 3d

// static inline auto refs
//     = std::tuple<DefaultIons_t<1, 1>, DefaultIons_t<2, 1>, DefaultIons_t<3, 1>>{}; // all d


template<typename Param>
struct ParticlesSynchronisedMoverPPTest : public ::testing::Test
{
    auto constexpr static dim         = Param::dim;
    auto constexpr static interp      = 1;
    auto constexpr static layout_mode = Param::layout_mode;
    auto constexpr static alloc_mode  = Param::alloc_mode;


    using Layout_t       = TestLayout<typename PLAMN_Types<SimOpts{dim, interp}>::Hybrid::Layout_t>;
    using RefParticles_t = ParticlesAoSMapped<dim>;
    using CmpParticles_t
        = Particles<ParticlesOptions{dim, layout_mode, StorageMode::VECTOR, alloc_mode}>;

    using RefIons_t = UsableIons_t<RefParticles_t, interp>;
    using CmpIons_t = UsableIons_t<CmpParticles_t, interp>;
    using DefIons   = DefaultIons<RefIons_t, dim, interp>;

    DefIons& ref = std::get<dim - 1>(refs);
    Layout_t const layout{cells};


    ParticlesSynchronisedMoverPPTest() {}

    auto make_ions() const { return from_ions<CmpParticles_t>(layout, *ref.init); }
};

// clang-format off
using Permutations_t = testing::Types< // ! notice commas !
//      TestParam<1, LayoutMode::AoS>

// DAMN_WITH_THRUST(
//     ,TestParam<1, LayoutMode::SoA>
//     ,TestParam<1, LayoutMode::SoAVX>
// )

//     ,TestParam<2, LayoutMode::AoS>

// DAMN_WITH_THRUST(
//     ,TestParam<2, LayoutMode::SoA>
//     ,TestParam<2, LayoutMode::SoAVX>
// )

     TestParam<3, LayoutMode::AoS>
    ,TestParam<3, LayoutMode::AoSMapped>

DAMN_WITH_MKN_AVX(
    // ,TestParam<3, LayoutMode::SoAVX>
)


DAMN_WITH_THRUST(
    // ,TestParam<3, LayoutMode::SoA>
    // ,TestParam<3, LayoutMode::SoAVX>
)


DAMN_WITH_MKN_GPU(
    // ,TestParam<1, LayoutMode::AoS, AllocatorMode::GPU_UNIFIED>   // 3
    // ,TestParam<1, LayoutMode::AoSPC, AllocatorMode::GPU_UNIFIED> // 4
    // ,TestParam<1, LayoutMode::AoSPC, AllocatorMode::GPU_UNIFIED,  /*impl=*/1> // 5
    // ,TestParam<1, LayoutMode::AoSPC, AllocatorMode::GPU_UNIFIED,  /*impl=*/2>
    // ,TestParam<1, LayoutMode::SoA, AllocatorMode::GPU_UNIFIED>
)
    // ,TestParam<2, LayoutMode::AoS>
    // ,TestParam<2, LayoutMode::AoSPC>
DAMN_WITH_MKN_GPU(
    // ,TestParam<2, LayoutMode::SoA>
    // ,TestParam<2, LayoutMode::AoS, AllocatorMode::GPU_UNIFIED>
    // ,TestParam<2, LayoutMode::AoSPC, AllocatorMode::GPU_UNIFIED>
    // ,TestParam<2, LayoutMode::AoSPC, AllocatorMode::GPU_UNIFIED, /*impl=*/1> // 13
    // ,TestParam<2, LayoutMode::AoSPC, AllocatorMode::GPU_UNIFIED, /*impl=*/2> // 14
    // ,TestParam<2, LayoutMode::SoA, AllocatorMode::GPU_UNIFIED>
)
    // /*,*/TestParam<3, LayoutMode::AoS>
    // ,TestParam<3, LayoutMode::AoSPC>
DAMN_WITH_THRUST(
    // ,TestParam<3, LayoutMode::SoA>
)

DAMN_WITH_MKN_GPU(
    ,TestParam<3, LayoutMode::AoS, damn::AllocatorMode::GPU_UNIFIED>

//     // ,TestParam<3, LayoutMode::AoSPC, AllocatorMode::GPU_UNIFIED>
//     // ,TestParam<3, LayoutMode::AoSPC, AllocatorMode::GPU_UNIFIED, /*impl=*/1> //
//     // ,TestParam<3, LayoutMode::AoSPC, AllocatorMode::GPU_UNIFIED, /*impl=*/2> // 22
// DAMN_WITH_THRUST(
//     ,TestParam<3, LayoutMode::SoA, AllocatorMode::GPU_UNIFIED>
// )
)

>;
// clang-format on

TYPED_TEST_SUITE(ParticlesSynchronisedMoverPPTest, Permutations_t);


TYPED_TEST(ParticlesSynchronisedMoverPPTest, updater)
{
    if constexpr (plamn::core::USE_SERIALIZATION)
        if (gen_bin)
            return;

    auto cmp_ions = this->make_ions();

    // if (do_cmp)
    // {
    //     DAMN_LOG_LINE_SS("Init check!")
    //     compare(*this->layout, *this->ref.init, *cmp_ions);
    // }

    compare(*this->layout,   //
            *this->ref.ions, //
            evolve(*cmp_ions, *this->layout));
}


} // namespace plamn::core


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    auto r = RUN_ALL_TESTS();
    DAMN_WITH_PHLOP(phlop::threaded::ScopeTimerMan::reset());
    return r;
}
