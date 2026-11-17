#include "plamn/core/def.hpp"

#include "plamn/plamn_core.hpp"
#include "plamn/simulator/simulator_def.hpp"

#include "damn/core/math/array.hpp"
#include "damn/core/math/numeric.hpp"
#include "plamn/core/numerics/particles_synchronised_mover/particles_synchronised_mover.hpp"
#include "plamn/core/models/electromag_initializer.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/numerics/interpolator/interpolating.hpp"
#include "plamn/core/numerics/interpolator/interpolating.hpp"
#include "plamn/core/data/particles/appending/particles_appending.hpp"
#include "plamn/core/data/particles/converting/particles_converting.hpp"

#include "plamn/simulator/simulator_def.hpp"


#include "tst/plamn/core/data/vecfield/test_vecfield_fixtures.hpp"
#include "tst/plamn/core/data/electromag/test_electromag_fixtures.hpp"
#include "tst/plamn/core/data/tensorfield/test_tensorfield_fixtures.hpp"
#include "tst/plamn/core/data/ion_population/test_ion_population_fixtures.hpp"
#include "tst/plamn/amr/resources_manager/test_resources_manager_fixtures.hpp"

#include "gtest/gtest.h"

#include <numeric>
#include <algorithm>
#include <type_traits>
#include <unordered_map>


namespace plamn::core
{

using Param  = damn::core::Span<double const> const&;
using Return = std::shared_ptr<damn::core::Span<double>>;

Return density(Param x)
{
    return std::make_shared<damn::core::VectorSpan<double>>(x.size(), 1);
}

Return vx(Param x)
{
    return std::make_shared<damn::core::VectorSpan<double>>(x.size(), 0);
}

Return vy(Param x)
{
    return std::make_shared<damn::core::VectorSpan<double>>(x.size(), 0);
}

Return vz(Param x)
{
    return std::make_shared<damn::core::VectorSpan<double>>(x.size(), 0);
}

Return vthx(Param x)
{
    return std::make_shared<damn::core::VectorSpan<double>>(x.size(), .1);
}

Return vthy(Param x)
{
    return std::make_shared<damn::core::VectorSpan<double>>(x.size(), .1);
}

Return vthz(Param x)
{
    return std::make_shared<damn::core::VectorSpan<double>>(x.size(), .1);
}

Return bx(Param x)
{
    return std::make_shared<damn::core::VectorSpan<double>>(x.size(), 0);
}

Return by(Param x)
{
    return std::make_shared<damn::core::VectorSpan<double>>(x.size(), 0);
}

Return bz(Param x)
{
    return std::make_shared<damn::core::VectorSpan<double>>(x.size(), 0);
}



std::size_t nbrPartPerCell = 1000;

using InitFunctionT = plamn::initializer::InitFunction<1>;

plamn::initializer::Dict createDict()
{
    plamn::initializer::Dict dict;

    dict["simulation"]["algo"]["ion_updater"]["pusher"]["name"] = std::string{"modified_boris"};

    dict["ions"]["nbrPopulations"]                          = std::size_t{2};
    dict["ions"]["pop0"]["name"]                            = std::string{"protons"};
    dict["ions"]["pop0"]["mass"]                            = 1.;
    dict["ions"]["pop0"]["particle_initializer"]["name"]    = std::string{"maxwellian"};
    dict["ions"]["pop0"]["particle_initializer"]["density"] = static_cast<InitFunctionT>(density);

    dict["ions"]["pop0"]["particle_initializer"]["bulk_velocity_x"]
        = static_cast<InitFunctionT>(vx);

    dict["ions"]["pop0"]["particle_initializer"]["bulk_velocity_y"]
        = static_cast<InitFunctionT>(vy);

    dict["ions"]["pop0"]["particle_initializer"]["bulk_velocity_z"]
        = static_cast<InitFunctionT>(vz);


    dict["ions"]["pop0"]["particle_initializer"]["thermal_velocity_x"]
        = static_cast<InitFunctionT>(vthx);

    dict["ions"]["pop0"]["particle_initializer"]["thermal_velocity_y"]
        = static_cast<InitFunctionT>(vthy);

    dict["ions"]["pop0"]["particle_initializer"]["thermal_velocity_z"]
        = static_cast<InitFunctionT>(vthz);


    dict["ions"]["pop0"]["particle_initializer"]["nbr_part_per_cell"]
        = static_cast<int>(nbrPartPerCell);
    dict["ions"]["pop0"]["particle_initializer"]["charge"] = 1.;
    dict["ions"]["pop0"]["particle_initializer"]["basis"]  = std::string{"cartesian"};

    dict["ions"]["pop1"]["name"]                            = std::string{"alpha"};
    dict["ions"]["pop1"]["mass"]                            = 1.;
    dict["ions"]["pop1"]["particle_initializer"]["name"]    = std::string{"maxwellian"};
    dict["ions"]["pop1"]["particle_initializer"]["density"] = static_cast<InitFunctionT>(density);

    dict["ions"]["pop1"]["particle_initializer"]["bulk_velocity_x"]
        = static_cast<InitFunctionT>(vx);

    dict["ions"]["pop1"]["particle_initializer"]["bulk_velocity_y"]
        = static_cast<InitFunctionT>(vy);

    dict["ions"]["pop1"]["particle_initializer"]["bulk_velocity_z"]
        = static_cast<InitFunctionT>(vz);


    dict["ions"]["pop1"]["particle_initializer"]["thermal_velocity_x"]
        = static_cast<InitFunctionT>(vthx);

    dict["ions"]["pop1"]["particle_initializer"]["thermal_velocity_y"]
        = static_cast<InitFunctionT>(vthy);

    dict["ions"]["pop1"]["particle_initializer"]["thermal_velocity_z"]
        = static_cast<InitFunctionT>(vthz);


    dict["ions"]["pop1"]["particle_initializer"]["nbr_part_per_cell"]
        = static_cast<int>(nbrPartPerCell);
    dict["ions"]["pop1"]["particle_initializer"]["charge"] = 1.;
    dict["ions"]["pop1"]["particle_initializer"]["basis"]  = std::string{"cartesian"};

    dict["electromag"]["name"]             = std::string{"EM"};
    dict["electromag"]["electric"]["name"] = std::string{"E"};
    dict["electromag"]["magnetic"]["name"] = std::string{"B"};

    dict["electromag"]["magnetic"]["initializer"]["x_component"] = static_cast<InitFunctionT>(bx);
    dict["electromag"]["magnetic"]["initializer"]["y_component"] = static_cast<InitFunctionT>(by);
    dict["electromag"]["magnetic"]["initializer"]["z_component"] = static_cast<InitFunctionT>(bz);

    return dict;
}
static auto init_dict = createDict();




template<auto opts_>
struct TestParam
{
    auto static constexpr opts         = opts_;
    static constexpr auto dimension    = opts.dimension;
    static constexpr auto interp_order = opts.interp_order;
};

template<auto layout_mode, auto opts>
struct updater_test_bits;

template<typename TestParam_t>
struct ParticlesSynchronisedMoverTest : public ::testing::Test
{
    auto static constexpr opts         = TestParam_t::opts;
    static constexpr auto dim          = opts.dimension;
    static constexpr auto interp_order = opts.interp_order;
    using DAMNTypes                    = plamn::core::PLAMN_Types<opts>;
    using Ions                         = DAMNTypes::Hybrid::Ions_t;
    using Electromag                   = DAMNTypes::Hybrid::Electromag_t;
    using Layout                       = DAMNTypes::Hybrid::Layout_t;
    using Particles                    = DAMNTypes::Hybrid::Particles_t;
    using ParticleInitializerFactory   = DAMNTypes::Hybrid::ParticleInitializerFactory_t;

    using ParticlesSynchronisedMover_t
        = std::conditional_t<Particles::layout_mode == LayoutMode::AoSMapped,
                             core::ParticlesSynchronisedMover<Particles, Layout>,
                             core::ParallelParticlesSynchronisedMover<Particles, Layout>>;
    using Boxing_t     = ParticlesSynchronisedMover_t::Boxing_t;
    using Interpolator = Interpolating<dim, interp_order>;

    using Hybrid_t                   = DAMNTypes::Hybrid;
    auto constexpr static field_opts = TensorFieldOptions<Hybrid_t>{};
    using UsableElectromag_t         = UsableElectromag<field_opts>;

    Interpolator interpolator;

    double const dt{0.01};

    // grid configuration
    Layout const layout{damn::core::Box<int, 1>{{0}, {99}}, {0.1}}; // 100 cells


    Boxing_t const boxing{layout, {layout.cells}};
    std::unordered_map<std::string, Boxing_t> const levelBoxing{{"patch_id", boxing}};

    static_assert(std::is_same_v<Electromag, typename UsableElectromag_t::Super>);
    UsableElectromag_t EM{layout, init_dict["electromag"]};

    UsableIons_t<opts> ions{layout, init_dict["ions"]};

    ParticlesSynchronisedMoverTest()
    {
        // ok all resources pointers are set to buffers
        // now let's initialize Electromag fields to user input functions
        // and ion population particles to user supplied moments

        assert(no_nans(ions.velocity()(Component::X)));
        ElectromagInitializer<dim>{init_dict["electromag"]}(*EM, layout);
        for (auto& pop : ions)
        {
            auto const& info         = pop.particleInitializerInfo();
            auto particleInitializer = ParticleInitializerFactory::create(info, pop.name());
            particleInitializer->loadParticles(pop.domainParticles(), layout);
            EXPECT_GT(pop.domainParticles().size(), 0ull);
            // particles_domain_is_valid(pop.domainParticles(), layout.cells);
        }
        assert(no_nans(ions.velocity()(Component::X)));

        // now all domain particles are loaded we need to manually insert
        // ghost particles (this is in reality SAMRAI's job)
        // these are needed if we want all used nodes to be complete


        // in 1D we assume left border is touching the level border
        // and right is touching another patch
        // so on the left no patchGhost but levelGhost(and old and new)
        // on the right no levelGhost but patchGhosts

        for (auto& pop : ions)
        {
            if constexpr (dim == 1)
            {
                auto const cells  = domain_box(layout, dual_centering<dim>());
                int firstPhysCell = cells.lower[0];
                int lastPhysCell  = cells.upper[0];
                auto firstAMRCell = to_amr(layout, damn::core::Point{firstPhysCell});
                auto lastAMRCell  = to_amr(layout, damn::core::Point{lastPhysCell});

                // we need to put levelGhost particles in the cell just to the
                // left of the first cell. In reality these particles should
                // come from splitting particles of the next coarser level
                // in this test we just copy those of the first cell
                // we also assume levelGhostOld and New are the same particles
                // for simplicity

                auto& domainPart        = pop.domainParticles();
                auto& levelGhostPartOld = pop.levelGhostParticlesOld();
                auto& levelGhostPartNew = pop.levelGhostParticlesNew();
                auto& levelGhostPart    = pop.levelGhostParticles();
                auto& patchGhostPart    = pop.patchGhostParticles();



                // copies need to be put in the ghost cell
                // we have copied particles be now their iCell needs to be udpated
                // our choice is :
                //
                // first order:
                //
                //   ghost| domain...
                // [-----]|[-----][-----][-----][-----][-----]
                //     ^      v
                //     |      |
                //     -------|
                //
                // second and third order:

                //   ghost        | domain...
                // [-----]|[-----][-----][-----][-----][-----][-----]
                //     ^      ^       v     v
                //     |      |       |     |
                //     -------|-------|     |
                //            ---------------

                // per_particle() dispatches per layout (flat for AoS/AoSMapped, per tile for
                // AoSCMTS), so no manual tile loop is needed here
                // the first particle_ghost_width domain cells, copied into the as many ghost
                // cells left of the domain
                auto constexpr pgw = static_cast<int>(Layout::options.particle_ghost_width);
                per_particle(domainPart, [&](auto const& part) {
                    if (part.iCell()[0] >= firstAMRCell[0]
                        and part.iCell()[0] < firstAMRCell[0] + pgw)
                    {
                        auto p{part};
                        p.iCell()[0] -= pgw;
                        levelGhostPartOld.push_back(p);
                    }
                });

                levelGhostPartOld.template on_appended<ParticleType::Ghost>();
                levelGhostPartNew = levelGhostPartOld;
                levelGhostPart    = levelGhostPartOld;

                EXPECT_EQ(pop.domainParticles().size(), 100 * nbrPartPerCell);
                EXPECT_EQ(levelGhostPartOld.size(),
                          Layout::options.particle_ghost_width * nbrPartPerCell);
                EXPECT_EQ(patchGhostPart.size(), 0);


            } // end 1D
        } // end pop loop

        assert(no_nans(ions.velocity()(Component::X)));
        plamn::core::depositParticles(ions, layout, interpolator, plamn::core::DomainDeposit{});
        plamn::core::depositParticles(ions, layout, interpolator, plamn::core::LevelGhostDeposit{});
        sumTileBorders();
        assert(no_nans(ions.velocity()(Component::X)));

        ions.computeChargeDensity();
        assert(no_nans(ions.velocity()(Component::X)));
        ions.computeBulkVelocity();
        assert(no_nans(ions.velocity()(Component::X)));
    } // end Ctor



    using Patch = test::HybridPatch<Ions, UsableElectromag_t>;


    auto update(auto const mode)
    {
        assert(no_nans(ions.velocity()(Component::X)));
        ParticlesSynchronisedMover_t ionUpdater{};

        if constexpr (Particles::layout_mode == LayoutMode::AoSMapped)
            ionUpdater.updatePopulations(*ions, *EM, boxing, dt, mode);
        else
        {
            // state = {layout, ions, em, electromag}: em is an unused owner (UsableElectromag
            // is move-only, hence emplace), electromag views this->EM
            std::vector<Patch> patches;
            patches.reserve(1);
            patches.emplace_back(layout, *ions, layout, *EM);

            auto const quantities = [&](int i) {
                return std::forward_as_tuple(patches[0].model.state.ions,
                                             patches[i].model.state.electromag);
            };

            assert(ions.massDensity().data() == patches[0].model.state.ions.massDensity().data());
            auto accessor = test::make_model_level_accessor(patches, quantities);
            ionUpdater.updatePopulations(accessor, levelBoxing, dt, mode);
        }
        assert(no_nans(ions.velocity()(Component::X)));
    }

    void fillIonsMomentsGhosts()
    {
        for (auto& pop : this->ions)
        {
            double alpha = 0.5;
            interpolator.template particleToMesh<ParticleType::LevelGhost>(
                pop.levelGhostParticlesNew(), layout, pop.particleDensity(), pop.flux(),
                /*coef = */ alpha);

            interpolator.template particleToMesh<ParticleType::LevelGhost>(
                pop.levelGhostParticlesOld(), layout, pop.particleDensity(), pop.flux(),
                /*coef = */ (1. - alpha));
        }

        sumTileBorders();
    }

    // tiles hold only their own partial deposits on shared border/ghost nodes; in a run the
    // messenger's border sum completes them. no messenger here: reduce (sums every tile's
    // ghost box) and copy the totals back so per-tile ions.update() sees full moments.
    // tiles are then complete, so checks read them with reduce_single (copy, no sum)
    void sumTileBorders()
    {
        if constexpr (is_field_tile_set_v<std::decay_t<decltype(ions.massDensity())>>)
        {
            auto const sum = [](auto& tiles) { copy_fields(tiles, *reduce(tiles)); };

            for (auto& pop : this->ions)
            {
                sum(pop.particleDensity()); // pop charge density is q * n, see Ions
                for (auto& component : pop.flux())
                    sum(component);
            }
        }
    }



    void checkMomentsHaveEvolved(auto const& ionsBufferCpy)
    {
        auto& populations = this->ions.getRunTimeResourcesViewList();

        auto const& protonParticleDensity = reduce_single(populations[0].particleDensity());
        auto const& protonChargeDensity   = reduce_single(populations[0].chargeDensity());
        auto const& protonFx              = reduce_single(populations[0].flux()(Component::X));
        auto const& protonFy              = reduce_single(populations[0].flux()(Component::Y));
        auto const& protonFz              = reduce_single(populations[0].flux()(Component::Z));

        auto const& alphaParticleDensity = reduce_single(populations[1].particleDensity());
        auto const& alphaChargeDensity   = reduce_single(populations[1].chargeDensity());
        auto const& alphaFx              = reduce_single(populations[1].flux()(Component::X));
        auto const& alphaFy              = reduce_single(populations[1].flux()(Component::Y));
        auto const& alphaFz              = reduce_single(populations[1].flux()(Component::Z));


        auto const nodes = domain_box(this->layout, primal_centering<dim>());
        auto const ix0   = nodes.lower[0];
        auto const ix1   = nodes.upper[0];

        auto nonZero = [&](auto const& field, std::string const type) {
            auto sum = 0.;
            for (auto ix = ix0; ix <= ix1; ++ix)
            {
                sum += std::abs(field(ix));
            }
            EXPECT_GT(sum, 0.);
            if (sum == 0)
                std::cout << "nonZero failed for " << type << " : " << field.name() << "\n";
        };

        auto check = [&](auto const& newField, auto const& og) {
            // assert(no_nans(newField));
            assert(no_nans(og));
            auto const& originalField = reduce_single(og);
            nonZero(newField, newField.name() + ":new");
            // nonZero(originalField, "originalField");
            for (auto ix = ix0; ix <= ix1; ++ix)
            {
                auto evolution = std::abs(newField(ix) - originalField(ix));
                //  should check that moments are still compatible with user inputs also
                EXPECT_GT(evolution, 0.0);
                if (evolution <= 0.0)
                    std::cout << newField.name() + ": after update : " << newField(ix)
                              << " before update : " << originalField(ix)
                              << " evolution : " << evolution << " ix : " << ix << "\n";
            }
        };


        check(protonParticleDensity, ionsBufferCpy[0].particleDensity());
        check(protonChargeDensity, ionsBufferCpy[0].chargeDensity());
        check(protonFx, ionsBufferCpy[0].flux()(Component::X));
        check(protonFy, ionsBufferCpy[0].flux()(Component::Y));
        check(protonFz, ionsBufferCpy[0].flux()(Component::Z));

        check(alphaParticleDensity, ionsBufferCpy[1].particleDensity());
        check(alphaChargeDensity, ionsBufferCpy[1].chargeDensity());
        check(alphaFx, ionsBufferCpy[1].flux()(Component::X));
        check(alphaFy, ionsBufferCpy[1].flux()(Component::Y));
        check(alphaFz, ionsBufferCpy[1].flux()(Component::Z));

        assert(no_nans(ionsBufferCpy.velocity()(Component::X)));
        check(reduce_single(ions.velocity()(Component::X)), ionsBufferCpy.velocity()(Component::X));
        check(reduce_single(ions.velocity()(Component::Y)), ionsBufferCpy.velocity()(Component::Y));
        check(reduce_single(ions.velocity()(Component::Z)), ionsBufferCpy.velocity()(Component::Z));
    }



    void checkDensityIsAsPrescribed()
    {
        auto const nodes = domain_box(this->layout, primal_centering<dim>());
        auto const ix0   = nodes.lower[0];
        auto const ix1   = nodes.upper[0];

        auto check = [&](auto const& density, auto const& function) {
            std::vector<std::size_t> ixes;
            std::vector<double> x;

            // We do not use the primal box as the last primal is incomplete.
            // This is because level ghosts are only interpolated if they enter the domain
            for_each_amr_local(this->layout, this->layout.cells,
                               [&](auto const& amr_idx, auto const& lcl_idx) {
                                   ixes.emplace_back(lcl_idx[0]);
                                   x.emplace_back(cell_center(layout, amr_idx)[0]);
                               });

            auto const span   = damn::core::make_const_span(x);
            auto functionXPtr = function(span); // keep alive
            EXPECT_EQ(functionXPtr->size(), (ix1 - ix0));

            auto& functionX = *functionXPtr;

            for (std::size_t i = 0; i < functionX.size(); i++)
            {
                auto ix   = ixes[i];
                auto diff = std::abs(density(ix) - functionX[i]);

                EXPECT_GE(0.07, diff);

                if (diff >= 0.07)
                    std::cout << i << " actual : " << density(ix)
                              << " prescribed : " << functionX[i] << " diff : " << diff
                              << " ix : " << ix << " max: " << functionX.size() << "\n";
            }
        };

        auto& populations           = this->ions.getRunTimeResourcesViewList();
        auto& protonParticleDensity = populations[0].particleDensity();
        auto& alphaParticleDensity  = populations[1].particleDensity();

        check(reduce_single(protonParticleDensity), density);
        check(reduce_single(alphaParticleDensity), density);
    }
};


// clang-format off
using Permutations = ::testing::Types<
    TestParam<SimOpts{1, 1}>
  , TestParam<SimOpts{1, 2}>
  , TestParam<SimOpts{1, 3}>
  , TestParam<SimOpts{.dimension = 1, .interp_order = 1, .layout_mode = LayoutMode::AoSPCTS}>
  , TestParam<SimOpts{.dimension = 1, .interp_order = 2, .layout_mode = LayoutMode::AoSPCTS}>
  , TestParam<SimOpts{.dimension = 1, .interp_order = 3, .layout_mode = LayoutMode::AoSPCTS}>
>;
// clang-format on
TYPED_TEST_SUITE(ParticlesSynchronisedMoverTest, Permutations, );




TYPED_TEST(ParticlesSynchronisedMoverTest, ionUpdaterTakesPusherParamsFromDictAtConstruction)
{
    typename ParticlesSynchronisedMoverTest<TypeParam>::ParticlesSynchronisedMover_t ionUpdater{};
}


// the following 3 tests are testing the fixture is well configured.


TYPED_TEST(ParticlesSynchronisedMoverTest, loadsDomainPatchAndLevelGhostParticles)
{
    auto check = [this](std::size_t nbrGhostCells, auto& pop) {
        EXPECT_EQ(cell_count(this->layout)[0] * nbrPartPerCell, pop.domainParticles().size());
        EXPECT_EQ(0, pop.patchGhostParticles().size());
        EXPECT_EQ(nbrGhostCells * nbrPartPerCell, pop.levelGhostParticlesOld().size());
        EXPECT_EQ(nbrGhostCells * nbrPartPerCell, pop.levelGhostParticlesNew().size());
        EXPECT_EQ(nbrGhostCells * nbrPartPerCell, pop.levelGhostParticles().size());
    };


    if constexpr (TypeParam::dimension == 1)
    {
        for (auto& pop : this->ions)
        {
            check(TestFixture::Layout::options.particle_ghost_width, pop);
        }
    }
}




TYPED_TEST(ParticlesSynchronisedMoverTest, loadsLevelGhostParticlesOnLeftGhostArea)
{
    int firstPhysCell = domain_box(this->layout, dual_centering<TypeParam::dimension>()).lower[0];
    auto firstAMRCell = to_amr(this->layout, damn::core::Point{firstPhysCell});

    if constexpr (TypeParam::dimension == 1)
    {
        for (auto& pop : this->ions)
        {
            auto const ghost = convert_to<LayoutMode::AoS>(pop.levelGhostParticles(), this->layout);

            // particle_ghost_width ghost cells left of the domain, nbrPartPerCell in each
            auto constexpr pgw
                = static_cast<int>(TestFixture::Layout::options.particle_ghost_width);
            for (int c = 1; c <= pgw; ++c)
                EXPECT_EQ(nbrPartPerCell, std::count_if(std::begin(ghost), std::end(ghost),
                                                        [&](auto const& particle) {
                                                            return particle.iCell()[0]
                                                                   == firstAMRCell[0] - c;
                                                        }));
            EXPECT_EQ(pgw * nbrPartPerCell, std::distance(std::begin(ghost), std::end(ghost)));
        }
    }
}




// start of DAMN TESTS



TYPED_TEST(ParticlesSynchronisedMoverTest, particlesUntouchedInMomentOnlyMode)
{
    typename ParticlesSynchronisedMoverTest<TypeParam>::ParticlesSynchronisedMover_t ionUpdater{};

    auto ionsBufferCpy = this->ions;

    this->update(UpdaterMode::domain_only);

    this->fillIonsMomentsGhosts();

    this->ions.update();

    auto& populations = this->ions.getRunTimeResourcesViewList();

    auto checkIsUnTouched = [&](auto const& og, auto const& cp) {
        auto const original = convert_to<LayoutMode::AoS>(og, this->layout);
        auto const cpy      = convert_to<LayoutMode::AoS>(cp, this->layout);

        // no particles should have moved, so none should have left the domain
        EXPECT_EQ(cpy.size(), original.size());
        for (std::size_t iPart = 0; iPart < original.size(); ++iPart)
        {
            EXPECT_EQ(cpy.iCell(iPart)[0], original.iCell(iPart)[0]);

            EXPECT_DOUBLE_EQ(cpy.delta(iPart)[0], original.delta(iPart)[0]);
            for (std::size_t iDir = 0; iDir < 3; ++iDir)
            {
                EXPECT_DOUBLE_EQ(cpy.v(iPart)[iDir], original.v(iPart)[iDir]);
            }
        }
    };

    auto& cpy_pops = ionsBufferCpy.getRunTimeResourcesViewList();

    checkIsUnTouched(populations[0].levelGhostParticles(), cpy_pops[0].levelGhostParticles());
    checkIsUnTouched(populations[0].levelGhostParticlesOld(), cpy_pops[0].levelGhostParticlesOld());
    checkIsUnTouched(populations[0].levelGhostParticlesNew(), cpy_pops[0].levelGhostParticlesNew());

    checkIsUnTouched(populations[1].levelGhostParticles(), cpy_pops[1].levelGhostParticles());
    checkIsUnTouched(populations[1].levelGhostParticlesOld(), cpy_pops[1].levelGhostParticlesOld());
    checkIsUnTouched(populations[1].levelGhostParticlesNew(), cpy_pops[1].levelGhostParticlesNew());
}




// TYPED_TEST(ParticlesSynchronisedMoverTest, particlesAreChangedInParticlesAndMomentsMode)
//{
//    typename ParticlesSynchronisedMoverTest<TypeParam>::ParticlesSynchronisedMover_t
//    ionUpdater{init_dict["simulation"]["pusher"]};
//
//    IonsBuffers ionsBufferCpy{this->ionsBuffers, this->layout};
//
//    ionUpdater.updatePopulations(this->ions, this->EM, this->boxing, this->dt,
//                                 UpdaterMode::particles_and_moments);
//
//    this->fillIonsMomentsGhosts();
//
//    this->ions.update();
//
//    auto& populations = this->ions.getRunTimeResourcesViewList();
//
//    EXPECT_NE(ionsBufferCpy.protonDomain.size(), populations[0].domainParticles().size());
//    EXPECT_NE(ionsBufferCpy.alphaDomain.size(), populations[1].domainParticles().size());
//
//    // cannot think of anything else to check than checking that the number of particles
//    // in the domain have changed after them having been pushed.
//}



TYPED_TEST(ParticlesSynchronisedMoverTest, momentsAreChangedInParticlesAndMomentsMode)
{
    typename ParticlesSynchronisedMoverTest<TypeParam>::ParticlesSynchronisedMover_t ionUpdater{};

    auto ionsBufferCpy = this->ions;

    assert(ionsBufferCpy.massDensity().data() != this->ions.massDensity().data());

    assert(no_nans(this->ions.velocity()(Component::X)));
    this->update(UpdaterMode::all);
    // mode all moves only: the domain deposit follows (after the exchange in a run)
    ParticlesSynchronisedMoverTest<TypeParam>::ParticlesSynchronisedMover_t::depositDomain(
        this->ions, this->layout);
    assert(no_nans(this->ions.velocity()(Component::X)));

    this->fillIonsMomentsGhosts();
    assert(no_nans(this->ions.velocity()(Component::X)));

    this->ions.update();

    // assert(no_nans(this->ions.velocity()(Component::X)));

    this->checkMomentsHaveEvolved(ionsBufferCpy);
    this->checkDensityIsAsPrescribed();
}




TYPED_TEST(ParticlesSynchronisedMoverTest, momentsAreChangedInMomentsOnlyMode)
{
    typename ParticlesSynchronisedMoverTest<TypeParam>::ParticlesSynchronisedMover_t ionUpdater{};

    auto ionsBufferCpy = this->ions;

    assert(ionsBufferCpy.massDensity().data() != this->ions.massDensity().data());

    this->update(UpdaterMode::domain_only);

    this->fillIonsMomentsGhosts();

    this->ions.update();

    this->checkMomentsHaveEvolved(ionsBufferCpy);
    this->checkDensityIsAsPrescribed();
}



TYPED_TEST(ParticlesSynchronisedMoverTest, thatNoNaNsExistOnPhysicalNodesMoments)
{
    typename ParticlesSynchronisedMoverTest<TypeParam>::ParticlesSynchronisedMover_t ionUpdater{};

    this->update(UpdaterMode::domain_only);

    this->fillIonsMomentsGhosts();

    this->ions.update();

    auto const nodes = domain_box(this->layout, primal_centering<TypeParam::dimension>());
    auto const ix0   = nodes.lower[0];
    auto const ix1   = nodes.upper[0];

    for (auto& pop : this->ions)
    {
        for (auto ix = ix0; ix <= ix1; ++ix)
        {
            auto const& density = reduce_single(pop.particleDensity());
            auto const& flux    = pop.flux();

            auto const& fx = reduce_single(flux(Component::X));
            auto const& fy = reduce_single(flux(Component::Y));
            auto const& fz = reduce_single(flux(Component::Z));

            EXPECT_FALSE(std::isnan(density(ix)));
            EXPECT_FALSE(std::isnan(fx(ix)));
            EXPECT_FALSE(std::isnan(fy(ix)));
            EXPECT_FALSE(std::isnan(fz(ix)));
        }
    }
}


template<std::size_t dim>
void pseudo_randomize_deltas(auto& parts)
{
    std::size_t k = 0;

    per_particle(parts, [&](Particle<dim>& p) {
        for (std::uint16_t i = 0; i < dim; ++i)
        {
            double dir = k % 2 == 0 ? -1 : 1;
            p.delta()[i] += dir * .03 * k;
            assert(p.delta()[i] > 0 and p.delta()[i] < 1);
            if (k == 9)
                k = 0;
            ++k;
        }
    });
}

template<typename Layout>
void populate_particles(auto& ions, Layout& layout)
{
    auto constexpr dim = Layout::dimension;
    Particle<dim> const particle{1.0 / nbrPartPerCell, damn::core::ConstArray<int, dim>(),
                                 damn::core::ConstArray<double, dim>(.5),
                                 damn::core::ConstArray<double, 3>(1)};
    auto const ghost_box = damn::core::grow(layout.cells, 1);

    auto const shift_particle = [](auto p, auto const bix) {
        p.iCell() = bix;
        return p;
    };

    for (auto& pop : ions)
    {
        pop.domainParticles().clear();
        pop.levelGhostParticles().clear();
        pop.patchGhostParticles().clear();

        pop.domainParticles().check();
        pop.levelGhostParticles().check();

        damn::core::for_each_point(layout.cells, [&](auto const& bix) {
            for (std::size_t i = 0; i < nbrPartPerCell; ++i)
                pop.domainParticles().emplace_back(shift_particle(particle, *bix));
        });

        damn::core::for_each_point(ghost_box, [&](auto const& bix) {
            if (not isIn(bix, layout.cells))
                for (std::size_t i = 0; i < nbrPartPerCell; ++i)
                    pop.levelGhostParticles().emplace_back(shift_particle(particle, *bix));
        });

        per_particle(pop.levelGhostParticles(),
                     [&](Particle<dim>& p) { EXPECT_TRUE(not isIn(p, layout.cells)); });

        pseudo_randomize_deltas<dim>(pop.domainParticles());
        pseudo_randomize_deltas<dim>(pop.levelGhostParticles());

        pop.domainParticles().check();
        pop.levelGhostParticles().check();
    }
}




// Overlapping same-level patches: cells in `nonOwnedBoxes` belong to another patch.
// Particles move only along x, at a fixed fraction of a cell per step (E = B = 0).
template<typename TestParam_t>
struct ParticlesSynchronisedMoverOwnershipTest : public ParticlesSynchronisedMoverTest<TestParam_t>
{
    using Super    = ParticlesSynchronisedMoverTest<TestParam_t>;
    using Boxing_t = Super::Boxing_t;
    using Box_t    = Boxing_t::Box_t;

    static constexpr double cellsPerStep = 0.5;

    ParticlesSynchronisedMoverOwnershipTest()
    {
        (*this->EM).E.zero();
        (*this->EM).B.zero();
    }

    // the patch alone on its level: its ghost cells are level ghost cells, where its level
    //  ghosts are (a level ghost lives in one patch, in cells no patch of the level covers)
    Boxing_t makeBoxing(std::vector<Box_t> const& nonOwned) const
    {
        return {this->layout, {this->layout.cells}, nonOwned};
    }

    static Box_t box(int lower, int upper) { return Box_t{{lower}, {upper}}; }

    void setVelocityX(auto& particles, double const v)
    {
        auto const dx = this->layout.dx[0];
        for (auto& particle : particles)
            particle.v()[0] = v * cellsPerStep * dx / this->dt;
    }

    // what level initialization does on overlapping patches
    void eraseNonOwned(std::vector<Box_t> const& nonOwned)
    {
        for (auto& pop : this->ions)
            delete_particles_in(pop.domainParticles(), nonOwned);
    }

    static std::size_t count(auto const& particles, auto&& predicate)
    {
        return damn::core::sum_from(particles,
                                    [&](auto const& p) { return predicate(p) ? 1ul : 0ul; });
    }

    // particles of `cell` whose displacement takes them into the next cell
    static auto crossing(auto const& particles, int const cell)
    {
        return count(particles, [&](auto const& p) {
            return p.iCell()[0] == cell and p.delta()[0] + cellsPerStep >= 1.;
        });
    }

    static double weights(auto const& particles)
    {
        return damn::core::sum_from(particles,
                                    [](auto const& particle) { return particle.weight(); });
    }

    static double total(auto const& field) { return sum(field); }

    // level ghosts move into the first domain cell, which is owned (domain) or not (handed to
    //  its owner, as a patch ghost)
    void checkLevelGhostExport(bool const withNonOwned)
    {
        using ParticlesSynchronisedMover_t = Super::ParticlesSynchronisedMover_t;
        ParticlesSynchronisedMover_t ionUpdater{};

        auto const nonOwned = withNonOwned ? std::vector{box(0, 9)} : std::vector<Box_t>{};
        auto const boxing   = this->makeBoxing(nonOwned);
        eraseNonOwned(nonOwned);

        std::vector<std::size_t> nbrDomain, nbrEntering;
        for (auto& pop : this->ions)
        {
            setVelocityX(pop.domainParticles(), 0);
            setVelocityX(pop.levelGhostParticles(), 1);
            nbrDomain.push_back(pop.domainParticles().size());
            nbrEntering.push_back(crossing(pop.levelGhostParticles(), -1));
            ASSERT_GT(nbrEntering.back(), 0u);
        }

        ionUpdater.updatePopulations(*this->ions, *this->EM, boxing, this->dt, UpdaterMode::all);

        std::size_t ipop = 0;
        for (auto& pop : this->ions)
        {
            auto const& domain = pop.domainParticles();
            EXPECT_EQ(domain.size(), nbrDomain[ipop] + (withNonOwned ? 0 : nbrEntering[ipop]));
            EXPECT_EQ(count(domain, [&](auto const& p) { return !boxing.isOwned(p.iCell()); }), 0u);
            EXPECT_EQ(
                count(pop.patchGhostParticles(), [](auto const& p) { return p.iCell()[0] == 0; }),
                withNonOwned ? nbrEntering[ipop] : 0u);
            ++ipop;
        }
    }

    void checkLevelGhostDeposit(bool const withNonOwned, UpdaterMode const mode)
    {
        using ParticlesSynchronisedMover_t = Super::ParticlesSynchronisedMover_t;
        ParticlesSynchronisedMover_t ionUpdater{};

        auto const nonOwned = withNonOwned ? std::vector{box(0, 9)} : std::vector<Box_t>{};
        auto const boxing   = this->makeBoxing(nonOwned);
        eraseNonOwned(nonOwned);

        std::vector<double> expected;
        for (auto& pop : this->ions)
        {
            setVelocityX(pop.domainParticles(), 0);
            setVelocityX(pop.levelGhostParticles(), 1);

            double enteringWeights = 0;
            for (auto const& p : pop.levelGhostParticles())
                if (p.iCell()[0] == -1 and p.delta()[0] + cellsPerStep >= 1.)
                    enteringWeights += p.weight();
            ASSERT_GT(enteringWeights, 0.);

            // domain_only: deposited where it lands, by the one patch holding it; all: by the
            //  patch whose domain it enters, after the exchange
            bool const here = mode == UpdaterMode::domain_only or !withNonOwned;
            expected.push_back(weights(pop.domainParticles()) + (here ? enteringWeights : 0.));
        }

        ionUpdater.updatePopulations(*this->ions, *this->EM, boxing, this->dt, mode);
        if (mode == UpdaterMode::all) // mode all moves only, the domain deposit follows
            ParticlesSynchronisedMover_t::depositDomain(*this->ions, this->layout);

        std::size_t ipop = 0;
        for (auto& pop : this->ions)
        {
            EXPECT_NEAR(total(pop.particleDensity()), expected[ipop], 1e-10 * expected[ipop]);
            ++ipop;
        }
    }

    // Level ghosts moving left by cellsPerHalfPush (up to 2: Boris throws past that), those of the
    // outermost ghost cell end up 2 cells past the ghost box after the first half push. The ghost
    // box selector partitions through the cell map: it only drops them before fields are
    // interpolated on them if the map reaches that far, else they also stay level ghosts.
    void checkLevelGhostsPastTheGhostBoxAreDropped(double const cellsPerHalfPush)
    {
        using ParticlesSynchronisedMover_t = Super::ParticlesSynchronisedMover_t;
        ParticlesSynchronisedMover_t ionUpdater{};

        auto const boxing = makeBoxing({});
        auto const pastGhostBox
            = [&](auto const& p) { return !isIn(damn::core::Point{p.iCell()}, boxing.ghostBox); };
        auto const twoCellsPast = [&](auto const& p) {
            return p.iCell()[0] == boxing.ghostBox.lower[0]
                   and p.delta()[0] - cellsPerHalfPush < -1.;
        };

        for (auto& pop : this->ions)
        {
            setVelocityX(pop.domainParticles(), 0);
            setVelocityX(pop.levelGhostParticles(), -4 * cellsPerHalfPush);
            ASSERT_GT(count(pop.levelGhostParticles(), twoCellsPast), 0u);
        }

        ionUpdater.updatePopulations(*this->ions, *this->EM, boxing, this->dt, UpdaterMode::all);

        for (auto& pop : this->ions)
        {
            EXPECT_EQ(count(pop.levelGhostParticles(), pastGhostBox), 0u);
        }
    }
};

// clang-format off
using OwnershipPermutations = ::testing::Types<
    TestParam<SimOpts{1, 1}>
  , TestParam<SimOpts{1, 2}>
  , TestParam<SimOpts{1, 3}>
>;
// clang-format on

TYPED_TEST_SUITE(ParticlesSynchronisedMoverOwnershipTest, OwnershipPermutations, );



TYPED_TEST(ParticlesSynchronisedMoverOwnershipTest,
           particleEnteringNonOwnedCellLeavesDomainAndIsNotDepositedHere)
{
    typename TestFixture::ParticlesSynchronisedMover_t ionUpdater{};

    auto const nonOwned = std::vector{TestFixture::box(60, 99)};
    auto const boxing   = this->makeBoxing(nonOwned);
    this->eraseNonOwned(nonOwned);

    std::vector<std::size_t> nbrDomain, nbrCrossing;
    for (auto& pop : this->ions)
    {
        this->setVelocityX(pop.domainParticles(), 1);
        this->setVelocityX(pop.levelGhostParticles(), 0);
        nbrDomain.push_back(pop.domainParticles().size());
        nbrCrossing.push_back(TestFixture::crossing(pop.domainParticles(), 59));
        ASSERT_GT(nbrCrossing.back(), 0u);
    }

    ionUpdater.updatePopulations(*this->ions, *this->EM, boxing, this->dt, UpdaterMode::all);
    TestFixture::ParticlesSynchronisedMover_t::depositDomain(*this->ions, this->layout);

    std::size_t ipop = 0;
    for (auto& pop : this->ions)
    {
        auto const& domain     = pop.domainParticles();
        auto const& patchGhost = pop.patchGhostParticles();

        EXPECT_EQ(domain.size(), nbrDomain[ipop] - nbrCrossing[ipop]);
        EXPECT_EQ(
            TestFixture::count(domain, [&](auto const& p) { return !boxing.isOwned(p.iCell()); }),
            0u);

        // the leaving particles wait in the patch ghost array for the exchange
        EXPECT_EQ(patchGhost.size(), nbrCrossing[ipop]);
        EXPECT_EQ(TestFixture::count(patchGhost, [](auto const& p) { return p.iCell()[0] == 60; }),
                  nbrCrossing[ipop]);

        // the leaving particles are deposited by the patch owning their cell, after the exchange
        auto const deposited = TestFixture::total(pop.particleDensity());
        auto const expected  = TestFixture::weights(domain);
        EXPECT_NEAR(deposited, expected, 1e-10 * expected);
        ++ipop;
    }
}



TYPED_TEST(ParticlesSynchronisedMoverOwnershipTest, levelGhostEnteringOwnedCellIsExported)
{
    this->checkLevelGhostExport(/*withNonOwned=*/false);
}

TYPED_TEST(ParticlesSynchronisedMoverOwnershipTest, levelGhostEnteringNonOwnedCellIsHandedOver)
{
    this->checkLevelGhostExport(/*withNonOwned=*/true);
}

TYPED_TEST(ParticlesSynchronisedMoverOwnershipTest,
           levelGhostEnteringOwnedCellIsDepositedInMomentsOnlyMode)
{
    this->checkLevelGhostDeposit(/*withNonOwned=*/false, UpdaterMode::domain_only);
}

TYPED_TEST(ParticlesSynchronisedMoverOwnershipTest,
           levelGhostEnteringNonOwnedCellIsDepositedInMomentsOnlyMode)
{
    this->checkLevelGhostDeposit(/*withNonOwned=*/true, UpdaterMode::domain_only);
}

TYPED_TEST(ParticlesSynchronisedMoverOwnershipTest, levelGhostEnteringOwnedCellIsDeposited)
{
    this->checkLevelGhostDeposit(/*withNonOwned=*/false, UpdaterMode::all);
}

TYPED_TEST(ParticlesSynchronisedMoverOwnershipTest,
           levelGhostEnteringNonOwnedCellIsNotDepositedHere)
{
    this->checkLevelGhostDeposit(/*withNonOwned=*/true, UpdaterMode::all);
}



TYPED_TEST(ParticlesSynchronisedMoverOwnershipTest, levelGhostsPushedPastTheGhostBoxAreDropped)
{
    this->checkLevelGhostsPastTheGhostBoxAreDropped(2.0); // the most Boris allows
}



TYPED_TEST(ParticlesSynchronisedMoverOwnershipTest, ownershipFollowsDomainAndNonOwnedBoxes)
{
    auto const boxing = this->makeBoxing({TestFixture::box(10, 19), TestFixture::box(50, 50)});
    auto const owned  = [&](int i) { return boxing.isOwned(std::array{i}); };

    EXPECT_FALSE(owned(-1));
    EXPECT_TRUE(owned(0));
    EXPECT_TRUE(owned(9));
    EXPECT_FALSE(owned(10));
    EXPECT_FALSE(owned(19));
    EXPECT_TRUE(owned(20));
    EXPECT_FALSE(owned(50));
    EXPECT_TRUE(owned(99));
    EXPECT_FALSE(owned(100));
}


// TYPED_TEST(ParticlesSynchronisedMoverTest, hardCodedRegressionTest0)
// {
//     using ParticlesSynchronisedMover      = TestFixture::ParticlesSynchronisedMover;
//     auto constexpr dim    = TestFixture::dim;
//     auto constexpr interp = TestFixture::interp_order;

//     std::array<double, 3> expectations
//         = {199.99999999999889, 199.99999999999659, 199.99999999999673};

//     populate_particles(this->ions, this->layout);

//     for (auto& pop : this->ions)
//     {
//         EXPECT_EQ(pop.domainParticles().size(), 100 * nbrPartPerCell);
//         EXPECT_EQ(pop.levelGhostParticles().size(), 2 * nbrPartPerCell);
//         EXPECT_EQ(pop.patchGhostParticles().size(), 0);
//     }

//     ParticlesSynchronisedMover ionUpdater{};
//     this->update(ionUpdater, UpdaterMode::domain_only);

//     std::int64_t icellSum = 0;
//     double densitySum = 0, fluxXSum = 0, fluxYSum = 0, fluxZSum = 0;
//     for (auto& pop : this->ions)
//     {
//         densitySum += plamn::core::sum(reduce(pop.density()));
//         fluxXSum += plamn::core::sum(reduce(pop.flux()[0]));
//         fluxYSum += plamn::core::sum(reduce(pop.flux()[1]));
//         fluxZSum += plamn::core::sum(reduce(pop.flux()[2]));
//         per_particle(pop.domainParticles(),
//                      [&](Particle<dim>& p) { icellSum += plamn::core::sum(p.iCell()); });
//     }

//     EXPECT_EQ(icellSum, 9900000);
//     EXPECT_DOUBLE_EQ(densitySum, expectations[interp - 1]);
//     EXPECT_DOUBLE_EQ(fluxXSum, expectations[interp - 1]);
//     EXPECT_DOUBLE_EQ(fluxYSum, expectations[interp - 1]);
//     EXPECT_DOUBLE_EQ(fluxZSum, expectations[interp - 1]);
// }



// TYPED_TEST(ParticlesSynchronisedMoverTest, hardCodedRegressionTest1)
// {
//     using ParticlesSynchronisedMover      = TestFixture::ParticlesSynchronisedMover;
//     auto constexpr dim    = TestFixture::dim;
//     auto constexpr interp = TestFixture::interp_order;

//     std::array<double, 3> expectations
//         = {199.99999999999889, 199.99999999999659, 199.99999999999673};

//     populate_particles(this->ions, this->layout);

//     double deltaSum = 0;
//     for (auto& pop : this->ions)
//     {
//         EXPECT_EQ(pop.domainParticles().size(), 100 * nbrPartPerCell);
//         EXPECT_EQ(pop.levelGhostParticles().size(), 2 * nbrPartPerCell);
//         EXPECT_EQ(pop.patchGhostParticles().size(), 0);
//         per_particle(pop.domainParticles(),
//                      [&](Particle<dim>& p) { deltaSum += plamn::core::sum(p.delta()); });
//     }
//     EXPECT_DOUBLE_EQ(deltaSum, 103333.29999993632);

//     ParticlesSynchronisedMover ionUpdater{};
//     this->update(ionUpdater, UpdaterMode::all);

//     std::size_t icellSum = 0, pCount = 0;
//     deltaSum          = 0;
//     double densitySum = 0, fluxXSum = 0, fluxYSum = 0, fluxZSum = 0;
//     for (auto& pop : this->ions)
//     {
//         densitySum += plamn::core::sum(reduce(pop.density()));
//         fluxXSum += plamn::core::sum(reduce(pop.flux()[0]));
//         fluxYSum += plamn::core::sum(reduce(pop.flux()[1]));
//         fluxZSum += plamn::core::sum(reduce(pop.flux()[2]));
//         per_particle(pop.domainParticles(),
//                      [&](Particle<dim>& p) { deltaSum += plamn::core::sum(p.delta()); });
//         per_particle(pop.domainParticles(),
//                      [&](Particle<dim>& p) { icellSum += plamn::core::sum(p.iCell()); });
//         pCount += pop.domainParticles().size();
//     }

//     EXPECT_EQ(pCount, 200000);
//     EXPECT_EQ(icellSum, 9900000);
//     EXPECT_DOUBLE_EQ(deltaSum, 123333.30000002828);
//     EXPECT_DOUBLE_EQ(densitySum, expectations[interp - 1]);
//     EXPECT_DOUBLE_EQ(fluxXSum, expectations[interp - 1]);
//     EXPECT_DOUBLE_EQ(fluxYSum, expectations[interp - 1]);
//     EXPECT_DOUBLE_EQ(fluxZSum, expectations[interp - 1]);
// }



} // namespace plamn::core


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
