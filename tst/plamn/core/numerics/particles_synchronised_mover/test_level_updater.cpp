// the particle updater over a level: a level is a set of patches tiling a box, each updated as
// the simulator updates it (its boxing from the level's other patches: damn's covered_by_level,
// as Simulator::rebox), the patch ghost particles then handed to the patch owning their cell
// (periodic images included) by box overlaps, no damn schedules. Each push, every patch split of
// the level must hold exactly the particles of the same level as one patch: the same push, the
// same exchanges, nothing duplicated or lost at patch borders and corners
//
//   level 0  periodic, no level border
//   level 1  a box inside a periodic level 1 domain, its level ghost particles in the ring
//            around it, each patch holding those in its ghost box (patches meeting on the level
//            border share them: PHAREHUB/PHARE#1579's corner case)
//
// moments as PHAREHUB/PHARE#1441 (and the simulator): after the exchange each patch deposits its
// domain particles (depositDomain), the level's moments are then every patch's sum on each node
// (the border sums, periodic images included): the same as the level as one patch, within the
// summation order.
//
// uniform E and B: the push is the same whichever patch holds a particle and the comparison is
// exact. Every layout picks its updater
// (ParticlesSynchronisedMoverFor_t): AoSMapped serial, AoS flat parallel, AoSCMTS tiled parallel
//
// env: PLAMN_CELLS (level cells per direction), PLAMN_PPC, PLAMN_PUSHES, PLAMN_TIMESTEP, PLAMN_SEED

#include "damn/amr/hierarchy.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/logger.hpp"
#include "damn/core/meta/type_meta.hpp"
#include "damn/core/sys/env.hpp"

#include "plamn/amr/resources.hpp"
#include "plamn/core/data/field/field_box.hpp"
#include "plamn/core/data/particles/particles.hpp"
#include "plamn/core/data/particles/appending/particles_appending.hpp"
#include "plamn/core/data/particles/comparing/particles_comparing.hpp"
#include "plamn/core/data/particles/converting/particles_converting.hpp"
#include "plamn/core/data/particles/sorting/particles_sorting.hpp"
#include "plamn/core/numerics/particles_synchronised_mover/particles_synchronised_mover.hpp" // IWYU pragma: keep
#include "plamn/simulator/simulator_def.hpp"

#include "tst/plamn/amr/resources_manager/test_resources_manager_fixtures.hpp"
#include "tst/plamn/core/data/electromag/test_electromag_fixtures.hpp"
#include "tst/plamn/core/data/grid/test_layout_fixtures.hpp"
#include "tst/plamn/core/data/ion_population/test_ion_population_fixtures.hpp"
#include "tst/plamn/core/data/particles/test_particles.hpp"

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>


namespace plamn::core
{
auto static const cells  = damn::core::get_env_as("PLAMN_CELLS", int{16});
auto static const ppc    = damn::core::get_env_as("PLAMN_PPC", std::size_t{10});
auto static const pushes = damn::core::get_env_as("PLAMN_PUSHES", std::size_t{4});
auto static const dt
    = damn::core::get_env_as("PLAMN_TIMESTEP", double{.015}); // dl .1, |v| <= 6: < 1 cell
auto static const seed = damn::core::get_env_as("PLAMN_SEED", std::size_t{1067});

enum class Updater : std::uint16_t { serial = 0, parallel };

template<std::size_t _dim, auto _layout_mode, auto _updater, bool _refined>
struct LevelParam
{
    auto constexpr static dim         = _dim;
    auto constexpr static layout_mode = _layout_mode;
    auto constexpr static updater     = _updater;
    auto constexpr static refined     = _refined;
};


// one layout and updater over a level: its patches, push, exchange, and the level's particles
template<typename Param>
struct LevelRun
{
    auto constexpr static dim     = Param::dim;
    auto constexpr static refined = Param::refined;
    auto constexpr static opts    = SimOpts{.dimension   = dim,
                                            .layout_mode = Param::layout_mode,
                                            .alloc_mode  = damn::AllocatorMode::CPU};

    using CoreTypes   = PLAMN_Types<opts>::Hybrid;
    using Layout_t    = CoreTypes::Layout_t;
    using Ions_t      = UsableIons_t<opts>;
    using EM_t        = UsableElectromag<TensorFieldOptions<CoreTypes>{}>;
    using Patch_t     = test::HybridPatch<Ions_t, EM_t>;
    using Particles_t = Patch_t::Model_t::particles_type;
    using AoS_t       = ParticlesAoS<dim>;
    using Box_t       = damn::core::Box<int, dim>;
    using Point_t     = damn::core::Point<int, dim>;
    using Mover_t     = std::conditional_t<Param::updater == Updater::serial,
                                           ParticlesSynchronisedMover<Particles_t, Layout_t>,
                                           ParticlesSynchronisedMoverFor_t<Particles_t, Layout_t>>;
    using Boxing_t    = Mover_t::Boxing_t;
    auto constexpr static pgw = Layout_t::options.particle_ghost_width;

    int const ln = refined ? 1 : 0;

    // level 0 cells; level 1: the domain twice as fine, the level box in its middle
    Box_t const level_box = refined ? Box_t{damn::core::ConstArray<int, dim>(cells / 2),
                                            damn::core::ConstArray<int, dim>(cells / 2 + cells - 1)}
                                    : Box_t{damn::core::ConstArray<int, dim>(0),
                                            damn::core::ConstArray<int, dim>(cells - 1)};

    damn::amr::Hierarchy<dim> hierarchy(std::vector<Box_t> const& patch_boxes) const
    {
        damn::amr::Hierarchy<dim> h;
        h.domain     = Box_t{damn::core::ConstArray<int, dim>(0),
                             damn::core::ConstArray<int, dim>(cells - 1)};
        h.dx         = damn::core::ConstArray<double, dim>(.1);
        h.periodic   = damn::core::ConstArray<bool, dim>(true);
        h.max_levels = 2;
        damn::amr::set_level(
            h, damn::amr::make_level(0, refined ? std::vector{h.domain} : patch_boxes));
        if (refined)
            damn::amr::set_level(h, damn::amr::make_level(1, patch_boxes));
        return h;
    }

    // the level box cut into patches of size (at most) n per direction
    std::vector<Box_t> split(int const n) const
    {
        std::vector<Box_t> boxes{level_box};
        for (std::size_t d = 0; d < dim; ++d)
        {
            std::vector<Box_t> cut;
            for (auto const& b : boxes)
                for (int lo = b.lower[d]; lo <= b.upper[d]; lo += n)
                {
                    auto c     = b;
                    c.lower[d] = lo;
                    c.upper[d] = std::min(lo + n - 1, b.upper[d]);
                    cut.push_back(c);
                }
            boxes = std::move(cut);
        }
        return boxes;
    }

    // the level's particles, once: domain on the level box, level ghosts on the ring around it
    AoS_t const domain = [&]() {
        auto ps = make_particles<AoS_t>(*TestLayout<Layout_t>{level_box});
        add_particles(ps, level_box, ppc);
        delta_disperse(ps, seed);
        vary_velocity(ps, -6, 6, seed);
        return ps;
    }();
    AoS_t const level_ghosts = [&]() {
        auto ps = make_particles<AoS_t>(*TestLayout<Layout_t>{level_box});
        if (refined)
        {
            add_ghost_particles(ps, level_box, ppc, pgw);
            delta_disperse(ps, seed + 1);
            vary_velocity(ps, -6, 6, seed + 1);
        }
        return ps;
    }();

    // into any layout: AoS appends into AoS and AoSMapped, AoSMapped into the tiled layouts
    template<auto type>
    static void append(AoS_t const& src, Particles_t& dst, Layout_t const& layout)
    {
        using enum LayoutMode;
        if constexpr (Particles_t::layout_mode == AoS or Particles_t::layout_mode == AoSMapped
                      or Particles_t::layout_mode == SoAVXChunked)
            append_particles<type>(src, dst);
        else
            append_particles<type>(convert_particles<ParticlesAoSMapped<dim>>(src, layout), dst);
    }

    static AoS_t in_box(AoS_t const& from, Box_t const& box)
    {
        auto out = from;
        std::erase_if(out.vector(), [&](auto const& p) { return !isIn(Point_t{p.iCell()}, box); });
        return out;
    }

    // the same E and B everywhere (the fixtures' default varies with the local index): a push is
    //  then the same whichever patch holds the particle
    static void uniform_fields(auto& em)
    {
        for (std::size_t c = 0; c < 3; ++c)
        {
            em.E[c].fill(std::array{.02, -.01, .005}[c]);
            em.B[c].fill(std::array{1., .2, .3}[c]);
        }
    }

    // a level of patches, its boxings and their ids
    struct Level
    {
        damn::amr::Hierarchy<dim> h;
        std::vector<damn::core::aggregate_adapter<Patch_t>> patches;
        std::unordered_map<std::string, Boxing_t> boxings;
    };

    Level make_level(int const n) const
    {
        auto const boxes = split(n);
        Level l{hierarchy(boxes), {}, {}};
        l.patches.reserve(boxes.size());
        for (std::size_t i = 0; i < boxes.size(); ++i)
        {
            Layout_t const layout = *TestLayout<Layout_t>{boxes[i]};
            Ions_t ions{layout, "protons"};
            auto& particles = ions.populations[0].particles;
            append<ParticleType::Domain>(in_box(domain, boxes[i]), particles.domain_particles,
                                         layout);
            append<ParticleType::LevelGhost>(in_box(level_ghosts, grow(boxes[i], pgw)),
                                             particles.level_ghost_particles, layout);
            auto& patch = l.patches.emplace_back(layout, std::move(ions));
            patch.id    = std::to_string(i);
            uniform_fields(patch.model.state.em);
            l.boxings.try_emplace(patch.id, Boxing_t{layout, damn::amr::covered_by_level(
                                                                 l.h, ln, grow(boxes[i], pgw))});
        }
        return l;
    }

    void push(Level& l) const
    {
        if constexpr (Param::updater == Updater::serial)
            for (auto& patch : l.patches)
                Mover_t{}.updatePopulations(patch.model.state.ions, patch.model.state.electromag,
                                            l.boxings.at(patch.id), dt, UpdaterMode::all);
        else
        {
            auto accessor = test::make_model_level_accessor(l.patches, [&](int i) {
                return std::forward_as_tuple(l.patches[i].model.state.ions,
                                             l.patches[i].model.state.electromag);
            });
            Mover_t{}.updatePopulations(accessor, l.boxings, dt, UpdaterMode::all);
        }
    }

    // patch ghost particles to the patch owning their cell, a periodic image's cell shifted
    void exchange(Level& l) const
    {
        std::vector<Point_t> shifts{Point_t{}};
        for (auto const& s : damn::amr::periodic_shifts(l.h, ln))
            shifts.push_back(s);

        std::vector<AoS_t> arriving;
        for (auto& patch : l.patches)
            arriving.push_back(make_particles<AoS_t>(patch.layout));
        for (auto& patch : l.patches)
        {
            auto& ghosts = patch.model.state.ions.populations[0].particles.patch_ghost_particles;
            for (auto p : convert_particles<AoS_t>(ghosts, patch.layout))
            {
                bool owned = false;
                for (std::size_t j = 0; j < l.patches.size() and !owned; ++j)
                    for (auto const& s : shifts)
                        if (auto const cell = Point_t{p.iCell()} + s;
                            isIn(cell, l.patches[j].layout.cells))
                        {
                            p.iCell() = *cell;
                            arriving[j].push_back(p);
                            owned = true;
                            break;
                        }
                EXPECT_TRUE(owned)
                    << "patch ghost particle owned by no patch: " << Point_t{p.iCell()};
            }
            ghosts.clear();
        }
        for (std::size_t j = 0; j < l.patches.size(); ++j)
            append<ParticleType::Domain>(
                arriving[j],
                l.patches[j].model.state.ions.populations[0].particles.domain_particles,
                l.patches[j].layout);
    }

    void step(Level& l) const
    {
        push(l);
        exchange(l);
        for (auto& patch : l.patches)
            Mover_t::depositDomain(patch.model.state.ions, patch.layout);
    }

    // node -> {density, flux x, y, z}: every patch's deposit summed on each node of the level
    //  (its domain nodes; on a periodic level 0 the nodes modulo the domain)
    using Node_t    = std::array<int, dim>;
    using Moments_t = std::map<Node_t, std::array<double, 4>>;
    Moments_t level_moments(Level& l) const
    {
        auto constexpr gw = Layout_t::ghost_width;
        auto const shell  = [&]() { // the level's nodes
            auto b = level_box;
            for (std::size_t d = 0; d < dim; ++d)
                b.upper[d] += 1;
            return b;
        }();
        auto const canonical = [&](Point_t p) {
            if (!refined) // periodic level 0: node cells == the level's first one
                for (std::size_t d = 0; d < dim; ++d)
                    p[d] = ((p[d] % cells) + cells) % cells;
            return p;
        };
        Moments_t sums;
        for (auto& patch : l.patches)
        {
            auto& pop         = *patch.model.state.ions.begin(); // the population
            auto const cells_ = patch.layout.cells;
            // fn(resource) of a field: tiled ones reduced, their tiles' deposits (ghosts included)
            //  summed into one patch grid
            auto const with_resource = [&](auto& field, auto&& fn) {
                if constexpr (is_field_tile_set_v<std::decay_t<decltype(field)>>)
                {
                    auto grid = reduce(field);
                    fn(amr::field_resource<gw>(grid));
                }
                else
                    fn(amr::field_resource<gw>(field));
            };
            auto const add = [&](auto& field, std::size_t const k) {
                with_resource(field, [&](auto const& res) {
                    damn::core::for_each_point(grow(Box_t{cells_.lower, cells_.upper + 1}, gw),
                                               [&](auto const& p) {
                                                   auto const node = canonical(Point_t{p});
                                                   if (!refined or isIn(node, shell))
                                                       sums[*node][k] += value(res, cells_, p);
                                               });
                });
            };
            add(pop.particleDensity(), 0);
            for (std::size_t c = 0; c < 3; ++c)
                add(pop.flux()[c], 1 + c);
        }
        return sums;
    }

    AoS_t level_domain(Level const& l) const
    {
        auto all = make_particles<AoS_t>(*TestLayout<Layout_t>{level_box});
        for (auto const& patch : l.patches)
            for (auto const& p : convert_particles<AoS_t>(
                     patch.model.state.ions.populations[0].particles.domain_particles,
                     patch.layout))
                all.push_back(p);
        // a total order (sort_particles orders by cell only: within a cell the order depends on
        //  the patches a particle went through)
        std::sort(all.vector().begin(), all.vector().end(), [](auto const& a, auto const& b) {
            if (a.iCell() != b.iCell())
                return a.iCell() < b.iCell();
            if (a.delta() != b.delta())
                return a.delta() < b.delta();
            if (a.v() != b.v())
                return a.v() < b.v();
            return a.weight() < b.weight();
        });
        return all;
    }
};


// each layout against itself as one patch (patch splits), and against AoSMapped's serial
//  updater with the same split (layouts / updaters)
template<typename Param>
struct LevelUpdaterTest : public ::testing::Test
{
    auto constexpr static dim = Param::dim;
    using Run_t               = LevelRun<Param>;
    using Ref_t
        = LevelRun<LevelParam<Param::dim, LayoutMode::AoSMapped, Updater::serial, Param::refined>>;

    // atol: exact by default (float_equals is |a - b| < atol)
    static void expect_same(auto const& r, auto const& c, std::string const& what,
                            double const atol = std::numeric_limits<double>::denorm_min())
    {
        ASSERT_EQ(r.size(), c.size()) << what;
        auto const report = compare_particles(r, c, atol);
        if (!report)
        { // how far apart: the largest |dv|, |ddelta|, and how many particles differ at all
            double dv = 0, ddelta = 0;
            std::size_t n = 0;
            for (std::size_t k = 0; k < r.size(); ++k)
            {
                bool differs = false;
                for (std::size_t d = 0; d < 3; ++d)
                    if (auto const x = std::abs(r[k].v()[d] - c[k].v()[d]); x > 0)
                        dv = std::max(dv, x), differs = true;
                for (std::size_t d = 0; d < dim; ++d)
                    if (auto const x = std::abs(r[k].delta()[d] - c[k].delta()[d]); x > 0)
                        ddelta = std::max(ddelta, x), differs = true;
                n += differs;
            }
            DAMN_LOG_LINE_SS(what << ": " << n << " of " << r.size() << " differ, max |dv| " << dv
                                  << " |ddelta| " << ddelta);
        }
        ASSERT_TRUE(report) << what << ": " << report.why();
    }

    // the same nodes, each moment within the summation order (relative to the largest)
    static void expect_same_moments(auto const& r, auto const& c, std::string const& what)
    {
        ASSERT_EQ(r.size(), c.size()) << what;
        std::array<double, 4> scale{}, worst{};
        for (auto const& [node, m] : r)
            for (std::size_t k = 0; k < 4; ++k)
                scale[k] = std::max(scale[k], std::abs(m[k]));
        for (auto const& [node, m] : r)
        {
            auto const it = c.find(node);
            ASSERT_TRUE(it != c.end()) << what;
            for (std::size_t k = 0; k < 4; ++k)
                worst[k] = std::max(worst[k], std::abs(m[k] - it->second[k]));
        }
        for (std::size_t k = 0; k < 4; ++k)
            EXPECT_LE(worst[k], 1e-14 * std::max(1., scale[k]))
                << what << ": " << std::array{"density", "flux x", "flux y", "flux z"}[k]
                << " max |diff| " << worst[k] << " of " << scale[k];
    }

    void run()
    {
        Run_t run;
        Ref_t ref_run;
        for (int const n : {cells / 2, cells / 4})
        {
            auto one = run.make_level(cells); // the level as one patch
            auto cmp = run.make_level(n);
            auto ref = ref_run.make_level(n); // AoSMapped, serial, the same split
            DAMN_LOG_LINE_SS(Run_t::Particles_t::type_id
                             << (Param::refined ? " level 1" : " level 0") << ", "
                             << cmp.patches.size() << " patches of " << n);
            for (std::size_t i = 0; i < pushes; ++i)
            {
                run.step(one);
                run.step(cmp);
                ref_run.step(ref);
                auto const what = "push " + std::to_string(i) + ", patches of " + std::to_string(n);
                expect_same(run.level_domain(one), run.level_domain(cmp), what + " (vs one patch)");
                expect_same_moments(run.level_moments(one), run.level_moments(cmp),
                                    what + " moments (vs one patch)");
                // another push implementation (MultiBoris): the same arithmetic in another
                //  order, a ULP or so per push, as test_multi_updater's tolerance (|v| <= 6)
                expect_same(ref_run.level_domain(ref), run.level_domain(cmp),
                            what + " (vs AoSMapped serial)", 1e-15 * 4 * 6 * (i + 1));
            }
        }
    }
};

// clang-format off
using LevelPermutations_t = testing::Types< // ! notice commas !
    LevelParam<1, LayoutMode::AoSMapped, Updater::serial, false>
   ,LevelParam<1, LayoutMode::AoSMapped, Updater::serial, true>
   ,LevelParam<2, LayoutMode::AoSMapped, Updater::serial, false>
   ,LevelParam<2, LayoutMode::AoSMapped, Updater::serial, true>
   ,LevelParam<2, LayoutMode::AoS, Updater::parallel, false>
   ,LevelParam<2, LayoutMode::AoS, Updater::parallel, true>
   ,LevelParam<2, LayoutMode::AoSCMTS, Updater::parallel, false>
   ,LevelParam<2, LayoutMode::AoSCMTS, Updater::parallel, true>
   ,LevelParam<2, LayoutMode::SoAVXChunked, Updater::parallel, false>
   ,LevelParam<2, LayoutMode::SoAVXChunked, Updater::parallel, true>
>;
// clang-format on

TYPED_TEST_SUITE(LevelUpdaterTest, LevelPermutations_t, );

TYPED_TEST(LevelUpdaterTest, patchSplitsHoldTheSameParticles)
{
    this->run();
}

} // namespace plamn::core


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
