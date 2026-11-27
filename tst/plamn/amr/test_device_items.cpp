// SoAVXChunked particles in unified memory leaving their patches: damn's device gather (as
//  columns, through plamn's device item hooks) gives what the host's gives

#include "damn/amr/copy.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/amr/topology.hpp"
#include "damn/core/def/damn_config.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/geometry/point.hpp"
#include "plamn/amr/resources.hpp"
#include "plamn/core/data/particles/particle.hpp"
#include "plamn/core/data/particles/particle_components.hpp"
#include "plamn/core/data/particles/particles.hpp"

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#if DAMN_HAVE_GPU && DAMN_HAVE_THRUST
namespace plamn::amr
{
namespace
{
    using Box2                 = damn::core::Box<int, 2>;
    std::uint32_t constexpr gw = 1;
    template<damn::AllocatorMode alloc>
    using Particles_t = core::Particles<core::ParticlesOptions{
        .dim = 2, .layout_mode = core::LayoutMode::SoAVXChunked, .alloc_mode = alloc}>;
    using Device_t    = Particles_t<damn::AllocatorMode::GPU_UNIFIED>;
    using Host_t      = Particles_t<damn::AllocatorMode::CPU>;

    static_assert(damn::amr::DeviceItemSource<ParticleResource<Device_t, gw>, 2>
                  and damn::amr::DeviceItemDestination<ParticleResource<Device_t, gw>, 2>);
    static_assert(!damn::amr::detail::OnDevice<ParticleResource<Host_t, gw>>);

    struct PatchData
    {
        explicit PatchData(Box2 const& cells)
            : device_ghost{core::make_particles<Device_t>(cells, gw)}
            , device_domain{core::make_particles<Device_t>(cells, gw)}
            , host_ghost{core::make_particles<Host_t>(cells, gw)}
            , host_domain{core::make_particles<Host_t>(cells, gw)}
        {
        }
        Device_t device_ghost, device_domain;
        Host_t host_ghost, host_domain;
    };
    // the plans' geometry (by name): either side, the same boxes
    void for_each_resource(PatchData& d, auto&& fn)
    {
        auto g = particle_resource<gw>(d.host_ghost);
        fn("patch_ghost", g);
        auto o = particle_resource<gw>(d.host_domain);
        fn("domain", o);
    }
    template<bool device>
    struct Leaving
    {
        auto src(PatchData& d) const
        {
            if constexpr (device)
                return particle_resource<gw>(d.device_ghost);
            else
                return particle_resource<gw>(d.host_ghost);
        }
        auto dst(PatchData& d) const
        {
            if constexpr (device)
                return particle_resource<gw>(d.device_domain);
            else
                return particle_resource<gw>(d.host_domain);
        }
    };

    // a particle's values, real particles only (a chunks' gap has no weight), sorted
    std::vector<std::array<double, 9>> values(auto const& particles)
    {
        std::vector<std::array<double, 9>> out;
        auto const c = core::components(particles);
        for (std::size_t i = 0; i < particles.size(); ++i)
            if (c.weight[i] != 0)
                out.push_back({c.weight[i], double(c.iCell[0][i]), double(c.iCell[1][i]),
                               c.delta[0][i], c.delta[1][i], c.v[0][i], c.v[1][i], c.v[2][i], 0});
        std::sort(out.begin(), out.end());
        return out;
    }
} // namespace

TEST(DeviceItems, leavingParticlesOnTheDeviceAreTheHosts)
{
    for (auto const periodic : {false, true})
    {
        damn::amr::Hierarchy<2> h;
        h.domain   = Box2{std::array{0, 0}, std::array{11, 7}};
        h.dx       = {.1, .1};
        h.periodic = {periodic, periodic};
        damn::amr::set_level(h, damn::amr::uniform_level_0(h, {4, 4}, 1));
        auto ld = damn::amr::allocate(h.levels[0], 0,
                                      [](auto const& patch) { return PatchData{patch.box}; });

        // per ghost cell outside its patch, as many particles as its row mod 3 (some none)
        damn::amr::for_each_patch(ld, [&](auto const& patch, auto& d) {
            damn::core::for_each_point(damn::core::grow(patch.box, int{gw}), [&](auto const& c) {
                if (isIn(c, patch.box))
                    return;
                for (int k = 0; k < ((c[1] % 3) + 3) % 3; ++k)
                {
                    core::Particle<2> p;
                    p.weight() = 1. + patch.id + .01 * k;
                    p.iCell()  = {c[0], c[1]};
                    p.delta()  = {.25, .5 + .1 * k};
                    p.v()      = {double(c[0]), double(c[1]), double(k)};
                    d.device_ghost.push_back(p);
                    d.host_ghost.push_back(p);
                }
            });
            d.device_ghost.sort();
            d.host_ghost.sort();
        });

        auto run = [&]<bool device>() { // plans of their own: a plan keeps its device table
            auto n = damn::amr::make_neighbours(h, 0, 0, damn::amr::single_node(), 3);
            damn::amr::copy(damn::amr::gather_leaving, n,
                            std::vector<damn::amr::ResourcePair>{{"patch_ghost", "domain"}},
                            std::array{Leaving<device>{}}, ld);
            for (auto const& [_, plan] : n.plans)
                EXPECT_EQ(static_cast<damn::amr::detail::ItemTable<2>*>(plan.device.get())->usable,
                          device);
        };
        run.template operator()<true>();
        run.template operator()<false>();

        std::size_t arrived = 0;
        damn::amr::for_each_patch(ld, [&](auto const& patch, auto& d) {
            auto const host = values(d.host_domain);
            EXPECT_EQ(values(d.device_domain), host) << periodic << " patch " << patch.id;
            EXPECT_TRUE(d.device_domain.sorted()) << patch.id;
            arrived += host.size();
        });
        EXPECT_GT(arrived, 0u);
    }
}

TEST(DeviceItems, unsortedSourcesThrow)
{
    damn::amr::Hierarchy<2> h;
    h.domain   = Box2{std::array{0, 0}, std::array{7, 7}};
    h.dx       = {.1, .1};
    h.periodic = {true, true};
    damn::amr::set_level(h, damn::amr::uniform_level_0(h, {4, 4}, 1));
    auto ld = damn::amr::allocate(h.levels[0], 0,
                                  [](auto const& patch) { return PatchData{patch.box}; });
    damn::amr::for_each_patch(ld, [&](auto const& patch, auto& d) {
        core::Particle<2> p;
        p.weight() = 1;
        p.iCell()  = {patch.box.lower[0] - 1, patch.box.lower[1]};
        d.device_ghost.push_back(p); // not sorted since
    });
    auto n = damn::amr::make_neighbours(h, 0, 0, damn::amr::single_node(), 3);
    EXPECT_THROW(damn::amr::copy(damn::amr::gather_leaving, n,
                                 std::vector<damn::amr::ResourcePair>{{"patch_ghost", "domain"}},
                                 std::array{Leaving<true>{}}, ld),
                 std::runtime_error);
}

} // namespace plamn::amr
#endif // DAMN_HAVE_GPU && DAMN_HAVE_THRUST


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
