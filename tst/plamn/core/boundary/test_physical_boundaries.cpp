#include "plamn/core/def.hpp"
// a 2D patch covering a non periodic domain, open in x and reflective in y

#include "damn/amr/hierarchy.hpp"
#include "damn/amr/physical_boundary.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"
#include "plamn/core/boundary/boundary.hpp"
#include "plamn/core/boundary/boundary_defs.hpp"
#include "plamn/core/boundary/physical_boundaries.hpp"
#include "plamn/core/models/quantities/mhd_quantities.hpp"
#include "plamn/initializer/data_provider.hpp"
#include "plamn/simulator/simulator_def.hpp"
#include "tst/plamn/core/data/grid/test_layout_fixtures.hpp"
#include "tst/plamn/core/data/vecfield/test_vecfield_fixtures.hpp"

#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>

namespace plamn::core
{
namespace
{
    auto constexpr opts = SimOpts{.dimension           = 2,
                                  .interp_order        = 1,
                                  .reconstruction_type = MHDOpts::ReconstructionType::Linear,
                                  .slope_limiter_type  = MHDOpts::SlopeLimiterType::VanLeer,
                                  .riemann_solver_type = MHDOpts::RiemannSolverType::Rusanov};
    using Types         = PLAMN_Types<opts>::MHD;
    using Layout_t      = Types::Layout_t;
    using Grid_t        = Types::Grid_t;
    auto constexpr static vf_opts = TensorFieldOptions<Types>{};
    using VecField_t              = UsableVecField<vf_opts>;
    using S                       = MHDQuantity::Scalar;
    using V                       = MHDQuantity::Vector;
    auto constexpr gw             = Layout_t::ghost_width;

    void fill(auto const& layout, auto& f)
    {
        double v = 1.;
        damn::core::for_each_point(ghost_box(layout, f), [&](auto const& ijk) { f(ijk) = v++; });
    }

    // the ghost (dual, local) index mirrored across a lower face along d
    auto mirror_lower(auto ijk, std::size_t const d)
    {
        ijk[d] = 2 * gw - 1 - ijk[d];
        return ijk;
    }
} // namespace

TEST(PhysicalBoundaries, facesThenCornersByTheirBoundaries)
{
    TestLayout<Layout_t> const layout{10}; // cells 0..9 in both directions: the whole domain
    damn::amr::Hierarchy<2> h;
    h.domain   = layout.cells;
    h.dx       = layout.dx;
    h.periodic = {false, false};
    h.levels.resize(1);

    initializer::Dict d;
    d["xlower"]["type"] = static_cast<int>(BoundaryType::Open);
    d["xupper"]["type"] = static_cast<int>(BoundaryType::Open);
    d["ylower"]["type"] = static_cast<int>(BoundaryType::Reflective);
    d["yupper"]["type"] = static_cast<int>(BoundaryType::Reflective);
    BoundaryManager<MHDQuantity> const manager{d, {S::rho}, {V::rhoV}};

    Grid_t rho{"rho", layout, S::rho};
    VecField_t rhoV{"rhoV", layout, V::rhoV};
    fill(layout, rho);
    for (std::size_t c = 0; c < 3; ++c)
        fill(layout, rhoV[c]);

    auto const regions = damn::amr::physical_boundary_regions(h, 0, layout.cells, gw);
    ASSERT_EQ(regions.size(), 8u); // 4 faces, 4 corners
    fill_physical_boundaries(manager, regions, layout, rho, rhoV);

    using P = damn::core::Point<std::uint32_t, 2>;
    for (std::uint32_t i = 0; i < gw; ++i)
    {
        // the x lower face (open): rho and rhoV Neumann along x
        P const x{i, gw + 3};
        EXPECT_EQ(rho(x), rho(mirror_lower(x, 0)));
        EXPECT_EQ(rhoV[1](x), rhoV[1](mirror_lower(x, 0)));
        // the y lower face (reflective): rhoV's y component (normal) odd, its x one even
        P const y{gw + 3, i};
        EXPECT_EQ(rhoV[1](y), -rhoV[1](mirror_lower(y, 1)));
        EXPECT_EQ(rhoV[0](y), rhoV[0](mirror_lower(y, 1)));
        // the lower corner: the y face's conditions (by direction), mirrored along y, filled
        //  after the x face whose ghosts it reads
        for (std::uint32_t j = 0; j < gw; ++j)
        {
            P const corner{i, j};
            EXPECT_EQ(rhoV[1](corner), -rhoV[1](mirror_lower(corner, 1)));
            EXPECT_EQ(rhoV[0](corner), rhoV[0](mirror_lower(corner, 1)));
        }
    }
}

TEST(PhysicalBoundaries, regionLocations)
{
    initializer::Dict d;
    for (auto const* loc : {"xlower", "xupper", "ylower", "yupper", "zlower", "zupper"})
        d[loc]["type"] = static_cast<int>(BoundaryType::Open);
    BoundaryManager<MHDQuantity> const m{d, {S::rho}, {V::rhoV}};
    EXPECT_EQ((region_location<MHDQuantity, 3>(m, {0, 1, 0})), BoundaryLocation::YUpper);
    EXPECT_EQ((region_location<MHDQuantity, 3>(m, {-1, 0, 1})), BoundaryLocation::ZUpper);
    EXPECT_EQ((region_location<MHDQuantity, 3>(m, {1, -1, 0})), BoundaryLocation::YLower);
    EXPECT_EQ((region_location<MHDQuantity, 3>(m, {1, 1, -1})), BoundaryLocation::ZLower);
}

} // namespace plamn::core

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
