#include "plamn/core/def.hpp"
// DivergenceFreeTransverseNeumann (field_boundary_condition.hpp) on an MHD B linear in the local
//  indexes: past each boundary its tangential components mirror the domain's, its normal one
//  makes every ghost cell's discrete divergence zero, on isotropic and anisotropic 2D meshes and
//  in 3D

#include "damn/core/geometry/box_spans.hpp"

#include "plamn/core/boundary/boundary_defs.hpp"
#include "plamn/core/data/grid/layout.hpp"
#include "plamn/core/data/grid/layout_defs.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"
#include "plamn/core/models/quantities/mhd_quantities.hpp"
#include "plamn/core/numerics/boundary_condition/field_boundary_condition.hpp"

#include "tst/plamn/core/numerics/boundary_condition/bc_test_fixtures.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace plamn::core::bc_test
{
using enum BoundaryLocation;

// B's components a i + b j (+ c k) + d over their whole allocation
std::array<std::array<double, 4>, 3> constexpr coefs2D{{
    {0.3, 0.7, 0.0, 1.0},
    {-0.2, 0.4, 0.0, 2.0},
    {0.5, -0.6, 0.0, 3.0},
}};
std::array<std::array<double, 4>, 3> constexpr coefs3D{{
    {0.3, 0.7, -0.2, 1.0},
    {-0.2, 0.4, 0.5, 2.0},
    {0.5, -0.6, 0.3, 3.0},
}};

template<std::size_t dim_, bool anisotropic = false>
struct DivFreeTransverseNeumannPatch : public ::testing::Test
{
    auto constexpr static dim = dim_;
    using Types               = MHDTypes<dim>;
    using Layout_t            = Types::Layout_t;
    auto constexpr static gw  = Layout_t::ghost_width;

    static std::array<double, dim> dx()
    {
        auto dl = uniform_dx<dim>();
        if constexpr (anisotropic)
            dl[1] = .2;
        return dl;
    }

    DivFreeTransverseNeumannPatch()
    {
        auto const& coefs = dim == 2 ? coefs2D : coefs3D;
        for (std::size_t c = 0; c < 3; ++c)
            damn::core::for_each_point(ghost_box(layout, Bvec[c]), [&](auto const& index) {
                double v = coefs[c][3];
                for (std::size_t d = 0; d < dim; ++d)
                    v += coefs[c][d] * static_cast<double>(index[d]);
                Bvec[c](index) = v;
            });
    }

    void apply(BoundaryLocation const loc, damn::core::Box<std::uint32_t, dim> const& ghostBox)
    {
        apply_vector(FieldCondition{.type = FieldConditionType::DivergenceFreeTransverseNeumann},
                     Bvec, loc, ghostBox, layout);
    }

    void checkTransverseMirrored(BoundaryLocation const loc,
                                 damn::core::Box<std::uint32_t, dim> const& ghostBox)
    {
        auto const side    = side_of(loc);
        auto const iNormal = static_cast<std::size_t>(direction_of(loc));
        for (std::size_t comp = 0; comp < 3; ++comp)
        {
            if (comp == iNormal)
                continue;
            auto const& bField = Bvec[comp];
            auto const centerings
                = centering<dim>(MHDQuantity::componentsQuantities(MHDQuantity::Vector::B)[comp]);
            damn::core::for_each_point(toCenteredBox(ghostBox, centerings), [&](auto const& index) {
                auto mirror     = index;
                mirror[iNormal] = boundary_mirrored(centerings[iNormal], side, n_cells[iNormal], gw,
                                                    index[iNormal]);
                EXPECT_DOUBLE_EQ(bField(index), bField(mirror))
                    << "comp=" << comp << " at " << static_cast<int>(loc);
            });
        }
    }

    // each ghost cell's discrete divergence, its differences divided by the mesh sizes if spaced
    void checkGhostCellsDivergenceFree(BoundaryLocation const loc,
                                       damn::core::Box<std::uint32_t, dim> const& ghostBox,
                                       bool const spaced)
    {
        damn::core::for_each_point(ghostBox, [&](auto const& cell) {
            double div = 0.;
            for (std::size_t d = 0; d < dim; ++d)
            {
                auto up = cell;
                ++up[d];
                div += (Bvec[d](up) - Bvec[d](cell)) / (spaced ? layout.dx[d] : 1.);
            }
            EXPECT_NEAR(div, 0.0, 1e-12) << "at " << static_cast<int>(loc);
        });
    }

    Layout_t layout = make_layout<Layout_t>(dx());
    VecField_t<Types> Bvec{"bc_test_B", layout, MHDQuantity::Vector::B};
};



struct DivFreeTransverseNeumannBC2D : public DivFreeTransverseNeumannPatch<2>
{
    void applyAndCheck(BoundaryLocation const loc)
    {
        auto const ghostBox = ghost_cells(layout, loc);
        apply(loc, ghostBox);
        checkTransverseMirrored(loc, ghostBox);
        checkGhostCellsDivergenceFree(loc, ghostBox, false);
    }
};

TEST_F(DivFreeTransverseNeumannBC2D, XLowerGhostsMirrorTransverseAndKillDivergence)
{
    applyAndCheck(XLower);
}

TEST_F(DivFreeTransverseNeumannBC2D, XUpperGhostsMirrorTransverseAndKillDivergence)
{
    applyAndCheck(XUpper);
}

TEST_F(DivFreeTransverseNeumannBC2D, YLowerGhostsMirrorTransverseAndKillDivergence)
{
    applyAndCheck(YLower);
}

TEST_F(DivFreeTransverseNeumannBC2D, YUpperGhostsMirrorTransverseAndKillDivergence)
{
    applyAndCheck(YUpper);
}


struct DivFreeTransverseNeumannBC2DAnisotropic : public DivFreeTransverseNeumannPatch<2, true>
{
    void applyAndCheckSpaced(BoundaryLocation const loc)
    {
        auto const ghostBox = ghost_cells(layout, loc);
        apply(loc, ghostBox);
        checkGhostCellsDivergenceFree(loc, ghostBox, true);
    }
};

TEST_F(DivFreeTransverseNeumannBC2DAnisotropic, XBoundariesKillDivergenceOnAnisotropicMesh)
{
    applyAndCheckSpaced(XLower);
    applyAndCheckSpaced(XUpper);
}

TEST_F(DivFreeTransverseNeumannBC2DAnisotropic, YBoundariesKillDivergenceOnAnisotropicMesh)
{
    applyAndCheckSpaced(YLower);
    applyAndCheckSpaced(YUpper);
}


struct DivFreeTransverseNeumannBC3D : public DivFreeTransverseNeumannPatch<3>
{
    void applyAndCheck(BoundaryLocation const loc)
    {
        auto const ghostBox = ghost_cells(layout, loc);
        apply(loc, ghostBox);
        checkTransverseMirrored(loc, ghostBox);
        checkGhostCellsDivergenceFree(loc, ghostBox, true);
    }
};

TEST_F(DivFreeTransverseNeumannBC3D, XBoundariesKillDivergence)
{
    applyAndCheck(XLower);
    applyAndCheck(XUpper);
}

TEST_F(DivFreeTransverseNeumannBC3D, YBoundariesKillDivergence)
{
    applyAndCheck(YLower);
    applyAndCheck(YUpper);
}

TEST_F(DivFreeTransverseNeumannBC3D, ZBoundariesKillDivergence)
{
    applyAndCheck(ZLower);
    applyAndCheck(ZUpper);
}

} // namespace plamn::core::bc_test

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
