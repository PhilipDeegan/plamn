#ifndef PLAMN_TEST_CORE_NUMERICS_BOUNDARY_CONDITION_BC_TEST_FIXTURES_HPP
#define PLAMN_TEST_CORE_NUMERICS_BOUNDARY_CONDITION_BC_TEST_FIXTURES_HPP

// field conditions' test patches (PHARE's hybrid_bc_test_fixtures.hpp and
//  mhd_bc_test_fixtures.hpp): 10 (x 8 (x 6)) cells from 0, the ghost cells past a boundary in local
//  index space (the whole patch, ghosts included, along the other directions), a hybrid rho
//  (primal) or B (Yee) holding an interior value over its domain nodes and a sentinel on its
//  ghosts. The field conditions are model agnostic: hybrid fields test them as PHARE's do

#include "plamn/core/def.hpp"

#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"

#include "plamn/core/boundary/boundary_defs.hpp"
#include "plamn/core/data/grid/layout_defs.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"
#include "plamn/simulator/simulator_def.hpp"

#include "tst/plamn/core/data/vecfield/test_vecfield_fixtures.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace plamn::core::bc_test
{
template<std::size_t dim>
auto constexpr hybrid_opts = SimOpts{.dimension = dim, .interp_order = 1};

template<std::size_t dim>
auto constexpr mhd_opts = SimOpts{.dimension           = dim,
                                  .interp_order        = 1,
                                  .reconstruction_type = MHDOpts::ReconstructionType::Constant,
                                  .slope_limiter_type  = MHDOpts::SlopeLimiterType::None,
                                  .riemann_solver_type = MHDOpts::RiemannSolverType::Rusanov};

template<std::size_t dim>
using HybridTypes = PLAMN_Types<hybrid_opts<dim>>::Hybrid;

template<std::size_t dim>
using MHDTypes = PLAMN_Types<mhd_opts<dim>>::MHD;

template<typename Types>
using VecField_t = UsableVecField<TensorFieldOptions<Types>{}>;

double constexpr interior_value = 5.;
double constexpr ghost_sentinel = 99.;

// cells per direction: x 10, y 8, z 6
std::array<std::uint32_t, 3> constexpr n_cells{10, 8, 6};

template<typename Layout_t>
Layout_t make_layout(std::array<double, Layout_t::dimension> const& dx)
{
    Layout_t layout;
    for (std::size_t d = 0; d < Layout_t::dimension; ++d)
    {
        layout.cells.lower[d] = 0;
        layout.cells.upper[d] = static_cast<int>(n_cells[d]) - 1;
    }
    layout.dx = dx;
    return layout;
}

template<std::size_t dim>
std::array<double, dim> uniform_dx(double const dl = .1)
{
    std::array<double, dim> dx;
    dx.fill(dl);
    return dx;
}

// the ghost cells past the boundary at `location` (local index space)
template<typename Layout_t>
auto ghost_cells(Layout_t const& layout, BoundaryLocation const location)
{
    auto constexpr dim = Layout_t::dimension;
    auto constexpr gw  = Layout_t::ghost_width;
    auto const shape   = layout.cells.shape();
    damn::core::Box<std::uint32_t, dim> box;
    for (std::size_t d = 0; d < dim; ++d)
    {
        box.lower[d] = 0;
        box.upper[d] = static_cast<std::uint32_t>(shape[d]) + 2 * gw - 1;
    }
    auto const n = static_cast<std::size_t>(direction_of(location));
    if (side_of(location) == Side::Lower)
        box.upper[n] = gw - 1;
    else
        box.lower[n] = gw + static_cast<std::uint32_t>(shape[n]);
    return box;
}

// the sentinel on the ghost nodes, the interior value (or fn(node)) over the domain's
template<typename Field, typename Layout_t>
void fill_interior(Field& field, Layout_t const& layout, auto&& fn)
{
    damn::core::for_each_point(ghost_box(layout, field),
                               [&](auto const& index) { field(index) = ghost_sentinel; });
    damn::core::for_each_point(domain_box(layout, field),
                               [&](auto const& index) { field(index) = fn(index); });
}

template<typename Field, typename Layout_t>
void fill_interior(Field& field, Layout_t const& layout)
{
    fill_interior(field, layout, [](auto const&) { return interior_value; });
}

// a hybrid rho (primal)
template<std::size_t dim_>
struct FieldBC : public ::testing::Test
{
    auto constexpr static dim = dim_;
    using Layout_t            = HybridTypes<dim>::Layout_t;
    using Grid_t              = HybridTypes<dim>::Grid_t;
    auto constexpr static gw  = Layout_t::ghost_width;
    auto constexpr static qty = HybridQuantity::Scalar::rho;

    FieldBC() { fill_interior(field, layout); }

    // the field's allocation along d
    std::uint32_t alloc(std::size_t const d) const { return ghost_box(layout, qty).upper[d] + 1; }
    std::uint32_t phys_start(std::size_t const d) const { return domain_box(layout, qty).lower[d]; }
    std::uint32_t phys_end(std::size_t const d) const { return domain_box(layout, qty).upper[d]; }

    Layout_t layout = make_layout<Layout_t>(uniform_dx<dim>());
    Grid_t field{"rho", layout, qty};
};

using FieldBC1D = FieldBC<1>;
using FieldBC2D = FieldBC<2>;
using FieldBC3D = FieldBC<3>;

// a hybrid B: By holds its local y index over its domain nodes if non_uniform_by
template<std::size_t dim_, bool non_uniform_by = false, bool anisotropic = false>
struct VecFieldBC : public ::testing::Test
{
    auto constexpr static dim    = dim_;
    using Types                  = HybridTypes<dim>;
    using Layout_t               = Types::Layout_t;
    auto constexpr static gw     = Layout_t::ghost_width;
    auto constexpr static vecQty = HybridQuantity::Vector::B;

    static std::array<double, dim> dx()
    {
        auto dl = uniform_dx<dim>();
        if constexpr (anisotropic)
            dl[1] = .2;
        return dl;
    }

    VecFieldBC()
    {
        for (std::size_t c = 0; c < 3; ++c)
            fill_interior(B[c], layout, [&](auto const& index) {
                if constexpr (non_uniform_by)
                    if (c == 1)
                        return static_cast<double>(index[1]);
                return interior_value;
            });
    }

    // component c's domain nodes along d (local index space)
    std::uint32_t phys_start(std::size_t const c, std::size_t const d) const
    {
        return domain_box(layout, B[c]).lower[d];
    }
    std::uint32_t phys_end(std::size_t const c, std::size_t const d) const
    {
        return domain_box(layout, B[c]).upper[d];
    }

    Layout_t layout = make_layout<Layout_t>(dx());
    VecField_t<Types> B{"B", layout, vecQty};
};

using VecFieldBC1D                        = VecFieldBC<1>;
using VecFieldBC2D                        = VecFieldBC<2>;
using VecFieldBC2DNonUniformBy            = VecFieldBC<2, true>;
using VecFieldBC2DNonUniformByAnisotropic = VecFieldBC<2, true, true>;
using VecFieldBC3D                        = VecFieldBC<3>;

// a (anti)symmetric condition's ghosts along direction n, both sides, for each component over its
//  domain nodes along the other directions: the first ghost past each boundary is sign times the
//  interior value (normal_sign for the normal component, tangential_sign for the others), and if
//  normal_boundary_zero the normal component's boundary nodes are 0
template<typename VecField, typename Layout_t>
void expect_reflected(VecField const& B, Layout_t const& layout, std::size_t const n,
                      double const normal_sign, double const tangential_sign,
                      bool const normal_boundary_zero)
{
    for (std::size_t c = 0; c < 3; ++c)
    {
        auto const box  = domain_box(layout, B[c]);
        auto const ps   = box.lower[n];
        auto const pe   = box.upper[n];
        auto const sign = c == n ? normal_sign : tangential_sign;
        auto plane      = box;
        plane.upper[n]  = plane.lower[n];
        damn::core::for_each_point(plane, [&](auto const& index) {
            auto const at = [&](std::uint32_t const i) {
                auto node = index;
                node[n]   = i;
                return B[c](node);
            };
            EXPECT_DOUBLE_EQ(at(ps - 1), sign * interior_value) << "comp=" << c << " lower ghost";
            EXPECT_DOUBLE_EQ(at(pe + 1), sign * interior_value) << "comp=" << c << " upper ghost";
            if (c == n and normal_boundary_zero)
            {
                EXPECT_DOUBLE_EQ(at(ps), 0.0) << "comp=" << c << " lower boundary";
                EXPECT_DOUBLE_EQ(at(pe), 0.0) << "comp=" << c << " upper boundary";
            }
        });
    }
}

} // namespace plamn::core::bc_test

#endif // PLAMN_TEST_CORE_NUMERICS_BOUNDARY_CONDITION_BC_TEST_FIXTURES_HPP
