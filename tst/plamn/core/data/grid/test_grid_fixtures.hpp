#ifndef PLAMN_TEST_CORE_DATA_TEST_GRID_FIXTURES_HPP
#define PLAMN_TEST_CORE_DATA_TEST_GRID_FIXTURES_HPP

#include "plamn/core/def.hpp"

#include "plamn/core/data/grid/grid.hpp"
#include "damn/core/math/equality.hpp"
#include "plamn/core/data/grid/grid_tiles.hpp"
#include "plamn/core/data/field/field_box.hpp"

#include "tst/plamn/core/data/field/test_field_fixtures.hpp"

namespace plamn::core
{


template<typename... T0s, typename... T1s>
damn::core::EqualityReport compare_reduced_fields(GridTileSet<T0s...> const& ref, Grid<T1s...> const& cmp,
                                      double const diff = 1e-15)
{
    if (!ref.size() != cmp.size())
        return damn::core::EqualityReport{false, "Grid/GridTileset shape/size mismatch"};
    auto tmp = cmp;
    tmp.zero();
    reduce_into(tmp, ref);
    return compare_fields(*tmp, *cmp, diff);
}


template<typename... T0s, typename... T1s>
damn::core::EqualityReport compare_reduced_fields(Grid<T0s...> const& ref, GridTileSet<T1s...> const& cmp,
                                      double const diff = 1e-15)
{
    if (ref.size() != cmp.size())
        return damn::core::EqualityReport{false, "Grid/GridTileset shape/size mismatch"};
    auto tmp = ref;
    tmp.zero();
    reduce_into(tmp, cmp);
    return compare_fields(*ref, *tmp, diff);
}


template<typename... T0s, typename... T1s>
damn::core::EqualityReport compare_fields(GridTileSet<T0s...> const& ref, Grid<T1s...> const& cmp,
                              double const diff = 1e-15)
{
    if (ref.size() != cmp.size())
        return damn::core::EqualityReport{false, "Grid/GridTileset shape/size mismatch"};
    auto tmp = cmp;
    tmp.zero();
    reduce_single(tmp, ref);
    return compare_fields(*tmp, *cmp, diff);
}


template<typename... T0s, typename... T1s>
damn::core::EqualityReport compare_fields(Grid<T0s...> const& ref, GridTileSet<T1s...> const& cmp,
                              double const diff = 1e-15)
{
    if (ref.size() != cmp.size())
        return damn::core::EqualityReport{false, "Grid/GridTileset shape/size mismatch"};
    auto tmp = ref;
    tmp.zero();
    reduce_single(tmp, cmp);
    return compare_fields(*ref, *tmp, diff);
}


template<typename... T0s, typename... T1s>
damn::core::EqualityReport compare_reduced_fields(Grid<T0s...> const& ref, Grid<T1s...> const& cmp,
                                      double const diff = 1e-15)
{
    if (ref.size() != cmp.size())
        return damn::core::EqualityReport{false, "Grid shape/size mismatch"};
    return compare_fields(*ref, *cmp, diff);
}


template<auto opts>
void zero_ghost_layer(basic::Field<opts>& f, auto const& layout)
{
    for (auto const& ghost : ghost_box(layout, f).remove(domain_box(layout, f)))
        damn::core::for_each_point(ghost, [&](auto const& bix) { f(bix) = 0; });
}

template<typename Layout_t, typename Grid_t, typename Field_t>
void zero_ghost_layer(FieldTileSet<Layout_t, Grid_t, Field_t>& ts, Layout_t const& layout)
{
    auto const pq = ts.physicalQuantity();
    for (auto const& ghost : ghost_box(layout, pq).remove(domain_box(layout, pq)))
        FieldBox{ts, layout, ghost}.op(0);
}

} // namespace plamn::core


#endif /*PLAMN_TEST_CORE_DATA_TEST_GRID_FIXTURES_HPP*/
