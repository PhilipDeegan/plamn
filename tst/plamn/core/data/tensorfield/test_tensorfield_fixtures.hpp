#ifndef PLAMN_TEST_CORE_DATA_TEST_TENSORFIELD_FIXTURES_HPP
#define PLAMN_TEST_CORE_DATA_TEST_TENSORFIELD_FIXTURES_HPP

#include "plamn/core/def.hpp"

#include "plamn/plamn_core.hpp"
#include "damn/core/def/damn_config.hpp"

#include "damn/core/math/array.hpp"
#include "damn/core/meta/for_N.hpp"
#include "damn/core/math/equality.hpp"
#include "plamn/core/data/tensorfield/tensorfield.hpp"
#include "plamn/core/data/tensorfield/usable_tensorfield.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"

#include "tst/plamn/core/data/grid/test_grid_fixtures.hpp"
#include "tst/plamn/core/data/field/test_field_fixtures.hpp"

#include <cstddef>

namespace plamn::core
{


template<auto o0, auto o1>
damn::core::EqualityReport compare_reduced_tensor_fields(UsableTensorField<o0> const& ref,
                                             UsableTensorField<o1> const& cmp, double const diff)
{
    static_assert(o0.rank == o1.rank);
    auto constexpr static N_elements = detail::tensor_field_dim_from_rank<o0.rank>();

    std::stringstream log;
    log << std::endl;
    for (std::size_t ci = 0; ci < N_elements; ++ci)
        if (auto eqr = compare_reduced_fields(ref[ci], cmp[ci], diff); !eqr)
            return eqr;
        else
            log << eqr.what() << std::endl;

    return damn::core::EqualityReport{true, log.str()};
}




template<typename F0, typename F1, typename PhysicalQuantity, std::size_t rank>
damn::core::EqualityReport compare_reduced_tensor_fields(TensorField<F0, PhysicalQuantity, rank> const& ref,
                                             TensorField<F1, PhysicalQuantity, rank> const& cmp,
                                             double const diff)
{
    auto constexpr static N_elements = detail::tensor_field_dim_from_rank<rank>();

    if (ref.componentNames() != cmp.componentNames())
        return damn::core::EqualityReport{false, "Tensorfield component mismatch"};

    auto const same_sizes = [&]() {
        return damn::core::for_N_all<N_elements>([&](auto i) { return ref[i].size() == cmp[i].size(); });
    }();

    if (!same_sizes)
        return damn::core::EqualityReport{false, "Tensorfield shape/size mismatch"};

    std::stringstream log;
    log << std::endl;
    for (std::size_t ci = 0; ci < N_elements; ++ci)
        if (auto eqr = compare_reduced_fields(ref[ci], cmp[ci], diff); !eqr)
            return eqr;
        else
            log << eqr.what() << std::endl;

    return damn::core::EqualityReport{true, log.str()};
}




template<auto o0, auto o1>
damn::core::EqualityReport compare_tensor_fields(UsableTensorField<o0> const& ref,
                                     UsableTensorField<o1> const& cmp, double const diff)
{
    static_assert(o0.rank == o1.rank);
    auto constexpr static N_elements = detail::tensor_field_dim_from_rank<o0.rank>();

    std::stringstream log;
    log << std::endl;
    for (std::size_t ci = 0; ci < N_elements; ++ci)
        if (auto eqr = compare_fields(ref[ci], cmp[ci], diff); !eqr)
            return eqr;
        else
            log << eqr.what() << std::endl;

    return damn::core::EqualityReport{true, log.str()};
}




template<typename F0, typename F1, typename PhysicalQuantity, std::size_t rank>
damn::core::EqualityReport compare_tensor_fields(TensorField<F0, PhysicalQuantity, rank> const& ref,
                                     TensorField<F1, PhysicalQuantity, rank> const& cmp,
                                     double const diff)
{
    auto constexpr static N_elements = detail::tensor_field_dim_from_rank<rank>();

    if (ref.componentNames() != cmp.componentNames())
        return damn::core::EqualityReport{false, "Tensorfield component mismatch"};

    auto const same_sizes = [&]() {
        return damn::core::for_N_all<N_elements>([&](auto i) { return ref[i].size() == cmp[i].size(); });
    }();

    if (!same_sizes)
        return damn::core::EqualityReport{false, "Tensorfield shape/size mismatch"};

    std::stringstream log;
    log << std::endl;
    for (std::size_t ci = 0; ci < N_elements; ++ci)
        if (auto eqr = compare_fields(ref[ci], cmp[ci], diff); !eqr)
            return eqr;
        else
            log << eqr.what() << std::endl;

    return damn::core::EqualityReport{true, log.str()};
}


template<auto o>
std::ostream& operator<<(std::ostream& out, UsableTensorField<o> const& ts)
{
    return (out << *ts);
}


template<typename F0, typename PhysicalQuantity, std::size_t rank>
void zero_ghost_layer(TensorField<F0, PhysicalQuantity, rank>& tf, auto const& layout)
{
    for (auto& f : tf)
        zero_ghost_layer(f, layout);
}

void zero_ghost_layer([[maybe_unused]] auto& layout, auto&... args)
{
    (zero_ghost_layer(args), ...);
}

} // namespace plamn::core


#endif /*PLAMN_TEST_CORE_DATA_TEST_TENSORFIELD_FIXTURES_HPP*/
