#ifndef PLAMN_CORE_DATA_TENSORFIELD_USABLE_TENSORFIELD_HPP
#define PLAMN_CORE_DATA_TENSORFIELD_USABLE_TENSORFIELD_HPP

// UsableTensorField: a TensorField view that owns its component grids and points at them,
// a TensorField with storage. UsableVecField is the rank 1 one

#include "plamn/core/def.hpp"
#include "damn/core/meta/for_N.hpp"
#include "plamn/core/data/tensorfield/tensorfield.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"
#include <array>
#include <cstddef>
#include <optional>
#include <ostream>
#include <string>

namespace plamn::core
{
// Usable* constructors given no_fill allocate only: production code writes everything
// before reading it; without it they fill on construction, as test fixtures expect
struct no_fill_t
{
};
inline constexpr no_fill_t no_fill{};

template<typename core_types>
struct TensorFieldOptions
{
    using Layout_t         = core_types::Layout_t;
    using Grid_t               = core_types::Grid_t;
    using Field_t              = core_types::Field_t;
    auto constexpr static opts = core_types::opts;
    std::size_t rank           = 2;

    auto constexpr with_rank(std::size_t const r) const
    {
        auto copy = *this;
        copy.rank = r;
        return copy;
    }
};


template<auto opts>
struct UsableTensorFieldImpl
{
    using Options      = decltype(opts);
    using Layout_t = Options::Layout_t;
    using Grid_t       = Options::Grid_t;
    using Field_t      = Options::Field_t;
    using Super_t      = TensorField<Field_t, HybridQuantity, opts.rank>;
};




/*
A UsableTensorField is an extension of the TensorField view that owns memory for components and sets
the view pointers. It is useful for tests to easily declare usable (== set views) tensors

Note: UsableTensorFields hold Grids that are default initialized to zero for convenience rather
than NaN (default grid init value)

*/
template<auto opts>
class UsableTensorField : public UsableTensorFieldImpl<opts>::Super_t
{
    using Options = decltype(opts);

    auto constexpr static N_elements = detail::tensor_field_dim_from_rank<opts.rank>();
    using Impl_t                     = UsableTensorFieldImpl<opts>;
    using This                       = UsableTensorField;

public:
    using Layout_               = Options::Layout_t;
    auto static constexpr dimension = Layout_::dimension;
    using Super                     = Impl_t::Super_t;
    using Grid_t                    = Impl_t::Grid_t;
    using Field_t                   = Impl_t::Field_t;
    auto static constexpr rank      = Super::rank;


protected:
    using tensor_t = Super::tensor_t;

    void _set()
    {
        for (std::size_t i = 0; i < N_elements; ++i)
            super()[i].setBuffer(&xyz[i]);
    }



public:
    UsableTensorField(std::string const& name, Layout_ const& layout, tensor_t qty,
                      std::optional<double> v = std::nullopt)
        : Super{name, qty}
        , xyz{make_grids(Super::componentNames(), layout, qty)}
    {
        if (v)
            for (std::size_t i = 0; i < N_elements; ++i)
                xyz[i].fill(*v);
        _set();
    }

    UsableTensorField(UsableTensorField&& that)
        : Super{std::forward<Super>(that)}
        , xyz{std::move(that.xyz)}
    {
        _set();
    }

    UsableTensorField(UsableTensorField const& that)
        : Super{that}
        , xyz{that.xyz}
    {
        _set();
    }


    void set_on(Super& tensorfield)
    {
        // used for setting on normal model tensorfields
        for (std::size_t i = 0; i < N_elements; ++i)
            tensorfield[i].setBuffer(&xyz[i]);
    }


    Super& super() { return *this; }
    Super& super() const { return *this; }

    Super& view() { return *this; }
    Super const& view() const { return *this; }

    auto& operator*() { return view(); }
    auto& operator*() const { return view(); }

    auto& grids() const { return xyz; }

    auto& operator[](std::size_t const i) DAMN_ALL { return xyz[i]; }
    auto& operator[](std::size_t const i) const DAMN_ALL { return xyz[i]; }


    template<auto o>
    bool isclose(UsableTensorField<o> const& that, double diff = 1e-15) const
    {
        return damn::core::for_N_all<N_elements>(
            [&](auto i) { return (*this)[i].isclose(that[i], diff); });
    }

    template<auto o>
    bool operator==(UsableTensorField<o> const& that) const
    {
        return damn::core::for_N_all<N_elements>([&](auto i) { return (*this)[i] == that[i]; });
    }

    void fill(double const v)
    {
        for (auto& c : xyz)
            c.fill(v);
    }

    template<auto o>
    friend std::ostream& operator<<(std::ostream&, UsableTensorField<o> const&);

protected:
    template<typename ComponentNames, typename Layout>
    auto static make_grids(ComponentNames const& compNames, Layout const& layout, tensor_t qty)
    {
        auto qts = HybridQuantity::componentsQuantities(qty);
        return damn::core::for_N<N_elements, damn::core::for_N_R_mode::make_array>(
            [&](auto i) { return Grid_t{compNames[i], layout, qts[i], 0}; });
    }

    std::array<Grid_t, N_elements> xyz;
};






template<auto opts>
using UsableVecField = UsableTensorField<opts.with_rank(1)>;

} // namespace plamn::core

#endif // PLAMN_CORE_DATA_TENSORFIELD_USABLE_TENSORFIELD_HPP
