#ifndef PLAMN_TENSORFIELD_HPP
#define PLAMN_TENSORFIELD_HPP

#include "mkn/kul/for.hpp"
#include "damn/core/def/damn_config.hpp"
#include "mkn/kul/meta.hpp"

#include "plamn/core/def.hpp"
#include "plamn/core/data/field/field.hpp"
#include "plamn/core/data/grid/grid_tiles.hpp"
#include "plamn/core/data/field/field_tiles.hpp"

#include <array>
#include <string>
#include <vector>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <algorithm>
#include <format>
#include <initializer_list>
#include <iterator>
#include <type_traits>
#include <version>
#include <cstdint>

namespace plamn::core
{
enum class Component : std::uint16_t;
} // namespace plamn::core

namespace plamn::core::detail
{
template<std::size_t rank>
constexpr static std::size_t tensor_field_dim_from_rank()
{
    static_assert(rank > 0 and rank < 3);
    if constexpr (rank == 1) // Vector field
        return 3;
    else if constexpr (rank == 2) // symmetric 3x3 tensor field
        return 6;
    return 0; // nvcc complains about no return
}

template<std::size_t rank, typename R = std::array<std::string, tensor_field_dim_from_rank<rank>()>>
R static tensor_field_names(std::string const& name)
{
    if constexpr (rank == 1)
        return {{name + "_x", name + "_y", name + "_z"}};

    else if constexpr (rank == 2)
        return {
            {name + "_xx", name + "_xy", name + "_xz", name + "_yy", name + "_yz", name + "_zz"}};
}

template<typename Field_t, std::size_t N, typename Qtys>
auto static tensor_field_make_fields(std::array<std::string, N> const& names, Qtys const& qtys)
{
    return mkn::kul::for_N_make_array<N>([&](auto i) { return Field_t{names[i], qtys[i]}; });
}

template<std::size_t rank>
auto static tensor_field_index_for(Component const component) DAMN_ALL
{
    auto const val = static_cast<std::underlying_type_t<Component>>(component);
    if constexpr (rank == 1)
        return val;
    else if constexpr (rank == 2)
        return val - detail::tensor_field_dim_from_rank<1>();
}

} // namespace plamn::core::detail

namespace plamn::core::basic
{

template<typename Field_t, std::size_t rank>
struct TensorField
{
    using This                             = TensorField;
    using field_type                       = Field_t;
    using value_type                       = Field_t::value_type;
    auto constexpr static N                = detail::tensor_field_dim_from_rank<rank>();
    static constexpr std::size_t dimension = Field_t::dimension;

    TensorField(std::array<Field_t, N> const& components) DAMN_ALL : components_{components}
    {
    }

    TensorField(std::array<Field_t, N>& components) DAMN_ALL : components_{components} {}
    TensorField() DAMN_ALL = default; // null views if Field_t is a view
    TensorField(TensorField const& source) DAMN_ALL            = default;
    TensorField(TensorField&& source) DAMN_ALL                 = default;
    TensorField& operator=(TensorField const& source) DAMN_ALL = default;
    TensorField& operator=(TensorField&& source) DAMN_ALL      = default;


    template<typename... Args>
    TensorField(Args&&... args)
        requires(sizeof...(Args) == N)
    DAMN_ALL : components_{args...}
    {
    }


    auto& operator[](std::size_t const i) DAMN_ALL { return components_[i]; }
    auto& operator[](std::size_t const i) const DAMN_ALL { return components_[i]; }
    auto& operator()() DAMN_ALL { return components_; }
    auto& operator()() const DAMN_ALL { return components_; }

    auto& operator()(Component component) DAMN_ALL
    {
        return components_[detail::tensor_field_index_for<rank>(component)];
    }

    auto& operator()(Component component) const DAMN_ALL
    {
        return components_[detail::tensor_field_index_for<rank>(component)];
    }

    auto begin() { return std::begin(components_); }
    auto begin() const { return std::begin(components_); }
    auto end() { return std::end(components_); }
    auto end() const { return std::end(components_); }
    auto constexpr size() const { return N; }

    void zero()
    {
        for (auto& component : components_)
            component.zero();
    }

    void setBuffer(std::nullptr_t /*ptr*/)
    {
        mkn::kul::for_N<N, mkn::kul::for_N_R_mode::forward_tuple>(
            [&](auto i) { components_[i].setBuffer(nullptr); });
    }

    template<typename Fields>
    void setBuffer(Fields* const fields)
    {
        if (!fields)
            throw std::runtime_error("use other fn");
        mkn::kul::for_N<N>([&](auto i) { components_[i].setBuffer(&(*fields)[i]); });
    }


    //! return true if the TensorField can be used to access component data
    MKN_KUL_FN_SIG_RET bool isUsable() const DAMN_ALL
    {
        return std::all_of(std::begin(components_), std::end(components_),
                           [](auto const& c) { return c.isUsable(); });
    }

    MKN_KUL_FN_SIG_RET bool isSettable() const DAMN_ALL
    {
        return std::all_of(std::begin(components_), std::end(components_),
                           [](auto const& c) { return c.isSettable(); });
    }

    template<typename V>
    auto as(auto&& a, auto&&... args) DAMN_ALL
    {
        V r{mkn::kul::for_N<N, mkn::kul::for_N_R_mode::make_array>(
            [&](auto i) { return a(components_[i], args...); })};
        DAMN_ASSERT(r.isUsable());
        return r;
    }

    template<typename V>
    auto as(auto&& a, auto&&... args) const DAMN_ALL
    {
        return V{mkn::kul::for_N<N, mkn::kul::for_N_R_mode::make_array>(
            [&](auto i) { return a(components_[i], args...); })};
    }

    std::array<Field_t, N> components_;
};


} // namespace plamn::core::basic

namespace plamn::core
{
template<typename Field_t, typename PhysicalQuantity, std::size_t rank_ = 1>
class TensorField : public basic::TensorField<Field_t, rank_>
{
protected:
    auto constexpr static N = detail::tensor_field_dim_from_rank<rank_>();

public:
    static constexpr std::size_t dimension = Field_t::dimension;
    static constexpr std::size_t rank      = rank_;

    using Super = basic::TensorField<Field_t, rank>;
    using This  = TensorField<Field_t, PhysicalQuantity, rank>;

protected:
    using Super::components_;

public:
    using physical_quantity_type = PhysicalQuantity;
    using field_type             = Field_t;
    using raw_array_t            = field_type (&)[N];
    using raw_array_ct           = field_type const (&)[N];
    using value_type             = Field_t::type;
    using tensor_t               = PhysicalQuantity::template TensorType<rank>;


    TensorField()                                     = delete;
    TensorField(TensorField const& source)            = default;
    TensorField(TensorField&& source)                 = default;
    TensorField& operator=(TensorField const& source) = default;
    TensorField& operator=(TensorField&& source)      = default;

    TensorField(auto&& name, tensor_t physQty, auto&& cnames, auto&& cqts)
        : Super{detail::tensor_field_make_fields<Field_t>(cnames, cqts)}
        , qty_{physQty}
        , name_{name}
        , physQties_{cqts}
        , componentNames_{cnames}
        , nameToIndex_{makeMap_()}
    {
    }

    TensorField(std::string const& name, tensor_t physQty)
        : TensorField{name, physQty, detail::tensor_field_names<rank>(name),
                      PhysicalQuantity::componentsQuantities(physQty)}
    {
    }



    //-------------------------------------------------------------------------
    //                  start the ResourcesUser interface
    //-------------------------------------------------------------------------




    void zero()
    {
        DAMN_DEBUG_DO({
            if (!this->isUsable())
                throw std::runtime_error(
                    "Error, cannot zero the TensorField because it is not usable");
        })

        for (auto& component : components_)
            component.zero();
    }


    //-------------------------------------------------------------------------
    //                  ends the ResourcesUser interface
    //-------------------------------------------------------------------------

    MKN_KUL_FN_SIG_RET std::string const& name() const { return name_; }




    MKN_KUL_FN_SIG_RET field_type& getComponent(Component component) DAMN_ALL
    {
        return components_[detail::tensor_field_index_for<rank>(component)];
    }

    MKN_KUL_FN_SIG_RET field_type const& getComponent(Component component) const DAMN_ALL
    {
        return components_[detail::tensor_field_index_for<rank>(component)];
    }


    MKN_KUL_FN_SIG_RET std::string getComponentName(Component component) const
    {
        return componentNames_[detail::tensor_field_index_for<rank>(component)];
    }



    MKN_KUL_FN_SIG_RET auto& components() const DAMN_ALL
    {                       // std::array can't work on gpu?
        return components_; // reinterpret_cast<raw_array_ct>(components_);
    }
    MKN_KUL_FN_SIG_RET auto& components() DAMN_ALL
    {
        return components_; // reinterpret_cast<raw_array_t>(components_);
    }


    void copyData(TensorField const& source)
    {
        DAMN_DEBUG_DO({
            if (!this->isUsable() || !source.isUsable())
                throw std::runtime_error("Error, unusable TensorField, cannot copyData");
        })

        mkn::kul::for_N<N>([&](auto i) { components_[i].copyData(source.components_[i]); });
    }



    MKN_KUL_FN_SIG_RET auto begin() { return std::begin(components_); }
    MKN_KUL_FN_SIG_RET auto begin() const { return std::cbegin(components_); }
    MKN_KUL_FN_SIG_RET auto end() { return std::end(components_); }
    MKN_KUL_FN_SIG_RET auto end() const { return std::cend(components_); }


    MKN_KUL_FN_SIG_RET auto& componentNames() const { return componentNames_; }
    MKN_KUL_FN_SIG_RET auto& physicalQuantity() const { return qty_; }
    MKN_KUL_FN_SIG_RET auto static constexpr size() { return N; }

    Super& operator*() { return *this; }
    Super const& operator*() const { return *this; }

    template<typename, typename, std::size_t>
    friend std::ostream& operator<<(std::ostream& out, TensorField const&);

private:
    auto makeMap_() const
    {
        std::unordered_map<std::string, std::size_t> m;
        for (std::size_t i = 0; i < N; ++i)
            m[componentNames_[i]] = i;
        return m;
    }


    tensor_t qty_;

    std::string const name_{"No Name"};
    std::array<typename PhysicalQuantity::Scalar, N> physQties_;
    std::array<std::string, N> const componentNames_;
    std::unordered_map<std::string, std::size_t> const nameToIndex_;
};



template<typename Field_t, typename PhysicalQuantity>
using SymTensorField = TensorField<Field_t, PhysicalQuantity, /*rank=*/2>;



template<typename Field_t, typename PhysicalQuantity, std::size_t rank>
inline std::ostream& operator<<(std::ostream& out,
                                TensorField<Field_t, PhysicalQuantity, rank> const& ts)
{
    auto constexpr dim = Field_t::dimension;

    out << ts.name() << std::endl;

    auto const names = detail::tensor_field_names<rank>(ts.name());
    for (std::size_t i = 0; i < ts.size(); ++i)
    {
        out << "component " << names[i] << std::endl;
        if constexpr (is_field_tile_set_v<Field_t>)
            for (auto const& tile : ts[i]())
                out << tile();

        else
            out << ts[i];
    }

    return out;
}

template<typename TensorField>
void check_tensor_field(std::vector<TensorField> const& vec)
{
#if DAMN_DEBUG
    for (auto const& tf : vec)
        check_tensor_field(mkn::kul::deref(tf));
#endif // DAMN_DEBUG
}


template<typename Field_t, std::size_t rank>
void check_tensor_field(basic::TensorField<Field_t, rank> const& tf, auto const& layout)
{
    static_assert(is_field_v<Field_t> || is_field_tile_set_v<Field_t>);

#if DAMN_DEBUG
    if constexpr (is_field_v<Field_t>)
    {
        for (auto const& field : tf)
            check_field(field, layout);
    }
    else if constexpr (is_field_tile_set_v<Field_t>)
    {
        using Tile_vt        = Field_t::value_type::value_type;
        using TensorField_vt = basic::TensorField<Tile_vt, rank>;

        for (std::size_t tidx = 0; tidx < tf[0].ntiles(); ++tidx)
            check_tensor_field(tf.template as<TensorField_vt>([&](auto& c) { return c()[tidx](); }),
                               tf[0][tidx].layout());
    }
    else
        static_assert(mkn::kul::dependent_false_v<Field_t>);
#endif // DAMN_DEBUG
}


template<typename Field_t, std::size_t rank>
void check_tensor_field(basic::TensorField<Field_t, rank> const& tf)
{
    static_assert(is_field_v<Field_t> || is_field_tile_set_v<Field_t>);

#if DAMN_DEBUG
    if constexpr (is_field_v<Field_t>)
    {
        for (auto const& field : tf)
            check_field(field);
    }
    else if constexpr (is_field_tile_set_v<Field_t>)
    {
        using Tile_vt        = Field_t::value_type::value_type;
        using TensorField_vt = basic::TensorField<Tile_vt, rank>;

        for (std::size_t tidx = 0; tidx < tf[0].ntiles(); ++tidx)
            check_tensor_field(
                tf.template as<TensorField_vt>([&](auto& c) { return c()[tidx](); }));
    }
    else
        static_assert(mkn::kul::dependent_false_v<Field_t>);
#endif // DAMN_DEBUG
}



void check_tensor_fields(auto const&... tfs)
{
#if DAMN_DEBUG
    (check_tensor_field(tfs), ...);
#endif // DAMN_DEBUG
}


} // namespace plamn::core



#endif /* PLAMN_TENSORFIELD_HPP */
