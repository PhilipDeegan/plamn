#ifndef PLAMN_CORE_DATA_FIELD_FIELD_EXPR_HPP
#define PLAMN_CORE_DATA_FIELD_FIELD_EXPR_HPP

// lazy pointwise field expressions: B - dt * curl(E) builds a small tree of views, evaluated
// once per point by assign(dst, box, expr); nothing is materialised. Stencils (Partial, curl,
// ...) are leaves that read shifted points; everything else is pointwise. An expression is plain
// views and scalars, so it is trivially copyable to the device.

#include "damn/core/geometry/point.hpp"
#include "mkn/kul/for.hpp"
#include "damn/core/data/ndarray/ndarray_view.hpp"
#include "damn/core/def/damn_config.hpp"
#include "damn/core/geometry/box_spans.hpp"

#include "plamn/core/def.hpp"
#include "plamn/core/data/grid/detail/detail.hpp"

#include <cstddef>
#include <type_traits>

namespace plamn::core
{
template<typename T>
concept Expr = requires { typename std::remove_cvref_t<T>::expr_tag; };

template<typename T>
concept ExprOperand = Expr<T> or std::is_arithmetic_v<std::remove_cvref_t<T>>;


namespace expr
{
    template<std::size_t dim, typename T>
    struct Leaf
    {
        using expr_tag = void;

        damn::core::NdArrayView<dim, T const> view;

        template<typename I>
        NO_DISCARD auto operator()(damn::core::Point<I, dim> const& i) const DAMN_ALL
        {
            return mkn::kul::for_N_fold<dim>([&](auto... d) { return view(i[d]...); });
        }
    };

    struct Scalar
    {
        using expr_tag = void;

        double value;

        NO_DISCARD double operator()(auto const&) const DAMN_ALL { return value; }
    };

    struct Plus
    {
        NO_DISCARD auto operator()(auto const a, auto const b) const DAMN_ALL { return a + b; }
    };
    struct Minus
    {
        NO_DISCARD auto operator()(auto const a, auto const b) const DAMN_ALL { return a - b; }
    };
    struct Times
    {
        NO_DISCARD auto operator()(auto const a, auto const b) const DAMN_ALL { return a * b; }
    };
    struct Divides
    {
        NO_DISCARD auto operator()(auto const a, auto const b) const DAMN_ALL { return a / b; }
    };

    template<typename Op, typename L, typename R>
    struct Binary
    {
        using expr_tag = void;

        L l;
        R r;

        NO_DISCARD auto operator()(auto const& i) const DAMN_ALL { return Op{}(l(i), r(i)); }
    };

    template<typename E>
    struct Negate
    {
        using expr_tag = void;

        E e;

        NO_DISCARD auto operator()(auto const& i) const DAMN_ALL { return -e(i); }
    };

    template<typename T>
    NO_DISCARD auto lift(T const& t)
    {
        if constexpr (Expr<T>)
            return t;
        else
            return Scalar{static_cast<double>(t)};
    }

    template<typename Op>
    NO_DISCARD auto binary(auto const& a, auto const& b)
    {
        using L = decltype(lift(a));
        using R = decltype(lift(b));
        return Binary<Op, L, R>{lift(a), lift(b)};
    }
} // namespace expr


// a field as an expression leaf (a view, no ownership)
template<typename Field>
NO_DISCARD auto of(Field const& f)
{
    using T = std::remove_cv_t<std::remove_reference_t<decltype(*f.data())>>;
    return expr::Leaf<Field::dimension, T>{{f.data(), f.shape()}};
}


template<ExprOperand A, ExprOperand B>
    requires(Expr<A> or Expr<B>)
NO_DISCARD auto operator+(A const& a, B const& b)
{
    return expr::binary<expr::Plus>(a, b);
}

template<ExprOperand A, ExprOperand B>
    requires(Expr<A> or Expr<B>)
NO_DISCARD auto operator-(A const& a, B const& b)
{
    return expr::binary<expr::Minus>(a, b);
}

template<ExprOperand A, ExprOperand B>
    requires(Expr<A> or Expr<B>)
NO_DISCARD auto operator*(A const& a, B const& b)
{
    return expr::binary<expr::Times>(a, b);
}

template<ExprOperand A, ExprOperand B>
    requires(Expr<A> or Expr<B>)
NO_DISCARD auto operator/(A const& a, B const& b)
{
    return expr::binary<expr::Divides>(a, b);
}

template<Expr E>
NO_DISCARD auto operator-(E const& e)
{
    return expr::Negate<E>{e};
}


// dst = e at every point of box (dst's local index space)
template<typename Dst, Expr E>
void assign(Dst& dst, auto const& box, E const& e)
{
    auto const fn = [] DAMN_ALL(auto const& ijk, auto& d, auto const& ex) {
        d(ijk) = ex(ijk);
    };

    if constexpr (Dst::alloc_mode == damn::AllocatorMode::CPU)
        damn::core::for_each_point(box, [&](auto const& ijk) { fn(ijk, dst, e); });
    else
        gpu::eval_on_box(fn, box, dst, e);
}

} // namespace plamn::core

#endif // PLAMN_CORE_DATA_FIELD_FIELD_EXPR_HPP
