#ifndef PLAMN_CORE_NUMERICS_DIFFERENTIAL_OPERATORS_HPP
#define PLAMN_CORE_NUMERICS_DIFFERENTIAL_OPERATORS_HPP

#include "damn/core/def/constants.hpp"
#include "damn/core/geometry/point.hpp"
#include "mkn/kul/for.hpp"
#include "damn/core/data/ndarray/ndarray_view.hpp"
#include "damn/core/def/damn_config.hpp"

#include "plamn/core/def.hpp"
#include "plamn/core/data/grid/layout_defs.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"
#include "plamn/core/data/vecfield/vecfield_component.hpp"
#include "plamn/core/models/quantities/quantity.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>

namespace plamn::core::detail
{
template<std::size_t d, typename T, std::size_t dim>
double shifted(auto const& array, damn::core::Point<T, dim> const& idx, int const shift) DAMN_ALL
{
    return mkn::kul::for_N_fold<dim>(
        [&](auto... i) { return array((i == d ? idx[i] + shift : idx[i])...); });
}

template<Component c>
constexpr std::size_t vector_component_index()
{
    static_assert(c == Component::X or c == Component::Y or c == Component::Z);
    return static_cast<std::size_t>(c);
}

template<auto dir, std::size_t dim>
constexpr bool direction_exists = static_cast<std::size_t>(dir) < dim;

constexpr QtyCentering flipped(QtyCentering const centering)
{
    return centering == QtyCentering::primal ? QtyCentering::dual : QtyCentering::primal;
}
} // namespace plamn::core::detail



namespace plamn::core
{
template<auto dir, std::size_t dim>
class Partial;

template<auto dir, std::size_t dim>
    requires(detail::direction_exists<dir, dim>)
class Partial<dir, dim>
{
    static constexpr auto d = static_cast<std::size_t>(dir);

public:
    using expr_tag = void;

    Partial(std::array<double, dim> const& dx, std::array<QtyCentering, dim> const& srcCentering,
            auto const& src)
        : view_{src.data(), src.shape()}
    {
        if (src.data() == nullptr)
            throw std::runtime_error("partial: the source has no data");

        shift_        = srcCentering[d] == QtyCentering::primal ? 0 : -1;
        scale_        = 1. / dx[d];
        centering_    = srcCentering;
        centering_[d] = detail::flipped(srcCentering[d]);
    }

    Partial(auto const& layout, auto const& src)
        : Partial{layout.dx, ::plamn::core::centering(layout, src), src} // not Partial::centering
    {
    }

    MKN_KUL_FN_SIG_RET std::array<QtyCentering, dim> centering() const DAMN_ALL
    {
        return centering_;
    }

    template<typename T>
    MKN_KUL_FN_SIG_RET double operator()(damn::core::Point<T, dim> const& idx) const DAMN_ALL
    {
        return scale_
               * (detail::shifted<d>(view_, idx, shift_ + 1)
                  - detail::shifted<d>(view_, idx, shift_));
    }

private:
    damn::core::NdArrayView<dim, double const> view_;
    int shift_;
    double scale_;
    std::array<QtyCentering, dim> centering_;
};

template<auto dir, std::size_t dim>
    requires(not detail::direction_exists<dir, dim>)
class Partial<dir, dim>
{
public:
    using expr_tag = void;

    template<typename Src>
    Partial(std::array<double, dim> const&, std::array<QtyCentering, dim> const& srcCentering,
            Src const&)
        : centering_{srcCentering}
    {
    }

    Partial(auto const& layout, auto const& src)
        : centering_{::plamn::core::centering(layout, src)}
    {
    }

    MKN_KUL_FN_SIG_RET std::array<QtyCentering, dim> centering() const DAMN_ALL
    {
        return centering_;
    }

    template<typename T>
    MKN_KUL_FN_SIG_RET double operator()(damn::core::Point<T, dim> const&) const DAMN_ALL
    {
        return 0.;
    }

private:
    std::array<QtyCentering, dim> centering_;
};


namespace detail
{
    template<Component c, std::size_t dim>
    class CurlComponent
    {
        static constexpr auto i = vector_component_index<c>();
        static constexpr auto j = (i + 1) % damn::core::NBR_COMPO;
        static constexpr auto k = (i + 2) % damn::core::NBR_COMPO;

    public:
        using expr_tag = void;

        CurlComponent(std::array<double, dim> const& dx,
                      std::array<std::array<QtyCentering, dim>, 3> const& cs, auto const& v)
            : dj_vk_{dx, cs[k], v[k]}
            , dk_vj_{dx, cs[j], v[j]}
        {
            check_();
        }

        CurlComponent(auto const& layout, auto const& v)
            : dj_vk_{layout, v[k]}
            , dk_vj_{layout, v[j]}
        {
            check_();
        }

        MKN_KUL_FN_SIG_RET std::array<QtyCentering, dim> centering() const DAMN_ALL
        {
            return dj_vk_.centering();
        }

        template<typename T>
        MKN_KUL_FN_SIG_RET double operator()(damn::core::Point<T, dim> const& idx) const DAMN_ALL
        {
            if constexpr (j >= dim && k < dim)
                return -dk_vj_(idx);
            else
                return dj_vk_(idx) - dk_vj_(idx);
        }

    private:
        void check_() const
        {
            if (dj_vk_.centering() != dk_vj_.centering())
                throw std::runtime_error("curl: the two derivatives land on different centerings");
        }

        Partial<static_cast<Direction>(j), dim> dj_vk_;
        Partial<static_cast<Direction>(k), dim> dk_vj_;
    };
} // namespace detail


template<std::size_t dim>
struct Curl
{
    detail::CurlComponent<Component::X, dim> x;
    detail::CurlComponent<Component::Y, dim> y;
    detail::CurlComponent<Component::Z, dim> z;

    template<std::size_t c>
    auto const& get() const DAMN_ALL
    {
        if constexpr (c == 0)
            return x;
        else if constexpr (c == 1)
            return y;
        else
            return z;
    }
};


template<std::size_t dim>
struct Grad
{
    Partial<Direction::X, dim> x;
    Partial<Direction::Y, dim> y;
    Partial<Direction::Z, dim> z;
};


template<std::size_t dim>
class Div
{
public:
    using expr_tag = void;

    Div(auto const& layout, auto const& v)
        : dx_vx_{layout, v[0]}
        , dy_vy_{layout, v[1]}
        , dz_vz_{layout, v[2]}
    {
        if (dx_vx_.centering() != dy_vy_.centering() or dx_vx_.centering() != dz_vz_.centering())
            throw std::runtime_error("div: the three derivatives land on different centerings");
    }

    MKN_KUL_FN_SIG_RET std::array<QtyCentering, dim> centering() const DAMN_ALL
    {
        return dx_vx_.centering();
    }

    template<typename T>
    MKN_KUL_FN_SIG_RET double operator()(damn::core::Point<T, dim> const& idx) const DAMN_ALL
    {
        return dx_vx_(idx) + dy_vy_(idx) + dz_vz_(idx);
    }

private:
    Partial<Direction::X, dim> dx_vx_;
    Partial<Direction::Y, dim> dy_vy_;
    Partial<Direction::Z, dim> dz_vz_;
};



template<auto dir, typename Layout>
auto partial(Layout const& layout, auto const& src)
{
    return Partial<dir, Layout::dimension>{layout, src};
}

// centering of a field from its quantity, no layout needed
template<std::size_t dim>
auto centering_of(auto const& f)
{
    return centering<dim>(quantity(f.physicalQuantity()));
}

template<std::size_t dim>
auto curl(std::array<double, dim> const& dx, auto const& v)
{
    std::array<std::array<QtyCentering, dim>, 3> const c{
        centering_of<dim>(v[0]), centering_of<dim>(v[1]), centering_of<dim>(v[2])};
    return Curl<dim>{{dx, c, v}, {dx, c, v}, {dx, c, v}};
}

template<typename Layout>
auto curl(Layout const& layout, auto const& v)
{
    return Curl<Layout::dimension>{{layout, v}, {layout, v}, {layout, v}};
}

template<typename Layout>
auto grad(Layout const& layout, auto const& src)
{
    return Grad<Layout::dimension>{{layout, src}, {layout, src}, {layout, src}};
}

template<typename Layout>
auto div(Layout const& layout, auto const& v)
{
    return Div<Layout::dimension>{layout, v};
}

template<typename Layout, typename Src>
MKN_KUL_FN_SIG_RET double
laplacian(Layout const& layout, Src const& f,
          damn::core::Point<std::size_t, Layout::dimension> const idx) DAMN_ALL
{
    static_assert(Src::dimension == Layout::dimension, "field and layout dimensions must match");

    // plain accumulation rather than sum(array): sum is host-only (std::accumulate)
    double lap = 0;
    mkn::kul::for_N<Layout::dimension>([&](auto ic) {
        constexpr auto d = ic();
        auto const inv   = 1.0 / layout.dx[d];
        lap += inv * inv
               * (detail::shifted<d>(f, idx, 1) - 2.0 * detail::shifted<d>(f, idx, 0)
                  + detail::shifted<d>(f, idx, -1));
    });
    return lap;
}

} // namespace plamn::core

#endif
