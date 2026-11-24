#ifndef PLAMN_CORE_MODELS_QUANTITIES_QUANTITY_HPP
#define PLAMN_CORE_MODELS_QUANTITIES_QUANTITY_HPP

// what a quantity is (Phys, Comp: the same in every model) and where it lives (centering:
// per model, since models discretise the same physics differently)

#include "damn/core/def/damn_config.hpp"

#include "plamn/core/def.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace plamn::core
{
enum class QtyCentering : std::uint8_t { primal = 0, dual = 1 };

enum class Phys : std::uint8_t {
    B,
    E,
    J,
    Rho,
    V,
    P,
    M, // momentum tensor
    Etot,
    RhoV,
    ScalarFlux, // comp = face direction
    VecFlux,    // comp = (component, face direction)
    AllPrimal,  // scratch, any quantity on primal nodes
    AllDual,    // scratch, any quantity at cell centres
};

// tensor components: (row, column); for VecFlux (vector component, face direction)
enum class Comp : std::uint8_t { scalar, x, y, z, xx, xy, xz, yx, yy, yz, zx, zy, zz };

struct Quantity
{
    Phys phys;
    Comp comp = Comp::scalar;
    std::array<QtyCentering, 3> centering{};
};

MKN_KUL_FN_SIG_RET constexpr bool same_physics(Quantity const& a, Quantity const& b) DAMN_ALL
{
    return a.phys == b.phys and a.comp == b.comp;
}

template<std::size_t dim>
MKN_KUL_FN_SIG_RET constexpr auto centering(Quantity const& q) DAMN_ALL
{
    std::array<QtyCentering, dim> c;
    for (std::size_t i = 0; i < dim; ++i)
        c[i] = q.centering[i];
    return c;
}


namespace centered
{
    auto constexpr static p = QtyCentering::primal;
    auto constexpr static d = QtyCentering::dual;

    auto constexpr static ppp = std::array{p, p, p};
    auto constexpr static ddd = std::array{d, d, d};
    auto constexpr static pdd = std::array{p, d, d}; // x face
    auto constexpr static dpd = std::array{d, p, d}; // y face
    auto constexpr static ddp = std::array{d, d, p}; // z face
    auto constexpr static dpp = std::array{d, p, p}; // x edge
    auto constexpr static pdp = std::array{p, d, p}; // y edge
    auto constexpr static ppd = std::array{p, p, d}; // z edge
} // namespace centered

} // namespace plamn::core

#endif // PLAMN_CORE_MODELS_QUANTITIES_QUANTITY_HPP
