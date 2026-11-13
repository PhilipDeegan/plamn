#ifndef PLAMN_CORE_HYBRID_HYBRID_QUANTITIES_HPP
#define PLAMN_CORE_HYBRID_HYBRID_QUANTITIES_HPP

#include "plamn/core/def.hpp"
#include "damn/core/def/enum_table.hpp"
#include "plamn/core/models/quantities/quantity.hpp"


#include <array>
#include <cstdint>
#include <tuple>
#include <stdexcept>


namespace plamn::core
{
class HybridQuantity
{
public:
    enum class Scalar : std::uint16_t {
        Bx = 0, // magnetic field components
        By,
        Bz,
        Ex, // electric field components
        Ey,
        Ez,
        Jx, // current density components
        Jy,
        Jz,
        rho, // charge density
        Vx,  // bulk velocity components
        Vy,
        Vz,
        P,   // pressure
        Mxx, // momentum tensor components
        Mxy,
        Mxz,
        Myy,
        Myz,
        Mzz,
        count,
        INVALID
    };
    enum class Vector : std::uint16_t { B = 0, E, J, V, count, INVALID };
    enum class Tensor : std::uint16_t { M = 0, count, INVALID };

    static constexpr auto vector_components = damn::core::make_enum_table<Vector, std::array<Scalar, 3>>({
        {Vector::B, {Scalar::Bx, Scalar::By, Scalar::Bz}},
        {Vector::E, {Scalar::Ex, Scalar::Ey, Scalar::Ez}},
        {Vector::J, {Scalar::Jx, Scalar::Jy, Scalar::Jz}},
        {Vector::V, {Scalar::Vx, Scalar::Vy, Scalar::Vz}},
    });

    static constexpr auto all_primal_field = Scalar::rho;

    template<std::size_t rank>
        requires(rank == 1 or rank == 2)
    using TensorType = std::conditional_t<rank == 1, Vector, Tensor>;


    NO_DISCARD static constexpr auto B() { return componentsQuantities(Vector::B); }
    NO_DISCARD static constexpr auto E() { return componentsQuantities(Vector::E); }
    NO_DISCARD static constexpr auto J() { return componentsQuantities(Vector::J); }
    NO_DISCARD static constexpr auto V() { return componentsQuantities(Vector::V); }

    NO_DISCARD static constexpr std::array<Scalar, 3> componentsQuantities(Vector const qty)
    {
        auto const idx = static_cast<std::size_t>(qty);
        if (idx >= vector_components.size())
            throw std::runtime_error("Error - invalid Vector");
        return vector_components[idx];
    }

    NO_DISCARD static constexpr std::array<Scalar, 6> componentsQuantities(Tensor /*qty*/)
    {
        // no condition, for now there's only then momentum tensor M
        return {{Scalar::Mxx, Scalar::Mxy, Scalar::Mxz, Scalar::Myy, Scalar::Myz, Scalar::Mzz}};
    }
};

auto inline componentsQuantities(HybridQuantity::Vector const qty)
{
    return HybridQuantity::componentsQuantities(qty);
}

auto inline componentsQuantities(HybridQuantity::Tensor const qty)
{
    return HybridQuantity::componentsQuantities(qty);
}


namespace hybrid
{
    using namespace centered;
    using S = HybridQuantity::Scalar;

    // Yee: B on faces, E and J on edges, moments on primal nodes
    auto constexpr static quantities = damn::core::make_enum_table<S, Quantity>({
        {S::Bx, {Phys::B, Comp::x, pdd}},    {S::By, {Phys::B, Comp::y, dpd}},
        {S::Bz, {Phys::B, Comp::z, ddp}},    {S::Ex, {Phys::E, Comp::x, dpp}},
        {S::Ey, {Phys::E, Comp::y, pdp}},    {S::Ez, {Phys::E, Comp::z, ppd}},
        {S::Jx, {Phys::J, Comp::x, dpp}},    {S::Jy, {Phys::J, Comp::y, pdp}},
        {S::Jz, {Phys::J, Comp::z, ppd}},    {S::rho, {Phys::Rho, Comp::scalar, ppp}},
        {S::Vx, {Phys::V, Comp::x, ppp}},    {S::Vy, {Phys::V, Comp::y, ppp}},
        {S::Vz, {Phys::V, Comp::z, ppp}},    {S::P, {Phys::P, Comp::scalar, ppp}},
        {S::Mxx, {Phys::M, Comp::xx, ppp}},  {S::Mxy, {Phys::M, Comp::xy, ppp}},
        {S::Mxz, {Phys::M, Comp::xz, ppp}},  {S::Myy, {Phys::M, Comp::yy, ppp}},
        {S::Myz, {Phys::M, Comp::yz, ppp}},  {S::Mzz, {Phys::M, Comp::zz, ppp}},
    });
} // namespace hybrid

NO_DISCARD constexpr Quantity quantity(HybridQuantity::Scalar const qty) DAMN_ALL
{
    auto const idx = static_cast<std::size_t>(qty);
    if (idx >= hybrid::quantities.size())
        damn::throw_runtime_error("Wrong hybridQuantity");
    return hybrid::quantities[idx < hybrid::quantities.size() ? idx : 0];
}

} // namespace plamn::core

#endif
