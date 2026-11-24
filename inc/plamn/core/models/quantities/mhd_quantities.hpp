#ifndef PLAMN_CORE_MHD_MHD_QUANTITIES_HPP
#define PLAMN_CORE_MHD_MHD_QUANTITIES_HPP

#include "mkn/kul/enum.hpp"
#include "mkn/kul/throw.hpp"
#include "damn/core/def/damn_config.hpp"

#include "plamn/core/def.hpp"
#include "plamn/core/models/quantities/quantity.hpp"

#include <array>
#include <cstdint>
#include <tuple>
#include <stdexcept>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace plamn::core
{
class MHDQuantity
{
public:
    enum class Scalar : std::uint16_t {
        rho = 0, // density
        Vx,      // velocity components
        Vy,
        Vz,
        Bx,
        By,
        Bz,
        P, // pressure

        Etot,  // total energy
        rhoVx, // momentum components
        rhoVy,
        rhoVz,

        Ex, // electric field components
        Ey,
        Ez,
        Jx, // current density components
        Jy,
        Jz,

        ScalarFlux_x,
        ScalarFlux_y,
        ScalarFlux_z,
        VecFluxX_x,
        VecFluxY_x,
        VecFluxZ_x,
        VecFluxX_y,
        VecFluxY_y,
        VecFluxZ_y,
        VecFluxX_z,
        VecFluxY_z,
        VecFluxZ_z,

        ScalarAllPrimal,
        VecAllPrimalX,
        VecAllPrimalY,
        VecAllPrimalZ,

        count
    };
    enum class Vector : std::uint16_t {
        V = 0,
        B,
        rhoV,
        E,
        J,
        VecFlux_x,
        VecFlux_y,
        VecFlux_z,
        VecAllPrimal,
        count
    };
    enum class Tensor : std::uint16_t { count = 0 };

    static constexpr auto vector_components
        = mkn::kul::make_enum_table<Vector, std::array<Scalar, 3>>({
            {Vector::V, {Scalar::Vx, Scalar::Vy, Scalar::Vz}},
            {Vector::B, {Scalar::Bx, Scalar::By, Scalar::Bz}},
            {Vector::rhoV, {Scalar::rhoVx, Scalar::rhoVy, Scalar::rhoVz}},
            {Vector::E, {Scalar::Ex, Scalar::Ey, Scalar::Ez}},
            {Vector::J, {Scalar::Jx, Scalar::Jy, Scalar::Jz}},
            {Vector::VecFlux_x, {Scalar::VecFluxX_x, Scalar::VecFluxY_x, Scalar::VecFluxZ_x}},
            {Vector::VecFlux_y, {Scalar::VecFluxX_y, Scalar::VecFluxY_y, Scalar::VecFluxZ_y}},
            {Vector::VecFlux_z, {Scalar::VecFluxX_z, Scalar::VecFluxY_z, Scalar::VecFluxZ_z}},
            {Vector::VecAllPrimal,
             {Scalar::VecAllPrimalX, Scalar::VecAllPrimalY, Scalar::VecAllPrimalZ}},
        });

    static constexpr auto all_primal_field = Scalar::ScalarAllPrimal;

    template<std::size_t rank>
        requires(rank == 1 or rank == 2)
    using TensorType = std::conditional_t<rank == 1, Vector, Tensor>;

    MKN_KUL_FN_SIG_RET static constexpr auto V() { return componentsQuantities(Vector::V); }
    MKN_KUL_FN_SIG_RET static constexpr auto B() { return componentsQuantities(Vector::B); }
    MKN_KUL_FN_SIG_RET static constexpr auto rhoV() { return componentsQuantities(Vector::rhoV); }

    MKN_KUL_FN_SIG_RET static constexpr auto E() { return componentsQuantities(Vector::E); }
    MKN_KUL_FN_SIG_RET static constexpr auto J() { return componentsQuantities(Vector::J); }

    MKN_KUL_FN_SIG_RET static constexpr auto VecFlux_x()
    {
        return componentsQuantities(Vector::VecFlux_x);
    }
    MKN_KUL_FN_SIG_RET static constexpr auto VecFlux_y()
    {
        return componentsQuantities(Vector::VecFlux_y);
    }
    MKN_KUL_FN_SIG_RET static constexpr auto VecFlux_z()
    {
        return componentsQuantities(Vector::VecFlux_z);
    }

    MKN_KUL_FN_SIG_RET static constexpr auto VecAllPrimal()
    {
        return componentsQuantities(Vector::VecAllPrimal);
    }

    MKN_KUL_FN_SIG_RET static constexpr std::array<Scalar, 3> componentsQuantities(Vector const qty)
    {
        auto const idx = static_cast<std::size_t>(qty);
        if (idx >= vector_components.size())
            throw std::runtime_error("Error - invalid Vector");
        return vector_components[idx];
    }

    MKN_KUL_FN_SIG_RET static constexpr auto B_items()
    {
        auto const& [Bx, By, Bz] = B();
        return std::make_tuple(std::make_pair("Bx", Bx), std::make_pair("By", By),
                               std::make_pair("Bz", Bz));
    }
    MKN_KUL_FN_SIG_RET static constexpr auto E_items()
    {
        auto const& [Ex, Ey, Ez] = E();
        return std::make_tuple(std::make_pair("Ex", Ex), std::make_pair("Ey", Ey),
                               std::make_pair("Ez", Ez));
    }
};


namespace mhd
{
    using namespace centered;
    using S = MHDQuantity::Scalar;

    // finite volume: conserved/primitive variables cell centred, B on faces, E and J on
    // edges, fluxes on the face normal to their direction
    auto constexpr static quantities = mkn::kul::make_enum_table<S, Quantity>({
        {S::rho, {Phys::Rho, Comp::scalar, ddd}},
        {S::Vx, {Phys::V, Comp::x, ddd}},
        {S::Vy, {Phys::V, Comp::y, ddd}},
        {S::Vz, {Phys::V, Comp::z, ddd}},
        {S::Bx, {Phys::B, Comp::x, pdd}},
        {S::By, {Phys::B, Comp::y, dpd}},
        {S::Bz, {Phys::B, Comp::z, ddp}},
        {S::P, {Phys::P, Comp::scalar, ddd}},
        {S::Etot, {Phys::Etot, Comp::scalar, ddd}},
        {S::rhoVx, {Phys::RhoV, Comp::x, ddd}},
        {S::rhoVy, {Phys::RhoV, Comp::y, ddd}},
        {S::rhoVz, {Phys::RhoV, Comp::z, ddd}},
        {S::Ex, {Phys::E, Comp::x, dpp}},
        {S::Ey, {Phys::E, Comp::y, pdp}},
        {S::Ez, {Phys::E, Comp::z, ppd}},
        {S::Jx, {Phys::J, Comp::x, dpp}},
        {S::Jy, {Phys::J, Comp::y, pdp}},
        {S::Jz, {Phys::J, Comp::z, ppd}},
        {S::ScalarFlux_x, {Phys::ScalarFlux, Comp::x, pdd}},
        {S::ScalarFlux_y, {Phys::ScalarFlux, Comp::y, dpd}},
        {S::ScalarFlux_z, {Phys::ScalarFlux, Comp::z, ddp}},
        {S::VecFluxX_x, {Phys::VecFlux, Comp::xx, pdd}},
        {S::VecFluxY_x, {Phys::VecFlux, Comp::yx, pdd}},
        {S::VecFluxZ_x, {Phys::VecFlux, Comp::zx, pdd}},
        {S::VecFluxX_y, {Phys::VecFlux, Comp::xy, dpd}},
        {S::VecFluxY_y, {Phys::VecFlux, Comp::yy, dpd}},
        {S::VecFluxZ_y, {Phys::VecFlux, Comp::zy, dpd}},
        {S::VecFluxX_z, {Phys::VecFlux, Comp::xz, ddp}},
        {S::VecFluxY_z, {Phys::VecFlux, Comp::yz, ddp}},
        {S::VecFluxZ_z, {Phys::VecFlux, Comp::zz, ddp}},
        {S::ScalarAllPrimal, {Phys::AllPrimal, Comp::scalar, ppp}},
        {S::VecAllPrimalX, {Phys::AllPrimal, Comp::x, ppp}},
        {S::VecAllPrimalY, {Phys::AllPrimal, Comp::y, ppp}},
        {S::VecAllPrimalZ, {Phys::AllPrimal, Comp::z, ppp}},
    });
} // namespace mhd

MKN_KUL_FN_SIG_RET constexpr Quantity quantity(MHDQuantity::Scalar const qty) DAMN_ALL
{
    auto const idx = static_cast<std::size_t>(qty);
    if (idx >= mhd::quantities.size())
        mkn::kul::throw_runtime_error("Wrong mhdQuantity");
    return mhd::quantities[idx < mhd::quantities.size() ? idx : 0];
}

} // namespace plamn::core

#endif
