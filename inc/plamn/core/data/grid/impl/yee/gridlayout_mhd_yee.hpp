#ifndef PLAMN_CORE_GRID_GRIDLAYOUTYEE_MHD_HPP
#define PLAMN_CORE_GRID_GRIDLAYOUTYEE_MHD_HPP

#include "plamn/core/def.hpp"
#include "damn/core/def/constants.hpp"
#include "damn/core/def/enum_table.hpp"
#include "plamn/core/data/grid/gridlayoutdefs.hpp"
#include "plamn/core/models/quantities/mhd_quantities.hpp"


#include <array>

namespace plamn::core::mhd
{

/**
 * @brief GridLayoutNdArrayImplYee class is a concrete GridLayoutNdArrayImpl used a Yee
 * type grid layout is needed.
 *
 * It provides methods related to grid layout operations:
 * - physical domain start/end indexes
 * - indexes of the first and last ghost nodes
 * - allocation sizes for Field attributes of other classes
 * - partial derivative operator (Faraday)
 * - physical coordinate given a field and a primal point (ix, iy, iz)
 * - cell centered coordinate given a primal point (ix, iy, iz)
 */
template<auto options>
class GridLayoutImplYee
{
    using Options = decltype(options);
    using Scalar  = Options::Scalar;

    static constexpr std::size_t dim = options.dimension;

public:
    struct GridData;
    static constexpr std::size_t dimension = dim;
    static constexpr std::string_view type = "yee";
    // The MHD layout reserves ghosts based on the reconstruction stencil width, plus the one
    // layer ampere loses by computing J on the shrinked ghost box.
    // static constexpr std::uint32_t reconstruction_nghosts = reconstruction_nghosts_;
    // Ghost width computed directly based on reconstruction stencil
    // static constexpr std::uint32_t ghost_width
    //     = nbrGhostsFromReconstruction<field_options.reconstruction_nghosts>();

    // centerings come from the model's quantity definitions (mhd::quantities)
    constexpr auto static initLayoutCentering_()
    {
        std::array<std::array<QtyCentering, damn::core::NBR_COMPO>, mhd::quantities.size()> table{};
        for (std::size_t i = 0; i < table.size(); ++i)
            table[i] = mhd::quantities[i].centering;
        return table;
    }

    //! says for each MHDQuantity::Quantity whether it is primal or dual, in each direction
    constexpr static std::array<std::array<QtyCentering, damn::core::NBR_COMPO>,
                                static_cast<std::size_t>(MHDQuantity::Scalar::count)> const
        qtyCentering_{initLayoutCentering_()};

    static std::size_t const dim_{dim};

    //! qtyCentering_ restricted to the first dim directions, indexed by MHDQuantity::Scalar
    constexpr static auto scalarCentering_ = [] {
        std::array<std::array<QtyCentering, dim>, qtyCentering_.size()> centerings{};
        for (std::size_t iQty = 0; iQty < centerings.size(); ++iQty)
            for (std::size_t iDir = 0; iDir < dim; ++iDir)
                centerings[iQty][iDir] = qtyCentering_[iQty][iDir];
        return centerings;
    }();

    //! centerings of each vector's components, indexed by MHDQuantity::Vector
    constexpr static auto vectorCentering_ = [] {
        std::array<std::array<std::array<QtyCentering, dim>, 3>,
                   MHDQuantity::vector_components.size()>
            centerings{};
        for (std::size_t iVec = 0; iVec < centerings.size(); ++iVec)
            for (std::size_t iComp = 0; iComp < 3; ++iComp)
                centerings[iVec][iComp] = scalarCentering_[static_cast<std::size_t>(
                    MHDQuantity::vector_components[iVec][iComp])];
        return centerings;
    }();

    // ------------------------------------------------------------------------
    //                          PUBLIC INTERFACE
    // ------------------------------------------------------------------------
public:
    NO_DISCARD constexpr static std::array<QtyCentering, dim>
    centering(MHDQuantity::Scalar const qty) DAMN_ALL
    {
        auto const idx = static_cast<std::size_t>(qty);
        if (idx >= scalarCentering_.size())
        {
            damn::throw_runtime_error("Wrong MHDQuantity");
            return scalarCentering_[0]; // unreachable on host, no exceptions on device
        }
        return scalarCentering_[idx];
    }

    NO_DISCARD constexpr static std::array<std::array<QtyCentering, dim>, 3>
    centering(MHDQuantity::Vector const qty) DAMN_ALL
    {
        auto const idx = static_cast<std::size_t>(qty);
        if (idx >= vectorCentering_.size())
        {
            damn::throw_runtime_error("Wrong MHDQuantity");
            return vectorCentering_[0]; // unreachable on host, no exceptions on device
        }
        return vectorCentering_[idx];
    }
};



template<auto options>
struct GridLayoutImplYee<options>::GridData
{
    static constexpr Direction dirX = Direction::X;
    static constexpr Direction dirY = Direction::Y;
    static constexpr Direction dirZ = Direction::Z;

    static constexpr QtyCentering primal = QtyCentering::primal;
    static constexpr QtyCentering dual   = QtyCentering::dual;

    static constexpr std::uint32_t idirX = static_cast<std::uint32_t>(Direction::X);
    static constexpr std::uint32_t idirY = static_cast<std::uint32_t>(Direction::Y);
    static constexpr std::uint32_t idirZ = static_cast<std::uint32_t>(Direction::Z);

    static constexpr std::uint32_t irho = static_cast<std::uint32_t>(MHDQuantity::Scalar::rho);

    static constexpr std::uint32_t iVx = static_cast<std::uint32_t>(MHDQuantity::Scalar::Vx);
    static constexpr std::uint32_t iVy = static_cast<std::uint32_t>(MHDQuantity::Scalar::Vy);
    static constexpr std::uint32_t iVz = static_cast<std::uint32_t>(MHDQuantity::Scalar::Vz);

    static constexpr std::uint32_t iBx = static_cast<std::uint32_t>(MHDQuantity::Scalar::Bx);
    static constexpr std::uint32_t iBy = static_cast<std::uint32_t>(MHDQuantity::Scalar::By);
    static constexpr std::uint32_t iBz = static_cast<std::uint32_t>(MHDQuantity::Scalar::Bz);

    static constexpr std::uint32_t iP = static_cast<std::uint32_t>(MHDQuantity::Scalar::P);

    static constexpr std::uint32_t iEtot = static_cast<std::uint32_t>(MHDQuantity::Scalar::Etot);

    static constexpr std::uint32_t irhoVx = static_cast<std::uint32_t>(MHDQuantity::Scalar::rhoVx);
    static constexpr std::uint32_t irhoVy = static_cast<std::uint32_t>(MHDQuantity::Scalar::rhoVy);
    static constexpr std::uint32_t irhoVz = static_cast<std::uint32_t>(MHDQuantity::Scalar::rhoVz);

    static constexpr std::uint32_t iEx = static_cast<std::uint32_t>(MHDQuantity::Scalar::Ex);
    static constexpr std::uint32_t iEy = static_cast<std::uint32_t>(MHDQuantity::Scalar::Ey);
    static constexpr std::uint32_t iEz = static_cast<std::uint32_t>(MHDQuantity::Scalar::Ez);

    static constexpr std::uint32_t iJx = static_cast<std::uint32_t>(MHDQuantity::Scalar::Jx);
    static constexpr std::uint32_t iJy = static_cast<std::uint32_t>(MHDQuantity::Scalar::Jy);
    static constexpr std::uint32_t iJz = static_cast<std::uint32_t>(MHDQuantity::Scalar::Jz);

    static constexpr std::uint32_t iScalarFlux_x
        = static_cast<std::uint32_t>(MHDQuantity::Scalar::ScalarFlux_x);
    static constexpr std::uint32_t iScalarFlux_y
        = static_cast<std::uint32_t>(MHDQuantity::Scalar::ScalarFlux_y);
    static constexpr std::uint32_t iScalarFlux_z
        = static_cast<std::uint32_t>(MHDQuantity::Scalar::ScalarFlux_z);

    static constexpr std::uint32_t iVecFluxX_x
        = static_cast<std::uint32_t>(MHDQuantity::Scalar::VecFluxX_x);
    static constexpr std::uint32_t iVecFluxY_x
        = static_cast<std::uint32_t>(MHDQuantity::Scalar::VecFluxY_x);
    static constexpr std::uint32_t iVecFluxZ_x
        = static_cast<std::uint32_t>(MHDQuantity::Scalar::VecFluxZ_x);

    static constexpr std::uint32_t iVecFluxX_y
        = static_cast<std::uint32_t>(MHDQuantity::Scalar::VecFluxX_y);
    static constexpr std::uint32_t iVecFluxY_y
        = static_cast<std::uint32_t>(MHDQuantity::Scalar::VecFluxY_y);
    static constexpr std::uint32_t iVecFluxZ_y
        = static_cast<std::uint32_t>(MHDQuantity::Scalar::VecFluxZ_y);

    static constexpr std::uint32_t iVecFluxX_z
        = static_cast<std::uint32_t>(MHDQuantity::Scalar::VecFluxX_z);
    static constexpr std::uint32_t iVecFluxY_z
        = static_cast<std::uint32_t>(MHDQuantity::Scalar::VecFluxY_z);
    static constexpr std::uint32_t iVecFluxZ_z
        = static_cast<std::uint32_t>(MHDQuantity::Scalar::VecFluxZ_z);

    static constexpr std::uint32_t iScalarAllPrimal
        = static_cast<std::uint32_t>(MHDQuantity::Scalar::ScalarAllPrimal);

    static constexpr std::uint32_t iVecAllPrimalX
        = static_cast<std::uint32_t>(MHDQuantity::Scalar::VecAllPrimalX);
    static constexpr std::uint32_t iVecAllPrimalY
        = static_cast<std::uint32_t>(MHDQuantity::Scalar::VecAllPrimalY);
    static constexpr std::uint32_t iVecAllPrimalZ
        = static_cast<std::uint32_t>(MHDQuantity::Scalar::VecAllPrimalZ);
};



} // namespace plamn::core::mhd


#endif // PLAMN_CORE_GRID_GRIDLAYOUTYEE_MHD_HPP
