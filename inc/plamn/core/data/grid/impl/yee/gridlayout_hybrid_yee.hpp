#ifndef PLAMN_CORE_GRID_IMPL_YEE_GRIDLAYOUT_HYBRID_YEE_HPP
#define PLAMN_CORE_GRID_IMPL_YEE_GRIDLAYOUT_HYBRID_YEE_HPP


#include "plamn/core/def.hpp"
#include "damn/core/def/constants.hpp"
#include "damn/core/def/enum_table.hpp"
#include "plamn/core/data/grid/gridlayoutdefs.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"


#include <array>


namespace plamn::core::hybrid
{


/**
 * @brief GridData provides constants used to initialize:
 * - hybridQuantity centerings
 * - physical start/end indexes
 * - ghost start/end indexes
 * - numbers of padding cells and physical cells
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


    // centerings come from the model's quantity definitions (hybrid::quantities)
    constexpr auto static initLayoutCentering_()
    {
        std::array<std::array<QtyCentering, damn::core::NBR_COMPO>, hybrid::quantities.size()> table{};
        for (std::size_t i = 0; i < table.size(); ++i)
            table[i] = hybrid::quantities[i].centering;
        return table;
    }



    //! says for each HybridQuantity::Quantity whether it is primal or dual, in each direction
    constexpr static std::array<std::array<QtyCentering, damn::core::NBR_COMPO>,
                                static_cast<std::size_t>(HybridQuantity::Scalar::count)> const
        qtyCentering_{initLayoutCentering_()};

    static std::size_t const dim_{dim};

    //! qtyCentering_ restricted to the first dim directions, indexed by HybridQuantity::Scalar
    constexpr static auto scalarCentering_ = [] {
        std::array<std::array<QtyCentering, dim>, qtyCentering_.size()> centerings{};
        for (std::size_t iQty = 0; iQty < centerings.size(); ++iQty)
            for (std::size_t iDir = 0; iDir < dim; ++iDir)
                centerings[iQty][iDir] = qtyCentering_[iQty][iDir];
        return centerings;
    }();

    //! centerings of each vector's components, indexed by HybridQuantity::Vector
    constexpr static auto vectorCentering_ = [] {
        std::array<std::array<std::array<QtyCentering, dim>, 3>,
                   HybridQuantity::vector_components.size()>
            centerings{};
        for (std::size_t iVec = 0; iVec < centerings.size(); ++iVec)
            for (std::size_t iComp = 0; iComp < 3; ++iComp)
                centerings[iVec][iComp] = scalarCentering_[static_cast<std::size_t>(
                    HybridQuantity::vector_components[iVec][iComp])];
        return centerings;
    }();

    // ------------------------------------------------------------------------
    //                          PUBLIC INTERFACE
    // ------------------------------------------------------------------------
public:
    NO_DISCARD constexpr static std::array<QtyCentering, dim>
    centering(HybridQuantity::Scalar const qty) DAMN_ALL
    {
        auto const idx = static_cast<std::size_t>(qty);
        if (idx >= scalarCentering_.size())
        {
            damn::throw_runtime_error("Wrong hybridQuantity");
            return scalarCentering_[0]; // unreachable on host, no exceptions on device
        }
        return scalarCentering_[idx];
    }

    NO_DISCARD constexpr static std::array<std::array<QtyCentering, dim>, 3>
    centering(HybridQuantity::Vector const qty) DAMN_ALL
    {
        auto const idx = static_cast<std::size_t>(qty);
        if (idx >= vectorCentering_.size())
        {
            damn::throw_runtime_error("Wrong hybridQuantity");
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

    static constexpr std::uint32_t iBx = static_cast<std::uint32_t>(HybridQuantity::Scalar::Bx);
    static constexpr std::uint32_t iBy = static_cast<std::uint32_t>(HybridQuantity::Scalar::By);
    static constexpr std::uint32_t iBz = static_cast<std::uint32_t>(HybridQuantity::Scalar::Bz);

    static constexpr std::uint32_t iEx = static_cast<std::uint32_t>(HybridQuantity::Scalar::Ex);
    static constexpr std::uint32_t iEy = static_cast<std::uint32_t>(HybridQuantity::Scalar::Ey);
    static constexpr std::uint32_t iEz = static_cast<std::uint32_t>(HybridQuantity::Scalar::Ez);

    static constexpr std::uint32_t iJx = static_cast<std::uint32_t>(HybridQuantity::Scalar::Jx);
    static constexpr std::uint32_t iJy = static_cast<std::uint32_t>(HybridQuantity::Scalar::Jy);
    static constexpr std::uint32_t iJz = static_cast<std::uint32_t>(HybridQuantity::Scalar::Jz);

    static constexpr std::uint32_t irho = static_cast<std::uint32_t>(HybridQuantity::Scalar::rho);

    static constexpr std::uint32_t iVx = static_cast<std::uint32_t>(HybridQuantity::Scalar::Vx);
    static constexpr std::uint32_t iVy = static_cast<std::uint32_t>(HybridQuantity::Scalar::Vy);
    static constexpr std::uint32_t iVz = static_cast<std::uint32_t>(HybridQuantity::Scalar::Vz);

    static constexpr std::uint32_t iMxx = static_cast<std::uint32_t>(HybridQuantity::Scalar::Mxx);
    static constexpr std::uint32_t iMxy = static_cast<std::uint32_t>(HybridQuantity::Scalar::Mxy);
    static constexpr std::uint32_t iMxz = static_cast<std::uint32_t>(HybridQuantity::Scalar::Mxz);
    static constexpr std::uint32_t iMyy = static_cast<std::uint32_t>(HybridQuantity::Scalar::Myy);
    static constexpr std::uint32_t iMyz = static_cast<std::uint32_t>(HybridQuantity::Scalar::Myz);
    static constexpr std::uint32_t iMzz = static_cast<std::uint32_t>(HybridQuantity::Scalar::Mzz);

    static constexpr std::uint32_t iP = static_cast<std::uint32_t>(HybridQuantity::Scalar::P);
};



} // namespace plamn::core::hybrid


#endif // PLAMN_CORE_GRID_IMPL_YEE_GRIDLAYOUT_HYBRID_YEE_HPP
