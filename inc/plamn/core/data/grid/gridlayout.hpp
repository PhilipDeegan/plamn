#ifndef PLAMN_CORE_GRID_GridLayout_HPP
#define PLAMN_CORE_GRID_GridLayout_HPP


#include "plamn/core/def.hpp"
#include "plamn/core/data/field/field.hpp"
#include "damn/core/meta/type_meta.hpp"
#include "damn/core/meta/for_N.hpp"
#include "damn/core/math/array.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/def/constants.hpp"
#include "plamn/core/data/grid/grid_tiles.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/geometry/point.hpp"

#include "gridlayoutdefs.hpp"
#include "plamn/core/data/grid/layout.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"

#include "plamn/core/data/grid/detail/detail.hpp"

#include <array>
#include <algorithm>
#include <tuple>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <functional>
#include <type_traits>

namespace plamn
{
namespace core
{
    template<auto options>
    concept HasInterpOrder = requires { options.interp_order; };



    NO_DISCARD constexpr int centering2int(QtyCentering c)
    {
        return static_cast<int>(c);
    }


    template<std::size_t interpOrder>
    NO_DISCARD std::uint32_t constexpr ghostWidthForParticles()
    {
        return (interpOrder % 2 == 0 ? interpOrder / 2 + 1 : (interpOrder + 1) / 2);
    }


    template<typename T, std::size_t s>
    NO_DISCARD auto boxFromNbrCells(std::array<T, s> nbrCells)
    {
        damn::core::Point<int, s> lower;
        damn::core::Point<int, s> upper;

        for (auto i = 0u; i < nbrCells.size(); ++i)
        {
            lower[i] = 0;
            upper[i] = static_cast<int>(nbrCells[i]) - 1;
        };

        return damn::core::Box{lower, upper};
    }

    /**
     * @brief A GridLayout object represents a uniform cartesian mesh.
     * It provides methods to manipulate physical quantities discretized on such
     * a mesh. From the client code, GridLayout is the object that knows
     * the centering of all quantities in the simulation. It knows where are
     * the physical domain indexes VS the ghost indexes for a grid.
     *
     * The specific knowledge of which quantity is where is encapsulated in the private
     * GridLayout implementation implT. For instance, the Yee lattice is
     * the implTYee of such implementations.
     *
     * GridLayout in itself gathers many methods to manipulate discretized quantities.
     * All these methods only depend on the concept of primal and dual nodes.
     * Primal nodes are nodes at integer multiple of the meshSize, dual node are positionned
     * at half multiples of the mesh size. In, say, 2D, a given quantity can for instance
     * be (primal,dual), meaning that it will be found at cell corners in the X direction
     * , but offset by dy/2 in the Y direction.
     *
     * GridLayout is templated by a specific implementation, such as implTYee.
     * This implementation brings the dimensionality and the interpolation order compile-time
     * constants to the GridLayout.
     *
     */
    template<auto options_>
    class GridLayout : public PatchLayout<options_>
    {
        using Options      = decltype(options_);
        using FieldOptions = Options::FieldOptions;
        using Vector       = FieldOptions::Vector;
        using Tensor       = FieldOptions::Tensor;
        using GridData     = Options::GridLayoutImpl::GridData;

    protected:
        GridLayout() = default;

    public:
        auto static constexpr options          = options_;
        static constexpr std::size_t dimension = options.dimension;

        using This     = GridLayout<options>;
        using implT    = Options::GridLayoutImpl;
        using AMRBox_t = damn::core::Box<int, dimension>;
        using Scalar   = FieldOptions::Scalar;

        /**
         * @brief Constructor of a GridLayout
         * @param meshSize is the size of the mesh in each direction
         * @param nbrCells is the number of physical cells of the grid
         * @param origin is the point of origin in physical units of the origin of the grid
         */
        GridLayout(std::array<double, dimension> const& meshSize,
                   std::array<std::uint32_t, dimension> const& nbrCells,
                   damn::core::Point<double, dimension> const& origin,
                   std::optional<AMRBox_t> const AMRBox = std::nullopt, int level_number = 0)
            : PatchLayout<options_>{AMRBox ? *AMRBox : boxFromNbrCells(nbrCells), meshSize, origin,
                                    level_number}
            , nbrPhysicalCells_{nbrCells}
        {
            if (this->cells.size() != boxFromNbrCells(nbrCells).size())
                throw std::runtime_error("Error - invalid AMR box, incorrect number of cells");

            inverseMeshSize_ = damn::core::generate_from([](auto const e) { return 1. / e; }, this->dx);
        }


        GridLayout(GridLayout const& that)            = default;
        GridLayout(GridLayout&& source)               = default;
        GridLayout& operator=(GridLayout const& that) = default;
        GridLayout& operator=(GridLayout&& source)    = default;


        /**
         * @brief origin return the lower point of the grid described by the GridLayout
         * in physical coordinates
         */
        NO_DISCARD damn::core::Point<double, dimension> origin() const noexcept { return PatchLayout<options_>::origin; }



        /**
         * @brief returns the mesh size in the 'dim' dimensions
         */
        NO_DISCARD std::array<double, dimension> const& meshSize() const noexcept DAMN_ALL
        {
            return this->dx;
        }



        NO_DISCARD double inverseMeshSize(Direction direction) const noexcept DAMN_ALL
        {
            return inverseMeshSize_[static_cast<std::uint32_t>(direction)];
        }



        NO_DISCARD std::array<double, dimension> inverseMeshSize() const noexcept
        {
            return inverseMeshSize_;
        }



        /**
         * @brief nbrCells returns the number of cells in the physical domain
         * described by the gridlayout
         */
        NO_DISCARD auto& nbrCells() const DAMN_ALL { return nbrPhysicalCells_; }


        NO_DISCARD auto const& AMRBox() const DAMN_ALL { return this->cells; }



        template<typename Centering, typename Direction>
        NO_DISCARD auto ghostStartToEnd(Centering const& centering, Direction const direction) const
        {
            return std::make_tuple(ghostStartIndex(centering, direction),
                                   ghostEndIndex(centering, direction));
        }

        template<HasPhysicalQuantity Field>
        NO_DISCARD auto ghostStartToEnd(Field const& field, Direction direction) const
        {
            return ghostStartToEnd(field.physicalQuantity(), direction);
        }

        template<typename Centering, typename Direction>
        NO_DISCARD auto physicalStartToEnd(Centering const& centering,
                                           Direction const direction) const
        {
            return std::make_tuple(physicalStartIndex(centering, direction),
                                   physicalEndIndex(centering, direction));
        }


        template<HasPhysicalQuantity Field>
        NO_DISCARD auto physicalStartToEnd(Field const& field, Direction direction) const
        {
            return physicalStartToEnd(field.physicalQuantity(), direction);
        }


        NO_DISCARD double cellVolume() const { return cell_volume(this->dx); }

        /**
         * @brief physicalStartIndex returns the index of the first node of a given
         * centering and in a given direction that is in the physical domain, i.e. not a ghost node.
         */
        NO_DISCARD std::uint32_t physicalStartIndex(QtyCentering centering,
                                                    Direction direction) const DAMN_ALL
        {
            return physical_start(options.field_ghost_width);
        }



        NO_DISCARD std::uint32_t physicalStartIndex(Scalar const& quantity,
                                                    Direction direction) const
        {
            return physical_start(options.field_ghost_width);
        }



        template<typename Field_t>
        NO_DISCARD std::uint32_t physicalStartIndex(Field_t const& field, Direction direction) const
        {
            return physicalStartIndex(field.physicalQuantity(), direction);
        }


        NO_DISCARD auto physicalStartIndex(QtyCentering centering) const
        {
            return damn::core::ConstArray<std::uint32_t, dimension>(physical_start(options.field_ghost_width));
        }



        /**
         * @brief physicalEndIndex returns the index of the last node of a given
         * centering and in a given direction that is in the physical domain, i.e. not a ghost node.
         */
        NO_DISCARD std::uint32_t physicalEndIndex(QtyCentering centering, Direction direction) const
        {
            return physical_end(centering, nbrPhysicalCells_[static_cast<std::uint32_t>(direction)], options.field_ghost_width);
        }



        NO_DISCARD std::uint32_t physicalEndIndex(Scalar const& quantity, Direction direction) const
        {
            return physical_end(implT::qtyCentering_[static_cast<std::uint32_t>(quantity)][static_cast<std::uint32_t>(direction)], nbrPhysicalCells_[static_cast<std::uint32_t>(direction)], options.field_ghost_width);
        }



        template<typename Field_t>
        NO_DISCARD std::uint32_t physicalEndIndex(Field_t const& field, Direction direction) const
        {
            return physicalEndIndex(field.physicalQuantity(), direction);
        }


        NO_DISCARD auto physicalEndIndex(QtyCentering centering) const
        {
            std::array<std::uint32_t, dimension> idx;
            for (std::size_t i = 0; i < dimension; ++i)
                idx[i] = physical_end(centering, nbrPhysicalCells_[i], options.field_ghost_width);
            return idx;
        }



        /**
         * @brief ghostStartIndex retuns the index of the first ghost node of a given centering
         * in a given direction. This is always zero by convention. This function exists only
         * for readibility reasons, not to have literal '0' in the code.
         */
        NO_DISCARD std::uint32_t ghostStartIndex([[maybe_unused]] QtyCentering centering,
                                                 [[maybe_unused]] Direction direction) const
        {
            // ghostStartIndex is always the first node
            return 0;
        }


        NO_DISCARD std::uint32_t ghostStartIndex([[maybe_unused]] Scalar const& quantity,
                                                 [[maybe_unused]] Direction direction) const
        {
            // ghostStartIndex is always the first node
            return 0;
        }


        template<typename Field_t>
        NO_DISCARD std::uint32_t ghostStartIndex(Field_t const& /*field*/,
                                                 Direction /*direction*/) const
        {
            // ghostStartIndex is always the first node
            return 0;
        }


        NO_DISCARD auto ghostStartIndex(QtyCentering /*centering*/) const
        {
            return std::array<std::uint32_t, dimension>{};
        }



        /**
         * @brief ghostEndIndex returns the index of the last ghost node of a given centering
         * and in a given direction.
         */
        NO_DISCARD std::uint32_t ghostEndIndex(QtyCentering centering, Direction direction) const
        {
            return ghost_end(centering, nbrPhysicalCells_[static_cast<std::uint32_t>(direction)], options.field_ghost_width);
        }



        NO_DISCARD std::uint32_t ghostEndIndex(Scalar const& quantity, Direction direction) const
        {
            return ghost_end(implT::qtyCentering_[static_cast<std::uint32_t>(quantity)][static_cast<std::uint32_t>(direction)], nbrPhysicalCells_[static_cast<std::uint32_t>(direction)], options.field_ghost_width);
        }



        template<typename Field_t>
        NO_DISCARD std::uint32_t ghostEndIndex(Field_t const& field, Direction direction) const
        {
            return ghostEndIndex(field.physicalQuantity(), direction);
        }


        NO_DISCARD auto ghostEndIndex(QtyCentering centering) const
        {
            std::array<std::uint32_t, dimension> idx;
            for (std::size_t i = 0; i < dimension; ++i)
                idx[i] = ghost_end(centering, nbrPhysicalCells_[i], options.field_ghost_width);
            return idx;
        }


        /**
         * @brief fieldNodeCoordinates returns the coordinate of a multidimensional index
         * associated with a given Field, in physical coordinates.
         */
        template<typename Field_t>
        NO_DISCARD damn::core::Point<double, dimension>
        fieldNodeCoordinates(Field_t const& field, damn::core::Point<int, dimension> amr_idx) const
        {
            return coordinate(this->dx, amr_idx, centering(field));
        }

        template<typename Field_t, typename... Indexes>
        NO_DISCARD damn::core::Point<double, dimension> fieldNodeCoordinates(Field_t const& field,
                                                                 Indexes... index) const
        {
            static_assert(sizeof...(Indexes) == dimension,
                          "Error dimension does not match number of arguments");

            return fieldNodeCoordinates(field, damn::core::Point<int, dimension>{index...});
        }


        /**
         * @brief cellCenteredCoordinates returns the coordinates in physical units
         * of a multidimensional index that is cell-centered.
         */
        NO_DISCARD damn::core::Point<double, dimension>
        cellCenteredCoordinates(damn::core::Point<int, dimension> coord) const
        {
            return cell_center(this->dx, coord);
        }

        template<typename... Indexes>
        NO_DISCARD damn::core::Point<double, dimension> cellCenteredCoordinates(Indexes... index) const
        {
            static_assert(sizeof...(Indexes) == dimension,
                          "Error dimension does not match number of arguments");

            return cellCenteredCoordinates(damn::core::Point<int, dimension>{index...});
        }




        template<typename Quantity>
        NO_DISCARD auto static constexpr nDNbrGhosts(Quantity /*centering*/ = QtyCentering::primal)
        { // Both dual and primal ghosts are the same!
            return damn::core::ConstArray<std::uint32_t, dimension>(options.field_ghost_width);
        }


        /**
         * @brief changeCentering changes primal into dual and vice versa.
         */
        NO_DISCARD auto static changeCentering(QtyCentering centering)
        {
            return flip(centering);
        }


        /**
         * @brief nextIndex returns the index of the next node of a given centering
         * from an index of the opposite centering.
         * For instance, if 'centering' is Qtycentering::dual, then 'indexCenter'
         * is interpreted as a primal index and the function returns the index
         * of the dual node just on the right of that primal node. This function is useful
         * for calculating derivatives.
         * The next index is not just indexCenter+1 because this depends on the number
         * of ghost nodes for dual and primal nodes.
         */
        NO_DISCARD auto static nextIndex(QtyCentering centering,
                                         std::uint32_t indexCenter) DAMN_ALL
        {
            return next_index(centering, indexCenter);
        }


        /**
         * @brief prevIndex does the same thing as nextIndex but returns the index
         * of the node of a given centering just to the left of indexCenter.
         */
        NO_DISCARD auto static prevIndex(QtyCentering centering,
                                         std::uint32_t indexCenter) DAMN_ALL
        {
            return prev_index(centering, indexCenter);
        }


        /**
         * @brief localToAMR returns the AMR index associated with the given local one.
         * This method only deals with **cell** indexes.
         */
        template<typename T>
        NO_DISCARD auto localToAMR(damn::core::Point<T, dimension> const& localPoint) const
        {
            return local_to_amr(this->cells, localPoint, options.field_ghost_width);
        }


        /**
         * @brief localToAMR returns the AMR box associated with the given local one.
         * This method only deals with **cell** indexes.
         */
        template<typename T>
        NO_DISCARD auto localToAMR(damn::core::Box<T, dimension> const& localBox) const
        {
            return local_to_amr(this->cells, localBox, options.field_ghost_width);
        }


        /**
         * @brief AMRToLocal returns the local index associated with the given AMR one.
         * This method only deals with **cell** indexes.
         */
        template<typename T>
        NO_DISCARD auto AMRToLocal(damn::core::Point<T, dimension> const& AMRPoint) const DAMN_ALL
        {
            return amr_to_local(this->cells, AMRPoint, options.field_ghost_width);
        }


        /**
         * @brief AMRToLocal returns the local Box associated with the given AMR one.
         * This method only deals with **cell** indexes.
         */
        template<typename T>
        NO_DISCARD auto AMRToLocal(damn::core::Box<T, dimension> const& AMRBox) const
        {
            return amr_to_local(this->cells, AMRBox, options.field_ghost_width);
        }



        template<auto src, auto dst, typename Field>
        NO_DISCARD static Field::type project(Field const& field, damn::core::Point<std::size_t, dimension> index)
            requires(HasPhysicalQuantity<Field>)
        DAMN_ALL
        {
            if constexpr (not DAMN_DEVICE_PASS) // host only: std::string, throw
            {
                DAMN_DEBUG_DO(
                    constexpr auto expected = centering(src);
                    if (centering(field) != expected) {
                        std::string msg = "GridLayout::project: field centering does not match "
                                          "the source quantity";
                        if constexpr (requires { field.name(); }) // basic:: tile views: no name
                            msg += ": " + field.name();
                        throw std::runtime_error(msg);
                    })
            }

            return core::project<centering(src), centering(dst)>(field, index);
        }


        /**
         * @brief Returns the index of @p index mirrored across a boundary.
         *
         * @param direction The direction normal to the boundary.
         * @param side Whether we reflect across the Lower or Upper boundary.
         * @param centering The staggering of the data along @p direction.
         * @param index The directional index to be reflected.
         *
         * @return The reflected directional index.
         */
        NO_DISCARD inline std::uint32_t boundaryMirrored(Direction direction, Side side,
                                                         QtyCentering centering,
                                                         std::uint32_t const index) const
        {
            return boundary_mirrored(centering, side, nbrPhysicalCells_[static_cast<std::uint32_t>(direction)], options.field_ghost_width, index);
        }

        template<std::size_t dim>
        NO_DISCARD inline damn::core::Point<std::uint32_t, dim>
        boundaryMirrored(Direction direction, Side side, QtyCentering centering,
                         damn::core::Point<std::uint32_t, dim> const point) const
        {
            std::size_t const iDir = static_cast<std::size_t>(direction);
            auto mirroredPoint     = point;
            // iDir < dim always holds for a valid call; the guard lets the compiler drop the
            // impossible out-of-bounds branch (avoids a GCC -Warray-bounds false positive in 1D).
            if (iDir < dim)
                mirroredPoint[iDir] = boundaryMirrored(direction, side, centering, point[iDir]);
            return mirroredPoint;
        }

        // ----------------------------------------------------------------------
        //                      LAYOUT SPECIFIC METHODS
        //
        // The methods below return results that need information specific to the
        // layout that is used. They thus all refer to the implT.
        // ----------------------------------------------------------------------



        /**
         * @brief returns the centering of a scalar hybrid quantity in each directions
         */
        NO_DISCARD constexpr static std::array<QtyCentering, dimension>
        centering(Scalar quantity) DAMN_ALL
        {
            return implT::centering(quantity);
        }



        /**
         * @brief returns the centering of a vector hybrid quantity in each directions
         */
        NO_DISCARD constexpr static std::array<std::array<QtyCentering, dimension>, 3>
        centering(Vector quantity) DAMN_ALL
        {
            return implT::centering(quantity);
        }

        NO_DISCARD constexpr static std::array<std::array<QtyCentering, dimension>, 6>
        centering(Tensor quantity) DAMN_ALL
        {
            return damn::core::for_N_make_array<6>(
                [](auto) { return damn::core::ConstArray<QtyCentering, dimension>(QtyCentering::primal); });
        }

        template<typename HasQuantity>
        NO_DISCARD constexpr static auto centering(HasQuantity const& hasQuantity)
            requires(HasPhysicalQuantity<HasQuantity>)
        DAMN_ALL
        {
            return centering(hasQuantity.physicalQuantity());
        }



        /**
         * @brief GridLayout<implT::dim>::allocSize
         * @return An std::array<std::uint32_t, dim> object, containing the size to which allocate
         * arrays of an HybridQuantity::Quantity 'qty' in every directions.
         */
        NO_DISCARD std::array<std::uint32_t, dimension> allocSize(Scalar qty) const DAMN_ALL
        {
            return alloc_shape(this->cells, centering(qty), options.field_ghost_width);
        }


        /**
         * @brief allocSizeDerived returns the shape of the array to be allocated to store
         * the derivative of a given quantity in a given direction.
         */
        NO_DISCARD std::array<std::uint32_t, dimension> allocSizeDerived(Scalar qty,
                                                                         Direction dir) const
        {
            auto c                            = centering(qty);
            c[static_cast<std::uint32_t>(dir)] = flip(c[static_cast<std::uint32_t>(dir)]);
            return alloc_shape(this->cells, c, options.field_ghost_width);
        }



        /** @brief return the centering of a given Field along a given direction
         */
        template<typename Field_t>
        NO_DISCARD QtyCentering fieldCentering(Field_t const& field, Direction dir) const
        {
            std::uint32_t iDir           = static_cast<std::uint32_t>(dir);
            std::uint32_t iQty           = static_cast<std::uint32_t>(field.physicalQuantity());
            constexpr auto& qtyCentering = implT::qtyCentering_;

            return qtyCentering[iQty][iDir];
        }


        /**
         * @brief nbrPhysicalNodes returns the number of nodes in each direction, that are node
         * ghost nodes
         */
        NO_DISCARD std::array<std::uint32_t, dimension> nbrPhysicalNodes(Scalar hybQty) const
        {
            return physical_nodes(this->cells, centering(hybQty));
        }


        /**
         * @brief derivedCentering this function returns the
         * centering (primal or dual) of a quantity after a first order derivation. dual becomes
         * primal and primal becomes dual. quantityCentering is used to know if the
         * HybridQuantity::Quantity 'qty' is primal or dual in the Direction 'dir'
         */
        NO_DISCARD QtyCentering derivedCentering(Scalar qty, Direction dir) const
        {
            return flip(centering(qty)[static_cast<std::uint32_t>(dir)]);
        }


        /**
         * @brief toFieldBox takes a cell-centered box, local or AMR, and creates a box
         * that is adequate for the specified quantity. The layout is used to know
         * the centering of the specified quantity.
         *
         * @see FieldGeometry::toFieldBox
         *
         * */
        template<typename T>
        NO_DISCARD static damn::core::Box<T, dimension> toFieldBox(damn::core::Box<T, dimension> const& box, Scalar qty)
        {
            return toCenteredBox(box, centering(qty));
        }


        auto AMRBoxFor(auto const& field) const { return amr_box(this->cells, centering(field)); }

        auto AMRGhostBoxFor(auto const& field) const
        {
            return amr_ghost_box(this->cells, centering(field), options.field_ghost_width);
        }


        template<auto direction>
        static damn::core::Point<std::size_t, dimension> next(damn::core::Point<std::size_t, dimension> index)
        {
            return core::next<direction>(index);
        }

        template<auto direction>
        static damn::core::Point<std::size_t, dimension> previous(damn::core::Point<std::size_t, dimension> index)
        {
            return core::previous<direction>(index);
        }

        // essentially box form of allocSize(...)
        template<typename Field>
        damn::core::Box<std::uint32_t, dimension> ghostBoxFor(Field const& field) const
        {
            return local_ghost_box(this->cells, centering(field), options.field_ghost_width);
        }

        template<typename Field>
        auto domainBoxFor(Field const& field) const
        {
            return local_domain_box(this->cells, centering(field), options.field_ghost_width);
        }

        template<typename Field, typename Fn, typename... Args>
        void evalOnBox(Field const& field, Fn&& fn, Args&&... args) const
        {
            evalOnAnyBox(field, domainBoxFor(field), fn, args...);
        }

        template<typename Field, typename Fn, typename... Args>
        void evalGrownOnBox(Field const& field, auto const& growby, Fn&& fn, Args&&... args) const
        {
            evalOnAnyBox(field, damn::core::grow(domainBoxFor(field), growby), fn, args...);
        }

        template<typename Field, typename Fn, typename... Args>
        void evalOnGhostBox(Field const& field, Fn&& fn, Args&&... args) const
        {
            evalOnAnyBox(field, ghostBoxFor(field), fn, args...);
        }

        template<typename Field, typename Fn, typename... Args>
        void evalOnBiggerBox(Field const& field, damn::core::Point<uint32_t, dimension> const& growby, Fn&& fn,
                             Args&&... args) const
        {
            evalOnAnyBox(field, damn::core::grow(domainBoxFor(field), growby), fn, args...);
        }

        template<typename Field, typename Fn, typename... Args>
        void evalOnShrinkedGhostBox(Field const& field, damn::core::Point<uint32_t, dimension> const& shrinkby,
                                    Fn&& fn, Args&&... args) const
        {
            evalOnAnyBox(field, damn::core::shrink(ghostBoxFor(field), shrinkby), fn, args...);
        }

        auto levelNumber() const DAMN_ALL { return this->level; }


        // function to take a part of a gridlayout
        This copy_as(damn::core::Box<int, dimension> const box) const
        {
            auto const shifted_origin = damn::core::Point<double, dimension>{this->dx} * box.lower;
            return {this->dx, box.shape().as_unsigned(), shifted_origin, box, this->level};
        }

        // call fn(amr_idx, lcl_idx) for each point of the AMR box
        void amr_lcl_idx(auto const& box, auto&& fn) const
        {
            damn::core::for_each_point_zip(box, AMRToLocal(box), fn);
        }
        void amr_lcl_idx(auto&& fn) const { amr_lcl_idx(AMRBox(), fn); }
        void domain_amr_lcl_idx(auto const& field, auto&& fn) const
        {
            amr_lcl_idx(AMRBoxFor(field), fn);
        }
        void ghost_amr_lcl_idx(auto const& field, auto&& fn) const
        {
            amr_lcl_idx(AMRGhostBoxFor(field), fn);
        }

    private:
        template<typename Field, typename Box_t, typename Fn, typename... Args>
        void static evalOnAnyBox([[maybe_unused]] Field const& feeld, Box_t const& box, Fn& fn,
                                 Args&... args)
        {
            static_assert(!is_field_tile_set_v<Field>);

            if constexpr (Field::alloc_mode == damn::AllocatorMode::CPU)
            {
                // span by span walk, see core/geometry/box_spans.hpp
                damn::core::for_each_point(box, [&](auto const& bix) { fn(bix, args...); });
            }
            else
            {
                gpu::GridLayout::evalOnBox(fn, box, args...);
            }
        }

    private:
        std::array<std::uint32_t, dimension> nbrPhysicalCells_;
        std::array<double, dimension> inverseMeshSize_;
        static constexpr GridData data{};


    };


} // namespace core
} // namespace plamn

#endif // GridLayout_HPP
