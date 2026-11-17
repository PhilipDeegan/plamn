#ifndef PLAMN_CORE_NUMERICS_PARTICLES_SYNCHRONISED_MOVER_DEF_HPP
#define PLAMN_CORE_NUMERICS_PARTICLES_SYNCHRONISED_MOVER_DEF_HPP

#include "damn/core/geometry/box.hpp"
#include "damn/core/data/tiles/tile_set_overlaps.hpp"
#include "damn/core/def/detail/mkn_gpu.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/sys/env.hpp"

#include "plamn/core/data/tiles/tile_set_overlaps.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"

#include <cstdint>
#include <array>
#include <cstddef>
#include <format>
#include <string>
#include <vector>

namespace plamn::core
{
enum class UpdaterMode : std::uint16_t { domain_only = 0, all };

template<typename Layout>
struct UpdaterSelectionBoxing;
template<typename Selector_t, typename Layout>
struct UpdaterCellMapSelectionBoxing;

template<typename Updater_t, typename Layout>
constexpr auto* cellmapselection_boxing_impl()
{
    using Selector_t = Updater_t::Pusher::ParticleSelector;
    return static_cast<UpdaterCellMapSelectionBoxing<Selector_t, Layout>*>(0);
}

template<typename Updater_t, typename Layout>
constexpr auto* selection_boxing_impl()
{
    using enum LayoutMode;
    if constexpr (Updater_t::Particles_t::layout_mode == AoSMapped)
        return cellmapselection_boxing_impl<Updater_t, Layout>();
    else
        return static_cast<UpdaterSelectionBoxing<Layout>*>(0);
}



template<typename Layout>
struct UpdaterSelectionBoxing
{
    auto constexpr static partGhostWidth = Layout::options.particle_ghost_width;
    using Layout_t                       = Layout;
    using Box_t                          = damn::core::Box<int, Layout_t::dimension>;

    UpdaterSelectionBoxing(Layout_t const& layout_, std::vector<Box_t> const& nonLevelGhostBox_,
                           std::vector<Box_t> const& nonOwnedBoxes_ = {})
        : layout{layout_}
        , nonLevelGhostBox{nonLevelGhostBox_}
        , nonOwnedBoxes{nonOwnedBoxes_}
    {
    }

    Layout_t const layout;
    std::vector<Box_t> const nonLevelGhostBox;
    // domain cells owned by an overlapping same-level patch, usually empty
    std::vector<Box_t> const nonOwnedBoxes;
    Box_t const domainBox = layout.cells;
    Box_t const ghostBox  = damn::core::grow(domainBox, partGhostWidth);

    static bool isOwned(auto const& cell, Box_t const& domainBox,
                        std::vector<Box_t> const& nonOwnedBoxes)
    {
        return isIn(cell, domainBox)
               and (nonOwnedBoxes.empty() or !isIn(damn::core::Point{cell}, nonOwnedBoxes));
    }

    // a particle belongs to this patch if and only if its cell is in the domain and owned
    bool isOwned(auto const& cell) const { return isOwned(cell, domainBox, nonOwnedBoxes); }
};


template<typename Selector_t, typename Layout>
struct UpdaterCellMapSelectionBoxing : public UpdaterSelectionBoxing<Layout>
{
    auto constexpr static partGhostWidth = Layout::options.particle_ghost_width;
    using Layout_t                       = Layout;
    using Box_t                          = damn::core::Box<int, Layout_t::dimension>;
    using Super                          = UpdaterSelectionBoxing<Layout>;

    UpdaterCellMapSelectionBoxing(Layout_t const& layout_,
                                  std::vector<Box_t> const& nonLevelGhostBox_,
                                  std::vector<Box_t> const& nonOwnedBoxes_ = {})
        : Super{layout_, nonLevelGhostBox_, nonOwnedBoxes_}
    {
    }

    Selector_t const noop = [](auto& particleRange) { return particleRange; };

    // lambda copy captures to detach from above references in case of class copy construct
    Selector_t const inOwnedDomain = [domainBox     = Super::domainBox,
                                      nonOwnedBoxes = Super::nonOwnedBoxes](auto& particleRange) {
        return particleRange.array().partition(particleRange, [&](auto const& cell) {
            return Super::isOwned(cell, domainBox, nonOwnedBoxes);
        });
    };

    Selector_t const inGhostBox = [ghostBox = Super::ghostBox](auto& particleRange) {
        return particleRange.array().partition(
            particleRange, [&](auto const& cell) { return isIn(cell, ghostBox); });
    };

    Selector_t const inNonLevelGhostBox
        = [nonLevelGhostBox = Super::nonLevelGhostBox](auto& particleRange) {
              return particleRange.array().partition(
                  particleRange, [&](auto const& cell) { return isIn(cell, nonLevelGhostBox); });
          };

    // the level ghost layer: ghost cells no patch of the level covers (a level ghost pushed into
    //  another patch's cell is that patch's: kept here it would come back as a duplicate,
    //  the level ghost duplicate fix)
    Selector_t const inGhostLayer
        = [ghostBox = Super::ghostBox, domainBox = Super::domainBox,
           nonLevelGhostBox = Super::nonLevelGhostBox](auto& particleRange) {
              return particleRange.array().partition(particleRange, [&](auto const& cell) {
                  return isIn(cell, ghostBox) and !isIn(cell, domainBox)
                         and !isIn(damn::core::Point{cell}, nonLevelGhostBox);
              });
          };

    Selector_t const outsideGhostBox = [ghostBox = Super::ghostBox](auto& particleRange) {
        return particleRange.array().partition(
            particleRange, [&](auto const& cell) { return !isIn(cell, ghostBox); });
    };
};



template<typename Layout>
struct UpdaterTileSetSelectionBoxing : public UpdaterSelectionBoxing<Layout>
{
    auto constexpr static dimension      = Layout::dimension;
    auto constexpr static partGhostWidth = Layout::options.particle_ghost_width;
    using Layout_t                       = Layout;
    using Box_t                          = damn::core::Box<int, Layout_t::dimension>;
    using Super                          = UpdaterSelectionBoxing<Layout>;
    using BoxSpanSet_t                   = damn::core::TileBoxSpanSet<dimension>;
    using scalar_t                       = HybridQuantity::Scalar;


    UpdaterTileSetSelectionBoxing(Layout_t const& layout_,
                                  std::vector<Box_t> const& nonLevelGhostBox_)
        : Super{layout_, nonLevelGhostBox_}
    {
    }

    static std::size_t constexpr field_quantities_count()
    {
        if constexpr (dimension == 1)
            return 2;
        if constexpr (dimension == 2)
            return 4;
        if constexpr (dimension == 3)
            return 7; // no ALL DUAL!
    }

    auto static field_quantities(auto const& layout)
    {
        // for all primal/dual permutations -- 1 = primal / 0 = dual
        if constexpr (dimension == 1)
            return std::array<BoxSpanSet_t, field_quantities_count()>{
                make_nd_span_set_for_qty(layout, scalar_t::By), // 0
                make_nd_span_set_for_qty(layout, scalar_t::Bx), // 1
            };
        if constexpr (dimension == 2)
            return std::array<BoxSpanSet_t, field_quantities_count()>{
                make_nd_span_set_for_qty(layout, scalar_t::Bz), // 0 0
                make_nd_span_set_for_qty(layout, scalar_t::By), // 0 1
                make_nd_span_set_for_qty(layout, scalar_t::By), // 1 0
                make_nd_span_set_for_qty(layout, scalar_t::Vx), // 1 1
            };
        if constexpr (dimension == 3)
            return std::array<BoxSpanSet_t, field_quantities_count()>{
                make_nd_span_set_for_qty(layout, scalar_t::Bz), // 0 0 1
                make_nd_span_set_for_qty(layout, scalar_t::By), // 0 1 0
                make_nd_span_set_for_qty(layout, scalar_t::Bx), // 1 0 0
                make_nd_span_set_for_qty(layout, scalar_t::Ex), // 0 1 1
                make_nd_span_set_for_qty(layout, scalar_t::Ey), // 1 0 1
                make_nd_span_set_for_qty(layout, scalar_t::Ez), // 1 1 0
                make_nd_span_set_for_qty(layout, scalar_t::Vx), // 1 1 1
            };
    }


    // std::array<BoxSpanSet_t, field_quantities_count()> fieldings =
    // field_quantities(Super::layout); BoxSpanSet_t particles{make_nd_span_set_from(Super::layout,
    // [](auto const& layout) {
    //     return box_from_zero_to_upper_minus_one(
    //         *grow(layout.cells, Layout_t::nbrParticleGhosts()).shape().as_unsigned());
    // })};
};


} // namespace plamn::core


// ###############################################

#if DAMN_HAVE_MKN_GPU || defined(_MKN_WITH_MKN_KUL_)
#include "mkn/kul/os.hpp"

namespace plamn::core::detail
{
auto static const timings_dir_str
    = damn::core::get_env_as("PLAMN_ASYNC_TIMES", std::string{".damn/async/multi_updater"});

static bool particles_synchronised_mover_io_setup = []() {
    mkn::kul::Dir timings{timings_dir_str};
    timings.mk();
    return true;
}();

} // namespace plamn::core::detail

#endif // DAMN_HAVE_MKN_GPU

// ###############################################


#endif // ION_UPDATER_DEF_HPP
