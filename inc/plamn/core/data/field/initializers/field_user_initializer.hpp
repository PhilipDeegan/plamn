#ifndef PLAMN_CORE_DATA_FIELD_INITIALIZERS_FIELD_USER_INITIALIZER_HPP
#define PLAMN_CORE_DATA_FIELD_INITIALIZERS_FIELD_USER_INITIALIZER_HPP

#include "damn/core/memory/span.hpp"

#include "plamn/core/data/grid/coordinates.hpp"
#include "plamn/core/data/field/field_tiles.hpp"
#include "plamn/core/data/grid/layout.hpp"
#include "plamn/core/models/quantities/quantity.hpp"

#include <tuple>
#include <cstddef>
#include <memory>

namespace plamn::core
{

class FieldUserFunctionInitializer
{
    void static _initialize(auto& field, auto const& layout, auto const& init)
    {
        auto const coords
            = coordinate_vectors(layout.dx, qty_ghost_box(layout, field), centering(layout, field));

        std::shared_ptr<damn::core::Span<double>> gridPtr // keep grid data alive
            = std::apply([&](auto const&... args) { return init(args...); }, coords.spans());

        damn::core::Span<double>& grid = *gridPtr;

        for (std::size_t i = 0; i < coords.size(); ++i)
            field(to_local(layout, coords.points[i])) = grid[i];
    }

public:
    template<typename Field>
    void static initialize(Field& field, auto const& layout, auto const& init)
    {
        auto constexpr static is_field_tile_set = is_field_tile_set_v<Field>;

        if constexpr (is_field_tile_set)
            for (auto& tile : field())
                _initialize(tile(), tile.layout(), init);
        else
            _initialize(field, layout, init);
    }
};


} // namespace plamn::core

#endif
