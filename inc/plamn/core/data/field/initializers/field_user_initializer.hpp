#ifndef PLAMN_CORE_DATA_FIELD_INITIALIZERS_FIELD_USER_INITIALIZER_HPP
#define PLAMN_CORE_DATA_FIELD_INITIALIZERS_FIELD_USER_INITIALIZER_HPP

#include "plamn/core/def.hpp"



#include "plamn/core/data/grid/grid_tiles.hpp"
#include "plamn/core/data/grid/coordinates.hpp"

#include <tuple>


namespace plamn::core
{

class FieldUserFunctionInitializer
{
    template<typename Field, typename GridLayout, typename InitFunction>
    void static _initialize(Field& field, GridLayout const& layout, InitFunction const& init)
    {
        auto const coords = coordinate_vectors(layout.meshSize(), layout.AMRGhostBoxFor(field),
                                               layout.centering(field));

        std::shared_ptr<damn::core::Span<double>> gridPtr // keep grid data alive
            = std::apply([&](auto const&... args) { return init(args...); }, coords.spans());

        damn::core::Span<double>& grid = *gridPtr;

        for (std::size_t i = 0; i < coords.size(); ++i)
            field(layout.AMRToLocal(coords.points[i])) = grid[i];
    }

public:
    template<typename Field, typename GridLayout, typename InitFunction>
    void static initialize(Field& field, GridLayout const& layout, InitFunction const& init)
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
