#ifndef PLAMN_CORE_DATA_GRID_COORDINATES_HPP
#define PLAMN_CORE_DATA_GRID_COORDINATES_HPP

#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/memory/span.hpp"
#include "damn/core/meta/for_N.hpp"
#include "damn/core/geometry/box_spans.hpp"

#include "plamn/core/def.hpp"
#include "plamn/core/data/grid/layout.hpp"

#include <array>
#include <vector>
#include <cstddef>
#include <cstdint>

namespace damn::core
{
template<typename Type, std::size_t dim>
struct Box;
} // namespace damn::core

namespace plamn::core
{
enum class QtyCentering : std::uint8_t;

// physical coordinates of every point of an AMR box, one vector per direction (as user
// init functions take them), with the AMR index of each entry in the same order
template<std::size_t dim>
struct CoordinateVectors
{
    std::array<std::vector<double>, dim> xyz;
    std::vector<damn::core::Point<int, dim>> points;

    auto size() const { return points.size(); }

    auto spans() const
    {
        return damn::core::for_N<dim>([&](auto i) { return damn::core::make_span(xyz[i]); });
    }
};


template<std::size_t dim>
auto coordinate_vectors(std::array<double, dim> const& dx, damn::core::Box<int, dim> const& amr,
                        std::array<QtyCentering, dim> const& centering)
{
    CoordinateVectors<dim> cv;
    cv.points.reserve(amr.size());
    for (auto& x : cv.xyz)
        x.reserve(amr.size());

    damn::core::for_each_point(amr, [&](auto const& p) {
        damn::core::Point<int, dim> const point{p};
        auto const x = coordinate(dx, point, centering);
        for (std::size_t i = 0; i < dim; ++i)
            cv.xyz[i].emplace_back(x[i]);
        cv.points.emplace_back(point);
    });

    return cv;
}

} // namespace plamn::core

#endif // PLAMN_CORE_DATA_GRID_COORDINATES_HPP
