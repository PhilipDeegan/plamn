#ifndef PLAMN_DIAGNOSTIC_PHAREH5_HPP
#define PLAMN_DIAGNOSTIC_PHAREH5_HPP

// phareh5 diagnostics (PHARE's default format: diagnostic/detail/h5writer.hpp and types/), the
// pieces; Simulator::dump_phareh5_ (simulator_diagnostics.hpp) writes them. A file per quantity:
//
//   /t/<time, 10 decimals>/pl<level>/p<rank>#<id>     a patch: attributes lower, upper (its
//                                                      cells), nbrCells, origin, mpi_rank
//     fields     <name>_<component> (B, E, vectors: x y z; tensors: xx xy xz yy yz zz; scalars:
//                <name>) on PHARE's ghost box of the patch (field_ghosts per interp order), in
//                the quantity's own (Yee) centering, C order (x slowest); attribute ghosts
//     particles  weight (n, 1), iCell (n, dim) int32, delta (n, dim), v (n, 3), memory order
//                (none when there are no particles); the patch's attribute ghosts
//                (particle_ghosts)
//   /           dimension, interpOrder, layoutType yee, domain_box (its upper), cell_width,
//               origin, periodicities; pop_mass and pop_charge for a population's quantity
//   /py_attrs   the diagnostic's attributes (pyphare's git hash, versions, the serialized
//               simulation)
//
// a dump at a time covers every level, or one (PHARE's dump_level, fine_dump_lvl_max): such a
// time holds that level only

#include "damn/core/def.hpp"
#include "damn/core/geometry/point.hpp"
#include "plamn/amr/resources.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"
#include "plamn/core/data/particles/particle_components.hpp"
#include "plamn/core/models/quantities/quantity.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>
#include <format>
#include <type_traits>

namespace damn::core
{
template<typename Type, std::size_t dim>
struct Box;
} // namespace damn::core

namespace plamn::diagnostic::phareh5
{
// PHARE master's ghost widths (core/models/options/hybrid_options.hpp): what its files hold,
//  whatever plamn's own widths
template<std::size_t interp>
std::uint32_t constexpr field_ghosts = std::array{2u, 4u, 4u}[interp - 1];
template<std::size_t interp>
std::uint32_t constexpr particle_ghosts = std::array{1u, 2u, 2u}[interp - 1];

// /ions/pop/protons/flux -> ions_pop_protons_flux.h5
NO_DISCARD inline std::string file_name(std::string const& quantity)
{
    auto name = quantity.substr(quantity.find_first_not_of('/'));
    std::replace(name.begin(), name.end(), '/', '_');
    return name + ".h5";
}

// what PHARE's core::to_string_with_precision(t, 10) returns: fixed, 10 decimals (it zeroes the
//  last digit of a copy, then returns the stream's string: every digit kept)
NO_DISCARD inline std::string time_key(double const t)
{
    std::ostringstream out;
    out.precision(10);
    out << std::fixed << t;
    return out.str();
}

NO_DISCARD inline std::string patch_path(std::string const& time, int const level, int const rank,
                                         std::uint32_t const id)
{
    return "/t/" + time + "/pl" + std::to_string(level) + "/p" + std::to_string(rank) + "#"
           + std::to_string(id);
}

// a patch's field on PHARE's ghost box (ghosts G of plamn's gw), C order (x slowest): its
//  shape, and its values into out
template<std::uint32_t gw, std::uint32_t G, typename Field, typename Float>
void field_values(Field const& field, damn::core::Box<int, Field::dimension> const& cells,
                  std::vector<std::size_t>& shape, std::vector<Float>& out)
{
    static_assert(G <= gw, "plamn's field ghosts narrower than PHARE's");
    auto constexpr dim    = Field::dimension;
    auto const resource   = amr::field_resource<gw>(const_cast<Field&>(field));
    auto const centerings = core::centering<dim>(field);

    damn::core::Point<int, dim> lower;
    shape.assign(dim, 0);
    std::size_t n = 1;
    for (std::size_t i = 0; i < dim; ++i)
    {
        lower[i] = cells.lower[i] - static_cast<int>(G);
        shape[i] = static_cast<std::size_t>(cells.upper[i] - cells.lower[i] + 1)
                   + (centerings[i] == core::QtyCentering::primal ? 1 : 0) + 2 * G;
        n *= shape[i];
    }
    out.resize(n);
    for (std::size_t k = 0; k < n; ++k)
    {
        damn::core::Point<int, dim> p;
        auto rest = k;
        for (std::size_t i = dim; i-- > 0;) // C order: the last direction fastest
        {
            p[i] = lower[i] + static_cast<int>(rest % shape[i]);
            rest /= shape[i];
        }
        out[k] = static_cast<Float>(value(resource, cells, p));
    }
}

// PHARE's particle datasets (ParticlePacker's keys and shapes): weight (n, 1), iCell (n, dim),
//  delta (n, dim), v (n, 3), each row-major
template<std::size_t dim>
struct ParticleColumns
{
    std::size_t n = 0;
    std::vector<double> weight, delta, v;
    std::vector<int> iCell;
};

template<typename Particles>
NO_DISCARD auto particle_columns(Particles const& particles)
{
    auto constexpr dim = std::decay_t<Particles>::dimension;
    ParticleColumns<dim> out;
    if (particles.size() == 0)
        return out;
    auto const c = core::components(particles);
    out.weight.reserve(particles.size());
    out.iCell.reserve(particles.size() * dim);
    out.delta.reserve(particles.size() * dim);
    out.v.reserve(particles.size() * 3);
    for (std::size_t j = 0; j < particles.size(); ++j)
    {
        if (c.weight[j] == 0) // SoAVXChunked's gaps
            continue;
        out.weight.push_back(c.weight[j]);
        for (std::size_t i = 0; i < dim; ++i)
        {
            out.iCell.push_back(c.iCell[i][j]);
            out.delta.push_back(c.delta[i][j]);
        }
        for (std::size_t i = 0; i < 3; ++i)
            out.v.push_back(c.v[i][j]);
    }
    out.n = out.weight.size();
    return out;
}

} // namespace plamn::diagnostic::phareh5

#endif // PLAMN_DIAGNOSTIC_PHAREH5_HPP
