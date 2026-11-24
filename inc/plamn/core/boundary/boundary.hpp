#ifndef PLAMN_CORE_BOUNDARY_BOUNDARY_HPP
#define PLAMN_CORE_BOUNDARY_BOUNDARY_HPP

// a model's physical boundaries (PHARE's core/boundary: Boundary, BoundaryFactory,
//  BoundaryManager), as values: a Boundary is its type, location and a FieldCondition per
//  quantity, made by make_boundary from its type (and an inflow's state):
//
//                ρ, Etot            ρV          B                         E
//   open         Neumann            Neumann     div free transverse Neum. -
//   reflective   Neumann            symmetric   div free transverse Neum. antisymmetric
//   inflow       Dirichlet ρ0, Etot Dirichlet   div free transverse Dir.  Dirichlet -v x B
//                (from P, γ)        ρ0 v        B0
//
//  (inflow: super magnetofast, constant extrapolation; other quantities: none). Reflective, open
//  and inflow are MHD's only (as PHARE). A BoundaryManager holds the domain's boundaries (each
//  face of a non periodic direction), from the input's simulation/grid/domain_boundaries; an edge
//  or corner takes one adjacent face's conditions: the highest direction (by default), or the
//  highest boundary type

#include "damn/core/def.hpp"

#include "plamn/core/boundary/boundary_defs.hpp"
#include "plamn/core/numerics/boundary_condition/field_boundary_condition.hpp"
#include "plamn/core/numerics/primitive_conservative/primitive_conservative.hpp"
#include "plamn/initializer/data_provider.hpp"

#include "dict.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <format>
#include <initializer_list>
#include <map>
#include <memory>

namespace plamn::core
{
class MHDQuantity;

// a super magnetofast inflow's prescribed state
struct InflowState
{
    double density  = 0.;
    double pressure = 0.;
    std::array<double, 3> velocity{};
    std::array<double, 3> B{};
};

template<typename Quantity>
struct Boundary
{
    using Scalar = Quantity::Scalar;
    using Vector = Quantity::Vector;

    BoundaryType type         = BoundaryType::None;
    BoundaryLocation location = BoundaryLocation::XLower;
    std::vector<std::pair<Scalar, FieldCondition>> scalars{};
    std::vector<std::pair<Vector, FieldCondition>> vectors{};

    // the quantity's condition, nullptr if it has none
    MKN_KUL_FN_SIG_RET FieldCondition const* condition(Scalar const q) const
    {
        return find_(scalars, q);
    }
    MKN_KUL_FN_SIG_RET FieldCondition const* condition(Vector const q) const
    {
        return find_(vectors, q);
    }

private:
    static FieldCondition const* find_(auto const& conditions, auto const q)
    {
        for (auto const& [quantity, condition] : conditions)
            if (quantity == q)
                return &condition;
        return nullptr;
    }
};

// the boundary of `type` at `location` for the given quantities; inflow needs its state and the
//  heat capacity ratio gamma (> 1)
template<typename Quantity>
MKN_KUL_FN_SIG_RET Boundary<Quantity>
make_boundary(BoundaryLocation const location, BoundaryType const type,
              std::vector<typename Quantity::Scalar> const& scalars,
              std::vector<typename Quantity::Vector> const& vectors,
              std::optional<InflowState> const& inflow = std::nullopt, double const gamma = 0.)
{
    using enum FieldConditionType;
    using S = Quantity::Scalar;
    using V = Quantity::Vector;

    Boundary<Quantity> b{.type = type, .location = location};
    auto const add = [&](auto const q, FieldCondition const c) {
        if constexpr (std::is_same_v<std::decay_t<decltype(q)>, S>)
            b.scalars.emplace_back(q, c);
        else
            b.vectors.emplace_back(q, c);
    };
    auto const none = FieldCondition{.type = None};

    if (type == BoundaryType::None)
    {
        for (auto const q : scalars)
            add(q, none);
        for (auto const q : vectors)
            add(q, none);
        return b;
    }
    if constexpr (!std::is_same_v<Quantity, MHDQuantity>)
        throw std::runtime_error("plamn: physical boundaries are the MHD model's only");
    else
    {
        if (type == BoundaryType::Reflective or type == BoundaryType::Open)
        {
            bool const reflective = type == BoundaryType::Reflective;
            for (auto const q : scalars)
                add(q, (reflective or q == S::rho or q == S::Etot) ? FieldCondition{.type = Neumann}
                                                                   : none);
            for (auto const q : vectors)
            {
                if (q == V::B)
                    add(q, FieldCondition{.type = DivergenceFreeTransverseNeumann});
                else if (q == V::rhoV)
                    add(q, FieldCondition{.type = reflective ? Symmetric : Neumann});
                else if (q == V::E and reflective)
                    add(q, FieldCondition{.type = AntiSymmetric});
                else
                    add(q, none);
            }
            return b;
        }
        if (type != BoundaryType::SuperMagnetofastInflow)
            throw std::runtime_error("plamn: unknown boundary type");
        if (!inflow)
            throw std::runtime_error("plamn: a super magnetofast inflow needs its state");
        if (!(gamma > 1.))
            throw std::runtime_error("plamn: a super magnetofast inflow needs gamma > 1, got "
                                     + std::to_string(gamma));

        auto const& [rho, P, v, B] = *inflow;
        auto const constant        = DirichletExtrapolation::Constant;
        auto const dirichlet       = [&](std::array<double, 3> const value) {
            return FieldCondition{.type = Dirichlet, .value = value, .extrapolation = constant};
        };
        double const Etot = eos_p_to_etot(gamma, rho, v[0], v[1], v[2], B[0], B[1], B[2], P);
        for (auto const q : scalars)
            add(q, q == S::rho    ? dirichlet({rho, 0., 0.})
                   : q == S::Etot ? dirichlet({Etot, 0., 0.})
                                  : none);
        for (auto const q : vectors)
        {
            if (q == V::rhoV)
                add(q, dirichlet({rho * v[0], rho * v[1], rho * v[2]}));
            else if (q == V::B)
                add(q, FieldCondition{.type          = DivergenceFreeTransverseDirichlet,
                                      .value         = B,
                                      .extrapolation = constant});
            else if (q == V::E) // ideal: E = -v x B
                add(q, dirichlet({v[2] * B[1] - v[1] * B[2], v[0] * B[2] - v[2] * B[0],
                                  v[1] * B[0] - v[0] * B[1]}));
            else
                add(q, none);
        }
        return b;
    }
}

// the boundary at `location` from its input (simulation/grid/domain_boundaries/<location>: type,
//  and for an inflow density, pressure, velocity/{x,y,z}, B/{x,y,z})
template<typename Quantity>
MKN_KUL_FN_SIG_RET Boundary<Quantity>
make_boundary(BoundaryLocation const location, initializer::Dict const& dict,
              std::vector<typename Quantity::Scalar> const& scalars,
              std::vector<typename Quantity::Vector> const& vectors, double const gamma = 0.)
{
    auto const type = cppdict::get_value(dict, "type", BoundaryType::None);
    std::optional<InflowState> inflow;
    if (type == BoundaryType::SuperMagnetofastInflow)
    {
        auto const xyz = [&](std::string const& key) {
            if (!dict.contains(key))
                throw std::runtime_error("plamn: a super magnetofast inflow needs its " + key);
            return std::array{dict[key]["x"].template to<double>(),
                              dict[key]["y"].template to<double>(),
                              dict[key]["z"].template to<double>()};
        };
        inflow = InflowState{.density  = dict["density"].template to<double>(),
                             .pressure = dict["pressure"].template to<double>(),
                             .velocity = xyz("velocity"),
                             .B        = xyz("B")};
    }
    return make_boundary<Quantity>(location, type, scalars, vectors, inflow, gamma);
}

// every non periodic direction's faces declare a boundary (not none)
template<std::size_t dim>
void validate_physical_boundaries(initializer::Dict const& grid)
{
    if (!grid.contains("periodicities"))
        return;
    constexpr std::array<char const*, 3> xyz{"x", "y", "z"};
    for (std::size_t d = 0; d < dim; ++d)
    {
        if (grid["periodicities"][xyz[d]].template to<bool>())
            continue;
        for (auto const* side : {"lower", "upper"})
        {
            auto const location = std::string{xyz[d]} + side;
            bool const declared = grid.contains("domain_boundaries")
                                  and grid["domain_boundaries"].contains(location)
                                  and cppdict::get_value(grid["domain_boundaries"][location],
                                                         "type", BoundaryType::None)
                                          != BoundaryType::None;
            if (!declared)
                throw std::runtime_error("plamn: direction " + std::string{xyz[d]}
                                         + " is not periodic but " + location
                                         + " declares no boundary");
        }
    }
}

template<typename Quantity>
struct BoundaryManager
{
    // which adjacent face an edge or a corner takes the conditions of
    enum class Priority : std::uint8_t { ByDirection, ByBoundaryType };

    std::array<std::optional<Boundary<Quantity>>, 6> boundaries{};
    Priority priority = Priority::ByDirection;

    BoundaryManager() = default;

    // from simulation/grid/domain_boundaries (absent: no physical boundary)
    BoundaryManager(initializer::Dict const& domain_boundaries,
                    std::vector<typename Quantity::Scalar> const& scalars,
                    std::vector<typename Quantity::Vector> const& vectors, double const gamma = 0.,
                    Priority const priority_ = Priority::ByDirection)
        : priority{priority_}
    {
        if (!domain_boundaries.isNode())
            return;
        for (auto const& [key, sub] : domain_boundaries)
        {
            auto const location = boundary_location(key);
            boundaries[static_cast<std::size_t>(location)]
                = make_boundary<Quantity>(location, *sub, scalars, vectors, gamma);
        }
    }

    MKN_KUL_FN_SIG_RET Boundary<Quantity> const* boundary(BoundaryLocation const location) const
    {
        auto const& b = boundaries[static_cast<std::size_t>(location)];
        return b ? &*b : nullptr;
    }

    // the face whose conditions a location takes: itself for a face, else one of its adjacent
    //  faces by the priority
    template<typename Location>
    MKN_KUL_FN_SIG_RET BoundaryLocation master(Location const location) const
    {
        if constexpr (std::is_same_v<Location, BoundaryLocation>)
            return location;
        else
        {
            auto const faces = adjacent_locations(location);
            if (priority == Priority::ByDirection)
                return *std::ranges::max_element(faces, {}, direction_of);
            return *std::ranges::max_element(faces, {}, [&](auto const face) {
                auto const* b = boundary(face);
                if (!b)
                    throw std::runtime_error("plamn: an edge or corner's face has no boundary");
                return b->type;
            });
        }
    }
};

} // namespace plamn::core

#endif // PLAMN_CORE_BOUNDARY_BOUNDARY_HPP
