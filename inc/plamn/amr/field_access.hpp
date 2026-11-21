#ifndef PLAMN_AMR_FIELD_ACCESS_HPP
#define PLAMN_AMR_FIELD_ACCESS_HPP

// damn's copies reach a patch's fields through accessors (damn/amr/copy.hpp), not by their names:
//  no name is built or compared per patch and piece
//
//   grid_table(patch)          its fixed grids' addresses, in for_each_fixed_field_grid's order,
//                              made on first use (the grids never move: patches are not movable,
//                              populations are held by unique_ptr)
//   FieldAccess<gw>            a resource pair's source and destination, each a grid table
//                              index, or a scratch grid's name (scratch grids come and go, so
//                              they are not in the table)
//   field_access<gw>(patch, pairs)  the pairs' accessors, the indices from one patch's
//                              enumeration: every patch of a model has the same table
//   level_field_access<cgw, fgw>(coarse_patch, fine_patch, pairs)
//                              refine / coarsen's: coarse, coarse_new and fine sides, each in
//                              its level's (model's) table; none if a name is no field of its
//                              patch (particles: by name)

#include "damn/amr/level_map.hpp"
#include "damn/amr/neighbours.hpp"
#include "plamn/amr/resources.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <format>
#include <type_traits>
#include <utility>

namespace plamn::amr
{
template<typename Patch>
auto& grid_table(Patch& patch)
{
    if (patch.grid_table_.empty())
        for_each_fixed_field_grid(patch, [&](auto const&, auto& grid) {
            patch.grid_table_.push_back(std::addressof(grid)); // Grid overloads operator&
        });
    return patch.grid_table_;
}

struct GridAt
{
    std::size_t index = 0;
    std::string scratch{}; // not empty: the scratch grid of that name, not the table's
};

template<std::uint32_t gw>
struct FieldAccess
{
    GridAt src_at, dst_at;

    static auto& grid_of(auto& patch, GridAt const& at)
        requires requires { patch.scratch; }
    {
        return at.scratch.empty() ? *grid_table(patch)[at.index] : patch.scratch.at(at.scratch);
    }
    static auto& grid_of(auto& patch, GridAt const& at)
        requires(!requires { patch.scratch; })
    {
        return *grid_table(patch)[at.index];
    }

    // data: a LevelData's patch data (a patch behind a unique_ptr)
    auto src(auto& data) const { return field_resource<gw>(grid_of(*data, src_at)); }
    auto dst(auto& data) const { return field_resource<gw>(grid_of(*data, dst_at)); }
};

// name's place on a patch: its grid table index, or its scratch grid; none if it is no field
inline auto grid_at_of(auto& patch)
{
    std::map<std::string, std::size_t, std::less<>> index;
    std::size_t k = 0;
    for_each_fixed_field_grid(patch, [&](auto const& name, auto&) { index.emplace(name, k++); });
    return [index = std::move(index), &patch](std::string const& name) -> std::optional<GridAt> {
        if (auto const it = index.find(name); it != index.end())
            return GridAt{it->second, {}};
        if constexpr (requires { patch.scratch; })
            if (patch.scratch.contains(name))
                return GridAt{0, name};
        return std::nullopt;
    };
}

template<std::uint32_t gw>
std::vector<FieldAccess<gw>> field_access(auto& patch,
                                          std::vector<damn::amr::ResourcePair> const& pairs)
{
    auto const at_of = grid_at_of(patch);
    auto const at    = [&](std::string const& name) {
        if (auto const at = at_of(name))
            return *at;
        throw std::runtime_error("plamn::amr::field_access: no field " + name);
    };
    std::vector<FieldAccess<gw>> access;
    access.reserve(pairs.size());
    for (auto const& pair : pairs)
        access.push_back({at(pair.src), at(pair.dst)});
    return access;
}

template<std::uint32_t cgw, std::uint32_t fgw>
struct LevelFieldAccess
{
    GridAt coarse_at, coarse_new_at, fine_at;

    // data: a LevelData's patch data (a patch behind a unique_ptr)
    auto coarse(auto& data) const
    {
        return field_resource<cgw>(FieldAccess<cgw>::grid_of(*data, coarse_at));
    }
    auto coarse_new(auto& data) const
    {
        return field_resource<cgw>(FieldAccess<cgw>::grid_of(*data, coarse_new_at));
    }
    auto fine(auto& data) const
    {
        return field_resource<fgw>(FieldAccess<fgw>::grid_of(*data, fine_at));
    }
};

template<std::uint32_t cgw, std::uint32_t fgw>
std::optional<std::vector<LevelFieldAccess<cgw, fgw>>>
level_field_access(auto& coarse_patch, auto& fine_patch,
                   std::vector<damn::amr::LevelPair> const& pairs)
{
    auto const coarse_at = grid_at_of(coarse_patch);
    auto const fine_at   = grid_at_of(fine_patch);
    std::vector<LevelFieldAccess<cgw, fgw>> access;
    access.reserve(pairs.size());
    for (auto const& pair : pairs)
    {
        auto const c = coarse_at(pair.coarse);
        auto const f = fine_at(pair.fine);
        auto const n = pair.coarse_new.empty() ? c : coarse_at(pair.coarse_new);
        if (!c or !f or !n)
            return std::nullopt; // a particle array, say: by name
        access.push_back({*c, *n, *f});
    }
    return access;
}

} // namespace plamn::amr

#endif // PLAMN_AMR_FIELD_ACCESS_HPP
