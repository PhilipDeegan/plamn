#ifndef PLAMN_SIMULATOR_SIMULATOR_GHOSTS_HPP
#define PLAMN_SIMULATOR_SIMULATOR_GHOSTS_HPP

// plamn::Simulator: ghost fills (patch and level ghosts), the debug check of unset points
// IWYU pragma: private, include "plamn/simulator/simulator.hpp"

#include "damn/amr/box_data.hpp"
#include "damn/amr/copy.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/level_map.hpp"
#include "damn/core/def.hpp"
#include "damn/core/geometry/box_spans.hpp"

#include "plamn/simulator/simulator.hpp" // IWYU pragma: keep (the class)
#include "plamn/amr/level_pairs.hpp"
#include "plamn/amr/refine.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <cstdint>
#include <deque>
#include "damn/amr/physical_boundary.hpp"
#include "plamn/core/boundary/physical_boundaries.hpp"

namespace damn::core
{
template<typename Type, std::size_t dim>
class Point;
template<typename Type, std::size_t dim>
struct Box;
} // namespace damn::core
namespace plamn::amr
{
template<typename Field, std::uint32_t gw>
struct FieldResource;
} // namespace plamn::amr
// IWYU pragma: private, include "plamn/simulator/simulator.hpp"

namespace plamn::simulator::detail
{
// the first point of a field resource's ghost box that is NaN (nothing filled it), points past
// a non periodic domain boundary skipped; std::nullopt for other resources
template<typename Resource, std::size_t dim>
MKN_KUL_FN_SIG_RET std::optional<damn::core::Point<int, dim>>
first_unset(Resource const& r, damn::core::Box<int, dim> const& cells,
            damn::core::Box<int, dim> const& domain, std::array<bool, dim> const& periodic)
{
    if constexpr (!damn::amr::BoxData<Resource, dim>)
        return std::nullopt;
    else
    {
        auto box          = ghost_box(r, cells);
        auto const inside = domain_box(r, domain);
        for (std::size_t d = 0; d < dim; ++d)
        {
            if (periodic[d])
                continue;
            box.lower[d] = std::max(box.lower[d], inside.lower[d]);
            box.upper[d] = std::min(box.upper[d], inside.upper[d]);
        }
        std::optional<damn::core::Point<int, dim>> at;
        damn::core::for_each_point(box, [&](auto const& p) {
            if (!at and std::isnan(value(r, cells, p)))
                at = p;
        });
        return at;
    }
}
} // namespace plamn::simulator::detail

namespace plamn
{
template<auto opts>
void Simulator<opts>::fill_magnetic_ghosts_(std::size_t const ln, std::string const& key,
                                            double const t)
{
    copy_(ln, damn::amr::fill_ghosts, key);
    if (ln > 0)
    {
        refine_(ln, amr::FieldRefiner{}, damn::amr::LevelRegion::LevelGhosts,
                amr::kept_vector_pairs("EM_B_old", key, "EM_B"), alpha_(ln, t));
        touch_up_level_ghosts_(ln, key);
    }
    check_set_(ln, key, "magnetic ghosts");
}

template<auto opts>
void Simulator<opts>::fill_electric_ghosts_(std::size_t const ln, std::string const& key,
                                            double const t)
{
    copy_(ln, damn::amr::fill_ghosts, key);
    if (ln > 0)
        refine_(ln, amr::FieldRefiner{}, damn::amr::LevelRegion::LevelGhosts,
                amr::kept_vector_pairs("EM_E_old", key, "EM_E"), alpha_(ln, t));
    check_set_(ln, key, "electric ghosts");
}

template<auto opts>
void Simulator<opts>::check_set_(std::size_t const ln, std::string const& key,
                                 std::string const& where)
{
    if constexpr (!DAMN_DEBUG)
        return;

    auto& ld          = data_(ln);
    auto const domain = damn::amr::level_domain(hierarchy_, static_cast<int>(ln));
    Leases leases; // tiled fields: checked on untiled temporaries
    leases.add(ld, {key + "_x", key + "_y", key + "_z"}, amr::Lease::read);
    for (std::size_t i = 0; i < damn::amr::size(ld); ++i)
        for (auto const* c : {"_x", "_y", "_z"})
            damn::amr::with_resource(ld.data[i], key + c, [&](auto const& r) {
                auto const& cells = ld.patches[i].box;
                auto const at
                    = simulator::detail::first_unset(r, cells, domain, hierarchy_.periodic);
                if (!at)
                    return;
                std::stringstream ss;
                ss << "plamn Simulator (" << where << "): " << key << c << " unset at " << *at
                   << " of level " << ln << " patch " << cells;
                throw std::runtime_error(ss.str());
            });
}

template<auto opts>
void Simulator<opts>::touch_up_level_ghosts_(std::size_t const ln, std::string const& key)
{
    auto& l       = levels_[ln];
    auto const dx = damn::amr::mesh_size(hierarchy_, static_cast<int>(ln));
    models_.visit(ln, [&](auto& model) {
        using Patch_t  = std::remove_cvref_t<decltype(model)>::Patch_t;
        using Resource = amr::FieldResource<typename resource_field_<Patch_t>::type,
                                            Patch_t::field_ghost_width>;
        auto& ld       = model.levels[static_cast<int>(ln)];
        // tiled B: touched up on its untiled temporaries, written back into the tiles
        static thread_local std::vector<std::string> names; // read by add only
        names.assign({key + "_x", key + "_y", key + "_z"});
        Leases leases;
        leases.add(ld, names, amr::Lease::fill);
        for (std::size_t i = 0; i < damn::amr::size(ld); ++i)
        {
            std::array<Resource, 3> B{};
            std::size_t nb = 0;
            for (auto const* c : {"_x", "_y", "_z"})
                damn::amr::with_resource(ld.data[i], key + c, [&](auto& r) {
                    if constexpr (std::is_same_v<std::decay_t<decltype(r)>, Resource>)
                        B[nb++] = r;
                });
            for (auto const& region : l.level_ghost_cells[i])
                amr::divergence_free_touch_up(std::array{&B[0], &B[1], &B[2]}, ld.patches[i].box,
                                              region, dx);
        }
    });
}

template<auto opts>
void Simulator<opts>::mhd_physical_boundaries_(std::size_t const ln, auto&& fn)
{
    auto constexpr gw = solver::MHDModel<opts>::Patch_t::field_ghost_width;
    auto& ld          = models_.mhd.levels[static_cast<int>(ln)];
    for (std::size_t i = 0; i < damn::amr::size(ld); ++i)
    {
        auto regions = damn::amr::physical_boundary_regions(hierarchy_, static_cast<int>(ln),
                                                            ld.patches[i].box, gw);
        if (!regions.empty())
            fn(*ld.data[i], std::move(regions));
    }
}

// patch ghosts, level ghosts (from the coarser level at t), then the physical boundaries'
template<auto opts>
void Simulator<opts>::fill_mhd_state_ghosts_(std::size_t const ln, std::size_t const s,
                                             double const t)
{
    copy_(ln, damn::amr::fill_ghosts, mhd_state_key(s));
    if (ln > 0)
        refine_mhd_level_ghosts_(ln, s, t);
    mhd_physical_boundaries_(ln, [&](auto& patch, auto regions) {
        auto& u = patch.states[s];
        core::fill_physical_boundaries(models_.mhd.boundaries, std::move(regions), patch.layout,
                                       u.rho, u.Etot, u.rhoV, u.B);
    });
}

template<auto opts>
void Simulator<opts>::fill_mhd_electric_ghosts_(std::size_t const ln, double)
{
    copy_(ln, damn::amr::fill_ghosts, "MHD_E");
    mhd_physical_boundaries_(ln, [&](auto& patch, auto regions) {
        core::fill_physical_boundaries(models_.mhd.boundaries, std::move(regions), patch.layout,
                                       patch.E);
    });
}

template<auto opts>
void Simulator<opts>::refine_mhd_level_ghosts_(std::size_t const ln, std::size_t const s,
                                               double const t)
{
    using Patch_t    = solver::MHDModel<opts>::Patch_t;
    auto const name  = [&](std::string const& q) { return Patch_t::state_name(q, s); };
    auto const alpha = alpha_(ln, t);
    // per state, the same every step: built once per thread (a deque: they stay where they are)
    static thread_local std::deque<std::vector<damn::amr::LevelPair>> by_state;
    while (by_state.size() <= s)
    {
        auto const k = by_state.size();
        auto& pairs  = by_state.emplace_back(std::vector<damn::amr::LevelPair>{
            {"rho_old", Patch_t::state_name("rho", k), "rho"},
            {"Etot_old", Patch_t::state_name("Etot", k), "Etot"}});
        for (auto const& p : amr::vector_pairs("rhoV_old", Patch_t::state_name("rhoV", k), "rhoV"))
            pairs.push_back(p);
    }
    refine_(ln, amr::FieldRefiner{}, damn::amr::LevelRegion::LevelGhosts, by_state[s], alpha);
    refine_(ln, amr::FieldRefiner{}, damn::amr::LevelRegion::LevelGhosts,
            amr::kept_vector_pairs("B_old", name("B"), "B"), alpha);
    touch_up_level_ghosts_(ln, name("B"));
}

} // namespace plamn

#endif // PLAMN_SIMULATOR_SIMULATOR_GHOSTS_HPP
