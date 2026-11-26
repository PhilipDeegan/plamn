#ifndef PLAMN_SIMULATOR_COUPLING_HPP
#define PLAMN_SIMULATOR_COUPLING_HPP

// what crosses the model boundary from an MHD level to the hybrid level over it:
// doc/simulator/coupling.md. One ion population for now: MHD's rho is that species' mass density

#include "damn/amr/level_map.hpp"
#include "damn/core/math/random.hpp"
#include "plamn/core/data/grid/patch_layout.hpp"
#include "plamn/core/data/ions/particle_initializers/maxwellian_particle_initializer.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"
#include "mkn/kul/random.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#include <format>

namespace damn::core
{
template<typename Type, std::size_t dim>
struct Box;
} // namespace damn::core
namespace plamn::simulator
{
template<auto opts>
struct HybridPatch;
template<auto opts>
struct MHDPatch;
} // namespace plamn::simulator

namespace plamn::simulator::coupling
{
template<typename>
inline constexpr bool is_mhd_patch_v = false;
template<auto opts>
inline constexpr bool is_mhd_patch_v<MHDPatch<opts>> = true;

// a level's patch data type is MHD's (LevelData holds patches by unique_ptr)
template<typename LevelData>
inline constexpr bool is_mhd_level_v
    = is_mhd_patch_v<std::decay_t<decltype(*std::declval<LevelData&>().data[0])>>;

// what the MHD level under a hybrid one holds a hybrid name as
inline std::string mhd_name(std::string const& hybrid)
{
    std::array<std::pair<std::string_view, std::string_view>, 4> constexpr prefixes{{
        {"EM_B_old_", "B_old_"}, // the old ones first: EM_B_ prefixes them too
        {"EM_E_old_", "E_old_"},
        {"EM_B_", "B_"},
        {"EM_E_", "E_"},
    }};
    for (auto const& [h, m] : prefixes)
        if (hybrid.starts_with(h))
            return std::string{m} + hybrid.substr(h.size());
    return hybrid;
}

inline std::vector<damn::amr::LevelPair> to_mhd_coarse(std::vector<damn::amr::LevelPair> pairs)
{
    for (auto& p : pairs)
    {
        p.coarse = mhd_name(p.coarse);
        if (!p.coarse_new.empty())
            p.coarse_new = mhd_name(p.coarse_new);
    }
    return pairs;
}

inline std::array<std::string, 5> const prim_names{"rho", "V_x", "V_y", "V_z", "P"};
inline std::string scratch_name(std::string const& prim)
{
    return "mhd_" + prim;
}

inline std::vector<damn::amr::LevelPair> const& prim_pairs()
{
    static auto const pairs = [] { // refined every step: built once (read only after)
        std::vector<damn::amr::LevelPair> out;
        out.reserve(prim_names.size());
        for (auto const& n : prim_names)
            out.push_back({n, scratch_name(n), ""});
        return out;
    }();
    return pairs;
}

template<auto opts>
void make_prim_scratch(HybridPatch<opts>& patch)
{
    for (auto const& n : prim_names)
        patch.scratch_grid(scratch_name(n), core::HybridQuantity::Scalar::AllDual);
}

// what a spawn's random streams are told apart by (spawn_seed): a coarse step's new level ghosts
//  by its end time's bits
struct Salt
{
    static constexpr std::uint64_t domain = 0, ghosts_at_init = 1;
    static std::uint64_t ghosts_new(double const t) { return std::bit_cast<std::uint64_t>(t); }
};

// the streams of a spawn: the population's seed (0x5eed if none), told apart from the
//  root loader's and from other spawns of the same cells by salt (e.g. a level ghost's coarse
//  step)
inline std::uint64_t spawn_seed(auto const& pop, std::uint64_t const salt)
{
    auto const& info = pop.particleInitializerInfo();
    std::optional<std::size_t> seed;
    if (info.contains("init") and info["init"].contains("seed"))
        seed = info["init"]["seed"].template to<std::optional<std::size_t>>();
    return damn::core::cell_key(seed ? *seed : std::size_t{0x5eed}, salt, 0, std::array<int, 1>{0});
}

// each population's particles (out(pop): which of its arrays) in cells (AMR boxes inside the
//  patch's scratch ghost box) from the MHD primitives on the patch's scratch grids; Te: the
//  isothermal electrons', level: the patch's level
template<auto opts>
void spawn(HybridPatch<opts>& patch, std::vector<damn::core::Box<int, opts.dimension>> const& cells,
           auto&& out, double const Te, std::uint64_t const salt)
{
    if (patch.populations.size() != 1)
        throw std::runtime_error("plamn coupling: one ion population for now");

    auto const& layout = patch.layout;
    auto const at      = [&](std::string const& prim, auto const& local) {
        return (*patch.scratch.at(scratch_name(prim)))(local); // the grid's field view
    };
    for (auto& pop : patch.ions)
    {
        auto const& info = pop.particleInitializerInfo();
        auto const nppc  = static_cast<std::uint32_t>(info["nbr_part_per_cell"].template to<int>());
        auto const cut_off = info.contains("density_cut_off")
                                 ? info["density_cut_off"].template to<double>()
                                 : 1e-16;
        auto const m = pop.mass(), q = pop.charge();
        auto const seed    = spawn_seed(pop, salt);
        auto const pop_key = mkn::kul::fnv1a64(pop.name());
        auto& particles    = out(pop);
        for (auto const& box : cells)
            core::for_each_amr_local(layout, box, [&](auto const& amr, auto const& local) {
                auto const n = at("rho", local) / m;
                if (!(n >= cut_off))
                    return;
                auto const Pi  = std::max(at("P", local) - Te * q * n, 0.);
                auto const vth = std::sqrt(Pi / (n * m));
                core::load_maxwellian_cell(
                    particles, amr.toArray(), n,
                    {at("V_x", local), at("V_y", local), at("V_z", local)}, {vth, vth, vth}, nppc,
                    damn::core::cell_key(seed, pop_key, layout.level, amr.toArray()));
            });
    }
}

} // namespace plamn::simulator::coupling

#endif // PLAMN_SIMULATOR_COUPLING_HPP
