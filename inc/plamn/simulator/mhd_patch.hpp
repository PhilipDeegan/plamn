#ifndef PLAMN_SIMULATOR_MHD_PATCH_HPP
#define PLAMN_SIMULATOR_MHD_PATCH_HPP

// one patch of an MHD level, what the time integration schemes use; its members and the
// resources damn's schedules may move: doc/simulator/mhd_patch.md
// not movable: the vector fields are views over their own grids

#include "damn/core/def.hpp"
#include "plamn/amr/untiled.hpp"
#include "plamn/core/data/field/level_fields.hpp"
#include "plamn/simulator/coupling_storage.hpp"
#include "plamn/core/models/mhd_state.hpp"
#include "plamn/core/models/quantities/mhd_quantities.hpp"
#include "plamn/plamn_core.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>
#include <utility>

namespace plamn::simulator
{
template<auto opts>
struct MHDPatch
{
    using Types      = core::PLAMN_Types<opts>::MHD;
    using Layout_t   = Types::Layout_t;
    using Grid_t     = Types::Grid_t;
    using field_type = Grid_t; // what diagnostics read (diagnostic/quantities.hpp)
    using S          = core::MHDQuantity::Scalar;
    using V          = core::MHDQuantity::Vector;

    auto static constexpr dimension         = Layout_t::dimension;
    auto static constexpr field_ghost_width = Layout_t::ghost_width;

    // a vector field view over its own grids (name_x, name_y, name_z)
    struct VecField : Types::VecField_t
    {
        using Super = Types::VecField_t;

        VecField(std::string const& name, Layout_t const& layout, V const qty,
                 std::optional<double> const value = std::nullopt)
            : Super{name, qty}
            , grids{grid_(0, layout, value), grid_(1, layout, value), grid_(2, layout, value)}
        {
            Super::setBuffer(&grids);
        }
        VecField(VecField const&)            = delete;
        VecField(VecField&&)                 = delete;
        VecField& operator=(VecField const&) = delete;
        VecField& operator=(VecField&&)      = delete;

        std::array<Grid_t, 3> grids;

    private:
        Grid_t grid_(std::size_t const i, Layout_t const& layout,
                     std::optional<double> const value) const
        {
            return Grid_t{Super::componentNames()[i], layout,
                          core::MHDQuantity::componentsQuantities(Super::physicalQuantity())[i],
                          value};
        }
    };

    using State_t  = core::MHDState<Grid_t, VecField>;
    using Fluxes_t = core::MHDFaceFluxes<Grid_t, VecField>;
    using CT_t     = core::MHDFaceCT<Grid_t, VecField>;

    // Hall MHD (the build's opts.Hall): the faces' current and rho averages (jt, rhot) the
    //  constrained transport reads, per face direction; in no other build
    auto static constexpr hall = opts.Hall;
    struct HallCT
    {
        VecField jt;
        Grid_t rhot;
    };
    struct NoHall
    {
    };

    // states[s]: a scheme's states, made at the patch's making
    struct States
    {
        std::vector<std::unique_ptr<State_t>> data;

        MKN_KUL_FN_SIG_RET State_t& operator[](std::size_t const s) { return *data[s]; }
        MKN_KUL_FN_SIG_RET State_t const& operator[](std::size_t const s) const { return *data[s]; }
        MKN_KUL_FN_SIG_RET std::size_t size() const { return data.size(); }
    };

    // a state's names: PHARE's for the advanced one, _s<s> after the others'
    MKN_KUL_FN_SIG_RET static std::string state_name(std::string const& name, std::size_t const s)
    {
        return s == 0 ? name : name + "_s" + std::to_string(s);
    }

    MHDPatch(Layout_t const& layout_, std::size_t const n_states)
        : layout{layout_}
    {
        for (std::size_t s = 0; s < n_states; ++s)
            states.data.emplace_back(new State_t{Grid_t{state_name("rho", s), layout, S::rho},
                                                 VecField{state_name("rhoV", s), layout, V::rhoV},
                                                 VecField{state_name("B", s), layout, V::B},
                                                 Grid_t{state_name("Etot", s), layout, S::Etot},
                                                 VecField{state_name("V", s), layout, V::V},
                                                 Grid_t{state_name("P", s), layout, S::P}});
        if constexpr (DAMN_DEBUG)
        {
            auto const nan
                = [](auto& grid) { grid.fill(std::numeric_limits<double>::quiet_NaN()); };
            for (auto& u : states.data)
                for_each_state_grid(*u, nan);
            for_each_state_grid(old, nan);
        }
    }

    MHDPatch(MHDPatch const&)            = delete;
    MHDPatch(MHDPatch&&)                 = delete;
    MHDPatch& operator=(MHDPatch const&) = delete;
    MHDPatch& operator=(MHDPatch&&)      = delete;

    // fn(grid) for every grid of a state
    static void for_each_state_grid(State_t& u, auto&& fn)
    {
        fn(u.rho);
        for (auto* v : {&u.rhoV, &u.B})
            for (auto& g : v->grids)
                fn(g);
        fn(u.Etot);
        for (auto& g : u.V.grids)
            fn(g);
        fn(u.P);
    }

    // fn(grid) for every grid of the patch
    void for_each_grid(auto&& fn)
    {
        auto const vector = [&](VecField& v) {
            for (auto& g : v.grids)
                fn(g);
        };
        for (auto& u : states.data)
            for_each_state_grid(*u, fn);
        for_each_state_grid(old, fn);
        vector(E);
        vector(butcherE);
        vector(flux_sum_E);
        vector(J);
        vector(E_diss);
        for (auto* faces : {&fluxes, &butcher, &flux_sum})
            for (auto& f : *faces)
            {
                fn(f.F_rho);
                vector(f.F_rhoV);
                fn(f.F_Etot);
            }
        for (auto& c : ct)
        {
            vector(c.vt);
            for (auto* g : {&c.aL, &c.aR, &c.dL, &c.dR})
                fn(*g);
        }
        if constexpr (hall)
            for (auto& h : hall_ct)
            {
                vector(h.jt);
                fn(h.rhot);
            }
        for_each_coupling_vector(*this, vector);
    }

    Layout_t layout;
    States states;
    State_t old{Grid_t{"rho_old", layout, S::rho}, VecField{"rhoV_old", layout, V::rhoV},
                VecField{"B_old", layout, V::B},   Grid_t{"Etot_old", layout, S::Etot},
                VecField{"V_old", layout, V::V},   Grid_t{"P_old", layout, S::P}};
    VecField E{"E", layout, V::E, 0.}, butcherE{"butcherE", layout, V::E, 0.};
    std::array<Fluxes_t, 3> fluxes{fluxes_("x", S::ScalarFlux_x, V::VecFlux_x),
                                   fluxes_("y", S::ScalarFlux_y, V::VecFlux_y),
                                   fluxes_("z", S::ScalarFlux_z, V::VecFlux_z)};
    std::array<Fluxes_t, 3> butcher{fluxes_("butcher_x", S::ScalarFlux_x, V::VecFlux_x),
                                    fluxes_("butcher_y", S::ScalarFlux_y, V::VecFlux_y),
                                    fluxes_("butcher_z", S::ScalarFlux_z, V::VecFlux_z)};
    std::array<CT_t, 3> ct{ct_("x", S::ScalarFlux_x, V::VecFlux_x),
                           ct_("y", S::ScalarFlux_y, V::VecFlux_y),
                           ct_("z", S::ScalarFlux_z, V::VecFlux_z)};
    std::array<Fluxes_t, 3> flux_sum{fluxes_("sum_x", S::ScalarFlux_x, V::VecFlux_x),
                                     fluxes_("sum_y", S::ScalarFlux_y, V::VecFlux_y),
                                     fluxes_("sum_z", S::ScalarFlux_z, V::VecFlux_z)};
    VecField flux_sum_E{"E_sum", layout, V::E, 0.};
    // resistive MHD (MHDSteps: eta > 0): J = curl B, the dissipative E = eta J; Hall: J too
    VecField J{"J", layout, V::J, 0.}, E_diss{"E_diss", layout, V::E, 0.};
    std::conditional_t<hall, std::array<HallCT, 3>, NoHall> hall_ct = hall_ct_();

    // under a hybrid level: E at the step's start, its E ghosts interpolate in time between it
    //  and E (simulator/coupling.hpp)
    struct UnderHybrid
    {
        explicit UnderHybrid(Layout_t const& layout)
            : E_old{"E_old", layout, V::E, 0.}
        {
        }
        VecField E_old;
    };
    [[no_unique_address]] CouplingStorage<
        std::conditional_t<opts.hybrid_enabled, UnderHybrid, void>, void> coupling{};

    // a face's fluxes' grids: F_rho, F_rhoV (x y z), F_Etot
    static std::array<Grid_t*, 5> flux_grids(Fluxes_t& f)
    {
        auto& v = f.F_rhoV.grids; // Grid overloads operator&
        return {std::addressof(f.F_rho), std::addressof(v[0]), std::addressof(v[1]),
                std::addressof(v[2]), std::addressof(f.F_Etot)};
    }
    static void for_each_flux_grid(Fluxes_t& f, auto&& fn)
    {
        for (auto* g : flux_grids(f))
            fn(*g);
    }

    // tiled fields: their untiled temporaries while a whole patch operation (damn's schedules)
    //  works on them
    using Untiled_t = amr::UntiledFor_t<Grid_t>;
    Untiled_t untiled;
    // the grids for_each_field_grid moves, in its order (see grid_table)
    std::vector<Grid_t*> grid_table_;

private:
    Fluxes_t fluxes_(std::string const& d, S const sf, V const vf) const
    {
        return Fluxes_t{Grid_t{"F_rho_" + d, layout, sf, 0.},
                        VecField{"F_rhoV_" + d, layout, vf, 0.},
                        Grid_t{"F_Etot_" + d, layout, sf, 0.}};
    }
    auto hall_ct_() const
    {
        if constexpr (hall)
        {
            auto const one = [&](std::string const& d, S const sf, V const vf) {
                return HallCT{VecField{"ct_jt_" + d, layout, vf, 0.},
                              Grid_t{"ct_rhot_" + d, layout, sf, 0.}};
            };
            return std::array{one("x", S::ScalarFlux_x, V::VecFlux_x),
                              one("y", S::ScalarFlux_y, V::VecFlux_y),
                              one("z", S::ScalarFlux_z, V::VecFlux_z)};
        }
        else
            return NoHall{};
    }
    CT_t ct_(std::string const& d, S const sf, V const vf) const
    {
        return CT_t{VecField{"ct_vt_" + d, layout, vf, 0.}, Grid_t{"ct_aL_" + d, layout, sf, 0.},
                    Grid_t{"ct_aR_" + d, layout, sf, 0.}, Grid_t{"ct_dL_" + d, layout, sf, 0.},
                    Grid_t{"ct_dR_" + d, layout, sf, 0.}};
    }
};

// the coupling's vector fields, where the patch has them (coupling_storage.hpp)
template<auto opts>
    requires(not opts.hybrid_enabled)
void for_each_coupling_vector(MHDPatch<opts>&, auto&&)
{
}
template<auto opts>
    requires(opts.hybrid_enabled)
void for_each_coupling_vector(MHDPatch<opts>& patch, auto&& vector)
{
    if (patch.coupling.under)
        vector(patch.coupling.under->E_old);
}

// a patch of the finest MHD level, a hybrid level over it: its coupling storage
template<auto opts>
    requires(opts.hybrid_enabled)
void make_under_hybrid(MHDPatch<opts>& patch)
{
    patch.coupling.under.emplace(patch.layout);
}

// a level's step start (prepare_step): E into E_old on the patches under a hybrid level (the
//  finest MHD level's), one level copy; nothing without the hybrid model
template<auto opts>
    requires(not opts.hybrid_enabled)
void keep_E_old(auto&&)
{
}
template<auto opts>
    requires(opts.hybrid_enabled)
void keep_E_old(auto&& level)
{
    core::level_copy<opts.alloc_mode>(
        level,
        [](auto& patch, auto& f) {
            if (patch.coupling.under)
                f.add(patch.coupling.under->E_old);
        },
        [](auto& patch, auto& f) {
            if (patch.coupling.under)
                f.add(patch.E);
        });
}

// fn(name, grid) for each field grid damn's schedules may move, by its name
template<auto opts>
void for_each_field_grid(MHDPatch<opts>& patch, auto&& fn)
{
    auto const scalar = [&](auto& grid) { fn(grid.name(), grid); };
    auto const vector = [&](auto& v) {
        for (auto& g : v.grids)
            scalar(g);
    };
    for (auto& u : patch.states.data)
    {
        scalar(u->rho);
        vector(u->rhoV);
        vector(u->B);
        scalar(u->Etot);
        vector(u->V);
        scalar(u->P);
    }
    scalar(patch.old.rho);
    vector(patch.old.rhoV);
    vector(patch.old.B);
    scalar(patch.old.Etot);
    vector(patch.E);
    for (std::size_t d = 0; d < MHDPatch<opts>::dimension; ++d)
        for (auto* f : {&patch.butcher[d], &patch.flux_sum[d]})
        {
            scalar(f->F_rho);
            vector(f->F_rhoV);
            scalar(f->F_Etot);
        }
    vector(patch.butcherE);
    vector(patch.flux_sum_E);
    for_each_coupling_vector(patch, vector); // last: the other patches' indices unchanged
}

// MHD patches have no scratch grids: every one is fixed (grid_table)
template<auto opts>
void for_each_fixed_field_grid(MHDPatch<opts>& patch, auto&& fn)
{
    for_each_field_grid(patch, fn);
}

template<auto opts>
void for_each_resource(MHDPatch<opts>& patch, auto&& fn)
{
    using Patch   = MHDPatch<opts>;
    auto const gw = Patch::field_ghost_width;
    for_each_field_grid(patch, [&](std::string const& name, auto& grid) {
        amr::field_resource_for<gw>(patch.untiled, name, grid, fn);
    });
}

// LevelData holds patches by unique_ptr
template<auto opts>
void for_each_resource(std::unique_ptr<MHDPatch<opts>>& patch, auto&& fn)
{
    for_each_resource(*patch, fn);
}

} // namespace plamn::simulator

#endif // PLAMN_SIMULATOR_MHD_PATCH_HPP
