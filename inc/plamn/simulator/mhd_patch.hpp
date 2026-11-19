#ifndef PLAMN_SIMULATOR_MHD_PATCH_HPP
#define PLAMN_SIMULATOR_MHD_PATCH_HPP

// MHDPatch: one patch of an MHD level, what the time integration schemes use
//  (core/numerics/time_integrator: the patch members they name)
//   states   the scheme's: states[0] the advanced one, the others its stages (n_states, the
//            scheme's: a run time choice)
//   old      states[0] at the start of the level's step (prepare_step): a finer level's ghosts
//            interpolate in time between it and states[0] (PHARE's MHDMessenger rhoOld_...),
//            the reflux redoes the step from it (PHARE's SolverMHD stateOld_)
//   E        the constrained transport electric field of the last flux computation
//   fluxes   per face direction, of the last flux computation; ct its upwind constrained
//            transport inputs; butcher, butcherE the schemes' weighted sums of them (the step's
//            fluxes and E, what a reflux replaces under a finer level)
//   J, E_diss  resistive MHD's: J = curl B, the dissipative E (eta J) added to the constrained
//            transport E and to the face fluxes (PHARE's DissipativeElectricField)
//   flux_sum, flux_sum_E  the step fluxes and E summed over the level's substeps in its coarser
//            level's step (PHARE's SolverMHD fluxSum_, fluxSumE_), what its reflux coarsens
//   storage: grids, named as their views. The states not filled (NaN in debug builds): the
//            simulator writes them before reading them. The others 0, as PHARE's (its fields
//            start zeroed): E's component without an equation (E_x in 1D) stays 0, and a
//            direction a lower dimension does not have has 0 fluxes
//
// not movable: the vector fields are views over their own grids; LevelData keeps patches behind
//  unique_ptr
//
// for_each_resource names what damn's schedules may move (PHARE's MHD names for the advanced
//  state): rho, rhoV_x..., B_x..., Etot, V_x..., P; a stage s > 0: rho_s<s>, rhoV_s<s>_x...;
//  the old state: rho_old, rhoV_old_x..., B_old_x..., Etot_old; E_x...; per direction d < dim
//  the refluxed step fluxes and their sums: F_rho_butcher_<d>, F_rhoV_butcher_<d>_x...,
//  F_Etot_butcher_<d>, F_rho_sum_<d>...; butcherE_x..., E_sum_x... (the last fluxes and the
//  constrained transport inputs never move)

#include "damn/core/def.hpp"
#include "plamn/amr/resources.hpp"
#include "plamn/core/models/mhd_state.hpp"
#include "plamn/core/models/quantities/mhd_quantities.hpp"
#include "plamn/plamn_core.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
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

    // states[s]: a scheme's states, made at the patch's making
    struct States
    {
        std::vector<std::unique_ptr<State_t>> data;

        NO_DISCARD State_t& operator[](std::size_t const s) { return *data[s]; }
        NO_DISCARD State_t const& operator[](std::size_t const s) const { return *data[s]; }
        NO_DISCARD std::size_t size() const { return data.size(); }
    };

    // a state's names: PHARE's for the advanced one, _s<s> after the others'
    NO_DISCARD static std::string state_name(std::string const& name, std::size_t const s)
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
                vector(f.F_B);
                fn(f.F_Etot);
            }
        for (auto& c : ct)
        {
            vector(c.vt);
            for (auto* g : {&c.aL, &c.aR, &c.dL, &c.dR})
                fn(*g);
        }
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
    // resistive MHD (MHDSteps: eta > 0): J = curl B, the dissipative E = eta J
    VecField J{"J", layout, V::J, 0.}, E_diss{"E_diss", layout, V::E, 0.};

    // a face's fluxes' grids: F_rho, F_rhoV (x y z), F_B (x y z), F_Etot
    static std::vector<Grid_t*> flux_grids(Fluxes_t& f)
    {
        std::vector<Grid_t*> grids{std::addressof(f.F_rho)}; // Grid overloads operator&
        for (auto* v : {&f.F_rhoV, &f.F_B})
            for (auto& g : v->grids)
                grids.push_back(std::addressof(g));
        grids.push_back(std::addressof(f.F_Etot));
        return grids;
    }
    static void for_each_flux_grid(Fluxes_t& f, auto&& fn)
    {
        for (auto* g : flux_grids(f))
            fn(*g);
    }

private:
    Fluxes_t fluxes_(std::string const& d, S const sf, V const vf) const
    {
        return Fluxes_t{
            Grid_t{"F_rho_" + d, layout, sf, 0.}, VecField{"F_rhoV_" + d, layout, vf, 0.},
            VecField{"F_B_" + d, layout, vf, 0.}, Grid_t{"F_Etot_" + d, layout, sf, 0.}};
    }
    CT_t ct_(std::string const& d, S const sf, V const vf) const
    {
        return CT_t{VecField{"ct_vt_" + d, layout, vf, 0.}, Grid_t{"ct_aL_" + d, layout, sf, 0.},
                    Grid_t{"ct_aR_" + d, layout, sf, 0.}, Grid_t{"ct_dL_" + d, layout, sf, 0.},
                    Grid_t{"ct_dR_" + d, layout, sf, 0.}};
    }
};

template<auto opts>
void for_each_resource(MHDPatch<opts>& patch, auto&& fn)
{
    using Patch   = MHDPatch<opts>;
    auto const gw = Patch::field_ghost_width;

    auto const scalar = [&](auto& grid) {
        auto r = amr::field_resource<gw>(grid);
        fn(grid.name(), r);
    };
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
    for (std::size_t d = 0; d < Patch::dimension; ++d)
        for (auto* f : {&patch.butcher[d], &patch.flux_sum[d]})
        {
            scalar(f->F_rho);
            vector(f->F_rhoV);
            scalar(f->F_Etot);
        }
    vector(patch.butcherE);
    vector(patch.flux_sum_E);
}

// LevelData holds patches by unique_ptr
template<auto opts>
void for_each_resource(std::unique_ptr<MHDPatch<opts>>& patch, auto&& fn)
{
    for_each_resource(*patch, fn);
}

} // namespace plamn::simulator

#endif // PLAMN_SIMULATOR_MHD_PATCH_HPP
