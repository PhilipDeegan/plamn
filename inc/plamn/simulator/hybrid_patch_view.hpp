#ifndef PLAMN_SIMULATOR_HYBRID_PATCH_VIEW_HPP
#define PLAMN_SIMULATOR_HYBRID_PATCH_VIEW_HPP

// a hybrid patch's state as views, apart from the grids that hold it (HybridPatch): what code
// outside the solver reads (diagnostics, python); valid while the patch is there

#include "plamn/core/data/electromag/electromag.hpp"
#include "damn/core/def.hpp"

#include <string>
#include <vector>

namespace plamn::simulator
{
template<typename VecField>
struct HybridPatchView
{
    using vecfield_type = VecField;
    using field_type    = VecField::field_type;

    struct Population
    {
        std::string name;
        field_type density, charge_density;
        VecField flux;
    };

    core::basic::Electromag<VecField> em;
    field_type mass_density, charge_density;
    VecField bulk_velocity, charge_flux;
    std::vector<Population> populations;
};

template<typename Patch>
MKN_KUL_FN_SIG_RET auto view(Patch& patch)
{
    using View = HybridPatchView<typename Patch::VecField_t>;
    View v{{patch.em.E, patch.em.B}, patch.ions.massDensity(), patch.ions.chargeDensity(),
           patch.ions.velocity(),    patch.ions.chargeFlux(),  {}};
    for (auto& pop : patch.ions)
        v.populations.push_back(
            {pop.name(), pop.particleDensity(), pop.chargeDensity(), pop.flux()});
    return v;
}
} // namespace plamn::simulator

#endif // PLAMN_SIMULATOR_HYBRID_PATCH_VIEW_HPP
