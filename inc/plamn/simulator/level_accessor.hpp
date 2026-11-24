#ifndef PLAMN_SIMULATOR_LEVEL_ACCESSOR_HPP
#define PLAMN_SIMULATOR_LEVEL_ACCESSOR_HPP

// a level's patches as the level-wide particle movers take them (ParticlesSynchronisedMoverFor_t's
// parallel and flat ones, their MultiBoris): per patch its id (the key of the boxings), layout,
// and args (ions, electromag), electromag the HybridPatch member pushed with (e.g. emAvg)

#include "damn/core/def.hpp"

#include <cstddef>
#include <string>
#include <tuple>

namespace plamn::simulator
{
template<typename LevelData, typename Patch, auto electromag>
struct LevelAccessor
{
    using Layout_t = Patch::Layout_t;
    struct Model_t
    {
        using particles_type  = Patch::Particles_t;
        using electromag_type = Patch::Electromag_t;
    };

    struct View
    {
        std::string id;
        Layout_t layout;
        std::tuple<typename Patch::Ions_t&, typename Patch::Electromag_t&> args;

        auto& patchID() const { return id; }
    };

    MKN_KUL_FN_SIG_RET std::size_t size() const { return data.data.size(); }

    View operator[](std::size_t const i)
    {
        auto& patch = *data.data[i];
        return View{id(i), patch.layout, {patch.ions, patch.*electromag}};
    }

    MKN_KUL_FN_SIG_RET std::string id(std::size_t const i) const
    {
        return std::to_string(data.patches[i].id);
    }

    LevelData& data;
};

} // namespace plamn::simulator

#endif // PLAMN_SIMULATOR_LEVEL_ACCESSOR_HPP
