#ifndef PLAMN_SOLVER_MODELS_HPP
#define PLAMN_SOLVER_MODELS_HPP

// the models of a simulation and which levels each owns: levels [0, mhd_levels) MHD, the rest
//  hybrid (PHARE's simulation/AMR/max_mhd_level). A disabled model is an empty specialization
//  (model_def.hpp); with one model enabled, every level is that model's and visit is a direct
//  call, without a run time test

#include "damn/core/def.hpp"
#include "plamn/solver/hybrid/model.hpp"
#include "plamn/solver/model_def.hpp"
#include "plamn/solver/mhd/model.hpp"
#include "dict.hpp"
#include "plamn/initializer/data_provider.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include <format>
#include <ranges>

namespace plamn::solver
{
template<typename Model>
inline constexpr bool is_mhd_model_v = false;

template<auto opts>
inline constexpr bool is_mhd_model_v<MHDModel<opts, true>> = true;

template<typename Model>
inline constexpr bool is_hybrid_model_v = false;

template<auto opts>
inline constexpr bool is_hybrid_model_v<HybridModel<opts, true>> = true;

template<auto opts>
struct Models
{
    static constexpr bool hybrid_on = has_hybrid_v<opts>;
    static constexpr bool mhd_on    = has_mhd_v<opts>;
    static_assert(hybrid_on or mhd_on, "plamn: a simulation needs a model");

    HybridModel<opts> hybrid{};
    MHDModel<opts> mhd{};
    int mhd_levels = 0; // levels below it are MHD

    MKN_KUL_FN_SIG_RET bool is_mhd(std::size_t const ln) const
    {
        if constexpr (!mhd_on)
            return false;
        else if constexpr (!hybrid_on)
            return true;
        else
            return static_cast<int>(ln) < mhd_levels;
    }

    // fn(the model owning level ln)
    decltype(auto) visit(std::size_t const ln, auto&& fn)
    {
        if constexpr (!mhd_on)
            return fn(hybrid);
        else if constexpr (!hybrid_on)
            return fn(mhd);
        else
            return is_mhd(ln) ? fn(mhd) : fn(hybrid);
    }

    // fn(model) for every enabled model
    void for_each(auto&& fn)
    {
        if constexpr (mhd_on)
            fn(mhd);
        if constexpr (hybrid_on)
            fn(hybrid);
    }

    // the levels from n on gone
    void truncate(std::size_t const n)
    {
        for_each([&](auto& model) { model.truncate(n); });
    }

    // the levels each model owns, from the input's simulation dict (max_levels: its
    //  AMR/max_nbr_levels): PHARE's max_mhd_level, by default (as pyphare) every level if MHD is
    //  the only model, none otherwise. Its models (pyphare's model_options), if given: the
    //  models the levels have, each enabled by the build
    void set_levels(initializer::Dict const& sim, int const max_levels)
    {
        std::vector<std::string> names;
        if (sim.contains("models"))
            names = sim["models"].template to<std::vector<std::string>>();
        auto const named = [&](std::string const& name) {
            return std::ranges::find(names, name) != names.end();
        };
        for (auto const& name : names)
            if (name != "HybridModel" and name != "MHDModel")
                throw std::runtime_error("plamn: no model " + name);
        if ((named("HybridModel") and !hybrid_on) or (named("MHDModel") and !mhd_on))
            throw std::runtime_error("plamn: the build does not enable the models of the input");

        bool const mhd_alone = names.empty() ? !hybrid_on : !named("HybridModel");
        mhd_levels = cppdict::get_value(sim, "AMR/max_mhd_level", mhd_alone ? max_levels : 0);
        if (mhd_levels < 0 or mhd_levels > max_levels)
            throw std::runtime_error("plamn: max_mhd_level " + std::to_string(mhd_levels)
                                     + " outside [0, max_nbr_levels " + std::to_string(max_levels)
                                     + "]");
        if (!mhd_on and mhd_levels > 0)
            throw std::runtime_error("plamn: MHD levels (max_mhd_level "
                                     + std::to_string(mhd_levels)
                                     + ") in a build without the MHD model");
        if (!hybrid_on and mhd_levels < max_levels)
            throw std::runtime_error("plamn: hybrid levels (max_mhd_level "
                                     + std::to_string(mhd_levels) + " < max_nbr_levels "
                                     + std::to_string(max_levels)
                                     + ") in a build without the hybrid model");
        if (!names.empty()
            and ((mhd_levels > 0) != named("MHDModel")
                 or (mhd_levels < max_levels) != named("HybridModel")))
            throw std::runtime_error("plamn: models and max_mhd_level disagree");
    }
};

} // namespace plamn::solver

#endif // PLAMN_SOLVER_MODELS_HPP
