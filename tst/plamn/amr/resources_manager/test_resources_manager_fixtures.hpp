#ifndef PLAMN_TEST_AMR_RESOURCES_MANAGER_FIXTURES_HPP
#define PLAMN_TEST_AMR_RESOURCES_MANAGER_FIXTURES_HPP

#include "plamn/core/def.hpp"

#include <string>
#include <type_traits>

namespace plamn::test
{


template<typename Ions_t, typename EM>
struct HybridPatch
{
    using Layout_t = Ions_t::layout_type;

    HybridPatch(auto&&... args)
        : model{args...}
    {
    }

    struct Model
    {
        using particles_type = Ions_t::particles_type;
        using electromag_type     = EM::Super;

        Model(auto&&... args)
            : state{args...}
        {
        }

        struct State
        {
            Layout_t layout;
            Ions_t ions;
            EM em{layout};
            electromag_type electromag = *em;
        };

        State state;
    };
    using Model_t = Model;

    auto& patchID() const { return id; }

    std::string id = "patch_id";
    Model model;
    Layout_t& layout = model.state.layout;
};


} // namespace plamn::test


namespace plamn::test::detail
{

template<typename ArgsGetter>
struct ArgsResolver
{
    using type = std::decay_t<std::invoke_result_t<ArgsGetter const&, int>>;
};

template<typename ArgsGetter>
using resolve_args_t = ArgsResolver<ArgsGetter>::type;

auto constexpr static default_no_args = [](int) { return std::tuple{}; };

} // namespace plamn::test::detail


namespace plamn::test
{

template<typename Patches, typename ArgsGetter>
struct UsableModelAccessor
{
    using Patch_t      = Patches::value_type;
    using Model_t      = Patch_t::Model_t;
    using Layout_t = Patch_t::Layout_t;
    using State_t      = std::decay_t<decltype(std::declval<Patch_t>().model.state)>;
    using Args_t       = detail::resolve_args_t<ArgsGetter>;

    explicit UsableModelAccessor(Patches& p, ArgsGetter getter = detail::default_no_args)
        : patches{p}
        , getter{getter}
    {
    }

    auto size() const { return patches.size(); }

    auto operator[](std::size_t i)
    {
        return PatchView{patches[i].patchID(), patches[i].layout, getter(i)};
    }

    struct PatchView
    {
        auto& patchID() const { return id; }

        std::string id;
        Layout_t layout;
        Args_t args;
    };


    Patches& patches;
    ArgsGetter getter;
};


template<typename Patches, typename ArgsGetter>
UsableModelAccessor(Patches, ArgsGetter) -> UsableModelAccessor<Patches, ArgsGetter>;


template<typename ArgsGetter = decltype(detail::default_no_args)>
auto make_model_level_accessor(auto& patches, ArgsGetter getter = detail::default_no_args)
{
    return UsableModelAccessor{patches, getter};
}


} // namespace plamn::test


#endif // PLAMN_TEST_AMR_RESOURCES_MANAGER_FIXTURES_HPP
