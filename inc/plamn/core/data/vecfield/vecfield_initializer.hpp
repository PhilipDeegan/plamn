#include <initializer_list>
#ifndef VECFIELD_INITIALIZER_HPP
#define VECFIELD_INITIALIZER_HPP

#include "plamn/initializer/data_provider.hpp"
#include "plamn/core/data/vecfield/vecfield_component.hpp"
#include "plamn/core/data/field/initializers/field_user_initializer.hpp"

#include <string>
#include <cstddef>
#include <stdexcept>

namespace plamn
{
namespace core
{
    template<std::size_t dimension>
    class VecFieldInitializer
    {
        static bool validate_dict(plamn::initializer::Dict const& dict);

        VecFieldInitializer(plamn::initializer::Dict const& dict, bool)
            : x_{dict["x_component"].template to<plamn::initializer::InitFunction<dimension>>()}
            , y_{dict["y_component"].template to<plamn::initializer::InitFunction<dimension>>()}
            , z_{dict["z_component"].template to<plamn::initializer::InitFunction<dimension>>()}
        {
        }

    public:
        VecFieldInitializer() = default;

        VecFieldInitializer(plamn::initializer::Dict const& dict)
            : VecFieldInitializer{dict, validate_dict(dict)}
        {
        }


        template<typename VecField, typename Layout>
        void initialize(VecField& v, Layout const& layout)
        {
            static_assert(Layout::dimension == VecField::dimension,
                          "dimension mismatch between vecfield and layout");

            FieldUserFunctionInitializer::initialize(v.getComponent(Component::X), layout, x_);
            FieldUserFunctionInitializer::initialize(v.getComponent(Component::Y), layout, y_);
            FieldUserFunctionInitializer::initialize(v.getComponent(Component::Z), layout, z_);
        }

    private:
        plamn::initializer::InitFunction<dimension> x_;
        plamn::initializer::InitFunction<dimension> y_;
        plamn::initializer::InitFunction<dimension> z_;
    };

} // namespace core

} // namespace plamn


namespace plamn::core
{
template<std::size_t dim>
bool VecFieldInitializer<dim>::validate_dict(plamn::initializer::Dict const& dict)
{
    for (auto const* key : {"x_component", "y_component", "z_component"})
        if (!dict.contains(key))
            throw std::runtime_error(std::string{"VecFieldInitializer: missing key "} + key);
    return true;
}

} // namespace plamn::core

#endif // VECFIELD_INITIALIZER_HPP
