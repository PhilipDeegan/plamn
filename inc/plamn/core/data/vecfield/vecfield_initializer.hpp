#include "plamn/core/def.hpp"
#ifndef VECFIELD_INITIALIZER_HPP
#define VECFIELD_INITIALIZER_HPP

#include "damn/initializer/data_provider.hpp"
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
        static bool validate_dict(damn::initializer::DAMNDict const& dict);

        VecFieldInitializer(damn::initializer::DAMNDict const& dict, bool)
            : x_{dict["x_component"].template to<damn::initializer::InitFunction<dimension>>()}
            , y_{dict["y_component"].template to<damn::initializer::InitFunction<dimension>>()}
            , z_{dict["z_component"].template to<damn::initializer::InitFunction<dimension>>()}
        {
        }

    public:
        VecFieldInitializer() = default;

        VecFieldInitializer(damn::initializer::DAMNDict const& dict)
            : VecFieldInitializer{dict, validate_dict(dict)}
        {
        }


        template<typename VecField, typename GridLayout>
        void initialize(VecField& v, GridLayout const& layout)
        {
            static_assert(GridLayout::dimension == VecField::dimension,
                          "dimension mismatch between vecfield and gridlayout");

            FieldUserFunctionInitializer::initialize(v.getComponent(Component::X), layout, x_);
            FieldUserFunctionInitializer::initialize(v.getComponent(Component::Y), layout, y_);
            FieldUserFunctionInitializer::initialize(v.getComponent(Component::Z), layout, z_);
        }

    private:
        damn::initializer::InitFunction<dimension> x_;
        damn::initializer::InitFunction<dimension> y_;
        damn::initializer::InitFunction<dimension> z_;
    };

} // namespace core

} // namespace plamn


namespace plamn::core
{
template<std::size_t dim>
bool VecFieldInitializer<dim>::validate_dict(damn::initializer::DAMNDict const& dict)
{
    for (auto const* key : {"x_component", "y_component", "z_component"})
        if (!dict.contains(key))
            throw std::runtime_error(std::string{"VecFieldInitializer: missing key "} + key);
    return true;
}

} // namespace plamn::core

#endif // VECFIELD_INITIALIZER_HPP
