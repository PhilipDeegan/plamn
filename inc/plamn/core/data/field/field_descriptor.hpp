#ifndef PLAMN_CORE_DATA_FIELD_FIELD_DESCRIPTOR_HPP
#define PLAMN_CORE_DATA_FIELD_FIELD_DESCRIPTOR_HPP

// a field described, no data: its name and its quantity (hence its centering), for a scalar, a
//  vector (components name_x, _y, _z) or a symmetric tensor (name_xx ... _zz). Quantity: a
//  model's quantities (e.g. HybridQuantity: its Scalar, Vector and Tensor)

#include <string>
#include <variant>

namespace plamn::core
{
template<typename Quantity>
struct ScalarFieldDescriptor
{
    std::string name;
    Quantity::Scalar qty;
};

template<typename Quantity>
struct VectorFieldDescriptor
{
    std::string name;
    Quantity::Vector qty;
};

template<typename Quantity>
struct TensorFieldDescriptor
{
    std::string name;
    Quantity::Tensor qty;
};

template<typename Quantity>
using FieldDescriptor
    = std::variant<ScalarFieldDescriptor<Quantity>, VectorFieldDescriptor<Quantity>,
                   TensorFieldDescriptor<Quantity>>;

} // namespace plamn::core

#endif /* PLAMN_CORE_DATA_FIELD_FIELD_DESCRIPTOR_HPP */
