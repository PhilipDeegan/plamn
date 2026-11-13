#ifndef PLAMN_CORE_DATA_VECFIELD_VECFIELD_HPP
#define PLAMN_CORE_DATA_VECFIELD_VECFIELD_HPP

#include "plamn/core/def.hpp"

#include <array>
#include <utility>
#include <algorithm>
#include <unordered_map>

#include "vecfield_component.hpp"
#include "plamn/core/data/tensorfield/tensorfield.hpp"

namespace plamn
{
namespace core
{
    template<typename Field_t, typename PhysicalQuantity>
    using VecField = TensorField<Field_t, PhysicalQuantity, /*rank=*/1>;



} // namespace core
} // namespace plamn

#endif
