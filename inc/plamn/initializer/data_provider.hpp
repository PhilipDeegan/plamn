#ifndef PLAMN_INITIALIZER_DATA_PROVIDER_HPP
#define PLAMN_INITIALIZER_DATA_PROVIDER_HPP

// simulation input: the dict passed (by const&) to model constructors, and the
// user functions it carries. Whoever builds the dict owns it; there is no global.

#include "damn/core/memory/span.hpp"

#include "plamn/core/def.hpp"

#include "dict.hpp"

#include <memory>
#include <string>
#include <vector>
#include <cstddef>
#include <optional>
#include <functional>
#include <type_traits>

namespace cppdict
{
/**
 * @brief reads a scoped enum stored in the dict as its underlying integer.
 *
 * overloads cppdict::get_value
 */
template<typename Enum, typename... Types>
    requires(std::is_enum_v<Enum>)
Enum get_value(Dict<Types...> const& dict, std::string const& path, Enum const default_value,
               char delimiter = '/')
{
    return static_cast<Enum>(
        get_value(dict, detail::split_string(path, delimiter), static_cast<int>(default_value)));
}
} // namespace cppdict


namespace plamn::initializer
{
template<typename ReturnType, std::size_t dim>
struct InitFunctionHelper
{
};

template<>
struct InitFunctionHelper<double, 1>
{
    using return_type = std::shared_ptr<damn::core::Span<double>>;
    using param_type  = damn::core::Span<double const> const&;
    using type        = std::function<return_type(param_type)>;
};

template<>
struct InitFunctionHelper<double, 2>
{
    using return_type = std::shared_ptr<damn::core::Span<double>>;
    using param_type  = damn::core::Span<double const> const&;
    using type        = std::function<return_type(param_type, param_type)>;
};

template<>
struct InitFunctionHelper<double, 3>
{
    using return_type = std::shared_ptr<damn::core::Span<double>>;
    using param_type  = damn::core::Span<double const> const&;
    using type        = std::function<return_type(param_type, param_type, param_type)>;
};

template<std::size_t dim>
using InitFunction = InitFunctionHelper<double, dim>::type;


using Dict = cppdict::Dict<bool, int, std::vector<int>, double, std::vector<double>, std::size_t,
                           std::optional<std::size_t>, std::string, std::vector<std::string>,
                           InitFunction<1>, InitFunction<2>, InitFunction<3>>;

} // namespace plamn::initializer

#endif // PLAMN_INITIALIZER_DATA_PROVIDER_HPP
