#ifndef PLAMN_CORE_UTILITIES_VARIANTS_HPP
#define PLAMN_CORE_UTILITIES_VARIANTS_HPP

// std::variant helpers

namespace plamn::core
{
// one visitor from several callables, e.g. std::visit(overloads{[](A&) {}, [](B&) {}}, v)
template<typename... Ts>
struct overloads : Ts...
{
    using Ts::operator()...;
};

} // namespace plamn::core

#endif /* PLAMN_CORE_UTILITIES_VARIANTS_HPP */
