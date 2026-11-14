#ifndef PLAMN_CORE_ERRORS_HPP
#define PLAMN_CORE_ERRORS_HPP

#include "dict.hpp"

#include <string>
#include <sstream>
#include <exception>
#include <stdexcept>

namespace plamn::core
{
// exception carrying key/value details, e.g. {"ID": "Updater::outsideGhostBox"}
class DictionaryException : public std::exception
{
public:
    DictionaryException() = default;
    DictionaryException(auto const& k, auto const& v) { (*this)(k, v); }

    using Dict_t = cppdict::Dict<std::string>;

    auto& operator[](std::string const& key) { return dict_[key]; }
    auto& operator[](std::string const& key) const { return dict_[key]; }
    auto& operator()(std::string const key, std::string const val)
    {
        dict_[key] = val;
        return *this;
    }

    std::string operator()() const
    {
        std::stringstream ss;
        dict_.visit(
            [&](std::string const& key, auto const& val) { ss << key << " " << val << std::endl; });
        return ss.str();
    }

    char const* what() const noexcept override
    {
        static thread_local std::string msg;
        return (msg = (*this)()).c_str();
    }

    std::string id() const
    {
        if (!dict_.contains("ID"))
            throw std::runtime_error("DictionaryException has no ID");
        return (*this)["ID"].template to<std::string>();
    }

private:
    Dict_t dict_;
};

} // namespace plamn::core

#endif /* PLAMN_CORE_ERRORS_HPP */
