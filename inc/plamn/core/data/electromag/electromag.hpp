#ifndef PLAMN_CORE_DATA_ELECTROMAG_ELECTROMAG_HPP
#define PLAMN_CORE_DATA_ELECTROMAG_ELECTROMAG_HPP

#include "damn/core/def/damn_config.hpp"

#include "plamn/core/def.hpp"
#include "plamn/core/data/tensorfield/tensorfield.hpp"
#include "plamn/core/models/quantities/hybrid_quantities.hpp"
#include "plamn/initializer/data_provider.hpp"

#include "dict.hpp"

#include <tuple>
#include <string>
#include <cstddef>

namespace plamn::core::basic
{


template<typename VecFieldT>
class Electromag
{
public:
    using vecfield_type                    = VecFieldT;
    static constexpr std::size_t dimension = VecFieldT::dimension;

    template<typename V>
    auto as(auto&& a, auto&&... args) DAMN_ALL
    {
        return V{{a(E, args...)}, {a(B, args...)}};
    }

    template<typename V>
    auto as(auto&& a, auto&&... args) const DAMN_ALL
    {
        return V{{a(E, args...)}, {a(B, args...)}};
    }

    VecFieldT E, B;
};


} // namespace plamn::core::basic


namespace plamn
{
namespace core
{
    template<typename VecFieldT>
    class Electromag : public basic::Electromag<VecFieldT>
    {
    public:
        using Super = basic::Electromag<VecFieldT>;
        using Super::B;
        using Super::E;
        using vecfield_type                    = VecFieldT;
        static constexpr std::size_t dimension = VecFieldT::dimension;

        explicit Electromag(std::string name)
            : Super{{name + "_E", HybridQuantity::Vector::E},
                    {name + "_B", HybridQuantity::Vector::B}}
        {
        }

        explicit Electromag(plamn::initializer::Dict const& dict)
            : Super{{dict["name"].template to<std::string>() + "_"
                         + dict["electric"]["name"].template to<std::string>(),
                     HybridQuantity::Vector::E},
                    {dict["name"].template to<std::string>() + "_"
                         + dict["magnetic"]["name"].template to<std::string>(),
                     HybridQuantity::Vector::B}}
        {
        }

        // initialization from the input: core/models/electromag_initializer.hpp


        //-------------------------------------------------------------------------
        //                  start the ResourcesUser interface
        //-------------------------------------------------------------------------

        NO_DISCARD bool isUsable() const { return E.isUsable() && B.isUsable(); }

        NO_DISCARD bool isSettable() const { return E.isSettable() && B.isSettable(); }

        NO_DISCARD auto getCompileTimeResourcesViewList() const
        {
            return std::forward_as_tuple(E, B);
        }

        NO_DISCARD auto getCompileTimeResourcesViewList() { return std::forward_as_tuple(E, B); }


        //-------------------------------------------------------------------------
        //                  ends the ResourcesUser interface
        //-------------------------------------------------------------------------

        void copyData(Electromag const& source)
        {
            E.copyData(source.E);
            B.copyData(source.B);
        }


        auto operator()() { return std::forward_as_tuple(E, B); }
        auto operator()() const { return std::forward_as_tuple(E, B); }




        Super& super() { return *this; }
        Super const& super() const { return *this; }
        auto& operator*() { return super(); }
        auto& operator*() const { return super(); }


    private:
    };


} // namespace core
} // namespace plamn


namespace plamn::core
{

void check_electromag(auto const& em) DAMN_ALL
{
#if DAMN_DEBUG && !DAMN_HAVE_GPU
    check_tensor_field(em.E);
    check_tensor_field(em.B);
#endif
}

} // namespace plamn::core


#endif
