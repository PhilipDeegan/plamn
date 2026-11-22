#ifndef PLAMN_CORE_MODELS_ELECTROMAG_INITIALIZER_HPP
#define PLAMN_CORE_MODELS_ELECTROMAG_INITIALIZER_HPP

// sets an Electromag from the user's input, simulation/electromag in the dict:
//   ElectromagInitializer<dim>{dict}(em, layout);
// B from the magnetic initializer functions; E is left to the model (Ohm's law)

#include "plamn/core/data/vecfield/vecfield_initializer.hpp"
#include "plamn/initializer/data_provider.hpp"
#include "dict.hpp"

#include <cstddef>

namespace plamn::core
{
template<std::size_t dim>
struct ElectromagInitializer
{
    explicit ElectromagInitializer(initializer::Dict const& dict)
        : B{dict["magnetic"]["initializer"]}
    {
    }

    template<typename Electromag, typename Layout>
    void operator()(Electromag& em, Layout const& layout) const
    {
        B.initialize(em.B, layout);
    }

    VecFieldInitializer<dim> B;
};

} // namespace plamn::core

#endif // PLAMN_CORE_MODELS_ELECTROMAG_INITIALIZER_HPP
