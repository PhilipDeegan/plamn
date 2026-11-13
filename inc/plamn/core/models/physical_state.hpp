#ifndef PLAMN_PHYSICAL_STATE_HPP
#define PLAMN_PHYSICAL_STATE_HPP

#include "plamn/core/def.hpp"




/**
 * @brief The PhysicalState class is an interface for concrete states holding data associated with a
 * concrete IPhysicalModel
 */
namespace plamn
{
namespace core
{
    enum class Model { MHD, Hybrid };

    class PhysicalStateInitializer
    {
    };


    class IPhysicalState
    {
    };

} // namespace core
} // namespace plamn



#endif // PHYSICAL_STATE_HPP
