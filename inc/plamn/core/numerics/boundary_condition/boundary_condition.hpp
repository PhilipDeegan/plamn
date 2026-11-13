#ifndef PLAMN_CORE_NUMERICS_BOUNDARY_CONDITION_BOUNDARY_CONDITION_HPP
#define PLAMN_CORE_NUMERICS_BOUNDARY_CONDITION_BOUNDARY_CONDITION_HPP

#include "plamn/core/def.hpp"
#include "damn/core/geometry/box.hpp"
#include "plamn/core/data/particles/partitioning/partitionner.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace damn::core
{
template<typename Type, std::size_t dim>
struct Box;
} // namespace damn::core


namespace plamn
{
namespace core
{
    template<std::size_t dim, std::size_t interpOrder>
    class BoundaryCondition
    {
    public:
        void setBoundaryBoxes(std::vector<damn::core::Box<int, dim>> boxes)
        {
            boundaryBoxes_ = std::move(boxes);
        }

        template<typename ParticleIterator>
        ParticleIterator applyOutgoingParticleBC(ParticleIterator begin, ParticleIterator /*end*/)
        {
            // TODO
            return ParticleIterator{begin}; // place holder
        }


    private:
        std::vector<damn::core::Box<int, dim>> boundaryBoxes_;
    };

} // namespace core
} // namespace plamn



#endif
