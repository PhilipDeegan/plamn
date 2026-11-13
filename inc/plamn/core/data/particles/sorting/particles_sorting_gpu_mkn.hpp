#ifndef PLAMN_CORE_DATA_PARTICLES_SORTING_PARTICLES_SORTING_GPU_HPP
#define PLAMN_CORE_DATA_PARTICLES_SORTING_PARTICLES_SORTING_GPU_HPP

#if DAMN_HAVE_THRUST

#include "plamn/core/def.hpp"
#include "damn/core/exec/kernels.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "plamn/core/data/particles/particles_def.hpp"

#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>


namespace plamn::core::detail
{


template<typename Particles>
class ParticleSorter<damn::AllocatorMode::GPU_UNIFIED, /*impl = */ 0, Particles>
{
public:
    using box_t = damn::core::Box<int, Particles::dimension>;

    void operator()(std::uint64_t l, std::uint64_t r)
    {
        DAMN_LOG_SCOPE(1, "ParticleSorter<AllocatorMode::GPU_UNIFIED>");

        if constexpr (Particles::layout_mode == LayoutMode::AoS)
        {
            auto ps = particles.view();
            thrust::sort(thrust::device, ps.begin() + l, ps.begin() + r,
                         [cf = cell_flattener] DAMN_ALL(auto const& a, auto const& b) {
                             return cf(a.iCell()) < cf(b.iCell());
                         });
        }
        else if constexpr (Particles::layout_mode == LayoutMode::SoA)
        {
            auto it = SoAIteratorAdaptor::make(particles);
            thrust::sort(thrust::device, it + l, it + r,
                         [cf = cell_flattener] DAMN_ALL(auto const& a, auto const& b) {
                             return cf(SoAIteratorAdaptor::iCell(a))
                                    < cf(SoAIteratorAdaptor::iCell(b));
                         });
        }
        else if constexpr (Particles::layout_mode == LayoutMode::AoSTS)
        {
            // each tile on its own, l/r don't apply to tiles
            for (auto& tile : particles())
            {
                auto& ps = tile();
                thrust::sort(thrust::device, ps.data(), ps.data() + ps.size(),
                             [cf = cell_flattener] DAMN_ALL(auto const& a, auto const& b) {
                                 return cf(a.iCell()) < cf(b.iCell());
                             });
            }
        }
        else
            throw std::runtime_error("no impl");
    }

    void operator()() { (*this)(0, particles.size()); }

    // by cell, then by delta within a cell
    auto static constexpr by_cell_deltas(auto const cf) DAMN_ALL
    {
        return [cf] DAMN_ALL(auto const& a, auto const& b) -> bool {
            auto const ca = cf(a.iCell()), cb = cf(b.iCell());
            if (ca != cb)
                return ca < cb;
            return as_tuple(a.delta()) < as_tuple(b.delta());
        };
    }


    auto static constexpr by_deltas() DAMN_ALL
    {
        return [](auto const& a, auto const& b) -> bool {
            return as_tuple(a.delta()) < as_tuple(b.delta());
        };
    }
    void by_deltas(std::uint64_t l, std::uint64_t r)
    {
        DAMN_LOG_SCOPE(1, "ParticleSorter<AllocatorMode::GPU_UNIFIED>");
        if constexpr (Particles::layout_mode == LayoutMode::AoS)
        {
            auto ps = particles.view();
            thrust::sort(thrust::device, ps.begin() + l, ps.begin() + r, by_deltas());
        }
        else if constexpr (Particles::layout_mode == LayoutMode::SoA)
        {
            auto it = SoAIteratorAdaptor::make(particles);
            thrust::sort(thrust::device, it + l, it + r, [](auto const& a, auto const& b) -> bool {
                return as_tuple(SoAIteratorAdaptor::delta(a))
                       < as_tuple(SoAIteratorAdaptor::delta(b));
            });
        }
        else
            throw std::runtime_error("no impl");
    }

    ParticleSorter& by_delta()
    {
        DAMN_LOG_SCOPE(1, "ParticleSorter<AllocatorMode::GPU_UNIFIED>");
        // assumes already sorted by icell
        if (particles.size() == 0)
            return *this;

        using enum LayoutMode;

        if constexpr (damn::core::any_in(Particles::layout_mode, AoSPC))
            damn::core::for_each_point(particles.local_box(), [&](auto const& bix) {
                auto& ps = particles(bix);
                thrust::sort(thrust::device, ps.data(), ps.data() + ps.size(), by_deltas());
            });
        else if constexpr (damn::core::any_in(Particles::layout_mode, SoAPC))
            damn::core::for_each_point(particles.local_box(), [&](auto const& bix) {
                auto& ps = particles(bix);
                auto it  = SoAIteratorAdaptor::make(ps);
                thrust::sort(thrust::device, it, it + ps.size(),
                             [](auto const& a, auto const& b) -> bool {
                                 return as_tuple(SoAIteratorAdaptor::delta(a))
                                        < as_tuple(SoAIteratorAdaptor::delta(b));
                             });
            });
        else if constexpr (Particles::layout_mode == AoSTS)
            // tiles mix cells: one sort by (cell, delta) per tile keeps the cell order
            for (auto& tile : particles())
            {
                auto& ps = tile();
                thrust::sort(thrust::device, ps.data(), ps.data() + ps.size(),
                             by_cell_deltas(cell_flattener));
            }
        else if constexpr (Particles::layout_mode == SoATS)
        {
            throw std::runtime_error("no impl");
        }
        else
        {
            auto const end = particles.end();
            auto beg       = particles.begin();
            auto lst       = particles.begin();
            while (lst != end)
            {
                lst = beg + 1;
                while (lst != end and lst.iCell() == beg.iCell())
                    ++lst;
                by_deltas(it_dist(particles.begin(), beg), it_dist(particles.begin(), lst));
                beg = lst;
            }
        }

        return *this;
    }


    void sort_by_key(std::uint64_t l, std::uint64_t r) // unused but for reference
    {
        auto ps = particles.view();
        std::vector<int, mkn::gpu::ManagedAllocator<int>> flats(r - l);
        auto fv = flats.data();
        damn::kernel::launch(flats.size(), [=, cf = cell_flattener] DAMN_ALL() {
            auto idx          = damn::kernel::idx() + l;
            fv[damn::kernel::idx()] = cf(ps.iCell(idx));
        });
        thrust::device_vector<int> indices(flats.size());
        thrust::sequence(indices.begin(), indices.end());
        thrust::sort_by_key(thrust::device, fv, fv + flats.size(), indices.begin());
        thrust::gather(thrust::device, indices.begin(), indices.end(), ps.begin() + l,
                       ps.begin() + l);
    }


    Particles& particles;
    box_t& domain_box;
    CellFlattener<box_t> cell_flattener{domain_box};
};


// template<typename Particles>
// class ParticleSorter<AllocatorMode::GPU, /*impl = */ 0, Particles>
// {
// public:
//     void operator()(std::int64_t /*l*/, std::int64_t /*r*/) { std::abort(); }

//     void operator()() { (*this)(0, particles.size() - 1); }

//     Particles& particles;
// };

} // namespace plamn::core::detail


#endif // DAMN_HAVE_THRUST

#endif /*PLAMN_CORE_DATA_PARTICLES_SORTING_PARTICLES_SORTING_GPU_HPP*/
