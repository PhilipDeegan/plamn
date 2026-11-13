#ifndef PLAMN_CORE_DATA_PARTICLES_SORTING_PARTICLES_SORTING_CPU_HPP
#define PLAMN_CORE_DATA_PARTICLES_SORTING_PARTICLES_SORTING_CPU_HPP

#include "plamn/core/def.hpp"

#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/meta/tuple.hpp"
#include "plamn/core/data/particles/particles_def.hpp"



#include <stdexcept>

namespace plamn::core::detail
{

template<typename V>
bool constexpr el_wise_less(V const& v0, V const& v1)
{
    for (std::int16_t i = v0.size() - 1; i >= 0; --i)
    {
        if (v0[i] < v1[i])
            return true;
        if (v0[i] > v1[i])
            return false;
    }
    return false;
}

template<typename V>
bool constexpr el_wise_gr8r(V const& v0, V const& v1)
{
    for (std::int16_t i = v0.size() - 1; i >= 0; --i)
    {
        if (v0[i] > v1[i])
            return true;
        if (v0[i] < v1[i])
            return false;
    }
    return false;
}

template<typename Particles>
class ParticleSorter<damn::AllocatorMode::CPU, /*impl = */ 0, Particles>
{
public:
    using box_t = damn::core::Box<int, Particles::dimension>;

    ParticleSorter(Particles& particles_, box_t domain)
        : particles{particles_}
        , domain_box{domain}
    {
    }



    ParticleSorter& operator()(std::int64_t const& l, std::int64_t const& r) // basically quicksort
    {
        auto constexpr static layout_mode = Particles::layout_mode;
        using enum LayoutMode;
        if constexpr (damn::core::any_in(layout_mode, SoA, SoAVX, SoAVXChunked))
        {
            DAMN_WITH_THRUST_ELSE_THROW({
                using Iterator_t = std::conditional_t<layout_mode == SoA, SoAIteratorAdaptor,
                                                      SoAVXIteratorAdaptor>;
                auto it          = Iterator_t::make(particles);
                std::sort(it + l, it + r, [cf = cell_flattener](auto const& a, auto const& b) {
                    return cf(Iterator_t::iCell(a)) < cf(Iterator_t::iCell(b));
                });
            })
        }
        else if constexpr (damn::core::any_in(layout_mode, AoSTS, SoATS, SoAVXTS))
        {
            // fix soa - not finished
            for (auto& tile : particles())
            {
                std::sort(tile().begin(), tile().end(),
                          [cf = cell_flattener](auto const& a, auto const& b) {
                              return cf(a.iCell()) < cf(b.iCell());
                          });
            }
        }

        else
        {
            static_assert(!damn::core::any_in(layout_mode, AoSPC, SoAPC, AoSPCTS),
                          "cell sorting not implemented for this layout (no flat iteration)");
            std::sort(particles.begin() + l, particles.begin() + r,
                      [cf = cell_flattener](auto const& a, auto const& b) {
                          return cf(a.iCell()) < cf(b.iCell());
                      });
        }

        return *this;
    }


    ParticleSorter& operator()()
    {
        (*this)(0, particles.size());
        return *this;
    }

    auto static constexpr by_deltas()
    {
        return [](auto const& a, auto const& b) -> bool {
            return damn::core::as_tuple(a.delta()) < damn::core::as_tuple(b.delta());
        };
    }

    void by_deltas(std::uint64_t const& l, std::uint64_t const& r)
    {
        using enum LayoutMode;

        if constexpr (damn::core::any_in(Particles::layout_mode, SoA))
        {
            DAMN_WITH_THRUST_ELSE_THROW({
                using Iterator_t = SoAIteratorAdaptor;
                auto it          = Iterator_t::make(particles);
                std::sort(it + l, it + r, [](auto const& a, auto const& b) -> bool {
                    return as_tuple(Iterator_t::delta(a)) < as_tuple(Iterator_t::delta(b));
                });
            })
        }
        else if constexpr (damn::core::any_in(Particles::layout_mode, SoAVX, SoAVXChunked))
        {
            DAMN_WITH_THRUST_ELSE_THROW({
                using Iterator_t = SoAVXIteratorAdaptor;
                auto it          = Iterator_t::make(particles);
                std::sort(it + l, it + r, [](auto const& a, auto const& b) -> bool {
                    return as_tuple(Iterator_t::template fdelta<Particles::dimension>(a))
                           < as_tuple(Iterator_t::template fdelta<Particles::dimension>(b));
                });
            })
        }
        else if constexpr (damn::core::any_in(Particles::layout_mode, AoSTS, SoATS, SoAVXTS))
        {
            throw std::runtime_error("no sort for AoSTS/SoATS");
        }
        else
        {
            std::sort(particles.begin() + l, particles.begin() + r, by_deltas());
        }
    }


    ParticleSorter& by_delta()
    {
        // assumes already sorted by icell
        if (particles.size() == 0)
            return *this;

        using enum LayoutMode;


        if constexpr (damn::core::any_in(Particles::layout_mode, AoSPC, SoAPC))
            damn::core::for_each_point(particles.local_box(), [&](auto const& bix) {
                std::sort(particles(bix).begin(), particles(bix).end(), by_deltas());
            });
        else if constexpr (damn::core::any_in(Particles::layout_mode, AoSTS, SoATS, SoAVXTS))
        {
            // tiles mix cells: by (cell, delta), a delta-only sort would lose the cell order
            // fix soa - not finished
            for (auto& tile : particles())
                std::sort(tile().begin(), tile().end(),
                          [cf = cell_flattener](auto const& a, auto const& b) {
                              auto const ca = cf(a.iCell()), cb = cf(b.iCell());
                              if (ca != cb)
                                  return ca < cb;
                              return damn::core::as_tuple(a.delta()) < as_tuple(b.delta());
                          });
        }
        else
        {
            static_assert(!damn::core::any_in(Particles::layout_mode, AoSPCTS),
                          "delta sorting not implemented for this layout (no flat iteration)");
            auto const end = particles.end();
            auto beg       = particles.begin();
            auto lst       = particles.begin();

            auto const check = [&]() { return lst != end and lst.iCell() == beg.iCell(); };

            while (lst != end)
            {
                lst = beg + 1;
                while (check())
                    ++lst;
                auto const s = it_dist(particles.begin(), beg);
                auto const e = it_dist(particles.begin(), lst);
                // if (e - s > 1)
                by_deltas(s, e);
                beg = lst;
            }
        }

        return *this;
    }


private:
    Particles& particles;
    box_t domain_box;
    CellFlattener<box_t> cell_flattener{domain_box};


    template<typename V>
    bool constexpr flat_less(V const& v0, V const& v1) const
    {
        return cell_flattener(v0) < cell_flattener(v1);
    }

    template<typename V>
    bool constexpr flat_gr8r(V const& v0, V const& v1) const
    {
        return cell_flattener(v0) > cell_flattener(v1);
    }
};


template<typename Particles>
class ParticleSorter<damn::AllocatorMode::CPU, /*impl = */ 1, Particles>
{
public:
    using box_t = damn::core::Box<int, Particles::dimension>;

    void operator()(std::int64_t const& l, std::int64_t const& r) // basically quicksort
    {
        auto i          = l;
        auto j          = r;
        auto const half = particles.iCell((l + r) / 2);
        do
        {
            while (el_wise_less(particles.iCell(i), half))
                i++;
            while (el_wise_gr8r(particles.iCell(j), half))
                j--;
            if (i <= j)
            {
                particles.swap(i, j);
                i++;
                j--;
            }
        } while (i <= j);
        if (l < j)
            (*this)(l, j);
        if (i < r)
            (*this)(i, r);
    }

    void operator()() { (*this)(0, particles.size() - 1); }


    Particles& particles;
    box_t domain_box;
};



} // namespace plamn::core::detail

#endif /*PLAMN_CORE_DATA_PARTICLES_SORTING_PARTICLES_SORTING_CPU_HPP*/
