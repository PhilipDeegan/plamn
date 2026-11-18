#ifndef PLAMN_CORE_DATA_PARTICLES_PARTICLES_HPP
#define PLAMN_CORE_DATA_PARTICLES_PARTICLES_HPP

#include "damn/core/def/damn_config.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/logger.hpp"
#include "mkn/kul/std/array.hpp"
#include "mkn/kul/float.hpp"
#include "mkn/kul/for.hpp"
#include "mkn/kul/meta.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "mkn/kul/equality.hpp"
#include "damn/core/sys/monitoring.hpp"
#include "plamn/core/def.hpp"
#include "plamn/core/data/particles/particle.hpp"
#include "plamn/core/data/particles/particles_def.hpp"
#include "plamn/core/data/particles/particles_detail.hpp"

// #include "plamn/core/data/particles/sorting/particles_sorting.hpp"
// #include "plamn/core/data/particles/particles_selector.hpp"
// #include "plamn/core/data/particles/partitioning/particles_partitioning.hpp"

#include <utility>
#include <sstream>
#include <stdexcept>
#include <cstddef>
#include <format>
#include <string>

namespace plamn::core
{

template<auto opts /* defaulted in details header */>
class Particles : public ResolvedParticles_t<opts>
{
    using This      = Particles<opts>;
    using internals = ParticlesResolver<opts>;

public:
    using Super      = ResolvedParticles_t<opts>;
    using value_type = ParticleDefaults<opts.dim>::Particle_t;
    using view_t     = Particles<opts.with_storage(StorageMode::SPAN)>;

    auto static constexpr options      = opts;
    auto static constexpr dimension    = opts.dim;
    auto static constexpr alloc_mode   = opts.alloc_mode;
    auto static constexpr layout_mode  = opts.layout_mode;
    auto static constexpr storage_mode = opts.storage_mode;
    auto static constexpr type_id      = internals::type_id;
    // auto static constexpr is_mapped    = internals::is_mapped; // torm
    auto static inline mon = damn::core::MemoryMonitor{std::string{type_id}};

    std::string static id() { return std::string{type_id}; }

    Particles(Particles&& that)
        : Super{std::forward<Super>(that)}
    {
        mon.move();
    }
    Particles(Particles const& that)
        : Super{that}
    {
        mon.copy();
    }

    Particles& operator=(Particles&& that)
    {
        mon.move_assign();
        super() = std::move(that.super());
        return *this;
    }
    Particles& operator=(Particles const& that)
    {
        mon.copy_assign();
        super() = that.super();
        return *this;
    }

    template<typename... Args>
    Particles(Args&&... args)
        requires(self_excluding_constructible<This, Super, Args...>())
    DAMN_ALL : Super{std::forward<Args>(args)...}
    {
        mon.create();
    }



    auto view() { return view_t{*this}; }
    auto view() const { return view_t{*this}; }

    auto view(std::size_t i) // to take only i particles and ignore the rest
    {
        view_t v{*this};
        v.super().resize(i);
        return v;
    }

    auto view(std::size_t const start, std::size_t const size)
    {
        return view_t{*this, start, size};
    }

    auto operator*() { return view(); }
    auto operator*() const { return view(); }

    auto begin()
    {
        if constexpr (requires { super().begin(); })
            return super().begin();
        else
            static_assert(mkn::kul::dependent_false_v<This>,
                          "iteration not supported for this layout");
    }
    auto begin() const
    {
        if constexpr (requires { super().begin(); })
            return super().begin();
        else
            static_assert(mkn::kul::dependent_false_v<This>,
                          "iteration not supported for this layout");
    }
    auto end()
    {
        if constexpr (requires { super().end(); })
            return super().end();
        else
            static_assert(mkn::kul::dependent_false_v<This>,
                          "iteration not supported for this layout");
    }
    auto end() const
    {
        if constexpr (requires { super().end(); })
            return super().end();
        else
            static_assert(mkn::kul::dependent_false_v<This>,
                          "iteration not supported for this layout");
    }

    Super& super() DAMN_ALL { return *this; }
    Super const& super() const { return *this; }

    template<auto _opts>
    friend std::ostream& operator<<(std::ostream& out, Particles<_opts> const&);
};

template<std::size_t dim>
using ParticlesAoS = Particles<ParticlesOptions{dim, LayoutMode::AoS}>;


template<std::size_t dim>
using ParticlesAoSMapped = Particles<ParticlesOptions{dim, LayoutMode::AoSMapped}>;

template<std::size_t dim>
using ParticlesSoA = Particles<ParticlesOptions{dim, LayoutMode::SoA}>;


template<auto opts>
std::ostream& operator<<(std::ostream& out, Particles<opts> const& arr)
{
    if constexpr (Particles<opts>::layout_mode == LayoutMode::SoAPC)
    {
    }
    else
        for (auto const& p : arr)
            out << p.copy();
    return out;
}




template<auto opts>
void empty(Particles<opts>& array)
{
    array.clear();
}


template<auto opts>
void swap(Particles<opts>& array1, Particles<opts>& array2)
{
    array1.swap(array2);
}

template<typename P0, typename P1>
mkn::kul::EqualityReport particle_compare(P0 const& p0, P1 const& p1, std::size_t const i = 0,
                                          double const atol = 1e-15)
{
    std::string idx = std::to_string(i);
    if (p0.iCell() != p1.iCell())
        return mkn::kul::EqualityReport{false, "icell mismatch at index: " + idx, i};

    if (!mkn::kul::float_equals(p0.v(), p1.v(), atol))
        return mkn::kul::EqualityReport{false, "v mismatch at index: " + idx, i};
    if (!mkn::kul::float_equals(p0.delta(), p1.delta(), atol))
        return mkn::kul::EqualityReport{false, "delta mismatch at index: " + idx, i};


    return mkn::kul::EqualityReport{true};
}


template<typename P0, typename P1>
mkn::kul::EqualityReport particles_equals(P0 const& ref, P1 const& cmp, double const atol = 1e-15)
{
    if (ref.size() != cmp.size())
        return mkn::kul::EqualityReport{false, "different sizes: " + std::to_string(ref.size())
                                                   + " vs " + std::to_string(cmp.size())};

    auto rit      = ref.begin();
    auto cit      = cmp.begin();
    std::size_t i = 0;

    for (; rit != ref.end(); ++rit, ++cit, ++i)
        if (auto const eq = particle_compare(*rit, *cit, i, atol); !eq)
            return eq;

    return mkn::kul::EqualityReport{true};
}

template<auto o>
mkn::kul::EqualityReport operator==(Particles<o> const& p0, Particles<o> const& p1)
{
    auto report = particles_equals(p0, p1);
    if (!report)
    {
        DAMN_LOG_LINE_STR(p0[report.idx].copy());
        DAMN_LOG_LINE_STR(p1[report.idx].copy());
    }
    return report;
}

template<auto o>
mkn::kul::EqualityReport operator==(Particles<o> const& p0, std::vector<Particle<o>> const& p1)
{
    return particles_equals(p0, p1);
}



template<typename Particles_t>
auto constexpr base_layout_type()
{
    using enum LayoutMode;
    auto constexpr layout_mode = Particles_t::layout_mode;
    // SoAVX: AoS staging (pack buffers, refine buffers), SoA has no ARRAY storage
    if constexpr (mkn::kul::any_in(layout_mode, AoS, AoSMapped, AoSPC, AoSTS, AoSCMTS, AoSPCTS,
                                   SoAVX, SoAVXChunked))
        return AoS;
    return SoA;
}



template<auto o>
void check_particles(Particles<o> const& particles, [[maybe_unused]] bool const print = false)
{
    using enum LayoutMode;

    DAMN_DEBUG_DO({
        if constexpr (is_tiled(o.layout_mode))
            particles.check();
    })
}


template<auto o>
void check_particles_views(Particles<o>& particles, [[maybe_unused]] bool const print = false)
{
    using enum LayoutMode;

    DAMN_DEBUG_DO({ check_particles(*particles); })
}

template<auto o>
void particles_domain_is_valid(Particles<o> const& particles, auto const& domain_box)
{
    std::size_t in_domain_box = 0, not_in_domain_box = 0;

    if constexpr (o.layout_mode == LayoutMode::AoSPCTS)
    {
        for (auto const& tile : particles())
        {
            auto const& pc = tile();

            // every particle bucketed anywhere in this tile's ghost box (domain +
            // halo) must actually sit inside the tile itself, and must be filed
            // under the same cell its own iCell() maps to
            damn::core::for_each_point(pc.ghost_box(), [&](auto const& bix) {
                auto const local           = pc.local_cell(bix);
                auto const& cell_particles = pc(local);
                for (std::size_t i = 0; i < cell_particles.size(); ++i)
                {
                    auto const& p = cell_particles[i];
                    if (not isIn(p, tile))
                    {
                        std::ostringstream oss;
                        oss << "particles_domain is not valid: particle outside tile"
                            << " storage_mode=" << static_cast<int>(o.storage_mode)
                            << " tile_box=" << tile << " bix=" << damn::core::Point{bix}
                            << " local=" << damn::core::Point{local} << " iCell=" << damn::core::Point{p.iCell()};
                        throw std::runtime_error(oss.str());
                    }
                    if (not mkn::kul::array_equals(pc.local_cell(p.iCell()), local))
                    {
                        std::ostringstream oss;
                        oss << "particles_domain is not valid: particle iCell does not "
                               "match its per-cell bucket"
                            << " storage_mode=" << static_cast<int>(o.storage_mode)
                            << " tile_box=" << tile << " bix=" << damn::core::Point{bix}
                            << " local=" << damn::core::Point{local} << " iCell=" << damn::core::Point{p.iCell()};
                        throw std::runtime_error(oss.str());
                    }
                }
            });
        }
    }
    else if constexpr (is_tiled(o.layout_mode))
    {
        for (auto const& tile : particles())
            for (auto const& p : tile())
                if (not isIn(p, tile))
                    throw std::runtime_error("particles_domain is not valid");
    }

    per_particle(particles, [&](auto const& p) {
        if (isIn(p, domain_box))
            ++in_domain_box;
        else
            ++not_in_domain_box;
    });

    if (not(not_in_domain_box == 0 and in_domain_box == particles.size()))
        throw std::runtime_error("Invalid particles");

    // recurse once into the SPAN view so both storage sides are checked
    // SPAN views are non-const (AoSSpan holds Span<Particle_t>), this check only reads
    if constexpr (o.storage_mode == StorageMode::VECTOR)
        particles_domain_is_valid(*const_cast<Particles<o>&>(particles), domain_box);
}

template<auto o>
void particles_ghost_is_valid(Particles<o> const& particles, auto const& domain_box,
                                   auto const& ghost_box)
{
    std::size_t in_ghost_layer = 0, not_in_ghost_layer = 0, outside_gb = 0;

    per_particle(particles, [&](auto const& p) {
        auto const ingb = isIn(p, ghost_box);
        auto const indb = isIn(p, domain_box);

        if (!ingb)
            ++outside_gb;
        else if (ingb and not indb)
            ++in_ghost_layer;
        else
            ++not_in_ghost_layer;
    });

    if constexpr (o.layout_mode == LayoutMode::AoSPCTS)
    {
        for (auto const& tile : particles())
        {
            auto const& pc = tile();

            // every particle bucketed anywhere in this tile's ghost box (domain +
            // halo) must actually sit in the halo, i.e. outside the patch's own
            // domain box, and must be filed under the same cell its own iCell() maps to
            damn::core::for_each_point(pc.ghost_box(), [&](auto const& bix) {
                auto const local           = pc.local_cell(bix);
                auto const& cell_particles = pc(local);
                for (std::size_t i = 0; i < cell_particles.size(); ++i)
                {
                    auto const& p = cell_particles[i];
                    if (isIn(p, domain_box))
                    {
                        std::ostringstream oss;
                        oss << "particles_ghost is not valid: particle inside domain box"
                            << " storage_mode=" << static_cast<int>(o.storage_mode)
                            << " tile_box=" << tile << " bix=" << damn::core::Point{bix}
                            << " local=" << damn::core::Point{local} << " iCell=" << damn::core::Point{p.iCell()};
                        throw std::runtime_error(oss.str());
                    }
                    if (not mkn::kul::array_equals(pc.local_cell(p.iCell()), local))
                    {
                        std::ostringstream oss;
                        oss << "particles_ghost is not valid: particle iCell does not "
                               "match its per-cell bucket"
                            << " storage_mode=" << static_cast<int>(o.storage_mode)
                            << " tile_box=" << tile << " bix=" << damn::core::Point{bix}
                            << " local=" << damn::core::Point{local} << " iCell=" << damn::core::Point{p.iCell()};
                        throw std::runtime_error(oss.str());
                    }
                }
            });
        }
    }
    else if constexpr (is_tiled(o.layout_mode))
    {
        for (std::size_t tidx = 0; tidx < particles().size(); ++tidx)
        {
            auto const& tile = particles()[tidx];
            for (std::size_t i = 0; i < tile().size(); ++i)
                if (isIn(tile()[i], domain_box))
                    throw std::runtime_error("particles_ghost is not valid");
        }
    }

    if (not(outside_gb == 0 and not_in_ghost_layer == 0 and in_ghost_layer == particles.size()))
        throw std::runtime_error("Invalid particles");
}


} // namespace plamn::core


#endif
