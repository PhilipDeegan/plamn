#ifndef PLAMN_CORE_DATA_PARTICLES_PARTICLE_HPP
#define PLAMN_CORE_DATA_PARTICLES_PARTICLE_HPP

#include "mkn/kul/std/array.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/def/damn_config.hpp"
#include "plamn/core/def.hpp"

#include <array>
#include <random>
#include <type_traits>
#include <cstddef>
#include <limits>
#include <sstream>
#include <string>

namespace plamn::core
{


template<size_t dim>
struct Particle
{
    static_assert(dim > 0 and dim < 4, "Only dimensions 1,2,3 are supported.");
    static auto constexpr dimension = dim;

    constexpr Particle() DAMN_ALL {}
    Particle(Particle const&)            = default;
    Particle(Particle&&)                 = default;
    Particle& operator=(Particle const&) = default;
    Particle& operator=(Particle&&)      = default;

    Particle(double const& a_weight, std::array<int, dim> const& cell,
             std::array<double, dim> const& a_delta,
             std::array<double, 3> const& a_v) DAMN_ALL //
        : weight_{a_weight}
        , iCell_{cell}
        , delta_{a_delta}
        , v_{a_v}
    {
    }


    template<typename Particle_t>
    Particle(Particle_t const& p) DAMN_ALL //
        : weight_{p.weight()}
        , iCell_{p.iCell()}
        , delta_{p.delta()}
        , v_{p.v()}
    {
    }


    //                                                             1d  2d  3d
    double weight_                 = 0;                                   // 8   8   8
    std::array<int, dim> iCell_    = mkn::kul::ConstArray<int, dim>();    // 12  16  20
    std::array<double, dim> delta_ = mkn::kul::ConstArray<double, dim>(); // 20  32  44
    std::array<double, 3> v_       = mkn::kul::ConstArray<double, 3>();   // 44  56  68

    MKN_KUL_FN_SIG_RET bool operator==(Particle<dim> const& that) const
    {
        return (this->weight_ == that.weight_) && //
               (this->iCell_ == that.iCell_) &&   //
               (this->delta_ == that.delta_) &&   //
               (this->v_ == that.v_);
    }

    auto& weight() DAMN_ALL { return weight_; }
    auto& weight() const DAMN_ALL { return weight_; }

    auto& iCell() DAMN_ALL { return iCell_; }
    auto& iCell() const DAMN_ALL { return iCell_; }

    auto& delta() DAMN_ALL { return delta_; }
    auto& delta() const DAMN_ALL { return delta_; }

    auto& v() DAMN_ALL { return v_; }
    auto& v() const DAMN_ALL { return v_; }

    template<std::size_t dimension>
    friend std::ostream& operator<<(std::ostream& out, Particle<dimension> const& particle);


    auto copy() const { return *this; }
};

// must stay trivially copyable: else every particle array growth / reserve silently relocates
//  by a loop of per element construct / destroy instead of a memmove
static_assert(std::is_trivially_copyable_v<Particle<1>>);
static_assert(std::is_trivially_copyable_v<Particle<2>>);
static_assert(std::is_trivially_copyable_v<Particle<3>>);


template<typename T = float>
struct ParticleDeltaDistribution
{
    template<typename Generator>
    MKN_KUL_FN_SIG_RET T operator()(Generator& generator)
    {
        return dist(generator);
    }
    std::uniform_real_distribution<T> dist{0, 1. - std::numeric_limits<T>::epsilon()};
};


template<typename T, std::size_t dim>
MKN_KUL_FN_SIG_RET auto cellAsPoint(std::array<T, dim> const& iCell) DAMN_ALL
{
    return damn::core::Point<int, dim>{iCell};
}


template<typename Particle>
MKN_KUL_FN_SIG_RET auto cellAsPoint(Particle const& particle) DAMN_ALL
{
    return cellAsPoint(particle.iCell());
}



template<std::size_t dim>
struct ParticlePosition
{
    static size_t const dimension = dim;

    std::array<int, dim> _iCell;
    std::array<double, dim> _delta;

    auto& iCell() const DAMN_ALL { return _iCell; }
    auto& delta() const DAMN_ALL { return _delta; }
};


template<size_t dim>
auto to_string(Particle<dim> const& particle)
{
    std::stringstream out;
    auto const append = [&](std::string const name, auto const& value) {
        out << name << "(" << value[0];
        for (std::size_t i = 1; i < dim; ++i)
            out << "," << value[i];
        out << "), ";
    };
    append("iCell", particle.iCell());
    append("delta", particle.delta());
    append("v", particle.v());
    out << "weight: " << particle.weight();
    return out.str();
}

template<template<std::size_t> typename Particle_t, std::size_t dim, typename Stream>
Stream& write_to_stream(Particle_t<dim> const& particle, Stream& out, bool const new_line = true)
{
    out << to_string(particle);
    if (new_line)
        out << '\n';
    return out;
}

template<std::size_t dim>
std::ostream& operator<<(std::ostream& out, Particle<dim> const& particle)
{
    return write_to_stream(particle, out);
}



template<std::size_t dim>
struct ParticleView
{
    static_assert(dim > 0 and dim < 4, "Only dimensions 1,2,3 are supported.");
    static constexpr std::size_t dimension = dim;

    double& weight;
    std::array<int, dim>& iCell;
    std::array<double, dim>& delta;
    std::array<double, 3>& v;
};


template<typename Box_t, typename RValue = std::size_t>
struct CellFlattener
{
    template<typename Icell>
    MKN_KUL_FN_SIG_RET RValue operator()(Icell const& icell) const DAMN_ALL
    {
        if constexpr (Box_t::dimension == 2)
            return icell[1] + icell[0] * shape[1];
        if constexpr (Box_t::dimension == 3)
            return icell[2] + icell[1] * shape[2] + icell[0] * shape[1] * shape[2];
        return icell[0];
    }

    Box_t const box;
    std::array<int, Box_t::dimension> shape = box.shape().toArray();
};


template<std::size_t dim, typename T>
inline constexpr auto is_damn_particle_type
    = std::is_same_v<Particle<dim>, T> or std::is_same_v<ParticleView<dim>, T>;


template<std::size_t dim, template<std::size_t> typename ParticleA,
         template<std::size_t> typename ParticleB>
    requires(is_damn_particle_type<dim, ParticleA<dim>>
             and is_damn_particle_type<dim, ParticleB<dim>>)
MKN_KUL_FN_SIG_RET bool operator==(ParticleA<dim> const& particleA, ParticleB<dim> const& particleB)
{
    return particleA.weight == particleB.weight and //
           particleA.iCell == particleB.iCell and   //
           particleA.delta == particleB.delta and   //
           particleA.v == particleB.v;
}

auto shift_particle(auto particle, auto const& shift)
{
    mkn::kul::array_op<mkn::kul::PlusEquals<int>>(particle.iCell(), shift);
    return particle;
}


} // namespace plamn::core


template<std::size_t dim>
void swap(plamn::core::Particle<dim>& a, plamn::core::Particle<dim>& b)
{
    if (&a != &b)
        std::swap(a, b);
}

#endif
