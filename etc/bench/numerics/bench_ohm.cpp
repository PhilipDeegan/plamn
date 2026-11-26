// Ohm's law over one patch (OhmTransformer): 2D and 3D, ideal, resistive + constant
//  hyper-resistivity, resistive + spatial hyper-resistivity, a small and a large grid each. Per
//  case: the fastest and the median call over PLAMN_REPEATS (default 20) after one untimed call,
//  and a checksum of E's bits (results bitwise the same across a change: the same checksum)
//
//   mkn build -p bench_ohm -a "-DNDEBUG -O3 -march=native" -x res/mkn/clang -g 0 -t 4
//   ./bin/bench_ohm/plamn

#include "plamn/core/numerics/ohm/ohm_transformer.hpp"
#include "plamn/plamn_core.hpp"
#include "plamn/simulator/simulator_def.hpp"

#include "tst/plamn/core/data/grid/test_layout_fixtures.hpp"
#include "tst/plamn/core/data/vecfield/test_vecfield_fixtures.hpp"

#include "mkn/kul/env.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace
{
using namespace plamn::core;

enum class Terms { ideal, hyper_constant, hyper_spatial };
char const* name_of(Terms const t)
{
    return t == Terms::ideal            ? "ideal"
           : t == Terms::hyper_constant ? "hyper constant"
                                        : "hyper spatial";
}

OhmInfo info_of(Terms const t)
{
    if (t == Terms::ideal)
        return {/*eta=*/0., /*nu=*/0., HyperMode::constant};
    if (t == Terms::hyper_constant)
        return {/*eta=*/1., /*nu=*/0.01, HyperMode::constant};
    return {/*eta=*/1., /*nu=*/0.01, HyperMode::spatial};
}

// smooth, non-zero values everywhere (ghosts included): n divides in spatial hyper-resistivity
void fill(auto& field, double const offset, double const phase)
{
    for (std::size_t i = 0; i < field.size(); ++i)
        field.data()[i] = offset + 0.5 * std::sin(0.01 * static_cast<double>(i) + phase);
}

std::uint64_t bits_of(auto const& E)
{
    std::uint64_t h = 0;
    for (std::size_t c = 0; c < 3; ++c)
        for (std::size_t i = 0; i < E[c].size(); ++i)
            h = std::rotl(h, 5) ^ std::bit_cast<std::uint64_t>(E[c].data()[i]);
    return h;
}

template<std::size_t dim>
void run(Terms const terms, std::uint32_t const cells, std::size_t const repeats)
{
    using HybridTypes      = PLAMN_Types<plamn::SimOpts{dim, 1}>::Hybrid;
    using Layout_t         = HybridTypes::Layout_t;
    auto constexpr fopts   = TensorFieldOptions<HybridTypes>{};
    using UsableVecField_t = UsableVecField<fopts>;
    using Grid_t           = Grid<damn::core::NdArrayVector<dim>, HybridQuantity::Scalar>;

    Layout_t layout = ::TestLayout<Layout_t>{0.1, cells}; // cells^dim, dx 0.1
    layout.level    = 2; // spatial hyper-resistivity's 1 / 4^level not 1
    Grid_t n{"n", HybridQuantity::Scalar::rho, alloc_shape(layout, HybridQuantity::Scalar::rho)};
    Grid_t P{"P", HybridQuantity::Scalar::P, alloc_shape(layout, HybridQuantity::Scalar::P)};
    UsableVecField_t V{"V", layout, HybridQuantity::Vector::V};
    UsableVecField_t B{"B", layout, HybridQuantity::Vector::B};
    UsableVecField_t J{"J", layout, HybridQuantity::Vector::J};
    UsableVecField_t E{"E", layout, HybridQuantity::Vector::E};
    fill(n, 2., 0.);
    fill(P, 2., .5);
    for (std::size_t c = 0; c < 3; ++c)
    {
        fill(V[c], 0., 1. + c);
        fill(B[c], 1., 2. + c);
        fill(J[c], 0., 3. + c);
        fill(E[c], 0., 0.);
    }

    OhmTransformer const ohm{info_of(terms)};
    ohm(layout, n, V, P, B, J, E); // untimed: first touches
    std::vector<double> us;
    for (std::size_t r = 0; r < repeats; ++r)
    {
        auto const t0 = std::chrono::steady_clock::now();
        ohm(layout, n, V, P, B, J, E);
        auto const t1 = std::chrono::steady_clock::now();
        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(us.begin(), us.end());
    std::printf("%zuD %-15s %5u^%zu %12.1f %12.1f   %016llx\n", dim, name_of(terms), cells, dim,
                us.front(), us[us.size() / 2], static_cast<unsigned long long>(bits_of(E)));
}
} // namespace

int main()
{
    auto const repeats = mkn::kul::env::get_as("PLAMN_REPEATS", std::size_t{20});
    std::printf("%-19s %8s %12s %12s   %s\n", "case", "grid", "min us", "median us", "E bits");
    for (auto const terms : {Terms::ideal, Terms::hyper_constant, Terms::hyper_spatial})
    {
        for (auto const cells : {256u, 1024u})
            run<2>(terms, cells, repeats);
        for (auto const cells : {32u, 128u})
            run<3>(terms, cells, repeats);
    }
    return 0;
}
