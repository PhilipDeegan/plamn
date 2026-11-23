#ifndef PLAMN_CORE_DATA_PARTICLES_AOS_TRANSPOSE_HPP
#define PLAMN_CORE_DATA_PARTICLES_AOS_TRANSPOSE_HPP

// 8 particle structures (Particle<dim>) to and from columns (weight, iCell, delta, v: SoAVX's), by
//  an 8 x 8 transpose of doubles in AVX-512 registers: each particle one masked row load / store of
//  its 64 bit words (weight, the cells packed two ints a word, delta, v), the words' columns one
//  register each. 3D's ninth word (v_z) per particle. Without AVX-512: per particle and field.
//
//   to_columns<dim>(ps, n, cols)     particles [0, n) (n <= 8) into the columns' slots [0, n),
//   slots
//                                    [n, 8) weight 0 (cells, delta, v unset)
//   from_columns<dim>(ps, n, cols)   the columns' slots [0, n) back into particles [0, n)
//   to_columns_each, from_columns_each  the same per particle and field (no AVX-512 needed)
//
//  Measured (harris 2D, AoS4SoAVXChunked): the transpose wins writing back (the push's REF) and
//  loading without cells (the deposit), the per particle copy wins loading a push's stage (its
//  cell ints then read per lane by partial blocks: whole vector stores forwarded piecewise)
//
//  cols: the columns at the block's first slot (pointers), c may be null (no cells, the deposit's)
//  PLAMN_AOS_TRANSPOSE_SCALAR: the per particle copy even with AVX-512 (to compare)

#include "plamn/core/data/particles/particle.hpp"

#include <cstddef>

#if defined(__AVX512F__) and !defined(PLAMN_AOS_TRANSPOSE_SCALAR)
#define PLAMN_AOS_TRANSPOSE_AVX512 1
#include <immintrin.h>
#endif

namespace plamn::core::aos_transpose
{

template<std::size_t dim>
struct Cols
{
    double* w;
    int* c[dim]; // null: cells not read / written
    double* d[dim];
    double* v[3];
};

template<std::size_t dim>
struct Words
{
    using P = Particle<dim>;
    // the 64 bit words of a particle: weight, the cells (two ints a word), delta, v
    static constexpr std::size_t cell_words = (dim + 1) / 2;
    static constexpr std::size_t delta_at   = 1 + cell_words;
    static constexpr std::size_t v_at       = delta_at + dim;
    static constexpr std::size_t n          = v_at + 3;
    static constexpr std::size_t in_rows    = n < 8 ? n : 8; // through the transpose
};

template<std::size_t dim>
inline void to_columns_each(Particle<dim> const* ps, std::size_t const n, Cols<dim> const& c)
{
    for (std::size_t l = 0; l < 8; ++l)
    {
        if (l >= n)
        {
            c.w[l] = 0;
            continue;
        }
        auto const& p = ps[l];
        c.w[l]        = p.weight();
        for (std::size_t k = 0; k < dim; ++k)
        {
            if (c.c[0])
                c.c[k][l] = p.iCell()[k];
            c.d[k][l] = p.delta()[k];
        }
        for (std::size_t k = 0; k < 3; ++k)
            c.v[k][l] = p.v()[k];
    }
}

template<std::size_t dim>
inline void from_columns_each(Particle<dim>* ps, std::size_t const n, Cols<dim> const& c)
{
    for (std::size_t l = 0; l < n; ++l)
    {
        auto& p    = ps[l];
        p.weight() = c.w[l];
        for (std::size_t k = 0; k < dim; ++k)
        {
            p.iCell()[k] = c.c[k][l];
            p.delta()[k] = c.d[k][l];
        }
        for (std::size_t k = 0; k < 3; ++k)
            p.v()[k] = c.v[k][l];
    }
}

namespace detail
{
#if defined(PLAMN_AOS_TRANSPOSE_AVX512)
    // rows r[0..8) of 8 doubles to columns, in place
    inline void transpose8(__m512d (&r)[8])
    {
        __m512d t[8], u[8];
        for (std::size_t i = 0; i < 4; ++i)
        {
            t[2 * i]     = _mm512_unpacklo_pd(r[2 * i], r[2 * i + 1]);
            t[2 * i + 1] = _mm512_unpackhi_pd(r[2 * i], r[2 * i + 1]);
        }
        u[0] = _mm512_shuffle_f64x2(t[0], t[2], 0x88);
        u[1] = _mm512_shuffle_f64x2(t[0], t[2], 0xDD);
        u[2] = _mm512_shuffle_f64x2(t[1], t[3], 0x88);
        u[3] = _mm512_shuffle_f64x2(t[1], t[3], 0xDD);
        u[4] = _mm512_shuffle_f64x2(t[4], t[6], 0x88);
        u[5] = _mm512_shuffle_f64x2(t[4], t[6], 0xDD);
        u[6] = _mm512_shuffle_f64x2(t[5], t[7], 0x88);
        u[7] = _mm512_shuffle_f64x2(t[5], t[7], 0xDD);
        r[0] = _mm512_shuffle_f64x2(u[0], u[4], 0x88);
        r[4] = _mm512_shuffle_f64x2(u[0], u[4], 0xDD);
        r[1] = _mm512_shuffle_f64x2(u[2], u[6], 0x88);
        r[5] = _mm512_shuffle_f64x2(u[2], u[6], 0xDD);
        r[2] = _mm512_shuffle_f64x2(u[1], u[5], 0x88);
        r[6] = _mm512_shuffle_f64x2(u[1], u[5], 0xDD);
        r[3] = _mm512_shuffle_f64x2(u[3], u[7], 0x88);
        r[7] = _mm512_shuffle_f64x2(u[3], u[7], 0xDD);
    }
#endif
} // namespace detail


template<std::size_t dim>
inline void to_columns(Particle<dim> const* ps, std::size_t const n, Cols<dim> const& c)
{
#if defined(PLAMN_AOS_TRANSPOSE_AVX512)
    using W = Words<dim>;
    static_assert(sizeof(Particle<dim>) == 8 * W::n, "Particle<dim>: words without holes");
    static constexpr __mmask8 row_mask = static_cast<__mmask8>((1u << W::in_rows) - 1);

    __m512d r[8];
    for (std::size_t l = 0; l < 8; ++l)
        r[l] = l < n ? _mm512_maskz_loadu_pd(row_mask, reinterpret_cast<double const*>(ps + l))
                     : _mm512_setzero_pd();
    detail::transpose8(r); // r[j]: word j of the 8 particles

    _mm512_storeu_pd(c.w, r[0]);
    if (c.c[0])
        for (std::size_t k = 0; k < dim; ++k)
        {
            __m512i const word = _mm512_castpd_si512(r[1 + k / 2]);
            __m512i const half = k % 2 ? _mm512_srli_epi64(word, 32) : word;
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c.c[k]), _mm512_cvtepi64_epi32(half));
        }
    for (std::size_t k = 0; k < dim; ++k)
        _mm512_storeu_pd(c.d[k], r[W::delta_at + k]);
    for (std::size_t k = 0; k < 3; ++k)
    {
        if (W::v_at + k < 8)
            _mm512_storeu_pd(c.v[k], r[W::v_at + k]);
        else // 3D's v_z
            for (std::size_t l = 0; l < 8; ++l)
                c.v[k][l] = l < n ? ps[l].v()[k] : 0.;
    }
#else
    to_columns_each(ps, n, c);
#endif
}

template<std::size_t dim>
inline void from_columns(Particle<dim>* ps, std::size_t const n, Cols<dim> const& c)
{
#if defined(PLAMN_AOS_TRANSPOSE_AVX512)
    using W = Words<dim>;
    static_assert(sizeof(Particle<dim>) == 8 * W::n, "Particle<dim>: words without holes");
    static constexpr __mmask8 row_mask = static_cast<__mmask8>((1u << W::in_rows) - 1);

    __m512d r[8];
    r[0] = _mm512_loadu_pd(c.w);
    for (std::size_t j = 0; j < W::cell_words; ++j)
    {
        __m512i word = _mm512_cvtepu32_epi64(
            _mm256_loadu_si256(reinterpret_cast<__m256i const*>(c.c[2 * j])));
        if (2 * j + 1 < dim)
            word = _mm512_or_si512(
                word, _mm512_slli_epi64(_mm512_cvtepu32_epi64(_mm256_loadu_si256(
                                            reinterpret_cast<__m256i const*>(c.c[2 * j + 1]))),
                                        32));
        r[1 + j] = _mm512_castsi512_pd(word);
    }
    for (std::size_t k = 0; k < dim; ++k)
        r[W::delta_at + k] = _mm512_loadu_pd(c.d[k]);
    for (std::size_t k = 0; k < 3; ++k)
        if (W::v_at + k < 8)
            r[W::v_at + k] = _mm512_loadu_pd(c.v[k]);
    for (std::size_t j = W::n; j < 8; ++j)
        r[j] = _mm512_setzero_pd();
    detail::transpose8(r); // r[l]: particle l's first words

    for (std::size_t l = 0; l < n; ++l)
        _mm512_mask_storeu_pd(reinterpret_cast<double*>(ps + l), row_mask, r[l]);
    if constexpr (W::n > 8) // 3D's v_z
        for (std::size_t l = 0; l < n; ++l)
            ps[l].v()[2] = c.v[2][l];
#else
    from_columns_each(ps, n, c);
#endif
}

} // namespace plamn::core::aos_transpose

#endif /* PLAMN_CORE_DATA_PARTICLES_AOS_TRANSPOSE_HPP */
