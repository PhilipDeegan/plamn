# AoS to columns transpose

`inc/plamn/core/data/particles/aos_transpose.hpp` converts blocks of 8 particle structures
(`Particle<dim>`) to and from columns (weight, iCell, delta, v: SoAVX's).

## Strategy

With AVX-512, by an 8 x 8 transpose of doubles in registers:

- each particle is one masked row load / store of its 64 bit words: weight, the cells packed two
  ints a word, delta, v;
- after the transpose, each word's column is one register;
- in 3D, the ninth word (v_z) goes per particle.

Without AVX-512, or with `PLAMN_AOS_TRANSPOSE_SCALAR` defined (to compare), the copy goes per
particle and field.

## Functions

| Function | |
|---|---|
| `to_columns<dim>(ps, n, cols)` | particles `[0, n)` (`n <= 8`) into the columns' slots `[0, n)`; slots `[n, 8)` get weight 0 (cells, delta, v unset) |
| `from_columns<dim>(ps, n, cols)` | the columns' slots `[0, n)` back into particles `[0, n)` |
| `to_columns_each`, `from_columns_each` | the same, per particle and field (no AVX-512 needed) |

`cols` holds the columns at the block's first slot (pointers). Its cells `c` may be null: no
cells read or written (the deposit's case).

## Measured

Harris 2D, `AoS4SoAVXChunked`:

| Use | Faster |
|---|---|
| writing back (the push's REF) | transpose |
| loading without cells (the deposit) | transpose |
| loading a push's stage | per particle copy |

When loading a push's stage, its cell ints are then read per lane by partial blocks: the
transpose's whole vector stores are forwarded piecewise.
