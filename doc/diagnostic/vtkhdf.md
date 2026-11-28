# pharevtkhdf

`inc/plamn/diagnostic/vtkhdf.hpp` holds the pieces of the pharevtkhdf diagnostics (the file
itself is `damn/hdf5/vtkhdf.hpp`'s), and what both formats share.

| Piece | What |
|---|---|
| `Diagnostic`, `diagnostics_from(dict)` | what pyphare declared, `simulation/diagnostics/<type>/<type><N>`: quantity, `write_timestamps`, `flush_every`, attributes. Types `electromag`, `fluid`, `mhd`, and `particle` (phareh5 only); others are not written |
| `due(diagnostic, t, dt)` | the schedule, below |
| `file_name(quantity)` | `/ions/pop/protons/flux` goes to `ions_pop_protons_flux.vtkhdf` |
| `primal_into(...)` | a patch's field written straight into its rows |

The quantities themselves (`hybrid_quantities`, `mhd_quantities`, `quantity_for`) are in
`diagnostic/quantities.hpp`.

## Schedule

A diagnostic is due when one of its timestamps is within `dt` of `t`, compared as floats.
Timestamps already passed are caught up.

## primal_into

A patch's field on the primal nodes of its cells, x fastest, written straight into its rows:

- dual directions are averaged from the neighbouring dual values, so ghosts are read;
- each value goes at `row * n_components + component`;
- in 1D and 2D, the block is repeated `x_times` to make a 3D view.

Tiled fields go through an untiled copy of their tiles.
