# Diagnostics

`inc/plamn/simulator/simulator_diagnostics.hpp` writes the diagnostics pyphare declares
(`diag_options` and its `*Diagnostics`, under `simulation/diagnostics`), a file per quantity, in
one of two formats:

| Format | What |
|---|---|
| phareh5 | PHARE's default (`diagnostic/phareh5.hpp`): fields in their own centering, ghosts included, and particles |
| pharevtkhdf | a step per dump, every level up to `max_nbr_levels`, an empty entry while a level is not there; no particles (nor in PHARE) |

PHARE's `flush_every` applies to both: a file is closed every `flush_every` dumps (0: never) and
reopened at the next.

Unknown quantities are an error at setup, not at a dump.

## phareh5

As PHARE's `H5Writer`, per level:

1. every rank creates every patch's datasets (`create_data_set_per_mpi`): creating is
   collective, so a rank with fewer patches passes empty ones;
2. each rank writes its own patches' data;
3. the patch attributes (`write_attributes_per_mpi`);
4. then the file's attributes.

## pharevtkhdf

A dump of a quantity goes level by level:

1. this rank's patches' rows (`damn::hdf5::vtkhdf::rows_of`) are laid out in one buffer
   (`diagnostics_.buffer`, reused across dumps);
2. each patch is converted straight into its slice (`plamn::diagnostic::primal_into`), the
   patches in parallel on damn's thread pool;
3. one write per rank (`damn::hdf5::vtkhdf::append_level`).

The timing scopes are dump, convert and write, per level.

## Coupled hierarchies

With MHD and hybrid levels, a field quantity is written on the levels whose model provides it:
its quantity table, or a hybrid computed quantity. The other levels get nothing (phareh5) or an
empty entry (pharevtkhdf), as a level that is not there.

B, E, the ions' mass density and bulk velocity are on both models (MHD: B, E, rho, V). A
quantity that no model owning levels has is an error at setup.
