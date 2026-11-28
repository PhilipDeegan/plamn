# phareh5

`inc/plamn/diagnostic/phareh5.hpp` holds the pieces of the phareh5 diagnostics, pharein's default
format, as pharesee reads it. `Simulator::dump_phareh5_`
(`simulator_diagnostics.hpp`) writes them, a file per quantity:
`/ions/pop/protons/flux` goes to `ions_pop_protons_flux.h5`.

## Layout

| Path | What |
|---|---|
| `/` | attributes: `dimension`, `interpOrder`, `layoutType` (yee), `domain_box` (its upper), `cell_width`, `origin`, `periodicities`; `pop_mass` and `pop_charge` for a population's quantity |
| `/py_attrs` | the diagnostic's attributes: pyphare's git hash, versions, the serialized simulation |
| `/t/<time>/pl<level>/p<rank>#<id>` | a patch; attributes `lower`, `upper` (its cells), `nbrCells`, `origin`, `mpi_rank` |

The time is written fixed, with 10 decimals.

### Fields

A dataset per component: `<name>_<component>` for vectors (`x y z`), tensors (`xx xy xz yy yz
zz`), B and E; `<name>` for scalars.

Values are on the patch's ghost box, in the quantity's own (Yee) centering, C order (x slowest),
with a `ghosts` attribute. The ghost widths are the format's per interpolation order (those of
jump 1, [ghost widths](../core/data/grid/ghost_width_calculator.md)), whatever plamn's own
widths.

### Particles

The datasets, each in memory (row-major) order:

| Dataset | Shape |
|---|---|
| `weight` | `(n, 1)` |
| `iCell` | `(n, dim)`, int32 |
| `delta` | `(n, dim)` |
| `v` | `(n, 3)` |

No datasets when there are no particles. The patch's `ghosts` attribute is the particle ghost
width.

## Levels

A dump at a time covers every level, or one only (`fine_dump_lvl_max`): such a time
holds that level only.
