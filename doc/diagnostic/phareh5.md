# phareh5

`inc/plamn/diagnostic/phareh5.hpp` holds the pieces of the phareh5 diagnostics, PHARE's default
format (PHARE's `diagnostic/detail/h5writer.hpp` and `types/`). `Simulator::dump_phareh5_`
(`simulator_diagnostics.hpp`) writes them, a file per quantity:
`/ions/pop/protons/flux` goes to `ions_pop_protons_flux.h5`.

## Layout

| Path | What |
|---|---|
| `/` | attributes: `dimension`, `interpOrder`, `layoutType` (yee), `domain_box` (its upper), `cell_width`, `origin`, `periodicities`; `pop_mass` and `pop_charge` for a population's quantity |
| `/py_attrs` | the diagnostic's attributes: pyphare's git hash, versions, the serialized simulation |
| `/t/<time>/pl<level>/p<rank>#<id>` | a patch; attributes `lower`, `upper` (its cells), `nbrCells`, `origin`, `mpi_rank` |

The time is written with 10 decimals, as PHARE's `core::to_string_with_precision(t, 10)`.

### Fields

A dataset per component: `<name>_<component>` for vectors (`x y z`), tensors (`xx xy xz yy yz
zz`), B and E; `<name>` for scalars.

Values are on PHARE's ghost box of the patch, in the quantity's own (Yee) centering, C order (x
slowest), with a `ghosts` attribute. The ghost widths are PHARE master's per interpolation order
(`core/models/options/hybrid_options.hpp`): what its files hold, whatever plamn's own widths.

### Particles

PHARE's `ParticlePacker` keys and shapes, each in memory (row-major) order:

| Dataset | Shape |
|---|---|
| `weight` | `(n, 1)` |
| `iCell` | `(n, dim)`, int32 |
| `delta` | `(n, dim)` |
| `v` | `(n, 3)` |

No datasets when there are no particles. The patch's `ghosts` attribute is the particle ghost
width.

## Levels

A dump at a time covers every level, or one only (PHARE's `dump_level`, with
`fine_dump_lvl_max`): such a time holds that level only.
