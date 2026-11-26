# Untiled temporaries of tiled fields

`inc/plamn/amr/untiled.hpp` gives untiled temporaries of a patch's tiled fields to what works on
whole patches: damn's schedules (copies, sums, refines, coarsens), and anything else that reads
or writes a patch field as one array.

## `Untiled<Tiled>`

One per patch.

| Call | What |
|---|---|
| `u.lease(name, tiled, layout, how)` | a temporary for `name`, every value set from `tiled` (see below) |
| `u.find(name)` | `name`'s temporary while leased, else `nullptr` |
| `u.release()` | ends every lease: `fill` and `sum` leases are written back into every tile's ghost box (outer and inner ghosts); `read` leases are dropped |

| `Lease` | The temporary's values |
|---|---|
| `read`, `fill` | the tiles' values: their domains, and the patch border ghosts their tiles hold |
| `sum` | zeroed, then every tile added, the tiles' partial ghost contributions included |

- A lease sets every value of its temporary: nothing carries over from an earlier use, so no NaN
  gets into a sum.
- A name leased twice in one operation (a name both source and destination) gets the same
  temporary.
- The temporaries are kept and reused, in a pool per physical quantity: a grid's centering, and
  so its shape on this patch.

`UntiledFor_t<Grid>` is `Untiled<Grid>` for tiled grids, else `NotTiled<Grid>`, a placeholder that
holds none.

## One off helpers

| Helper | What |
|---|---|
| `with_untiled(field, fn)` | `fn` on an untiled field of `field`'s values (the field itself when not tiled), for an occasional whole patch read (diagnostics) |
| `modify_untiled(field, fn)` | as `with_untiled`, then the values written back into every tile's ghost box |
| `field_resource_for<gw>(untiled, name, grid, fn)` | the field resource damn sees for a patch's grid: a tiled grid's leased temporary (none outside a lease), an untiled grid itself |
| `with_untiled_resources<gw>(grids, fn)` | `fn(std::array{&rx, &ry, &rz})` over a vector's three components, through `modify_untiled` |
