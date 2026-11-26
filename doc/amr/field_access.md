# Field accessors for damn's copies

`inc/plamn/amr/field_access.hpp` gives damn's copies (`damn/amr/copy.hpp`) and inter level
operations a patch's fields through accessors rather than by their names, so no name is built or
compared per patch and piece. Names are still resolved once, when the accessors are made.

## Pieces

| Name | What |
|---|---|
| `grid_table(patch)` | the addresses of the patch's fixed grids, in `for_each_fixed_field_grid`'s order, made on first use |
| `GridAt` | a place on a patch: a grid table index, or a scratch grid's name |
| `FieldAccess<gw>` | a resource pair's source and destination, each a `GridAt` |
| `field_access<gw>(patch, pairs)` | the pairs' accessors, the indices taken from one patch's enumeration |
| `LevelFieldAccess<cgw, fgw>` | refine / coarsen's: the coarse, coarse_new and fine sides, each in its own level's (model's) table |
| `level_field_access<cgw, fgw>(coarse_patch, fine_patch, pairs)` | those accessors, or none if a name is no field of its patch |

## Why the indices hold

- The grids never move: patches are not movable, and populations are held by `unique_ptr`. The
  table's addresses stay valid for the patch's life.
- Every patch of a model has the same table, so indices from one patch's enumeration hold for all
  of them.
- Scratch grids come and go, so they are not in the table: they are reached by name.

## Fallback

`level_field_access` returns none when a pair is not a field on its patch (a particle array, for
example); the caller then goes by name.
