# Box copies

`inc/plamn/core/data/field/box_copies.hpp` copies boxes between fields of different shapes: a
patch's tiles and its untiled temporaries (`amr/untiled.hpp`, see [untiled](../../../amr/untiled.md)).
Same shaped fields need none of this: the allocator's copy, `core::level_copy`.

## `BoxCopies<alloc_mode, dim, T>`

Collects items (`dst[dst box] = src[src box]`, or `+=`), runs them at once.

| Call | What |
|---|---|
| `c.zero(field)` | `field` zeroed at the start of the next run, before any copy |
| `c.add(dst, dst_box, src, src_box, sum)` | an item: local boxes of the same shape, both fields C ordered |
| `c.run()` | the zeros, then every item; the collector emptied (its storage kept) |

- Unified memory: the zeros are one `LevelFields` launch, the items one launch over all their
  values: a thread finds its item by binary search over the items' prefix offsets (the items in
  managed memory, written by the host, never the values). Sums are atomic: tile ghost boxes
  overlap, so the order of a value's terms, and its last bits, vary run to run (as the GPU
  deposit's).
- Host memory: item by item, a span (the last dimension) at a time, in the order added (sums in
  tile order).
- `box_copies<alloc_mode, dim>()` is the calling thread's collector, kept between runs: no
  allocation once large enough. A run must not start between another user's adds and its run.

## Untiled temporaries

| Builder | Items |
|---|---|
| `owned_into(c, tmp, tiles)` | every `tmp` value from the tile owning it: its cell box, border tiles grown out to the patch ghost box (`BoxExpander`); these partition `tmp`, nothing is zeroed (`read`, `fill` leases) |
| `summed_into(c, tmp, tiles)` | `tmp` zeroed, then every tile's ghost box added (`sum` leases) |
| `into_tile_ghosts(c, tiles, tmp)` | every tile's ghost box, outer and inner ghosts, from `tmp` (release of `fill` / `sum` leases) |

The simulator's `lease_` / `release_` collect over a level's patches and run once: a launch per
lease or release of a level (two for a sum: the zeros, then the adds). Regrids lease patch by
patch (a launch per patch: occasional).
