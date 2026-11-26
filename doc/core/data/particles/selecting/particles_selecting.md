# Particle selection

`inc/plamn/core/data/particles/selecting/particles_selecting.hpp`: the particles of an array whose
cells are in a box, visited in place, nothing copied.

| Call | Visits |
|---|---|
| `for_each_range_in(src, box, fn)` | `fn(start, size)` per contiguous range of selected particles |
| `for_each_particle_in(src, box, fn)` | `fn(index)` per selected particle, over the ranges |
| `particle_indexes_in(src, box)` | the indexes collected, for callers that need them kept |

Each layout selects by what it knows of its cells (`ParticlesSelector<layout, alloc>`):

| Layout | Ranges |
|---|---|
| `SoAVXChunked`, `AoS4SoAVXChunked` | their per cell ranges when sorted, else their columns read |
| `AoSMapped` | its CellMap's per cell index lists, the box's cells only |
| `SoAVX` | its columns read |

Indexes that do not come as ranges (CellMap lists, column scans) are merged into runs of
consecutive ones (`IndexRuns`), so callers copy whole ranges (a `memcpy` per range and column).

Only the layouts that move particles between patches select (`amr/resources.hpp`, see
[resources.md](../../../../amr/resources.md)): the sorted ones, and `SoAVX`. Unsorted `AoS` layouts,
used in tests and benchmarks, go through damn's default scan. Unified memory selects on the host,
as the CPU does.
