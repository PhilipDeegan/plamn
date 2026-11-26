# Particle components

`components(particles)` in `inc/plamn/core/data/particles/particle_components.hpp` gives every
particle component as a `mkn::kul::StridedSpan` over the array's own memory, whatever the layout:

| Layout | A component is | Stride |
|---|---|---|
| `AoS`, `AoSMapped` | a member of `Particle` | `sizeof(Particle)` |
| `SoA` | a component of a member's array | `sizeof(std::array<., n>)` |
| `SoAVX`, `SoAVXChunked` | a column | contiguous |

This is what the DataWrangler hands python (live numpy views) and what restarts write and read in
place: neither copies, nor needs to know the layout.

`for_each` visits the components in file order: `weight`, `iCell_x..`, `delta_x..`, `v_x`, `v_y`,
`v_z`.

## Limits

The arrays must be host readable (`HasComponents`): host memory, or unified memory for the SoAVX
columns, read between kernels. Tiled layouts (one span per tile) are not supported yet.
