# Field boundary conditions

`inc/plamn/core/numerics/boundary_condition/field_boundary_condition.hpp` fills a field's ghosts
past a physical boundary of the domain from its domain values, as PHARE's
`core/numerics/boundary_condition` does.

A condition is applied on a box of ghost cells (local index space) beyond the boundary at
`location`. Each field is filled over the nodes of its own centering there.

## Mirrors

A ghost's mirror is the node mirrored across the boundary:

- across the boundary node for a primal direction,
- across the boundary face for a dual direction.

## Conditions

| Condition | Effect |
|---|---|
| `Neumann` | zero gradient: a ghost takes its mirror's value |
| `Dirichlet` | the value `v` at the boundary: `v` (constant extrapolation, and on a boundary node), else `2 v - mirror` (linear) |
| `Symmetric` | a scalar: Neumann; a vector: its normal component Dirichlet 0 (linear), its tangential ones Neumann |
| `AntiSymmetric` | a scalar: Dirichlet 0 (linear); a vector: its normal component Neumann, its tangential ones Dirichlet 0 |
| `DivergenceFreeTransverseNeumann` / `DivergenceFreeTransverseDirichlet` | B: its tangential components Neumann (or Dirichlet at the condition's values), then its normal one set face by face outward so that each ghost cell's discrete divergence is zero |

## Divergence free normal component

Given the tangential components in the ghost cells, the normal component is set face by face
away from the domain, starting from the boundary face (which belongs to the domain):

$$
B_n(\text{next face}) = B_n(\text{face}) - \Delta n \sum_{t \neq n} \frac{\partial B_t}{\partial t}
$$

The faces nearest the domain go first, so each face reads one already set.

## FieldCondition

A `FieldCondition` holds the condition type, Dirichlet's values (one per component) and its
extrapolation. It is plain data, since many boundaries carry one per quantity.
