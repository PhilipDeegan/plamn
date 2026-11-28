# Physical boundaries

`inc/plamn/core/boundary/boundary.hpp` holds a model's physical boundaries
(`Boundary`, `BoundaryFactory`, `BoundaryManager`), as values.

## Boundary

A `Boundary` is its type, its location and a `FieldCondition` per quantity. `make_boundary`
makes one from its type (and, for an inflow, its state):

| Type | `ρ`, `Etot` | `ρV` | `B` | `E` |
|---|---|---|---|---|
| open | Neumann | Neumann | div free, transverse Neumann | - |
| reflective | Neumann | symmetric | div free, transverse Neumann | antisymmetric |
| inflow | Dirichlet `ρ0`; `Etot` Dirichlet (from `P`, `γ`) | Dirichlet `ρ0 v` | div free, transverse Dirichlet `B0` | Dirichlet `-v x B` |

An inflow is super magnetofast, with constant extrapolation. Other quantities have no condition.
Reflective, open and inflow are MHD's only.

## BoundaryManager

A `BoundaryManager` holds the domain's boundaries, one per face of a non periodic direction, from
the input's `simulation/grid/domain_boundaries`.

An edge or a corner takes the conditions of one adjacent face, chosen by `Priority`:

| Priority | Face |
|---|---|
| `ByDirection` (default) | the adjacent face of the highest direction |
| `ByBoundaryType` | the adjacent face of the highest boundary type |
