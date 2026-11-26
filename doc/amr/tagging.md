# Tagging

`inc/plamn/amr/tagging.hpp` picks the cells of a level to refine from its fields, after PHARE's
tagging.

1. A quantity (B, or one component of it) is read at cell centres (`cell_centred`): each primal
   direction is averaged over the cell's two nodes.
2. A criterion gives one indicator per cell, from the quantity's components around it.
3. A cell is tagged if any quantity's indicator is above that quantity's threshold.

## Criteria

A criterion has `indicator(components, cells, cell, dx)` and a `reach`: the cells it reads on
either side.

| Criterion | Indicator | Reach |
|---|---|---|
| `DefaultCriterion` (PHARE's default), 1D | per component, `\|avg(c-1..c+3) - avg(c-2..c+2)\| / (1 + \|avg(c-2..c+2)\|)`; the L2 norm over the components | 3 |
| `DefaultCriterion`, 2D / 3D | the max over components and directions of `\|F(c+2e) - F(c)\| / (1 + \|F(c+e) - F(c)\|)` | 2 |
| `LohnerCriterion` | Lohner's estimator, as PLUTO's (Mignone et al. 2012, eq. 50-51) | 1 |

`LohnerCriterion` takes a vector as a whole (norms over its components). Along each direction,
with `a-`, `a0`, `a+` the values at `c - e`, `c`, `c + e`:

```
n = |a+ - 2 a0 + a-|
D = |a+ - a0| + |a0 - a-| + reltol (|a-| + 2 |a0| + |a+|) + abstol
indicator = sqrt(sum_d n^2 / sum_d D^2)
```

## Input

`tagging_from(dict, method_key)` reads the input's `simulation/AMR/refinement/tagging` node:

| Key | What |
|---|---|
| `<method_key>` | a model's method (`hybrid_method`, `mhd_method`, as PHARE's TaggerFactory): `"default"` (the default) or `"lohner-per-direction"` |
| `nbr_quantities`, `Q<i>/name`, `Q<i>/threshold` | the quantities to tag on: `B`, `Bx`, `By` or `Bz` |
| `threshold` | without `nbr_quantities`: B at this threshold (default 0.1) |
| `params/reltol`, `params/abstol` | Lohner's tolerances (default 0.01, 1e-8) |

Quantity names are model free: they map to `B_x`.. resource names (an MHD level's); a hybrid
level's resources are `EM_B_x`.., which the caller's resource lookup resolves.
