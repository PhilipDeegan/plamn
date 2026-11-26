# Particles as counts per cell

`plamn/counts.py` stands in for a run's particles with one count per cell: a cheap model to
assess regrids and partitions over many configurations (memory: a number per cell, not ~250 B
per particle).

```python
c = Counts.from_density(cells, dl, density, ppc)  # pyphare's loading: ppc where the density
                                                  #  is at or above the cut-off
c.advance(dt, vbulk, vth)                         # the particles move: counts follow
c.box_sum(lower, upper)                           # the particles in a box (O(1))
c.refined(boxes, ratio, refined_nbr)              # a refined level's counts
tags = c.tags(threshold)                          # a density gradient tagging
```

## `advance`

Each particle moves by `vbulk dt` plus a thermal displacement (a Gaussian of `vth dt` per
direction, a Maxwellian's), from a uniform position in its cell. A cell's count is dealt to the
cells it can reach with the probability of landing in each, as sequential binomials: the counts
stay integers and the total is exact. It goes direction by direction, each particle's
displacements being independent.

### Numerical diffusion

The counts' variance grows by `(vth dt / dx)^2 + 1/6` cells^2 per step. The `1/6` is the
cell's own discretization (where in the cell a particle is): a numerical diffusion of counting by
cells.

### No velocities

Counts keep no velocities: one `advance(T)` spreads a blob ballistically (`vth T`, as particles
streaming freely), while `n` advances of `T / n` make a random walk (`vth T / sqrt(n)`). Advance
once per regrid interval.

### No fields

The counts drift and spread; they do not feel B or E (a Harris sheet diffuses, it does not
reconnect). Regrids follow the particles, not the physics.
