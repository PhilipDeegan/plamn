# Particle accessors for damn's copies and refines

`inc/plamn/amr/particle_access.hpp` gives damn's particle copies and refines (damn's `copy.hpp`,
`inter_level.hpp`) a patch's particle arrays through accessors rather than by their names. An
accessor is:

- a population: its index in the patch's populations (the ions' order, the same on every patch);
- one of its arrays: a member of the patch's `PopulationData`, as a template argument.

The names (`array_name`) are still what damn plans with.

## Arrays by role

Each array has a role, enforced by concepts: using an array in the wrong role does not compile.

| Role | Arrays | Notes |
|---|---|---|
| sources | `domain` (the coarse side of splits), `patch_ghost` (`gather_leaving`) | read only: their resources hold a `Particles const*` |
| destinations | `domain` (`gather_leaving`, interior splits), `level_ghost_old` (init / regrid / restart splits), `level_ghost_new` (a coarse step's splits) | appended to |
| `level_ghost` | neither | never in a schedule: it is made from the old and new arrays |

## Schedules

| Type | What |
|---|---|
| `ParticleCopies<Patch, src, dst>` | `damn::amr::copy`'s pairs (names, for the plans) and accessors, one per population |
| `ParticleRefines<Patch, coarse, fine>` | `damn::amr::refine`'s, with the same `Patch` on the coarse and fine sides |
| `ParticleRoutes<Patch>` | a hybrid model's schedules, by role, made once per model from its population names |

A refine's `coarse_new` is its `coarse`: particles have no time interpolation.
