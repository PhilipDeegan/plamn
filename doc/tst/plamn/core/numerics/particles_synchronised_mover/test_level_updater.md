# The particle updater over a level

`tst/plamn/core/numerics/particles_synchronised_mover/test_level_updater.cpp`

A level is a set of patches tiling a box, each updated as the simulator updates it: its boxing
from the level's other patches (damn's `covered_by_level`, as `Simulator::rebox`), the patch ghost
particles then handed to the patch owning their cell (periodic images included) by box overlaps,
without damn schedules. Each push, every patch split of the level must hold the particles of the
same level as one patch: the same push, the same exchanges, nothing duplicated or lost at patch
borders and corners.

- level 0: periodic, no level border
- level 1: a box inside a periodic level 1 domain, its level ghost particles in the ring around
  it. Patches meeting on the level border hold cells of the ring in common: the corner case where
  a level ghost could be held twice

Level ghosts are handled as in the simulator: each is held by one patch, its cell's owner (the
nearest, `damn::amr::owned_elsewhere`), handed over as a patch ghost when it enters another
patch's domain, and split again every 2 pushes (a coarse step).

Moments follow PHARE's deposit after the exchange (and the simulator's): after the exchange each
patch deposits its domain particles (`depositDomain`), the level's moments are then every patch's
sum on each node (the border sums, periodic images included): the same as the level as one patch,
within the summation order.

E and B are uniform: the push is the same whichever patch holds a particle, within rounding (the
interpolation weights come from patch local indexes). Every layout picks its updater
(`ParticlesSynchronisedMoverFor_t`): AoSMapped serial, AoS flat parallel, AoSCMTS tiled parallel;
on a GPU also AoSPCTS and SoAVX in unified memory, updated on the device.

Crossing particles (`CrossingTest`): every cell's 3^dim - 1 particles (2, 8, 26) at its centre,
each moving to a different neighbour cell's centre in one push, no E or B. Each patch's domain,
patch ghost and level ghost particles after a push are then known exactly, the same for every
layout and updater; after the patch ghost swap, every level cell holds one particle from each
neighbour.

Environment: `PLAMN_CELLS` (level cells per direction), `PLAMN_PPC`, `PLAMN_PUSHES`,
`PLAMN_TIMESTEP`, `PLAMN_SEED`.
