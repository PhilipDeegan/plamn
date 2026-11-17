---
name: project-damn-phare-rewrite
description: "DAMN = Distributed Adaptive Mesh Numerics; from-scratch PHARE rewrite replacing SAMRAI, in eq at inc/damn, src/damn, tst/damn"
metadata:
  node_type: memory
  type: project
  originSessionId: 1ff89a38-7b05-4995-a1de-373db8ea6831
  modified: 2026-11-13T14:23:23.626Z
---

Started 2026-11-13. DAMN (Distributed Adaptive Mesh Numerics) is Philip's from-scratch rewrite of PHARE that replaces SAMRAI. It lives in the eq repo under inc/damn, src/damn and tst/damn, with includes written as "damn/..." and "tst/damn/...".

Seeded by copying, unchanged apart from include paths, the include closure of UsableIons (tests/core/data/ion_population/test_ion_population_fixtures.hpp) plus the updater and partitioning tests from ~/git/phare/mkn (branch tiling_gpu, the copy Philip wants as the source). The plan is to copy everything first and decide the final shape afterwards.

The local eq AMR (BCC, cell-tree, odd27) was reviewed and judged not reusable as a box/patch base; only patterns carry over (the CellSlab slab allocator, the Morton key, the phlop macros, HighFive streaming). For the design direction discussed for memory, see [[project-damn-memory-design]].

Module layout: damn_core (mkn profile; test_damn_core, run as `build test -p test_damn_core -dKO -t 8`, static libs during development). There will be no "amr" module; the hierarchy module will be called "hier".

Status 2026-11-13 (Philip made a separate DAMN repo; port it there next session):
- Done in eq: the PHARE changes of then merged in (an MPI-only one not copied). utilities/ split into geometry, meta, math, memory, exec, sys. ParticleArray renamed to Particles. Resolver simplified. GridLayout now forwards to the free functions in data/grid/layout.hpp, plus coordinates.hpp. Quantity DAO (models/quantities/quantity.hpp). Faraday, Ampere, Ohm ported. Expression layer (data/field/field_expr.hpp). Faraday is an expression, bit-identical to the old kernel. res/mkn GPU settings (clang_cuda works on the RTX 5080). CI step runs test_damn_core.
- Next: Ampere then Ohm as expressions; centering from Quantity at call sites; delete GridLayout; reference tests need pyphare-generated files (not yet in mkn).
- Open GPU issue: fixed a buf_count race in Interpolating::on_tiles_push_deposit (the same unfixed code is in ~/git/phare/mkn tiling_gpu, interpolating.hpp:636-644). compute-sanitizer racecheck still flags part_buf write (line 625) vs dep reads, and there are device asserts at ndarray_base.hpp:63 only under the sanitizer. Unresolved.

**Why:** this scopes future work on inc/damn and explains why it does not follow eq/DST conventions.
**How to apply:** the PHARE source is ~/git/phare/mkn, not stage or master. External dependencies come from ~/git/phare/mkn/subprojects (menum, mkn/gpu, mkn/kul, googletest) and are not yet vendored into tpp/ or wired into mkn.yaml.
