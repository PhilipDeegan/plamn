# Conventions

damn's [CONVENTIONS.md](https://github.com/PhilipDeegan/damn/blob/master/CONVENTIONS.md) applies
here as written: comments, code and gates. In short, for comments:

- a comment says what the code cannot: why, edge cases, subtle footguns, workarounds,
  optimization strategies; terse docstrings when they say more than the signature
- no narration of the next lines, no lists of users or callers, no history ("previously",
  "now", "fixed by"): that belongs in the commit and the PR
- long explanations go in `doc/` as markdown (SVGs welcome), with a one-line pointer in the
  code. `doc/` mirrors the source tree: `inc/plamn/simulator/simulator.hpp` is explained in
  `doc/simulator/simulator.md`, `etc/bench/harris/harris.py` in `doc/etc/bench/harris/harris.md`

and for memory and threads:

- code that runs every step does not allocate: buffers are owned by what is worked on, per
  thread (`damn::core::scratch`, a `thread_local damn::MinimizingVector`), or fixed size batches
- every buffer has one owner, an object or a thread; no false sharing between threads

## plamn's own

- model components (closures, ...) are owned by the model, and operations run on whole levels;
  patches hold storage and views
- intermediate field values go in a patch's temporaries (its extra fields, untiled temporaries,
  scratch grids), allocated with the patch; a patch is worked on by one thread at a time, so its
  temporaries need no lock and are never shared between threads
- a model level's advance is `plamn::solver::advance(model, level, ...)`, no `*TimeStepper` /
  `*Integrator` types
- derived quantities are computed on demand, only on the levels and steps that read them
- restarts store what the next step reads first; the rest is rebuilt on load
- "tiles" are threading tiles; clustering's boxes are tag boxes
- defaults may differ from the reference code for efficiency; reference-identical behaviour is
  a selected mode
- long-term cases and tools live in `etc/experiments/<area>/`, `etc/bench/`, `etc/compare/` or
  `tst/`
- python tests have no import side effects: code runs in test cases or under `__main__`
