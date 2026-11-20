#!/usr/bin/env python3
"""The cost terms of a hybrid patch's work (plamn.amr.cost_fit): busy seconds per substep as
particles + cells + ghost cells + a constant per patch, fitted over runs of different patch sizes
(the same cells cut in patches of 5^2 .. 50^2 cells: the per patch and ghost terms apart from the
per cell one), with one population absent from the middle (particles apart from cells).

  PYTHONPATH=.:<pyphare> python3 etc/bench/amr/cost_terms.py [--steps 4] [--sizes 5 10 20 50]
                                                              [--out phare_outputs/bench/cost]

Time only -g 0 builds (./mkn.py --dim 2 -- -g 0: NDEBUG). One rank: no exchange in the busy time.
"""

import argparse
import os
from pathlib import Path

import numpy as np

CELLS, DL = (100, 100), (0.2, 0.2)
PPC_HIGH, PPC_LOW = 90, 120


def config(ph, patch, steps):
    ph.global_vars.sim = None
    ph.clearDict()
    sim = ph.Simulation(time_step_nbr=steps, time_step=0.001, cells=CELLS, dl=DL,
                        largest_patch_size=(patch, patch), hyper_resistivity=0.001,
                        resistivity=0.001, strict=True)

    def const(v):
        return lambda x, y: v + 0.0 * x

    def middle_empty(x, y):
        rho = 0.1 + 0.0 * y
        rho[(y > 5) & (y < 15)] = 0
        return rho

    def pop(density, ppc, seed):
        return {"charge": 1, "density": density, "nbr_part_per_cell": ppc, "init": {"seed": seed},
                **{f"vbulk{c}": const(0.0) for c in "xyz"},
                **{f"vth{c}": const(0.3) for c in "xyz"}}

    ph.MaxwellianFluidModel(bx=const(1.0), by=const(0.0), bz=const(0.0),
                            brotons=pop(const(1.0), PPC_HIGH, 12334),
                            protons=pop(middle_empty, PPC_LOW, 43210))
    ph.ElectronModel(closure="isothermal", Te=0.0)
    return sim


def run(patch, steps, out):
    import plamn
    import plamn.backend.phare as ph

    plamn.patch()
    path = Path(out) / f"patch_{patch}.jsonl"
    os.environ.update(PLAMN_DECOMPOSITION=str(path), PLAMN_DECOMPOSITION_EVERY="1")
    try:
        simulator = ph.Simulator(config(ph, patch, steps), log_to_file=False).setup().initialize()
        for _ in range(steps):
            simulator.advance()
        simulator.reset()
    finally:
        for key in ("PLAMN_DECOMPOSITION", "PLAMN_DECOMPOSITION_EVERY"):
            os.environ.pop(key)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--sizes", type=int, nargs="+", default=[5, 10, 20, 50])
    p.add_argument("--out", default="phare_outputs/bench/cost")
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)

    from plamn.amr import cost_fit, read_decomposition

    runs = [read_decomposition(run(size, a.steps, a.out)) for size in a.sizes]
    terms = ("particles", "cells", "ghost", "patch")
    fit = cost_fit(runs, terms=terms)
    print(f"{fit['n']} patch substeps, patch sizes {a.sizes}, r2 {fit['r2']:.4f}")
    for t in terms:
        print(f"  {t:9}: {fit[t]:.3e} s")
    print(f"  a cell's work in particles: {fit['cells'] / fit['particles']:.1f}, a ghost cell's: "
          f"{fit['ghost'] / fit['particles']:.1f}, a patch's: {fit['patch'] / fit['particles']:.0f}")
    for size, records in zip(a.sizes, runs):
        f = cost_fit(records)
        print(f"  patch {size:3}: particles {f['particles']:.3e}, cells {f['cells']:.3e}, "
              f"r2 {f['r2']:.3f}")


if __name__ == "__main__":
    main()
