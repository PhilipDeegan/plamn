#!/usr/bin/env python3
"""The cost terms of a hybrid patch's work (plamn.amr.cost_fit): busy seconds per substep as
particles + cells + ghost cells + a constant per patch, fitted over runs of different patch sizes
(the same cells cut in patches of 5^2 .. 50^2 cells: the per patch and ghost terms apart from the
per cell one), with one population absent from the middle (particles apart from cells).

  PYTHONPATH=.:<pyphare> python3 etc/bench/amr/cost_terms.py [--ndim 2] [--steps 4]
                          [--sizes 5 10 20 50] [--ppc 90 120] [--out phare_outputs/bench/cost]
  (3D: --ndim 3 --cells 40 --sizes 5 10 20 40 --ppc 30 40, a 3D module: ./mkn.py --dim 3)

Time only -g 0 builds (./mkn.py --dim 2 -- -g 0: NDEBUG). One rank: no exchange in the busy time.
"""

import argparse
import os
from pathlib import Path

import numpy as np

def config(ph, patch, steps, ndim, cells, ppc):
    ph.global_vars.sim = None
    ph.clearDict()
    sim = ph.Simulation(time_step_nbr=steps, time_step=0.001, cells=(cells,) * ndim,
                        dl=(0.2,) * ndim, largest_patch_size=(patch,) * ndim,
                        hyper_resistivity=0.001, resistivity=0.001, strict=True)
    ly = cells * 0.2

    def const(v):
        return lambda *x: v + 0.0 * x[0]

    def middle_empty(*x):  # no particles in the middle half of y
        y = x[1]
        rho = 0.1 + 0.0 * y
        rho[(y > ly / 4) & (y < 3 * ly / 4)] = 0
        return rho

    def pop(density, ppc, seed):
        return {"charge": 1, "density": density, "nbr_part_per_cell": ppc, "init": {"seed": seed},
                **{f"vbulk{c}": const(0.0) for c in "xyz"},
                **{f"vth{c}": const(0.3) for c in "xyz"}}

    ph.MaxwellianFluidModel(bx=const(1.0), by=const(0.0), bz=const(0.0),
                            brotons=pop(const(1.0), ppc[0], 12334),
                            protons=pop(middle_empty, ppc[1], 43210))
    ph.ElectronModel(closure="isothermal", Te=0.0)
    return sim


def run(patch, steps, out, ndim, cells, ppc):
    import plamn
    import plamn.backend.phare as ph

    plamn.patch()
    path = Path(out) / f"patch_{patch}.jsonl"
    os.environ.update(PLAMN_DECOMPOSITION=str(path), PLAMN_DECOMPOSITION_EVERY="1")
    try:
        sim = config(ph, patch, steps, ndim, cells, ppc)
        simulator = ph.Simulator(sim, log_to_file=False).setup().initialize()
        for _ in range(steps):
            simulator.advance()
        simulator.reset()
    finally:
        for key in ("PLAMN_DECOMPOSITION", "PLAMN_DECOMPOSITION_EVERY"):
            os.environ.pop(key)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ndim", type=int, default=2)
    p.add_argument("--cells", type=int, default=100, help="per direction")
    p.add_argument("--ppc", type=int, nargs=2, default=[90, 120], help="the two populations'")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--sizes", type=int, nargs="+", default=[5, 10, 20, 50])
    p.add_argument("--out", default="phare_outputs/bench/cost")
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)

    from plamn.amr import cost_fit, read_decomposition

    runs = [read_decomposition(run(size, a.steps, a.out, a.ndim, a.cells, a.ppc))
            for size in a.sizes]
    terms = ("particles", "cells", "patch")
    fit = cost_fit(runs, terms=terms)
    print(f"{fit['n']} patch substeps, patch sizes {a.sizes}, r2 {fit['r2']:.4f}")
    for t in terms:
        print(f"  {t:9}: {fit[t]:.3e} s")
    print(f"  {a.ndim}D: a cell's work in particles: {fit['cells'] / fit['particles']:.1f}, a "
          f"patch's: {fit['patch'] / fit['particles']:.0f}")
    for size, records in zip(a.sizes, runs):
        f = cost_fit(records)
        print(f"  patch {size:3}: particles {f['particles']:.3e}, cells {f['cells']:.3e}, "
              f"r2 {f['r2']:.3f}")


if __name__ == "__main__":
    main()
