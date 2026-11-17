#!/usr/bin/env python3
"""SoAVXChunked's gaps (particles_soavx_chunked.hpp) against a sort on every change of cell:
harris 2D or 3D (the comparison ladder's, etc/compare/phare/cases.py: tagging, 3 levels), no
diagnostics, one process (one rank, one thread) per run, each slack (PLAMN_CHUNK_SLACK, 0: no
gaps, the post push sync and the exchange sort each time) in turn, repeated. --vbulk: the
protons drift along x (about along B: little v x B), every particle's cell changing the same way

  PYTHONPATH=python:<pyphare> python3 etc/bench/particles/chunk_gaps.py \\
      [--case harris_2d|harris_3d] [--vbulk 0] [--steps 20] [--ppc 100] \\
      [--slacks 0,.05,.1,.2] [--repeat 2]

Per run: the steps' wall time and, from the scope timers (PHARE_SCOPE_TIMING=1), the time of
the scopes the gaps change: the post push sync (resettle, or the sync then a sort), the
exchange's arrivals (copy) and SoAVXChunked's sorts (and their number). Needs the SoAVXChunked
module of the dimension (./mkn.py --dim 2 --layout SoAVXChunked).
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

# the scopes reported (substrings of their keys), their times summed over the run
SCOPES = {
    "sync": "FlatParallelParticlesSynchronisedMover::post_move_sync",
    "on_moved": "FlatParallelParticlesSynchronisedMover::on_moved",
    "resettle": "SoAVXChunked::resettle",
    "sort": "SoAVXChunked::sort",
}


def child(case, vbulk, steps, ppc):
    os.environ["LADDER_PPC"] = str(ppc)
    sys.path.insert(0, str(ROOT / "etc" / "compare" / "phare"))
    import cases

    cases.PPC = ppc
    cases._diagnostics = lambda *args, **kwargs: None  # timing the steps, not the dumps

    import pyplamn

    pyplamn.patch()
    from pyphare.simulator.simulator import Simulator, startMPI

    import pyphare.pharein as ph

    if vbulk:
        model = ph.MaxwellianFluidModel

        def drifting(**kwargs):
            kwargs["protons"]["vbulkx"] = lambda *xyz: 0.0 * xyz[0] + vbulk
            return model(**kwargs)

        ph.MaxwellianFluidModel = drifting

    startMPI()
    cases.CASES[case]["steps"] = steps
    sim = cases.simulation(case, out=os.getcwd())
    sim.model_layout = "SoAVXChunked"
    simulator = Simulator(sim, log_to_file=False).setup().initialize()
    start = time.perf_counter()
    for _ in range(steps):
        simulator.advance()
    seconds = time.perf_counter() - start
    levels = simulator.cpp_sim.n_levels()
    particles = sum(simulator.cpp_sim.particle_count(ln) for ln in range(levels))
    simulator.reset()  # writes the scope timers
    print(f"RESULT {seconds} {levels} {particles}")


def scope_times(path):
    """{scope: seconds} summed over every call, from a phlop scope timer file (ns)."""
    from phlop.timing import scope_timer

    lines = [l for l in Path(path).read_text().splitlines(True) if not l.startswith("#")]
    timers = scope_timer.lines_parser(lines)  # without damn's '#clock' header
    totals = {name: 0.0 for name in SCOPES}
    totals["sorts"] = 0
    stack = list(timers.roots)
    while stack:
        node = stack.pop()
        stack.extend(node.c)
        key = timers(node.k)
        for name, scope in SCOPES.items():
            if key.endswith(scope):
                totals[name] += node.t / 1e9
                totals["sorts"] += name == "sort"
    return totals


def run(slack, args):
    with tempfile.TemporaryDirectory() as tmp:
        paths = [str(Path(p).resolve()) for p in os.environ.get("PYTHONPATH", "").split(":") if p]
        env = dict(os.environ, PLAMN_CHUNK_SLACK=str(slack), PHARE_SCOPE_TIMING="1",
                   PYTHONPATH=":".join([str(ROOT / "python")] + paths))  # run in tmp
        env.pop("PHARE_THREAD_POOLS", None)
        env.pop("PHARE_THREADS_PER_POOL", None)
        out = subprocess.run(
            [sys.executable, __file__, "--child", "--case", args.case, "--vbulk",
             str(args.vbulk), "--steps", str(args.steps), "--ppc", str(args.ppc)],
            cwd=tmp, env=env, capture_output=True, text=True,
        )
        if out.returncode:
            sys.exit(out.stdout[-3000:] + out.stderr[-3000:])
        line = next(l for l in out.stdout.splitlines() if l.startswith("RESULT"))
        seconds, levels, particles = line.split()[1:]
        timings = Path(tmp) / ".phare" / "timings" / "rank.0.txt"
        scopes = scope_times(timings) if timings.exists() else {}
        return float(seconds), int(levels), int(particles), scopes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="harris_2d", choices=["harris_2d", "harris_3d"])
    parser.add_argument("--vbulk", type=float, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--ppc", type=int, default=100)
    parser.add_argument("--slacks", default="0,.05,.1,.2")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        return child(args.case, args.vbulk, args.steps, args.ppc)

    print(f"{args.case}, vbulk x {args.vbulk}, {args.ppc} ppc, {args.steps} steps, 1 rank, 1 thread")
    print(f"{'slack':>6} {'steps s':>8} {'levels':>6} {'particles':>10} "
          + " ".join(f"{name:>9}" for name in SCOPES) + f" {'sorts':>6}")
    for _ in range(args.repeat):
        for slack in [float(s) for s in args.slacks.split(",")]:
            seconds, levels, particles, scopes = run(slack, args)
            print(f"{slack:>6} {seconds:>8.2f} {levels:>6} {particles:>10} "
                  + " ".join(f"{scopes.get(name, float('nan')):>9.3f}" for name in SCOPES)
                  + f" {scopes.get('sorts', 0):>6}", flush=True)


if __name__ == "__main__":
    main()
