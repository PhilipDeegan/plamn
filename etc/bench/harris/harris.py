#!/usr/bin/env python3
"""One coarse step of PHARE tiling_gpu's harris (tests/simulator/layouts/test_harris_{2,3}d_layouts.py)
on each code, timed: wall clock of the setup + initialization and of the step, and the scope
timers (PHARE_SCOPE_TIMING=1, .phare/timings/rank.<r>.txt in each code's directory).

  etc/bench/harris/harris.py [--codes master pr_1601 plamn] [--ndim 2] [--levels 3]
                             [--steps 1] [--ppc N] [--out DIR] [--tiling-gpu DIR]
                             [--plamn-layouts AoSMapped SoAVXChunked] [--np N]
                             [--no-scope-timing] [--env K=V ...] [--label L] [--bind B]
                             [--exec FILE]

The runs, the input and reading the timers: doc/etc/bench/harris/harris.md
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAMN = HERE.parents[2]
PHARES = Path("/mkn/git/phare")
TILING_GPU = PHARES / "tiling_gpu"
CODES = ["master", "pr_1601", "plamn"]


def load_exec_module():
    """plamn/exec.py by path: importing the plamn package loads its C++ module"""
    spec = importlib.util.spec_from_file_location("plamn_exec", PLAMN / "plamn" / "exec.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_plamn(code):
    return code.startswith("plamn")


def pythonpath(code):
    if is_plamn(code):
        return [PLAMN, PHARES / "master" / "pyphare"]
    root = PHARES / code
    if not (root / "build" / "pybindlibs").is_dir():
        raise RuntimeError(f"{code}: no CMake build in {root / 'build'}")
    return [root / "build", root / "pyphare"]


def load_config(path):
    """tiling_gpu's test module, its tests.* imports stubbed: only config() is used"""
    for name, attrs in {
        "tests": {},
        "tests.diagnostic": {"dump_all_diags": lambda *a, **k: None},
        "tests.simulator": {"SimulatorTest": object},
        "tests.simulator.layouts": {"compare_hierarchies": None},
    }.items():
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod
    spec = importlib.util.spec_from_file_location("harris_layouts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.dump_all_diags = lambda *a, **k: None  # no diagnostics
    return module


def simulation_without_unknown_keywords(ph, overrides, dropped):
    """ph.Simulation, minus the keywords this pyphare rejects (master: "invalid arguments - k1 k2")"""
    Simulation = ph.Simulation

    def make(**kwargs):
        kwargs.update(overrides)
        kwargs.pop("diag_options", None)
        while True:
            try:
                return Simulation(**kwargs)
            except ValueError as e:
                marker = "invalid arguments - "
                if marker not in str(e):
                    raise
                ph.global_vars.sim = None
                for key in str(e).split(marker, 1)[1].split():
                    kwargs.pop(key, None)
                    dropped.append(key)

    return make


def particles_per_level(simulator, pop="protons"):
    """domain particles per level, from the data wrangler (the ladder's reader, both codes)"""
    sys.path.insert(0, str(PLAMN / "etc/compare/phare"))
    from run_case import domain_particles

    dw = simulator.data_wrangler()
    counts = []
    for ilvl in range(dw.getNumberOfLevels()):
        got = domain_particles(dw, ilvl, pop)
        counts.append(0 if got is None else len(got[3]))
    return counts


def child(args):
    """runs in the code's own process and cwd: one coarse step, timed"""
    # PHARE makes the timings directory on rank 0 only: on another host (--exec) it is missing
    Path(".phare/timings").mkdir(parents=True, exist_ok=True)
    if is_plamn(args.code):
        import plamn

        plamn.patch()
    import pyphare.pharein as ph
    from pyphare.simulator.simulator import Simulator, startMPI

    startMPI()
    module = load_config(
        Path(args.tiling_gpu) / "tests/simulator/layouts" / f"test_harris_{args.ndim}d_layouts.py"
    )
    module.final_time = module.time_step * args.steps
    if args.ppc:
        module.ppc = args.ppc
    dropped = []
    # config() gets a pyphare.pharein of its own: the simulator's isinstance(sim, ph.Simulation)
    #  needs the real class
    proxy = types.ModuleType("pharein_for_config")
    proxy.__dict__.update(ph.__dict__)
    proxy.Simulation = simulation_without_unknown_keywords(
        ph, {"max_nbr_levels": args.levels}, dropped
    )
    module.ph = proxy
    ph.global_vars.sim = None
    layout = args.code.split("_", 1)[1] if is_plamn(args.code) else "AoSMapped"
    sim = module.config(layout)[0]
    if is_plamn(args.code):
        # PHARE master's pyphare (plamn's) rejects model_layout, dropped above: plamn picks its
        #  module from the attribute
        sim.model_layout = layout

    t0 = time.perf_counter()
    simulator = Simulator(sim, log_to_file=False).setup().initialize()
    init_s = time.perf_counter() - t0
    # under mpirun (--np): rank 0 writes the result, no particle counts (the wrangler gathers
    #  every particle on rank 0)
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
    n_ranks = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
    initial = particles_per_level(simulator) if n_ranks == 1 else None  # outside the timed regions
    t0 = time.perf_counter()
    for _ in range(args.steps):
        simulator.advance()
    steps_s = time.perf_counter() - t0
    final = particles_per_level(simulator) if n_ranks == 1 else None
    levels = simulator.data_wrangler().getNumberOfLevels()
    loaded = type(simulator.cpp_sim).__module__ if is_plamn(args.code) else "-"
    if args.plots:
        plot_fields(simulator, Path(args.plots), args.code)
    simulator.reset()
    result = {
        "code": args.code,
        "module": loaded,
        "init_s": init_s,
        "steps_s": steps_s,
        "steps": args.steps,
        "levels": levels,
        "particles_initial": initial,
        "particles_final": final,
        "ppc": module.ppc,
        "cells": list(module.cells),
        "dropped": sorted(set(dropped)),
        "ranks": n_ranks,
    }
    if rank == 0:
        Path("result.json").write_text(json.dumps(result))
    return 0


def plot_fields(simulator, out, code):
    """B's components and the ion density at the end, every level (pharesee's plot2d), from the
    live hierarchy (hierarchy_from_sim): <out>/<code>_<quantity>.png, 2D runs"""
    import matplotlib

    matplotlib.use("Agg")
    from pyphare.pharesee.hierarchy.fromsim import hierarchy_from_sim

    out.mkdir(parents=True, exist_ok=True)
    for qty in ("EM_B", "density"):
        hier = hierarchy_from_sim(simulator, qty=qty)
        for q in hier.quantities():
            hier.plot2d(qty=q, filename=str(out / f"{code}_{q}.png"), plot_patches=True)


def perf_env():
    """phlop's per scope hardware counters (PHLOP_PERF_LIGHT, PHLOP_PERF_CTX): exact counts per
    scope, no sampling. The codes' perf_light scopes count with them (plamn: mkn.py --scope-flags
    <lvl>=time,perf_light; a PHARE build with per level scope flags: -DPHARE_LOG_SCOPE_FLAGS_<lvl>=...)"""
    out = subprocess.run(
        [sys.executable, "-m", "phlop.app.pfm.stat", "--env", "perf_light=light"],
        env=dict(os.environ, PYTHONPATH=str(PLAMN / "tpp/damn/tpp/phlop")),
        capture_output=True,
        text=True,
    ).stdout
    env = {}
    for line in out.splitlines():
        if line.startswith("export ") and "=" in line:
            key, value = line[len("export ") :].split("=", 1)
            env[key] = value.strip("'\"")
    return env


def parent(args):
    out = Path(args.out).resolve()
    rows = []
    counters = perf_env()
    print("hardware counters per scope:", counters.get("PHLOP_PERF_LIGHT", "none"))
    codes = [
        c
        for code in args.codes
        for c in ([f"plamn_{lay}" for lay in args.plamn_layouts] if code == "plamn" else [code])
    ]
    extra = dict(kv.split("=", 1) for kv in args.env)
    if args.exec:  # the ranks over the exec file's hosts
        exec_file = load_exec_module()
        spec = exec_file.load(args.exec)
        args.np = args.np or exec_file.total_ranks(spec)
    args.np = args.np or 1
    for code in codes:
        name = f"{code}_{args.label}" if args.label else code
        cwd = out / name
        cwd.mkdir(parents=True, exist_ok=True)
        timing = {"PHARE_SCOPE_TIMING": "1", **counters} if args.scope_timing else {}
        env = dict(os.environ, **timing, **extra)
        env["PYTHONPATH"] = os.pathsep.join(str(p) for p in pythonpath(code))
        if args.exec:
            env_names = ["PYTHONPATH", *timing, *extra]
            mpi = exec_file.mpirun(spec, cwd.resolve(), env_names, args.np, args.bind)
        else:
            bind = {"core": ["--bind-to", "core"], "none": ["--bind-to", "none"]}.get(
                args.bind or "core", ["--map-by", args.bind, "--bind-to", "core"]
            )
            mpi = ["mpirun", "-n", str(args.np)] + bind if args.np > 1 else []
        cmd = mpi + [sys.executable, str(Path(__file__).resolve()), "--child", code]
        cmd += ["--ndim", str(args.ndim), "--levels", str(args.levels), "--steps", str(args.steps)]
        cmd += ["--plots", str(Path(args.plots).resolve())] if args.plots else []
        cmd += ["--tiling-gpu", str(args.tiling_gpu)] + (
            ["--ppc", str(args.ppc)] if args.ppc else []
        )
        log = cwd / "run.log"
        print(f"{code}: running in {cwd}", flush=True)
        with open(log, "w") as f:
            if subprocess.call(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT):
                print(f"{code}: failed, see {log}")
                continue
        rows.append(dict(json.loads((cwd / "result.json").read_text()), code=name))

    print(
        f"\nharris {args.ndim}D, {args.steps} coarse step(s), max {args.levels} levels, {args.np} rank(s)"
    )
    print(f"{'code':20} {'levels':>6} {'ppc':>4} {'init s':>9} {'step(s) s':>10}  dropped keywords")
    for r in rows:
        print(
            f"{r['code']:20} module {r.get('module', '-')}, particles per level, initial {r['particles_initial']} final {r['particles_final']}"
        )
    for r in rows:
        print(
            f"{r['code']:20} {r['levels']:>6} {r['ppc']:>4} {r['init_s']:>9.3f} {r['steps_s']:>10.3f}  "
            f"{' '.join(r['dropped']) or '-'}"
        )
    phlop = f"PYTHONPATH={PLAMN / 'tpp/damn/tpp/phlop'} python3 -m phlop.timing"
    timers = {r["code"]: out / r["code"] / ".phare/timings/rank.0.txt" for r in rows}
    print("\nscope timers (phlop, perf -v: the counts per scope):")
    for code, path in timers.items():
        print(f"  {phlop}.scope_timer times {path}")
        print(f"  {phlop}.scope_timer perf -v {path}")
    # scope_compare matches scopes by path: PHARE codes against each other only (plamn's scopes
    #  have other names below the roots)
    phare = [code for code in timers if not is_plamn(code)]
    for code in phare[1:]:
        print(f"  {phlop}.scope_compare {timers[phare[0]]} {timers[code]}")
    return 0 if len(rows) == len(codes) else 1


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--codes", nargs="+", default=CODES, help="PHARE dirs under /mkn/git/phare, or plamn"
    )
    p.add_argument("--ndim", type=int, default=2, choices=[2, 3])
    p.add_argument("--levels", type=int, default=3, help="max_nbr_levels")
    p.add_argument("--steps", type=int, default=1, help="coarse steps")
    p.add_argument(
        "--plots", default=None, help="a PNG per quantity per code here at the end (2D, 1 rank)"
    )
    p.add_argument(
        "--np",
        type=int,
        default=None,
        help="MPI ranks per code (mpirun -n, a rank per core: no taskset); default 1, or the exec file's total",
    )
    p.add_argument(
        "--exec",
        default=os.environ.get("PLAMN_EXEC"),
        help="exec file (plamn/exec.py): the hosts the ranks go to (default $PLAMN_EXEC)",
    )
    p.add_argument(
        "--plamn-layouts", nargs="+", default=["AoSMapped"], help="plamn runs once per layout"
    )
    p.add_argument("--ppc", type=int, default=None, help="particles per cell (default: the test's)")
    p.add_argument("--out", default="bench_out/harris", help="one directory per code below it")
    p.add_argument(
        "--tiling-gpu", default=str(TILING_GPU), help="checkout holding the harris tests"
    )
    p.add_argument(
        "--no-scope-timing", dest="scope_timing", action="store_false", help="wall clock only"
    )
    p.add_argument("--env", nargs="*", default=[], help="K=V for the codes' environment")
    p.add_argument("--label", default=None, help="the run's directory: <code>_<label>")
    p.add_argument(
        "--bind", default=None, help="core (default), none, or an mpirun --map-by spec; overrides the exec file's"
    )
    p.add_argument("--child", dest="code", default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    return child(args) if args.code else parent(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
