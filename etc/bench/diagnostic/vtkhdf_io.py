#!/usr/bin/env python3
"""pharevtkhdf diagnostics IO: is the write saturating, and what does contention cost?

    python3 etc/bench/diagnostic/vtkhdf_io.py                     # the sweep, V-cache cores (0-7)
    python3 etc/bench/diagnostic/vtkhdf_io.py --cores 0-15        # every physical core
    python3 etc/bench/diagnostic/vtkhdf_io.py --ranks 1 4 --pools 1 2 --cells 256 --dumps 3
    python3 etc/bench/diagnostic/vtkhdf_io.py --subfiles 0 2 4 --stripes 1 4 --ioc-threads 1 4
        # MPI-IO (0) and subfiling (n subfiles, stripe MiB, threads per I/O concentrator): MPI and
        # HDF5 with subfiling, source /opt/hpc/env.sh, a module built with -x res/mkn/hpc

A 2D run (two populations, a refinement box), its EM and fluid diagnostics dumped --dumps times
without advancing (the same data each dump: only the dump is measured), for each ranks x thread
pools with ranks * pools <= the cores given (physical cores only: no SMT siblings, pinned with
taskset; mpirun --bind-to none inside them). Per configuration, per dump (the median):

    MB         bytes written by all ranks
    dump ms    the whole dump (barriers around it), MB/s from it
    convert    to primal, into the level buffers (phlop scope Simulator::dump_vtkhdf_::convert,
    write      HDF5's resizes and the slab writes  ...::write), the slowest rank's
    write MB/s MB over the slowest rank's write
    posix MB/s each rank writing its share to its own file (numpy, no HDF5): what the page
               cache takes without HDF5 or one shared file
    ctxsw      context switches per dump, voluntary / involuntary, summed over ranks (getrusage:
               blocking IO waits / preemption, contention shows as the latter)
    close ms   closing the files still open after the last dump (flush_every 0), the simulator going
    fsync ms   the files flushed to the device after the last dump (rank 0): the dumps above
               land in the page cache

Run from the plamn root, the 2D module built (./mkn.py --dim 2, with PHARE_SCOPE_TIMING's
scopes: the default). PYPHARE: the PHARE checkout pyphare comes from. --momentum adds the
momentum tensors (a deposit and a border sum: not IO). Results: --json writes them
"""

import argparse
import json
import os
import resource
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PYPHARE = Path(os.environ.get("PYPHARE", Path.home() / "git/phare/mkn"))


def cores_of(spec):
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


# one configuration, on every rank of an mpirun ---------------------------------------------


def simulation(args, out):
    import numpy as np
    import pyphare.pharein as ph

    ph.global_vars.sim = None
    ph.clearDict()
    sim = ph.Simulation(
        time_step_nbr=args.dumps,  # the timestamps within the run; it never advances
        time_step=0.001,
        cells=[args.cells] * 2,
        dl=[0.3] * 2,
        hyper_resistivity=0.001,
        largest_patch_size=args.patch,
        smallest_patch_size=args.patch // 2,
        refinement_boxes={
            "L0": {
                "B0": [
                    [args.cells // 4] * 2,
                    [args.cells // 4 + args.cells // 4 - 1] * 2,
                ]
            }
        },
        diag_options={
            "format": "pharevtkhdf",
            "options": {"dir": out, "mode": "overwrite"},
        },
    )

    def wave(x, y):
        L = sim.simulation_domain()
        return np.sin(2 * np.pi * x / L[0]) * np.cos(2 * np.pi * y / L[1])

    def const(v):
        return lambda x, y: v + 0 * x

    def pop(mass, vx, seed):
        return {
            "mass": mass,
            "charge": 1.0,
            "density": lambda x, y: 1.0 + 0.2 * wave(x, y),
            "vbulkx": const(vx),
            "vbulky": const(0.1),
            "vbulkz": const(0.0),
            **{f"vth{c}": const(0.3) for c in "xyz"},
            "nbr_part_per_cell": args.ppc,
            "init": {"seed": seed},
        }

    ph.MaxwellianFluidModel(
        bx=lambda x, y: 1.0 + 0.1 * wave(x, y),
        by=const(0.2),
        bz=const(0.5),
        protons=pop(1.0, 0.5, 1),
        alpha=pop(4.0, 1.0, 2),
    )
    ph.ElectronModel(closure="isothermal", Te=0.12)
    timestamps = [k * 0.001 for k in range(args.dumps)]
    fluid = ["charge_density", "mass_density", "bulkVelocity", "charge_flux"]
    pop_q = ["density", "charge_density", "flux"]
    if args.momentum:
        fluid.append("momentum_tensor")
        pop_q.append("momentum_tensor")
    for q in ["E", "B"]:
        ph.ElectromagDiagnostics(
            quantity=q, write_timestamps=timestamps, flush_every=args.flush_every
        )
    for q in fluid:
        ph.FluidDiagnostics(
            quantity=q, write_timestamps=timestamps, flush_every=args.flush_every
        )
    for name in ["protons", "alpha"]:
        for q in pop_q:
            ph.FluidDiagnostics(
                quantity=q,
                write_timestamps=timestamps,
                population_name=name,
                flush_every=args.flush_every,
            )
    return sim


def scope_ms(path, key):
    """the scope key's total run time in ms, every occurrence, in a phlop timing file: "#" header
    lines, "<id> <name>" lines, a blank line, then "<depth> <id> <start>:<run ns>" lines (parsed
    here: an older phlop's scope_timer does not take the header)"""
    names, total, body = {}, 0, False
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            continue
        if not line.strip():
            body = True
            continue
        bits = line.split()
        if not body:
            names[int(bits[0])] = bits[1]
        elif names.get(int(bits[1])) == key:
            total += int(bits[2].split(":")[1])
    return total / 1e6


def run(args):
    """one configuration: every rank dumps, writes its numbers, rank 0 gathers them"""
    import numpy as np
    import plamn

    plamn.patch()
    from pyphare.simulator.simulator import Simulator, startMPI

    startMPI()
    etc = plamn.etc
    rank, size = etc.mpi_rank(), etc.mpi_size()
    out = "diags"

    simulator = Simulator(simulation(args, out), log_to_file=False).setup().initialize()
    cpp = simulator.cpp_sim
    dumps = []
    for k in range(args.dumps):
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        etc.mpi_barrier()
        t0 = time.perf_counter()
        cpp.dump_diagnostics(k * 0.001, 0.001)
        etc.mpi_barrier()
        t1 = time.perf_counter()
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        dumps.append(
            dict(
                ms=(t1 - t0) * 1e3,
                nvcsw=r1.ru_nvcsw - r0.ru_nvcsw,
                nivcsw=r1.ru_nivcsw - r0.ru_nivcsw,
            )
        )

    # bytes: every file's size after the last dump, over the dumps (rank 0)
    etc.mpi_barrier()
    # the simulator gone: files still open (flush_every 0, or not yet due) closed, timed, the
    #  scope timing files written
    del cpp  # the last reference to the C++ simulator but pyphare's: reset() destroys it
    etc.mpi_barrier()
    t0 = time.perf_counter()
    simulator.reset()
    etc.mpi_barrier()
    close_ms = (time.perf_counter() - t0) * 1e3

    # the driver asked for is the one used: subfiling leaves subfiles (a module without it ignores
    #  DAMN_H5_SUB_FILES)
    if (n := int(os.environ.get("DAMN_H5_SUB_FILES", "0"))) and not any(
        Path(out).rglob(f"*.subfile_*_of_{n}")
    ):
        raise RuntimeError(f"DAMN_H5_SUB_FILES={n} but no files in {n} subfiles")
    # every file: subfiling writes subfiles and a config beside each .vtkhdf
    mb = (
        sum(
            p.stat().st_blocks * 512 for p in Path(out).rglob("*") if p.is_file()
        )  # allocated: subfiles hold unwritten stripe space
        / 1e6
        / args.dumps
    )

    # the posix baseline: this rank's share of a dump, to its own file
    share = np.ones(int(mb * 1e6 / size / 4), dtype=np.float32)
    etc.mpi_barrier()
    t0 = time.perf_counter()
    share.tofile(f"posix.{rank}.bin")
    etc.mpi_barrier()
    posix_s = time.perf_counter() - t0

    fsync_ms = 0.0
    if rank == 0:
        t0 = time.perf_counter()
        for p in [f for f in Path(out).rglob("*") if f.is_file()] + list(
            Path(".").glob("posix.*.bin")
        ):
            fd = os.open(p, os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)
        fsync_ms = (time.perf_counter() - t0) * 1e3

    Path(f"rank.{rank}.json").write_text(json.dumps(dict(dumps=dumps)))
    etc.mpi_barrier()
    if rank != 0:
        return

    ranks = [json.loads(Path(f"rank.{r}.json").read_text()) for r in range(size)]
    per_dump = [
        dict(
            ms=max(r["dumps"][k]["ms"] for r in ranks),
            nvcsw=sum(r["dumps"][k]["nvcsw"] for r in ranks),
            nivcsw=sum(r["dumps"][k]["nivcsw"] for r in ranks),
        )
        for k in range(args.dumps)
    ]
    med = lambda key: statistics.median(d[key] for d in per_dump)  # noqa: E731
    result = dict(
        ranks=size,
        pools=int(os.environ.get("PHARE_THREAD_POOLS", "1")),
        mb=mb,
        dump_ms=med("ms"),
        posix_mbs=mb / posix_s,
        nvcsw=med("nvcsw"),
        nivcsw=med("nivcsw"),
        fsync_ms=fsync_ms,
        close_ms=close_ms,
    )
    print("RESULT " + json.dumps(result), flush=True)


# the sweep, outside MPI -----------------------------------------------------------------------


def sweep(args):
    cores = cores_of(args.cores)
    # (subfiles, stripe MiB, IOC threads): 0 subfiles is MPI-IO, the others only for subfiling
    drivers = [
        (n, mib, ioc)
        for n in args.subfiles
        for mib in (args.stripes if n else [0])
        for ioc in (args.ioc_threads if n else [0])
    ]
    configs = [
        (d, r, p)
        for d in drivers
        for r in args.ranks
        for p in args.pools
        if r * p <= len(cores)
    ]
    work = Path(args.work).resolve()
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(ROOT), str(PYPHARE / "pyphare")]),
        PHARE_SCOPE_TIMING="1",
        OMP_NUM_THREADS="1",
    )
    results = []
    print(
        f"cores {args.cores}, {args.cells}^2 cells, patches {args.patch}, {args.dumps} dumps, flush_every {args.flush_every}"
    )
    head = f"{'subfiles':>8} {'stripe':>6} {'ioc':>3} {'ranks':>5} {'pools':>5} {'MB':>7} {'dump ms':>8} {'MB/s':>7} {'convert':>8} {'write':>7} {'write MB/s':>10} {'posix MB/s':>10} {'ctxsw v/i':>11} {'fsync ms':>8} {'close ms':>8}"
    print(head)
    for (subfiles, mib, ioc), ranks, pools in configs:
        d = work / f"sf{subfiles}_s{mib}_i{ioc}_r{ranks}_p{pools}"
        driver_env = dict(DAMN_H5_SUB_FILES=str(subfiles))
        if subfiles:
            driver_env.update(
                H5FD_SUBFILING_STRIPE_SIZE=str(mib << 20),
                H5FD_IOC_THREAD_POOL_SIZE=str(ioc),
            )
        label = (
            f"{subfiles:>8} {mib if subfiles else '-':>6} {ioc if subfiles else '-':>3}"
        )
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        cmd = ["taskset", "-c", ",".join(map(str, cores)), "mpirun", "-n", str(ranks)]
        cmd += [
            "--bind-to",
            "none",
            sys.executable,
            str(Path(__file__).resolve()),
            "run",
        ]
        cmd += [
            f"--{k}={v}"
            for k, v in dict(
                cells=args.cells, patch=args.patch, ppc=args.ppc, dumps=args.dumps
            ).items()
        ]
        cmd += ["--momentum"] if args.momentum else []
        cmd += [f"--flush-every={args.flush_every}"]
        p = subprocess.run(
            cmd,
            cwd=d,
            env=dict(env, PHARE_THREAD_POOLS=str(pools), **driver_env),
            capture_output=True,
            text=True,
        )
        lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if p.returncode or not lines:
            print(
                f"{label} {ranks:>5} {pools:>5}  failed:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}"
            )
            continue
        r = json.loads(lines[-1][7:])
        r.update(subfiles=subfiles, stripe_mib=mib, ioc_threads=ioc)
        # the scopes, the slowest rank's (the timing files are written as the simulators go)
        timings = [d / f".phare/timings/rank.{k}.txt" for k in range(ranks)]
        for key, name in [("convert_ms", "convert"), ("write_ms", "write")]:
            r[key] = (
                max(scope_ms(t, f"Simulator::dump_vtkhdf_::{name}") for t in timings)
                / args.dumps
                if all(t.exists() for t in timings)
                else 0.0
            )
        results.append(r)
        mbs = r["mb"] / (r["dump_ms"] / 1e3)
        wmbs = r["mb"] / (r["write_ms"] / 1e3) if r["write_ms"] else float("nan")
        print(
            f"{label} {ranks:>5} {pools:>5} {r['mb']:>7.1f} {r['dump_ms']:>8.1f} {mbs:>7.0f} {r['convert_ms']:>8.1f}"
            f" {r['write_ms']:>7.1f} {wmbs:>10.0f} {r['posix_mbs']:>10.0f}"
            f" {r['nvcsw']:>5.0f}/{r['nivcsw']:<5.0f} {r['fsync_ms']:>8.1f} {r['close_ms']:>8.1f}",
            flush=True,
        )
        if not args.keep:
            shutil.rmtree(d, ignore_errors=True)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))


def main():
    a = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    a.add_argument("mode", nargs="?", default="sweep", choices=["sweep", "run"])
    a.add_argument(
        "--cores", default="0-7", help="physical cores (default: the V-cache CCD)"
    )
    a.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8])
    a.add_argument("--pools", type=int, nargs="+", default=[1, 2, 4])
    a.add_argument(
        "--flush-every", type=int, default=1, help="dumps per file open (0: never)"
    )
    a.add_argument(
        "--subfiles", type=int, nargs="+", default=[0], help="0: MPI-IO, n: subfiling"
    )
    a.add_argument(
        "--stripes", type=int, nargs="+", default=[32], help="subfiling stripe MiB"
    )
    a.add_argument(
        "--ioc-threads", type=int, nargs="+", default=[4], help="per concentrator"
    )
    a.add_argument("--cells", type=int, default=512)
    a.add_argument("--patch", type=int, default=64)
    a.add_argument(
        "--ppc", type=int, default=2, help="particles per cell (the moments' source)"
    )
    a.add_argument("--dumps", type=int, default=5)
    a.add_argument("--momentum", action="store_true")
    a.add_argument("--work", default="phare_outputs/bench_vtkhdf_io")
    a.add_argument("--json")
    a.add_argument(
        "--keep", action="store_true", help="keep each configuration's files"
    )
    args = a.parse_args()
    (run if args.mode == "run" else sweep)(args)


if __name__ == "__main__":
    main()
