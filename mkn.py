#!/usr/bin/env python3
"""Builds plamn's python simulator modules: one per build permutation, from the py_sim profile

    ./mkn.py                              # 1D, interp 1: plamn_cpp_1_1_2_AoSMapped_CPU
    ./mkn.py --dim 1 2 --interp 1 2       # every combination
    ./mkn.py --dim 2 --refined 5          # a refined particle number other than the default
    ./mkn.py --dim 2 -- -W                # anything after -- goes to mkn
    ./mkn.py --scope-flags 1=time,mem,perf_all 2=time,perf_all   # phlop per scope metrics
    ./mkn.py --sync-pools                 # damn's thread pools run tasks on the calling thread
    ./mkn.py --dim 2 --diag-doubles       # diagnostics in double precision, as PHARE_DIAG_DOUBLES=1
    ./mkn.py --dim 2 --cell-jump 1        # PHARE's ghost widths (particles may move 1 cell a step)
    ./mkn.py --mhd                        # MHD only, 1D: plamn_cpp_1_AoS_CPU_Linear_VanLeer_Rusanov_false
    ./mkn.py --mhd --limiter VanLeer MinMod   # every combination of the MHD choices

The permutation is set with compile definitions (PLAMN_SIM_*, see src/python3/plamn_sim.cpp)
and the module named as plamn looks for it: plamn_cpp_<dim>_<interp>_<refined>_<layout>_<alloc>,
pyphare's simulator id; MHD only (--mhd, no hybrid: interp 0), pyphare's MHD id
plamn_cpp_<dim>_<layout>_<alloc>_<reconstruction>_<limiter>_<riemann>_<hall> (the time
integration scheme is a run time choice). Run from the plamn root; py_etc is built once first unless --no-etc.
The modules use MPI: built with res/mkn/mpi's settings (mpicxx) unless -x is given after --.

Scopes (DAMN_LOG_SCOPE(level, "name")) up to --log-level (default 2) are phlop scopes; what each
level records is chosen at build time with --scope-flags <level>=<any of time,mem,perf_light,
perf_all,io> (default: time), and at run time: PHARE_SCOPE_TIMING=1 writes .phare/timings/rank.<r>.txt,
PHLOP_SCOPE_MEM=1 adds memory, PHLOP_SCOPE_IO=1 the thread's I/O (/proc/thread-self/io),
PHLOP_PERF_LIGHT / PHLOP_PERF_ALL hardware counters (see
damn/core/def/phlop.hpp); python3 -m phlop.timing.scope_timer times|perf|variance reads them.
Hardware counters count the calling thread: --sync-pools (DAMN_THREAD_POOL_SYNC) for serial
numbers that include the pools' work.
"""

import argparse
import itertools
import shlex
import subprocess
import sys

# pyphare's valid refined particle numbers, the first being the default
# (pyphare/pharein/simulation.py valid_refined_particle_nbr)
VALID_REFINED = {
    1: {1: [2, 3], 2: [2, 3, 4], 3: [2, 3, 4, 5]},
    2: {1: [4, 5, 8, 9], 2: [4, 5, 8, 9, 16], 3: [4, 5, 8, 9, 25]},
    3: {
        1: [6, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 21, 26, 27],
        2: [6, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 21, 26, 27, 64],
        3: [6, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 21, 26, 27, 125],
    },
}
# plamn::core::LayoutMode (core/data/particles/particles_def.hpp)
LAYOUTS = ["AoS", "AoSMapped", "AoSCMTS", "SoA", "AoSTS", "AoSPC", "SoAVX", "SoATS", "SoAVXTS",
           "SoAPC", "AoSPCTS", "SoAVXChunked"]
ALLOCATORS = ["CPU", "GPU_UNIFIED"]


def module_name(dim, interp, refined, layout, allocator):
    return f"plamn_cpp_{dim}_{interp}_{refined}_{layout}_{allocator}"


def mhd_module_name(dim, layout, allocator, reconstruction, limiter, riemann, hall=False):
    """pyphare's simulator id of an MHD only simulation (pyphare/cpp/__init__.py)"""
    return (
        f"plamn_cpp_{dim}_{layout}_{allocator}_{reconstruction}_{limiter}_{riemann}_"
        + ("true" if hall else "false")
    )


# the MHD choices ported (plamn::solver::mhd::reconstruction static_asserts the rest)
RECONSTRUCTIONS = ["Constant", "Linear", "WENO3", "WENOZ", "MP5"]
LIMITERS = ["VanLeer", "MinMod", "None"]  # None: the non linear reconstructions
RIEMANNS = ["Rusanov", "HLL"]


def has_name_option():
    """mkn's -N/--name (build directory bin/<name> for the profile): newer than some mkn builds"""
    try:
        return "-N/--name" in subprocess.run(["mkn", "-h"], capture_output=True, text=True).stdout
    except OSError:
        return False


def mkn(profile, extra, jobs, defines=(), out=None, own_dir=False):
    # mkn rebuilds by timestamps, not compile arguments: each permutation builds in its own
    # directory (-N), or, with an mkn without -N, in a cleaned bin/<profile> (ccache keeps it
    # cheap)
    cmd = ["mkn"]
    if own_dir and not has_name_option():
        cmd += ["clean"]
    cmd += ["build", "-p", profile, "-dKO", "-t", str(jobs)]
    if own_dir and has_name_option():
        cmd += ["-N", out]
    if out:
        cmd += ["-o", out]
    # mkn takes one -a: extra's (e.g. warning flags after --) merged into the defines'
    extra, args = list(extra), list(defines)
    for flag in ("-a", "--arg"):
        while flag in extra:
            at = extra.index(flag)
            args.append(extra[at + 1])
            del extra[at : at + 2]
    if args:
        cmd += ["-a", " ".join(args)]
    cmd += extra
    print("+", shlex.join(cmd), flush=True)
    return subprocess.call(cmd)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dim", type=int, nargs="+", default=[1], choices=[1, 2, 3])
    parser.add_argument("--interp", type=int, nargs="+", default=[1], choices=[1, 2, 3])
    parser.add_argument(
        "--refined", type=int, default=None, help="default: pyphare's first valid one"
    )
    parser.add_argument(
        "--layout", nargs="+", choices=LAYOUTS, help="default AoSMapped; --mhd: AoS (pyphare's)"
    )
    parser.add_argument("--allocator", nargs="+", default=["CPU"], choices=ALLOCATORS)
    parser.add_argument("--mhd", action="store_true", help="MHD only modules (no hybrid)")
    parser.add_argument("--reconstruction", nargs="+", default=["Linear"], choices=RECONSTRUCTIONS)
    parser.add_argument("--limiter", nargs="+", default=["VanLeer"], choices=LIMITERS)
    parser.add_argument("--riemann", nargs="+", default=["Rusanov"], choices=RIEMANNS)
    parser.add_argument("--no-etc", action="store_true", help="don't build py_etc")
    parser.add_argument("--jobs", type=int, default=8, help="mkn -t (CI: 4)")
    parser.add_argument(
        "--scope-print",
        action="store_true",
        help="DAMN_LOG_SCOPE_PRINT: scopes print their names (PHARE_LOG=RANK_FILES: a file per rank), to compare ranks for deadlocks",
    )
    parser.add_argument(
        "--sync-pools", action="store_true", help="DAMN_THREAD_POOL_SYNC: no pool threads"
    )
    parser.add_argument("--log-level", type=int, choices=[0, 1, 2, 3], help="DAMN_LOG_LEVEL")
    parser.add_argument(
        "--cell-jump",
        type=int,
        help="PLAMN_PARTICLE_ALLOWED_CELL_JUMP (default 2; 1: PHARE's ghost widths)",
    )
    parser.add_argument(
        "--diag-doubles",
        action="store_true",
        help="PLAMN_DIAG_DOUBLES: diagnostics in double precision (PHARE_DIAG_DOUBLES)",
    )
    parser.add_argument(
        "--scope-flags",
        nargs="+",
        default=[],
        metavar="LEVEL=FLAGS",
        help="phlop scope flags per log level, e.g. 1=time,mem,perf_all,io",
    )
    if "--" in argv:
        at = argv.index("--")
        argv, extra = argv[:at], argv[at + 1 :]
    else:
        extra = []
    args = parser.parse_args(argv)
    if args.layout is None:  # pyphare's: an MHD only simulation's model_layout is AoS
        args.layout = ["AoS"] if args.mhd else ["AoSMapped"]
    if "-x" not in extra:
        extra += ["-x", "res/mkn/mpi"]

    scoping = [] if args.log_level is None else [f"-DDAMN_LOG_LEVEL={args.log_level}"]
    if args.scope_print:  # DAMN_FORCE_LOG_LINE: printed in NDEBUG (-g 0) builds too
        scoping += ["-DDAMN_LOG_SCOPE_PRINT", "-DDAMN_FORCE_LOG_LINE"]
    if args.sync_pools:
        scoping += ["-DDAMN_THREAD_POOL_SYNC=1"]
    if args.diag_doubles:
        scoping += ["-DPLAMN_DIAG_DOUBLES=1"]
    if args.cell_jump is not None:
        scoping += [f"-DPLAMN_PARTICLE_ALLOWED_CELL_JUMP={args.cell_jump}"]
    for flags in args.scope_flags:
        level, _, keys = flags.partition("=")
        valid = {"time", "mem", "perf_light", "perf_all", "io"}
        if level not in ("1", "2", "3") or not keys or not set(keys.split(",")) <= valid:
            print(f"--scope-flags {flags}: <1|2|3>=<any of {','.join(sorted(valid))}>")
            return 1
        scoping += [f"-DDAMN_LOG_SCOPE_FLAGS_{level}={keys}"]

    if not args.no_etc and mkn("py_etc", extra, args.jobs):
        return 1

    if args.mhd:
        for dim, layout, allocator, reconstruction, limiter, riemann in itertools.product(
            args.dim, args.layout, args.allocator, args.reconstruction, args.limiter, args.riemann
        ):
            name = mhd_module_name(dim, layout, allocator, reconstruction, limiter, riemann)
            defines = [
                f"-DPLAMN_SIM_DIM={dim}",
                "-DPLAMN_SIM_INTERP=0",
                "-DPLAMN_SIM_REFINED=0",
                f"-DPLAMN_SIM_LAYOUT={layout}",
                f"-DPLAMN_SIM_ALLOCATOR={allocator}",
                f"-DPLAMN_SIM_RECONSTRUCTION={reconstruction}",
                f"-DPLAMN_SIM_LIMITER={limiter}",
                f"-DPLAMN_SIM_RIEMANN={riemann}",
                f"-DPLAMN_SIM_MODULE={name}",
            ] + scoping
            if mkn("py_sim", extra, args.jobs, defines, out=name, own_dir=True):
                return 1
        return 0

    for dim, interp, layout, allocator in itertools.product(
        args.dim, args.interp, args.layout, args.allocator
    ):
        valid = VALID_REFINED[dim][interp]
        refined = args.refined if args.refined is not None else valid[0]
        if refined not in valid:
            print(f"refined {refined} is not valid for {dim}D interp {interp}: {valid}")
            return 1
        name = module_name(dim, interp, refined, layout, allocator)
        defines = [
            f"-DPLAMN_SIM_DIM={dim}",
            f"-DPLAMN_SIM_INTERP={interp}",
            f"-DPLAMN_SIM_REFINED={refined}",
            f"-DPLAMN_SIM_LAYOUT={layout}",
            f"-DPLAMN_SIM_ALLOCATOR={allocator}",
            f"-DPLAMN_SIM_MODULE={name}",
        ] + scoping
        if mkn("py_sim", extra, args.jobs, defines, out=name, own_dir=True):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
