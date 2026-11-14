#!/usr/bin/env python3
"""Builds plamn's python simulator modules: one per build permutation, from the py_sim profile

    ./mkn.py                              # 1D, interp 1: plamn_cpp_1_1_2_AoSMapped_CPU
    ./mkn.py --dim 1 2 --interp 1 2       # every combination
    ./mkn.py --dim 2 --refined 5          # a refined particle number other than the default
    ./mkn.py --dim 2 -- -x res/mkn/mpi    # anything after -- goes to mkn

The permutation is set with compile definitions (PLAMN_SIM_*, see src/python3/plamn_sim.cpp)
and the module named as pyplamn looks for it: plamn_cpp_<dim>_<interp>_<refined>_<layout>_<alloc>,
pyphare's simulator id. Run from the plamn root; py_etc is built once first unless --no-etc.
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


def has_name_option():
    """mkn's -N/--name (build directory bin/<name> for the profile): newer than some mkn builds"""
    try:
        return "-N/--name" in subprocess.run(["mkn", "-h"], capture_output=True, text=True).stdout
    except OSError:
        return False


def mkn(profile, extra, defines=(), out=None, own_dir=False):
    # mkn rebuilds by timestamps, not compile arguments: each permutation builds in its own
    # directory (-N), or, with an mkn without -N, in a cleaned bin/<profile> (ccache keeps it
    # cheap)
    cmd = ["mkn"]
    if own_dir and not has_name_option():
        cmd += ["clean"]
    cmd += ["build", "-p", profile, "-dKO", "-t", "8"]
    if own_dir and has_name_option():
        cmd += ["-N", out]
    if out:
        cmd += ["-o", out]
    if defines:
        cmd += ["-a", " ".join(defines)]
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
    parser.add_argument("--layout", nargs="+", default=["AoSMapped"], choices=LAYOUTS)
    parser.add_argument("--allocator", nargs="+", default=["CPU"], choices=ALLOCATORS)
    parser.add_argument("--no-etc", action="store_true", help="don't build py_etc")
    if "--" in argv:
        at = argv.index("--")
        argv, extra = argv[:at], argv[at + 1 :]
    else:
        extra = []
    args = parser.parse_args(argv)

    if not args.no_etc and mkn("py_etc", extra):
        return 1

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
        ]
        if mkn("py_sim", extra, defines, out=name, own_dir=True):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
