#!/usr/bin/env python3
"""PHARE vs plamn, case by case (cases.py): each case runs under two codes, then
hierarchy_overlap compares the diagnostics. Run from the plamn root:

  python3 etc/compare/phare/test_ladder.py [-k advance_L1]     # unittest options
  LADDER_REF=master LADDER_CMP=tiling_gpu python3 etc/compare/phare/test_ladder.py

A code is ``plamn`` or a PHARE checkout with a CMake build (build/pybindlibs): one of PHARES'
names below or a path. Comparing the PHARE branches with each other as well as with plamn
says whether a difference comes from plamn or from the branch plamn was ported from
(tiling_gpu).

Each case twice: PHARE's default patches ("default") and small patches ("small"), as
tiling_gpu's layouts tests isolate one mechanism per case. Tolerances per kind (B, E, moments)
as there; a failure prints the largest difference per quantity / level / time and where.

Environment (defaults in brackets):
  LADDER_REF     reference code [master]
  LADDER_CMP     compared code [plamn]
  PHARE_DIR      the PHARE checkout whose pyphare plamn runs with [/mkn/git/phare/master]
  PHARESEE_DIR   PHAREHUB/PHARE#1595 checkout, the reader [/mkn/git/phare/pr_1595]
  LADDER_NP      MPI ranks per run [1]
  LADDER_OUT     where the runs write [phare_outputs/compare]
  LADDER_FORMAT  phareh5 (fields in their centering, particles, every time read directly) or
                 pharevtkhdf (PHAREHUB/PHARE#1595's reader, particles from the data wrangler)
                 [phareh5]
  LADDER_FINE_DUMP  1: fine_dump_lvl_max too (cases.py's _diag_options) [1]
  LADDER_NDIM    the dimension (2 or 3; harris_2d is 2D only), runs in <case>_<variant>_<n>d
                 but for 2D [2]
PHARE checkouts built with PHARE_DIAG_DOUBLES=1, plamn's module with
./mkn.py --dim <n> --diag-doubles --cell-jump 1: every code writes doubles, PHARE's ghost
widths. 3D: PHARE's cpp_3_1_6 (master) / cpp_3_1_6_AoSMapped_CPU (tiling branches).
"""

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAMN = HERE.parents[2]

SMALL_PATCHES = {"harris_2d": 20}  # largest_patch_size of the "small" variant, else 7
MIN_OVERLAP = 0.9  # tagged levels (harris_2d); refinement boxes must overlap fully
TOLERANCE = {"EM_B": 1e-14, "EM_E": 1e-13, "moments": 1e-13}  # atol per kind
PHARES = {  # name: PHARE checkout with a CMake build in build/
    "master": "/mkn/git/phare/master",
    # master (dc62fdd4e) + PHAREHUB/PHARE#1583 (#1579-#1581 in it) + #1582: patch layout
    #  independent level ghosts, ordered border sums, no ion border max, level 0 B ghosts
    "next": "/mkn/git/phare/next",
    "tiling_gpu": "/mkn/git/phare/tiling_gpu",
    "tiling_multithreaded": "/mkn/git/phare/tiling_multithreaded",
}


def setting(name, default):
    return os.environ.get(name, default)


def phare_dir(code):
    return Path(PHARES.get(code, code))


def environment(code):
    """PYTHONPATH of a run_case.py process for ``code`` (plamn, or a PHARE name / path)."""
    if code == "plamn":
        paths = [PLAMN / "python", Path(setting("PHARE_DIR", PHARES["master"])) / "pyphare"]
    else:
        phare = phare_dir(code)
        if not (phare / "build" / "pybindlibs").is_dir():
            raise RuntimeError(f"{code}: no CMake build in {phare / 'build'}")
        paths = [phare / "build", phare / "pyphare"]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in paths)
    return env


def run(code, case, out, largest_patch_size, steps=None):
    n = int(setting("LADDER_NP", "1"))
    kind = "plamn" if code == "plamn" else "phare"
    cmd = [sys.executable, str(HERE / "run_case.py"), kind, case, str(out)]
    cmd += ["--ndim", str(ndim())]
    if steps is not None:
        cmd += ["--steps", str(steps)]
    if largest_patch_size:
        cmd += ["--largest-patch-size", str(largest_patch_size)]
    if setting("LADDER_FINE_DUMP", "1") == "1":
        cmd += ["--fine-dump"]
    cmd += ["--format", setting("LADDER_FORMAT", "phareh5")]
    if n > 1:
        cmd = ["mpirun", "--oversubscribe", "-n", str(n)] + cmd
    shutil.rmtree(out, ignore_errors=True)  # no files of an earlier run (another format)
    log = Path(f"{out}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        code_ = subprocess.call(cmd, env=environment(code), cwd=PLAMN, stdout=f, stderr=f)
    if code_:
        raise RuntimeError(f"{' '.join(cmd)} failed ({code_}), see {log}")


def ndim():
    return int(setting("LADDER_NDIM", "2"))


def tolerance(stem):
    return TOLERANCE.get(stem, TOLERANCE["moments"]), 0.0


class LadderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reader = Path(setting("PHARESEE_DIR", "/mkn/git/phare/pr_1595")) / "pyphare"
        sys.path.insert(0, str(reader))
        sys.path.insert(0, str(HERE))

    def compare(self, case, variant):
        import hierarchy_overlap

        largest = SMALL_PATCHES.get(case, 7) if variant == "small" else None
        ref, cmp = setting("LADDER_REF", "master"), setting("LADDER_CMP", "plamn")
        suffix = "" if ndim() == 2 else f"_{ndim()}d"
        base = Path(setting("LADDER_OUT", PLAMN / "phare_outputs" / "compare"))
        base = base / f"{case}_{variant}{suffix}"
        names = {code: Path(code).name for code in (ref, cmp)}  # a path: its last part
        for code in (ref, cmp):
            run(code, case, base / names[code], largest)
        args = dict(min_overlap=MIN_OVERLAP, margin=2, tolerance=tolerance)
        if setting("LADDER_FORMAT", "phareh5") == "phareh5":
            report = hierarchy_overlap.compare_h5_runs(base / names[ref], base / names[cmp], **args)
        else:
            report = hierarchy_overlap.compare_runs(base / names[ref], base / names[cmp], **args)
            hierarchy_overlap.compare_particles(base / names[ref], base / names[cmp], report)
        print(f"\n==== {case} ({variant} patches): {ref} vs {cmp}")
        hierarchy_overlap.print_report(report, values=False)
        self.assertTrue(report, f"{case} ({variant}): {len(report.failed)} failure(s)")

    def check(self, case):
        for variant in ("default", "small"):
            with self.subTest(variant=variant):
                self.compare(case, variant)

    def test_1_init_L0(self):
        self.check("init_L0")

    def test_2_init_L1(self):
        self.check("init_L1")

    def test_3_advance_L0(self):
        self.check("advance_L0")

    def test_4_advance_L1(self):
        self.check("advance_L1")

    def test_5_init_L2(self):
        self.check("init_L2")

    def test_5_init_L2_nested(self):
        self.check("init_L2_nested")

    def test_6_advance_L2(self):
        self.check("advance_L2")

    def test_6_advance_L2_nested(self):
        self.check("advance_L2_nested")

    def test_7_harris_2d(self):
        if ndim() != 2:
            self.skipTest("harris_2d is 2D")
        self.check("harris_2d")


if __name__ == "__main__":
    unittest.main()
