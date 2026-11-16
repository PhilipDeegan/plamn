#!/usr/bin/env python3
"""PHARE vs plamn, case by case (cases.py): each case runs under both codes, then
hierarchy_overlap compares the diagnostics (PHARE the reference). Run from the plamn root:

  python3 etc/compare/phare/test_ladder.py [-k advance_L1]     # unittest options

Each case twice: PHARE's default patches ("default") and small patches ("small"), as
tiling_gpu's layouts tests isolate one mechanism per case. Tolerances per kind (B, E, moments)
as there; a failure prints the largest difference per quantity / level / time and where.

Environment (defaults in brackets):
  PHARE_DIR      PHARE checkout built with PHARE_DIAG_DOUBLES=1 [/mkn/git/phare/master]
  PHARESEE_DIR   PHAREHUB/PHARE#1595 checkout, the reader [/mkn/git/phare/pr_1595]
  LADDER_NP      MPI ranks per run [1]
  LADDER_OUT     where the runs write [phare_outputs/compare]
plamn's 2D module must be built with ./mkn.py --dim 2 --diag-doubles (PHARE writes doubles).
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAMN = HERE.parents[2]

SMALL_PATCHES = {"harris_2d": 20}  # largest_patch_size of the "small" variant, else 7
MIN_OVERLAP = 0.9  # tagged levels (harris_2d); refinement boxes must overlap fully
TOLERANCE = {"EM_B": 1e-14, "EM_E": 1e-13, "moments": 1e-13}  # atol per kind


def setting(name, default):
    return os.environ.get(name, default)


def environment(code):
    """PYTHONPATH of a run_case.py process for ``code``."""
    phare = Path(setting("PHARE_DIR", "/mkn/git/phare/master"))
    paths = [phare / "build", phare / "pyphare"] if code == "phare" else []
    if code == "plamn":
        paths = [PLAMN / "python", phare / "pyphare"]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in paths)
    return env


def run(code, case, out, largest_patch_size):
    n = int(setting("LADDER_NP", "1"))
    cmd = [sys.executable, str(HERE / "run_case.py"), code, case, str(out)]
    if largest_patch_size:
        cmd += ["--largest-patch-size", str(largest_patch_size)]
    if n > 1:
        cmd = ["mpirun", "--oversubscribe", "-n", str(n)] + cmd
    log = Path(f"{out}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        code_ = subprocess.call(cmd, env=environment(code), cwd=PLAMN, stdout=f, stderr=f)
    if code_:
        raise RuntimeError(f"{' '.join(cmd)} failed ({code_}), see {log}")


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
        base = Path(setting("LADDER_OUT", PLAMN / "phare_outputs" / "compare")) / f"{case}_{variant}"
        for code in ("phare", "plamn"):
            run(code, case, base / code, largest)
        report = hierarchy_overlap.compare_runs(
            base / "phare", base / "plamn", min_overlap=MIN_OVERLAP, margin=2,
            tolerance=tolerance)
        print(f"\n==== {case} ({variant} patches)")
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

    def test_5_advance_L2(self):
        self.check("advance_L2")

    def test_6_harris_2d(self):
        self.check("harris_2d")


if __name__ == "__main__":
    unittest.main()
