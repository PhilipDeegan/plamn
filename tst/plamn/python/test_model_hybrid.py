"""etc/bench/amr/model_hybrid.py against its references (etc/bench/amr/ref/<case>.json and .npz):
the decomposition emulation's partitions (levels, finest + spill, processes and threads, the chop
under finer levels), kept so the same steps in damn give the same owners and numbers, or show
where and why they differ.

Both backends (--backend python: the partitions in model_hybrid.py; cpp: damn::amr::decompose
through plamn_etc, the simulator's code) against the same references. Owners compare exactly (a
partition is deterministic); a difference names the row, the level and
the patches that moved. Times, imbalances and shares compare at a relative 1e-9 (summation order
differs between python and C++), the largest difference printed.

    python3 -m tst.plamn.python.test_model_hybrid
    PLAMN_MODEL_HYBRID_SLOW=1 ...  also the big cases (minutes)
    a reference again (after an intended change): the case's command in REFERENCES with
    --json/--owners into etc/bench/amr/ref
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "etc" / "bench" / "amr" / "model_hybrid.py"
REF = ROOT / "etc" / "bench" / "amr" / "ref"
RTOL = 1e-9

# case: (its arguments, slow), the references made with these and --json/--owners
REFERENCES = {
    # the gist weak scaling case (100^3 level 0 cells a 200 ranks, refined 45% / 48%), 800 ranks
    "gist_800": ("--cells 400 100 100 --fractions 0.45 0.48 --partition levels finest "
                 "--spill 0.05 --topologies 800:40 --threads 1 20 --locality "
                 "--patch 20 20:5/20:10/20", False),
    # harris 400x800x200, 1000 ranks: level 1 over level 0 42% -> 97% on its own rank
    "harris_1000": ("--partition finest --spill 0.05 --topologies 100:10 1000:10 --locality "
                    "--no-exchanges --patch 20 20:5/20:10/20", True),
    # the gist case at 6400 cores: one process a NUMA zone (20 threads) against one a core
    "gist_6400_threads": ("--cells 3200 100 100 --fractions 0.45 0.48 --partition levels finest "
                          "--spill 0.05 --topologies 6400:40 --threads 1 20 --locality "
                          "--no-exchanges --patch 20:5/20:10/20 40:10/40:20/40", True),
}


def run(case, out, backend):
    args = REFERENCES[case][0].split() + ["--backend", backend]
    subprocess.run([sys.executable, str(BENCH), *args, "--json", str(out / f"{case}.json"),
                    "--owners", str(out / f"{case}.npz")], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    return json.loads((out / f"{case}.json").read_text()), np.load(out / f"{case}.npz")


def numbers(row):
    """a row's floats by name (owners and counts apart)"""
    out = {"step": row["step"], "serial": row["serial"]}
    for ln, lvl in enumerate(row["levels"]):
        out[f"L{ln} imbalance"] = lvl["imbalance"]
        for group in ("ghosts", "over_coarser"):
            for k, v in lvl.get(group, {}).items():
                out[f"L{ln} {group} {k}"] = v
    return out


class ModelHybridTest(unittest.TestCase):
    def check(self, case):
        if REFERENCES[case][1] and not os.environ.get("PLAMN_MODEL_HYBRID_SLOW"):
            self.skipTest("big case: PLAMN_MODEL_HYBRID_SLOW=1")
        for backend in ("python", "cpp"):  # the reference, and damn's decompose (a run's)
            with self.subTest(backend=backend):
                self.check_backend(case, backend)

    def check_backend(self, case, backend):
        ref = json.loads((REF / f"{case}.json").read_text())
        ref_owners = np.load(REF / f"{case}.npz")
        with tempfile.TemporaryDirectory() as tmp:
            got, got_owners = run(case, Path(tmp), backend)
            rows = {r["key"]: r for r in got["rows"]}
            self.assertEqual(sorted(rows), sorted(r["key"] for r in ref["rows"]))
            problems, worst = [], (0.0, "")
            for want in ref["rows"]:
                row = rows[want["key"]]
                for ln, (w, g) in enumerate(zip(want["levels"], row["levels"])):
                    for k in ("patches", "cells", "particles"):
                        if w[k] != g[k]:
                            problems.append(f"{want['key']} L{ln} {k}: {g[k]} != {w[k]}")
                    if w["owners_sha256"] != g["owners_sha256"]:
                        name = f"{want['key']}_L{ln}"
                        a, b = got_owners[name], ref_owners[name]
                        moved = np.nonzero(a != b)[0] if a.shape == b.shape else []
                        problems.append(f"{want['key']} L{ln} owners: {len(moved)} of {len(b)} "
                                        f"patches moved (first {[int(i) for i in moved[:5]]})")
                ws, gs = numbers(want), numbers(row)
                for k, v in ws.items():
                    diff = abs(gs[k] - v) / max(abs(v), 1e-300)
                    worst = max(worst, (diff, f"{want['key']} {k}"))
                    if diff > RTOL:
                        problems.append(f"{want['key']} {k}: {gs[k]} vs {v} (rel {diff:.3e})")
            print(f"{case} ({backend}): largest relative difference {worst[0]:.3e} ({worst[1]})")
            self.assertEqual(problems, [], "\n".join(problems))

    def test_gist_800(self):
        self.check("gist_800")

    def test_harris_1000(self):
        self.check("harris_1000")

    def test_gist_6400_threads(self):
        self.check("gist_6400_threads")


if __name__ == "__main__":
    unittest.main()
