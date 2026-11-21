"""reference samples of tst/plamn/python/test_coupled.py's uniform plasma at high ppc (little
noise): level 1's density per node and particle count at the sample steps, for MHD under hybrid
(coupled) and hybrid under hybrid, averaged over seeds. The test compares its low ppc runs with
them. Too heavy for CI: made locally, the JSON checked in (etc/bench/coupling/ref).

from the plamn root, with py_etc, the 1D hybrid and the coupled modules built:
    PYTHONPATH=.:$PHARE/pyphare:tst/plamn/python \\
      python3 etc/bench/coupling/uniform.py --ppc 10000 --seeds 1 2 \\
      --out etc/bench/coupling/ref/uniform.json
"""

import argparse
import json
import sys
import time


def main(argv):
    import numpy as np

    import test_coupled as tc

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppc", type=int, default=10000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--out", required=True)
    a = parser.parse_args(argv)

    ref = {
        "ppc": a.ppc,
        "seeds": a.seeds,
        "steps": tc.STEPS,
        "samples": list(tc.SAMPLES),
        "dt": tc.DT,
        "te": tc.TE,
    }
    for name, coupled in (("coupled", True), ("hybrid", False)):
        runs = []
        for seed in a.seeds:
            t0 = time.time()
            runs.append(tc.run(coupled, seed, ppc=a.ppc))
            print(f"{name} seed {seed}: {time.time() - t0:.0f} s", flush=True)
        ref[name] = {
            str(step): {
                "density": np.mean([r["samples"][step]["density"] for r in runs], axis=0).tolist(),
                "count": float(np.mean([r["samples"][step]["count"] for r in runs])),
            }
            for step in tc.SAMPLES
        }
    with open(a.out, "w") as f:
        json.dump(ref, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
