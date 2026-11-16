#!/usr/bin/env python3
"""Runs one case of cases.py under one code, diagnostics into OUT.

  run_case.py phare|plamn CASE OUT [--ndim 2] [--largest-patch-size N]

A case with steps runs them all; an init case only initializes (dumps t=0). One process per
code: PHARE's and plamn's python modules both register as pyphare's simulator, so the caller
sets PYTHONPATH (see test_ladder.py's environment()).
"""

import argparse
import sys
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("code", choices=["phare", "plamn"])
    p.add_argument("case")
    p.add_argument("out")
    p.add_argument("--ndim", type=int, default=2)
    p.add_argument("--largest-patch-size", type=int, default=None)
    a = p.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).parent))
    import cases

    if a.code == "plamn":
        import pyplamn

        pyplamn.patch()
    from pyphare.simulator.simulator import Simulator, startMPI

    startMPI()
    sim = cases.simulation(a.case, a.out, a.ndim, a.largest_patch_size)
    simulator = Simulator(sim, log_to_file=False).setup().initialize()
    if cases.CASES[a.case]["steps"]:
        simulator.run()
    simulator.reset()
    return 0


if __name__ == "__main__":
    sys.exit(main())
