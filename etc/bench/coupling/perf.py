"""coupled run timing (MHD under hybrid): L0 MHD 1000 cells dl 0.2, L1 hybrid over
cells 400-599, 100 ppc, Te 0.5, dt 0.001, 2000 steps. Prints the advance loop's wall time.

Local only, not in CI (CI checks correctness). Time -g 0 builds; SoAVXChunked with mkn.avx:
    ./mkn.py --coupled --dim 1 --layout SoAVXChunked -- -g 0 -w mkn.avx
then from the plamn root (profile: perf record -e cycles:u -F 2000 before python3):
    PYTHONPATH=.:$PHARE/pyphare:tst/plamn/python \\
      python3 etc/bench/coupling/perf.py --layout SoAVXChunked
"""

import argparse
import sys
import time


def const(v):
    return lambda x: v + 0.0 * x


def main(argv):
    import plamn.backend.phare as ph

    p = argparse.ArgumentParser()
    p.add_argument("--layout", default="AoSMapped")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--ppc", type=int, default=100)
    a = p.parse_args(argv)

    ph.Simulation(
        time_step=0.001,
        time_step_nbr=a.steps,
        cells=(1000,),
        dl=(0.2,),
        refinement_boxes={"L0": {"B0": [(400,), (599,)]}},
        hyper_resistivity=0.0,
        resistivity=0.0,
        eta=0.0,
        nu=0.0,
        gamma=5.0 / 3.0,
        hall=False,
        max_mhd_level=1,
        reconstruction="Linear",
        limiter="VanLeer",
        riemann="Rusanov",
        mhd_timestepper="TVDRK2",
        model_options=["MHDModel", "HybridModel"],
        model_layout=a.layout,
    )
    ph.MHDModel(
        density=const(1.0), vx=const(0.0), vy=const(0.0), vz=const(0.0),
        bx=const(1.0), by=const(0.0), bz=const(0.0), p=const(1.0),
        protons={"charge": 1, "mass": 1, "nbr_part_per_cell": a.ppc, "init": {"seed": 1}},
    )
    ph.ElectronModel(closure="isothermal", Te=0.5)

    simulator = ph.Simulator(ph.global_vars.sim, log_to_file=False).setup().initialize()
    cpp = simulator.cpp_sim
    t0 = time.time()
    for _ in range(a.steps):
        simulator.advance()
    dt = time.time() - t0
    print(f"{a.layout}: {a.steps} steps {dt:.1f} s, L1 particles {cpp.particle_count(1)}",
          flush=True)
    simulator.reset()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
