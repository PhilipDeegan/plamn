"""3D hybrid, 10^3 cells, 100 ppc, one level, a drifting plasma (cells and patches crossed): a
layout / allocator run, timed; B, E, the density and the particle count saved to compare runs

    PYTHONPATH=.:$PHARE/pyphare python3 etc/bench/gpu/uniform_3d.py SoAVXChunked GPU_UNIFIED 20 out.npz
"""
import sys
import time
import numpy as np


def main(layout, allocator, steps, out):
    import plamn.backend.phare as ph

    sys.argv = [sys.argv[0]]
    const = lambda v: (lambda x, y, z: v + 0.0 * x)
    ph.Simulation(time_step=0.001, time_step_nbr=steps, cells=(10, 10, 10), dl=(0.2, 0.2, 0.2),
                  interp_order=1, refinement_boxes={}, hyper_resistivity=0.0,
                  resistivity=0.0, model_layout=layout, allocator=allocator)
    ph.MaxwellianFluidModel(
        bx=const(1.0), by=const(0.1), bz=const(0.0),
        protons={"charge": 1, "mass": 1, "nbr_part_per_cell": 100, "init": {"seed": 1},
                 "density": const(1.0), "vbulkx": const(0.5), "vbulky": const(0.0),
                 "vbulkz": const(0.0), "vthx": const(0.3), "vthy": const(0.3), "vthz": const(0.3)})
    ph.ElectronModel(closure="isothermal", Te=0.1)
    sim = ph.Simulator(ph.global_vars.sim, log_to_file=False).setup().initialize()
    c = sim.cpp_sim
    n0 = c.particle_count(0)
    t0 = time.time()
    for _ in range(steps):
        sim.advance()
    dt = time.time() - t0
    np.savez(out, count=c.particle_count(0), density=np.asarray(c.level_values("protons_density", 0)),
             bx=np.asarray(c.level_values("EM_B_x", 0)), ey=np.asarray(c.level_values("EM_E_y", 0)))
    print(f"{layout} {allocator}: {steps} steps {dt:.2f} s, particles {n0} -> {c.particle_count(0)}")
    sim.reset()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
