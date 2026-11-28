"""a small 1D periodic Alfven wave"""

import numpy as np


def config(cells=50, time_step_nbr=10, final_time=0.1):
    import plamn.backend.phare as ph

    sim = ph.Simulation(
        smallest_patch_size=cells,
        largest_patch_size=cells,
        time_step_nbr=time_step_nbr,
        final_time=final_time,
        cells=cells,
        dl=1,
        hyper_resistivity=0.001,
    )

    L = sim.simulation_domain()[0]

    def density(x):
        return 1.0

    def bx(x):
        return 1.0

    def by(x):
        return 0.01 * np.cos(2 * np.pi * x / L)

    def bz(x):
        return 0.01 * np.sin(2 * np.pi * x / L)

    def vx(x):
        return 0.0

    def vy(x):
        return 0.01 * np.cos(2 * np.pi * x / L)

    def vz(x):
        return 0.01 * np.sin(2 * np.pi * x / L)

    def vth(x):
        return 0.01

    ph.MaxwellianFluidModel(
        bx=bx,
        by=by,
        bz=bz,
        protons={
            "charge": 1,
            "density": density,
            "vbulkx": vx,
            "vbulky": vy,
            "vbulkz": vz,
            "vthx": vth,
            "vthy": vth,
            "vthz": vth,
        },
    )
    ph.ElectronModel(closure="isothermal", Te=0.0)
    return sim
