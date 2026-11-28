"""a periodic 1D MHD shock tube (Brio-Wu like states): the input of a watched run
(python3 -m plamn ... mhd_tube) and of tests"""

import numpy as np

import plamn.backend.phare as ph


def config(cells=200, time_step_nbr=20, time_step=0.1, largest_patch_size=50):
    def left(x):
        return (x > cells / 4) & (x < 3 * cells / 4)

    sim = ph.Simulation(
        time_step=time_step,
        time_step_nbr=time_step_nbr,
        cells=(cells,),
        dl=(1.0,),
        largest_patch_size=largest_patch_size,
        hyper_resistivity=0.0,
        resistivity=0.0,
        eta=0.0,
        nu=0.0,
        gamma=5.0 / 3.0,
        reconstruction="Linear",
        limiter="VanLeer",
        riemann="Rusanov",
        mhd_timestepper="TVDRK2",
        hall=False,
        model_options=["MHDModel"],
    )
    ph.MHDModel(
        density=lambda x: np.where(left(x), 1.0, 0.125),
        vx=lambda x: 0.0 * x,
        vy=lambda x: 0.0 * x,
        vz=lambda x: 0.0 * x,
        bx=lambda x: 0.75 + 0.0 * x,
        by=lambda x: np.where(left(x), 1.0, -1.0),
        bz=lambda x: 0.0 * x,
        p=lambda x: np.where(left(x), 1.0, 0.1),
    )
    return sim
