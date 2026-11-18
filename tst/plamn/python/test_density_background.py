"""A population's density_background (plamn.patch()): the constant part of its density, added to
its density function. Loading 1 + 0.5 sin(x) as one function or as a background of 1 plus
0.5 sin(x) gives the same particles bit for bit (both add 1.0 to the same values once); a
background above the density cut-off populates every cell (counted without sampling).

    python3 -m unittest tst.plamn.python.test_density_background
"""

import unittest

import numpy as np

import plamn.backend.phare as ph

cells = 100
dl = 0.3
ppc = 50


def initialized(background):
    """a 1D Simulator, initialized: density 1 + 0.5 sin, as one function or background + rest"""
    ph.global_vars.sim = None
    ph.clearDict()
    sim = ph.Simulation(
        time_step=0.001, time_step_nbr=1, cells=cells, dl=dl, hyper_resistivity=0.001
    )
    length = cells * dl

    def wave(x):
        return 0.5 * np.sin(2 * np.pi * x / length)

    def const(v):
        return lambda x: v + 0 * x

    density = (
        dict(density=wave, density_background=1.0)
        if background
        else dict(density=lambda x: 1.0 + wave(x))
    )
    ph.MaxwellianFluidModel(
        bx=const(1.0),
        by=const(0.0),
        bz=const(0.0),
        protons={
            "charge": 1,
            **density,
            "vbulkx": const(0.0),
            "vbulky": const(0.0),
            "vbulkz": const(0.0),
            "vthx": const(0.3),
            "vthy": const(0.3),
            "vthz": const(0.3),
            "nbr_part_per_cell": ppc,
            "init": {"seed": 1337},
        },
    )
    ph.ElectronModel(closure="isothermal", Te=0.12)
    return ph.Simulator(sim, log_to_file=False).setup().initialize()


class DensityBackgroundTest(unittest.TestCase):
    def test_background_is_added_to_the_density(self):
        out = {}
        for background in (False, True):
            simulator = initialized(background)
            cpp = simulator.cpp_sim
            out[background] = (
                cpp.total_particle_count(0),
                np.asarray(cpp.all_domain_values("protons_density", 0)).copy(),
            )
            simulator.reset()
        self.assertEqual(out[True][0], cells * ppc)  # every cell populated
        self.assertEqual(out[True][0], out[False][0])
        np.testing.assert_array_equal(out[True][1], out[False][1])

    def test_background_must_not_be_negative(self):
        ph.global_vars.sim = None
        ph.Simulation(time_step=0.001, time_step_nbr=1, cells=cells, dl=dl)
        with self.assertRaises(ValueError):
            ph.MaxwellianFluidModel(protons={"density_background": -1.0})

    def tearDown(self):
        ph.global_vars.sim = None


if __name__ == "__main__":
    unittest.main()
