"""PHARE's tests/simulator/test_multi_species_momentum.py on plamn, level 0 only (plamn has one
level): read live through the DataWrangler, no diagnostics

Two ion species of different q/m drifting relative to each other in a uniform plasma, with a
uniform magnetic field and no current. With J = 0 the total force on the ions is
(sum_s q_s n_s V_s - Ne Ve) x B, which vanishes only if the electron velocity is the
charge-weighted ion velocity. The total ion momentum, i.e. the mass-weighted bulk velocity, must
then stay constant. Using the mass-weighted velocity in Ohm's law instead makes it rotate
(the multi species momentum fix).
"""

import unittest

import numpy as np

import plamn.backend.phare as ph

time_step = 0.005
time_step_nbr = 100
ppc = 100

# protons drifting across B through sodium at rest. With the mass-weighted velocity in Ohm's
# law, dVi_z/dt ~ 0.2 here, so Vi_z ~ 0.1 at the end of the run
protons = {"mass": 1.0, "density": 1.0, "vbulky": 1.0}
sodium = {"mass": 23.0, "density": 0.1, "vbulky": 0.0}
vth = 0.1

tolerance = 0.01


def config():
    ph.global_vars.sim = None
    ph.clearDict()
    sim = ph.Simulation(
        time_step=time_step,
        time_step_nbr=time_step_nbr,
        cells=50,
        dl=0.2,
        largest_patch_size=10,  # several patches: every rank has some on a few ranks
        hyper_resistivity=0.002,
        resistivity=0.001,
        strict=True,
    )

    def constant(value):
        return lambda x: value + 0.0 * x

    def population(species, seed):
        return {
            "charge": 1,
            "mass": species["mass"],
            "density": constant(species["density"]),
            "vbulkx": constant(0.0),
            "vbulky": constant(species["vbulky"]),
            "vbulkz": constant(0.0),
            "vthx": constant(vth),
            "vthy": constant(vth),
            "vthz": constant(vth),
            "nbr_part_per_cell": ppc,
            "init": {"seed": seed},
        }

    ph.MaxwellianFluidModel(
        bx=constant(1.0),
        by=constant(0.0),
        bz=constant(0.0),
        protons=population(protons, seed=1),
        sodium=population(sodium, seed=2),
    )
    ph.ElectronModel(closure="isothermal", Te=0.1)
    return sim


def level_bulk_velocity(simulator, ilvl):
    """mean of the ion bulk velocity over the physical nodes of a level"""
    patch_level = simulator.data_wrangler().getPatchLevel(ilvl)
    if not patch_level.getVi("x"):  # this rank holds none
        return None
    mean = []
    for component in ("x", "y", "z"):
        values = [
            np.asarray(patch.data)[
                patch.nGhosts : patch.nGhosts + patch.upper[0] - patch.lower[0] + 2
            ]
            for patch in patch_level.getVi(component)
        ]
        mean.append(np.concatenate(values).mean())
    return np.array(mean)


class MultiSpeciesMomentumTest(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, "simulator"):
            self.simulator.reset()

    def test_ion_momentum_is_conserved(self):
        self.simulator = ph.Simulator(config(), log_to_file=False).setup().initialize()
        self.assertEqual(self.simulator.data_wrangler().getNumberOfLevels(), 1)

        initial = level_bulk_velocity(
            self.simulator, 0
        )  # this rank's region: uniform plasma
        for _ in range(time_step_nbr):
            self.simulator.advance()
        final = level_bulk_velocity(self.simulator, 0)
        if initial is None:
            return

        expected_vy = (
            protons["mass"] * protons["density"] * protons["vbulky"]
            + sodium["mass"] * sodium["density"] * sodium["vbulky"]
        ) / (protons["mass"] * protons["density"] + sodium["mass"] * sodium["density"])

        print(f"L0 initial {initial} final {final}")
        np.testing.assert_allclose(initial[1], expected_vy, atol=tolerance)
        np.testing.assert_allclose(final, initial, atol=tolerance)


if __name__ == "__main__":
    unittest.main()
