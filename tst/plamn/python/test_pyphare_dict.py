"""pyphare's populateDict, through pyplamn.patch(), fills plamn's C++ Dict

run from the plamn root:
    PYTHONPATH=python:$PHARE/pyphare:tst/plamn/python python3 -m unittest discover -s tst/plamn/python
"""

import unittest

import numpy as np

import pyplamn

pyplamn.patch()

import alfven1d  # noqa: E402
import pyphare.pharein as ph  # noqa: E402


class PyphareDictTest(unittest.TestCase):
    def setUp(self):
        ph.global_vars.sim = None
        ph.clearDict()
        self.sim = alfven1d.config(cells=50, time_step_nbr=10, final_time=0.1)
        ph.populateDict()
        self.dict = pyplamn.current_dict()

    def tearDown(self):
        ph.clearDict()

    def test_dictator_is_the_python_owned_plamn_dict(self):
        from pyphare.pharein.initialize import dict_populator

        self.assertIsInstance(self.dict, pyplamn.etc.Dict)
        self.assertIs(dict_populator.dictator(), self.dict)
        self.assertIs(dict_populator.DictPopulator().pp, self.dict)

    def test_simulation_values(self):
        d = self.dict
        self.assertEqual(d.get_int("simulation/dimension"), 1)
        self.assertEqual(d.get_int("simulation/interp_order"), 1)
        self.assertEqual(d.get_int("simulation/grid/nbr_cells/x"), 50)
        self.assertEqual(d.get_double("simulation/grid/meshsize/x"), 1.0)
        self.assertEqual(d.get_int("simulation/time_step_nbr"), 10)
        self.assertAlmostEqual(d.get_double("simulation/time_step/value"), 0.01)
        self.assertEqual(d.get_vector_string("simulation/models"), ["HybridModel"])
        self.assertEqual(d.get_size_t("simulation/ions/nbrPopulations"), 1)
        self.assertEqual(d.get_string("simulation/ions/pop0/name"), "protons")

    def test_enums_are_plamns(self):
        d = self.dict
        self.assertEqual(
            d.get_int("simulation/model_layout"), pyplamn.etc.LayoutMode.AoSMapped.value
        )
        self.assertEqual(d.get_int("simulation/allocator"), pyplamn.etc.AllocatorMode.CPU.value)
        self.assertEqual(
            d.get_int("simulation/time_step/mode"), pyplamn.etc.TimeStepType.constant.value
        )

    def test_init_functions_are_called_from_cpp(self):
        d = self.dict
        L = self.sim.simulation_domain()[0]
        x = np.linspace(0, L, 7)

        by = d.eval_init_function_1d("simulation/electromag/magnetic/initializer/y_component", x)
        np.testing.assert_allclose(by, 0.01 * np.cos(2 * np.pi * x / L))

        # a user function returning a scalar comes back one value per point
        density = d.eval_init_function_1d(
            "simulation/ions/pop0/particle_initializer/density", x
        )
        np.testing.assert_allclose(density, np.ones_like(x))

    def test_clear_dict_starts_a_new_dict(self):
        old = self.dict
        ph.clearDict()
        self.assertIsNot(pyplamn.current_dict(), old)
        self.assertEqual(pyplamn.current_dict().paths(), [])


if __name__ == "__main__":
    unittest.main()
