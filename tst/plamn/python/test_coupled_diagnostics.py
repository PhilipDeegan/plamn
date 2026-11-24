"""Diagnostics of a coupled hierarchy (level 0 MHD, level 1 hybrid): a field quantity is written on
the levels whose model has it, the others get nothing (phareh5) or an empty entry (pharevtkhdf):
B and the ions' mass density on both (MHD: its rho), MHD's P on level 0 only, a population's
density on level 1 only. A uniform plasma at rest: the mass density is 1 on both levels.

needs py_etc and the coupled 1D module (./mkn.py --coupled --dim 1), then from the plamn root:
    PYTHONPATH=.:$PHARE/pyphare:tst/plamn/python python3 -m phlop.run.test_cases -i tst
"""

import shutil
import unittest
from pathlib import Path

import h5py
import numpy as np

import plamn.backend.phare as ph

CELLS = 32
DX = 0.2
DT = 0.001
STEPS = 4
OUT = Path("phare_outputs/plamn_test_coupled_diagnostics")


def const(v):
    return lambda x: v + 0.0 * x


def run(fmt):
    out = OUT / fmt
    shutil.rmtree(out, ignore_errors=True)
    ph.global_vars.sim = None
    ph.clearDict()
    ph.Simulation(
        time_step=DT,
        time_step_nbr=STEPS,
        cells=(CELLS,),
        dl=(DX,),
        refinement_boxes={"L0": {"B0": [(8,), (23,)]}},
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
        diag_options={"format": fmt, "options": {"dir": str(out), "mode": "overwrite"}},
    )
    ph.MHDModel(
        density=const(1.0), vx=const(0.0), vy=const(0.0), vz=const(0.0),
        bx=const(1.0), by=const(0.0), bz=const(0.0), p=const(1.0),
        protons={"charge": 1, "mass": 1, "nbr_part_per_cell": 200, "init": {"seed": 1}},
    )
    ph.ElectronModel(closure="isothermal", Te=0.5)
    times = [0.0, STEPS * DT]
    ph.ElectromagDiagnostics(quantity="B", write_timestamps=times)
    ph.FluidDiagnostics(quantity="mass_density", write_timestamps=times)
    ph.FluidDiagnostics(quantity="density", write_timestamps=times, population_name="protons")
    ph.MHDDiagnostics(quantity="P", write_timestamps=times)
    simulator = ph.Simulator(ph.global_vars.sim, log_to_file=False).setup().initialize()
    for _ in range(STEPS):
        simulator.advance()
    simulator.reset()
    return out


def vtkhdf_levels(path):
    """per level, the last step's values (None: an empty entry)"""
    with h5py.File(path, "r") as h:
        out = []
        for ln in range(2):
            boxes = h[f"VTKHDF/Steps/Level{ln}/NumberOfAMRBox"][-1]
            if boxes == 0:
                out.append(None)
                continue
            off = h[f"VTKHDF/Steps/Level{ln}/PointDataOffset/data"][-1]
            out.append(h[f"VTKHDF/Level{ln}/PointData/data"][off:])
        return out


def phareh5_levels(path):
    """the levels with patches at the last time"""
    with h5py.File(path, "r") as h:
        last = sorted(h["t"].keys(), key=float)[-1]
        return sorted(int(k[2:]) for k in h["t"][last].keys() if k.startswith("pl"))


class CoupledDiagnosticsTest(unittest.TestCase):
    def test_pharevtkhdf(self):
        out = run("pharevtkhdf")
        b0, b1 = vtkhdf_levels(out / "EM_B.vtkhdf")
        self.assertIsNotNone(b0)
        self.assertIsNotNone(b1)
        np.testing.assert_allclose(b0[:, 0], 1.0)
        np.testing.assert_allclose(b1[:, 0], 1.0)

        m0, m1 = vtkhdf_levels(out / "ions_mass_density.vtkhdf")
        np.testing.assert_allclose(m0, 1.0)  # MHD's rho
        self.assertTrue(np.all(np.abs(m1 - 1.0) < 0.3))  # the particles' (200 ppc)

        p0, p1 = vtkhdf_levels(out / "mhd_P.vtkhdf")
        np.testing.assert_allclose(p0, 1.0)
        self.assertIsNone(p1)

        d0, d1 = vtkhdf_levels(out / "ions_pop_protons_density.vtkhdf")
        self.assertIsNone(d0)
        self.assertIsNotNone(d1)

    def test_phareh5(self):
        out = run("phareh5")
        self.assertEqual(phareh5_levels(out / "EM_B.h5"), [0, 1])
        self.assertEqual(phareh5_levels(out / "ions_mass_density.h5"), [0, 1])
        self.assertEqual(phareh5_levels(out / "mhd_P.h5"), [0])
        self.assertEqual(phareh5_levels(out / "ions_pop_protons_density.h5"), [1])


if __name__ == "__main__":
    unittest.main()
