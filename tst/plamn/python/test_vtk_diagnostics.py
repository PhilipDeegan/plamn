"""pharevtkhdf diagnostics, as PHARE's tests/simulator/test_vtk_diagnostics.py: a 2D run with a
refinement box (2 levels), EM and fluid diagnostics at t0 and after a step, read back with h5py:

- every file's layout (assert_vtkhdf_layout, PHARE's): per level and step, Steps entries,
  boxes, rows = sum of prod(cells + 1) * X_TIMES, the X_TIMES copies equal, nodes shared by boxes
  of a level agree
- the values: each rank's patches, converted to primal again here (numpy, from the live views of
  the DataWrangler: dual directions averaged), equal their rows in the files at the last step
- a level with fewer boxes than ranks; an unsupported quantity raises

Run with any number of ranks, e.g. mpirun -n 3 python3 -m unittest tst.plamn.python.test_vtk_diagnostics
"""

import shutil
import unittest
from pathlib import Path

import numpy as np

import pyplamn

pyplamn.patch()

import pyphare.pharein as ph  # noqa: E402
from pyphare.simulator.simulator import Simulator, startMPI  # noqa: E402

ndim = 2
X_TIMES = {1: 4, 2: 2, 3: 1}[ndim]
cells, dl, timestep = 40, 0.3, 0.001
populations = {"protons": (1.0, 1.0), "alpha": (4.0, 2.0)}  # name: (mass, charge)

# per component, primal (True) or dual per direction (Yee, PHARE's gridlayout_hybrid_yee)
YEE = {
    "B": [(True, False, False), (False, True, False), (False, False, True)],
    "E": [(False, True, True), (True, False, True), (True, True, False)],
}
FLUID = [
    "charge_density",
    "mass_density",
    "bulkVelocity",
    "charge_flux",
    "momentum_tensor",
]
POP = ["density", "charge_density", "flux", "momentum_tensor"]
TENSOR = ["xx", "xy", "xz", "yy", "yz", "zz"]
VX, VY, VTH = (0.5, 1.0), 0.1, 0.3  # per population (order of populations), all, all


def simulation(out, box=((10, 10), (19, 19))):
    ph.global_vars.sim = None
    ph.clearDict()
    sim = ph.Simulation(
        time_step_nbr=1,
        time_step=timestep,
        cells=[cells] * ndim,
        dl=[dl] * ndim,
        hyper_resistivity=0.001,
        refinement_boxes={"L0": {"B0": [list(box[0]), list(box[1])]}},
        diag_options={
            "format": "pharevtkhdf",
            "options": {"dir": out, "mode": "overwrite"},
        },
    )

    def const(v):
        return lambda x, y: v + 0 * x

    def density(x, y):
        L = sim.simulation_domain()
        return 1.0 + 0.2 * np.cos(2 * np.pi * x / L[0]) * np.sin(2 * np.pi * y / L[1])

    def pop(mass, charge, vx, seed):
        return {
            "mass": mass,
            "charge": charge,
            "density": density,
            "vbulkx": const(vx),
            "vbulky": const(VY),
            "vbulkz": const(0.0),
            **{f"vth{c}": const(VTH) for c in "xyz"},
            "nbr_part_per_cell": 50,
            "init": {"seed": seed},
        }

    ph.MaxwellianFluidModel(
        bx=lambda x, y: 1.0 + 0.1 * np.sin(2 * np.pi * y / sim.simulation_domain()[1]),
        by=const(0.2),
        bz=const(0.5),
        **{
            name: pop(m, q, VX[i], 1234 + i)
            for i, (name, (m, q)) in enumerate(populations.items())
        },
    )
    ph.ElectronModel(closure="isothermal", Te=0.12)
    timestamps = np.array([0.0, timestep])
    for q in ["E", "B"]:
        ph.ElectromagDiagnostics(quantity=q, write_timestamps=timestamps, flush_every=1)
    for q in FLUID:
        ph.FluidDiagnostics(quantity=q, write_timestamps=timestamps, flush_every=1)
    for name in populations:
        for q in POP:
            ph.FluidDiagnostics(
                quantity=q,
                write_timestamps=timestamps,
                population_name=name,
                flush_every=1,
            )
    return sim


def comps_of(q):
    if q == "momentum_tensor":
        return TENSOR
    return "xyz" if q in ("bulkVelocity", "charge_flux", "flux") else None


def files():
    """quantity file: the live accessors of a level (DataWrangler), per component"""
    out = {"EM_B": ("B", "xyz"), "EM_E": ("E", "xyz")}
    for q in FLUID:
        out[f"ions_{q}"] = (q, comps_of(q))
    for name in populations:
        for q in POP:
            out[f"ions_pop_{name}_{q}"] = ((q, name), comps_of(q))
    return out


def live(level, what, component):
    """this rank's patch datas of a quantity's component on a level, from the DataWrangler
    (every quantity written has a getter)"""
    if isinstance(what, tuple):
        q, name = what
        return {
            "density": lambda: level.getDensity(name),
            "charge_density": lambda: level.getN(name),
            "flux": lambda: level.getFlux(component, name),
            "momentum_tensor": lambda: level.getMomentumTensor(component, name),
        }[q]()
    return {
        "B": lambda: level.getB(component),
        "E": lambda: level.getE(component),
        "charge_density": lambda: level.getNi(),
        "mass_density": lambda: level.getMassDensity(),
        "bulkVelocity": lambda: level.getVi(component),
        "charge_flux": lambda: level.getChargeFlux(component),
        "momentum_tensor": lambda: level.getMomentumTensor(component),
    }[what]()


def primal(pd, centering):
    """pd's values on the primal nodes of its cells, x fastest (dual directions averaged)"""
    a = np.asarray(pd.data)
    g = pd.nGhosts
    ncells = np.asarray(pd.upper) - np.asarray(pd.lower) + 1
    out = np.zeros(ncells + 1)
    corners = 0
    for shift in np.ndindex(*(1 if p else 2 for p in centering[:ndim])):
        idx = []
        for i in range(ndim):
            if centering[i]:  # primal: the nodes
                idx.append(slice(g, g + ncells[i] + 1))
            else:  # dual: cells node - 1 and node
                lo = g - 1 + shift[i]
                idx.append(slice(lo, lo + ncells[i] + 1))
        out += a[tuple(idx)]
        corners += 1
    return (out / corners).ravel(order="F")


def assert_vtkhdf_layout(test, path):
    """PHARE's: every level and step consistent"""
    import h5py

    with h5py.File(path, "r") as f:
        root = f["VTKHDF"]
        nsteps = int(root["Steps"].attrs["NSteps"])
        test.assertEqual(nsteps, root["Steps/Values"].shape[0])
        ilvl = 0
        while f"Level{ilvl}/PointData/data" in root:
            data = root[f"Level{ilvl}/PointData/data"][:]
            boxes = root[f"Level{ilvl}/AMRBox"][:]
            steps = root[f"Steps/Level{ilvl}"]
            for k in ("AMRBoxOffset", "NumberOfAMRBox", "PointDataOffset/data"):
                test.assertEqual(steps[k].shape[0], nsteps, f"{path} {ilvl} {k}")
            for s in range(nsteps):
                b0, nb = int(steps["AMRBoxOffset"][s]), int(steps["NumberOfAMRBox"][s])
                row = int(steps["PointDataOffset/data"][s])
                nodes = {}
                for box in boxes[b0 : b0 + nb]:
                    test.assertTrue((box[2 * ndim :] == 0).all())
                    lo, up = box[0 : 2 * ndim : 2], box[1 : 2 * ndim : 2]
                    points = int(np.prod(up - lo + 2))
                    block = data[row : row + points * X_TIMES]
                    for copy in range(1, X_TIMES):  # the 3D view's copies
                        np.testing.assert_array_equal(
                            block[copy * points : (copy + 1) * points], block[:points]
                        )
                    # shared nodes agree between boxes
                    grid = block[:points].reshape(*(up - lo + 2)[::-1], -1)
                    for ij in np.ndindex(*(up - lo + 2)):
                        node = tuple(lo + np.array(ij))
                        v = grid[tuple(reversed(ij))]
                        if node in nodes:
                            np.testing.assert_allclose(
                                v, nodes[node], rtol=1e-5, atol=1e-6
                            )
                        nodes[node] = v
                    row += points * X_TIMES
                if s + 1 < nsteps:
                    test.assertEqual(row, int(steps["PointDataOffset/data"][s + 1]))
            ilvl += 1
        return ilvl


class VtkDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        startMPI()
        np_ = pyplamn.etc.mpi_size()
        self.out = str(
            Path("phare_outputs/plamn_vtk") / f"{self.id().split('.')[-1]}_np{np_}"
        )
        self.simulator = None

    def tearDown(self):
        if self.simulator is not None:
            self.simulator.reset()
        pyplamn.etc.mpi_barrier()
        if pyplamn.etc.mpi_rank() == 0:
            shutil.rmtree(self.out, ignore_errors=True)

    def run_it(self, **kwargs):
        sim = simulation(self.out, **kwargs)
        self.simulator = Simulator(sim, log_to_file=False).setup().initialize()
        self.simulator.advance()  # dumps at t0 (initialize) and after the step
        pyplamn.etc.mpi_barrier()
        return self.simulator

    def test_dump_diags(self):
        import h5py

        simulator = self.run_it()
        dw = simulator.data_wrangler()
        n_levels = simulator.cpp_sim.n_levels()
        self.assertEqual(n_levels, 2)
        for name, (what, comps) in files().items():
            path = Path(self.out) / f"{name}.vtkhdf"
            self.assertTrue(path.exists(), path)
            self.assertEqual(assert_vtkhdf_layout(self, path), n_levels)

            with h5py.File(path, "r") as f:
                if name.startswith("ions_pop_"):
                    pop = name.split("_")[2]
                    self.assertEqual(f.attrs["pop_mass"], populations[pop][0])
                    self.assertEqual(f.attrs["pop_charge"], populations[pop][1])
                for ln in range(n_levels):
                    level = dw.getPatchLevel(ln)
                    root = f["VTKHDF"]
                    steps = root[f"Steps/Level{ln}"]
                    s = int(root["Steps"].attrs["NSteps"]) - 1  # the last
                    b0, nb = int(steps["AMRBoxOffset"][s]), int(
                        steps["NumberOfAMRBox"][s]
                    )
                    row = int(steps["PointDataOffset/data"][s])
                    data = root[f"Level{ln}/PointData/data"][:]
                    at = {}  # box -> its rows
                    for box in root[f"Level{ln}/AMRBox"][b0 : b0 + nb]:
                        lo, up = box[0 : 2 * ndim : 2], box[1 : 2 * ndim : 2]
                        points = int(np.prod(up - lo + 2))
                        at[(tuple(lo), tuple(up))] = data[row : row + points]
                        row += points * X_TIMES
                    for c, component in enumerate(comps or "x"):
                        for pd in live(level, what, component if comps else None):
                            key = (tuple(pd.lower), tuple(pd.upper))
                            self.assertIn(key, at, f"{name} {ln}")
                            centering = (
                                YEE[what][c]
                                if what in YEE
                                else (True, True, True)  # moments: primal
                            )
                            np.testing.assert_allclose(
                                at[key][:, c],
                                primal(pd, centering),
                                rtol=1e-6,
                                atol=1e-6,
                                err_msg=f"{name} level {ln} {key} {component}",
                            )

    def test_momentum_tensor(self):
        """the total is the populations' sum; on level 0 a Maxwellian's mean M_ii / n is
        m (V_i^2 + vth^2) (50 particles per cell: within a few percent)"""
        import h5py

        self.run_it()

        def level0(name):
            with h5py.File(Path(self.out) / f"{name}.vtkhdf", "r") as f:
                root = f["VTKHDF"]
                s = int(root["Steps"].attrs["NSteps"]) - 1
                steps = root["Steps/Level0"]
                row = int(steps["PointDataOffset/data"][s])
                rows = sum(
                    int(np.prod(b[1 : 2 * ndim : 2] - b[0 : 2 * ndim : 2] + 2))
                    * X_TIMES
                    for b in root["Level0/AMRBox"][
                        int(steps["AMRBoxOffset"][s]) : int(steps["AMRBoxOffset"][s])
                        + int(steps["NumberOfAMRBox"][s])
                    ]
                )
                return root["Level0/PointData/data"][row : row + rows]

        total = level0("ions_momentum_tensor")
        pops = [level0(f"ions_pop_{p}_momentum_tensor") for p in populations]
        np.testing.assert_allclose(total, sum(pops), rtol=1e-5, atol=1e-6)
        for i, (name, (mass, _)) in enumerate(populations.items()):
            n = level0(f"ions_pop_{name}_density")[:, 0].mean()
            m = pops[i].mean(axis=0) / n
            np.testing.assert_allclose(m[0], mass * (VX[i] ** 2 + VTH**2), rtol=0.05)
            np.testing.assert_allclose(m[3], mass * (VY**2 + VTH**2), rtol=0.05)
            np.testing.assert_allclose(
                m[1], mass * VX[i] * VY, rtol=0.1, atol=0.02 * mass
            )

    def test_level_without_patches_on_some_ranks(self):
        import h5py

        self.run_it(box=((10, 10), (12, 12)))
        path = Path(self.out) / "EM_B.vtkhdf"
        assert_vtkhdf_layout(self, path)
        with h5py.File(path, "r") as f:
            n_boxes = f["VTKHDF/Steps/Level1/NumberOfAMRBox"][:]
            if pyplamn.etc.mpi_size() > 1:
                self.assertLess(int(n_boxes.max()), pyplamn.etc.mpi_size())

    def test_unsupported_quantity_raises(self):
        """pyphare accepts no quantity plamn does not write: one set past its checks"""
        sim = simulation(self.out)
        ph.FluidDiagnostics(quantity="charge_density", write_timestamps=[0.0])
        list(sim.diagnostics.values())[-1].quantity = "/ions/pressure_tensor"
        with self.assertRaises(Exception):
            self.simulator = Simulator(sim, log_to_file=False).setup()


if __name__ == "__main__":
    unittest.main()
