"""tiling_gpu's tests/simulator/layouts (test_layouts_3d.py, interp 1): every particle layout
against the reference, AoSMapped, through the simulator, initialized and advanced, with
refinement boxes (no tagging), on level 0 only, 1 and 2 levels:

  init_L0, init_L1          initialized, fields and particles at t0
  advance_L0, advance_L1    one step
  advance_L2                four steps

tiling_gpu's model (a periodic gaussian density bump, B and the bulk velocity cos products), its
cells, particles per cell, boxes and tolerances. Fields (B, E, every moment) on every level,
after initialization and after every step, within atol; the level's domain particles too,
sorted (layouts keep particles in their own order).

plamn builds one module per layout (./mkn.py --dim 3 --layout AoSMapped AoSCMTS SoAVXChunked):
each run is its own process (python3 this_file --run ...), which writes what it holds to an
npz; a layout whose module is not built is skipped.
"""

import importlib.util
import itertools
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

NDIM = 3
INTERP = 1
REFINED = 6  # pyphare's default refined particle number in 3D, interp 1
CELLS = 14
PPC = 15
DL = 0.2
TIME_STEP = 0.001
REF = ("AoSMapped", "CPU")
LAYOUTS = [("AoSCMTS", "CPU"), ("SoAVXChunked", "CPU")]

L0_BOX = (1, 12)
L1_BOX = (4, 21)

CASES = {
    "init_L0": dict(steps=0, boxes=[], atol=dict(b=0, e=1e-14, moments=1e-14, particles=0)),
    "init_L1": dict(
        steps=0, boxes=[L0_BOX], atol=dict(b=0, e=1e-14, moments=1e-14, particles=0)
    ),
    "advance_L0": dict(
        steps=1, boxes=[], atol=dict(b=1e-14, e=1e-12, moments=1e-14, particles=1e-14)
    ),
    "advance_L1": dict(
        steps=1, boxes=[L0_BOX], atol=dict(b=1e-15, e=1e-14, moments=1e-14, particles=1e-15)
    ),
    "advance_L2": dict(
        steps=4,
        boxes=[L0_BOX, L1_BOX],
        atol=dict(b=5e-15, e=5e-14, moments=1e-14, particles=1e-15),
    ),
}

FIELDS = {
    "b": [f"EM_B_{c}" for c in "xyz"],
    "e": [f"EM_E_{c}" for c in "xyz"],
    "moments": ["protons_density", "protons_charge_density", "ions_charge_density"]
    + [f"protons_flux_{c}" for c in "xyz"]
    + [f"ions_charge_flux_{c}" for c in "xyz"],
}
PARTICLE_COLUMNS = (
    [f"iCell_{c}" for c in "xyz"]
    + [f"delta_{c}" for c in "xyz"]
    + [f"v_{c}" for c in "xyz"]
    + ["weight"]
)


def module_built(layout, allocator):
    name = f"pyplamn.plamn_cpp_{NDIM}_{INTERP}_{REFINED}_{layout}_{allocator}"
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


# ---------------------------------------------------------------- a run, in its own process


def setup_model(ph, sim):
    """tiling_gpu's tests/simulator/advance/test_advance_hybrid.py model"""
    import numpy as np

    L = sim.simulation_domain()

    def density(*xyz):
        Ls = np.array(L)
        hL = Ls / 2
        return 0.3 + np.exp(
            sum(
                -((Ls[i] / np.pi * np.sin(np.pi * (xyz[i] - hL[i]) / Ls[i])) ** 2)
                for i in range(len(xyz))
            )
        )

    def cos_product(*xyz):
        return np.asarray(
            [0.1 * np.cos(2 * np.pi * xyz[i] / L[i]) for i in range(len(xyz))]
        ).prod(axis=0)

    def vth(*xyz):
        return 0.01 + np.zeros_like(xyz[0])

    ph.MaxwellianFluidModel(
        bx=lambda *xyz: 1.0 + 0 * xyz[0],
        by=cos_product,
        bz=cos_product,
        protons={
            "charge": 1,
            "density": density,
            "vbulkx": cos_product,
            "vbulky": cos_product,
            "vbulkz": cos_product,
            "vthx": vth,
            "vthy": vth,
            "vthz": vth,
            "nbr_part_per_cell": PPC,
            "init": {"seed": 1337},
        },
    )
    ph.ElectronModel(closure="isothermal", Te=0.12)


def state(simulator, np):
    """per level: the fields, and the domain particles sorted by every column"""
    cpp = simulator.cpp_sim
    out = {}
    dw = simulator.data_wrangler()
    for ln in range(cpp.n_levels()):
        for names in FIELDS.values():
            for q in names:
                out[f"{q}@{ln}"] = np.asarray(cpp.level_values(q, ln))
        columns = {c: [] for c in PARTICLE_COLUMNS}
        for pd in dw.getPatchLevel(ln).getParticles("protons"):
            components = pd.data.components()
            for c in PARTICLE_COLUMNS:
                columns[c].append(np.array(components[c], dtype=float))
        rows = np.stack([np.concatenate(columns[c]) for c in PARTICLE_COLUMNS], axis=1)
        rows = rows[rows[:, PARTICLE_COLUMNS.index("weight")] != 0]  # SoAVXChunked's gaps
        out[f"particles@{ln}"] = rows[np.lexsort(rows.T[::-1])]
    out["levels"] = np.array([cpp.n_levels()])
    return out


def run(case, layout, allocator, out):
    import numpy as np

    import pyplamn

    pyplamn.patch()
    import pyphare.pharein as ph
    from pyphare.core.box import nDBox
    from pyphare.simulator.simulator import Simulator, startMPI

    startMPI()
    c = CASES[case]
    ph.global_vars.sim = None
    ph.clearDict()
    sim = ph.Simulation(
        time_step_nbr=max(c["steps"], 1),
        time_step=TIME_STEP,
        cells=[CELLS] * NDIM,
        dl=[DL] * NDIM,
        interp_order=INTERP,
        refinement_boxes={
            f"L{i}": {"B0": nDBox(NDIM, *box)} for i, box in enumerate(c["boxes"])
        },
        model_layout=layout,
        allocator=allocator,
        hyper_resistivity=0.001,
    )
    setup_model(ph, sim)
    simulator = Simulator(sim, log_to_file=False).setup().initialize()
    arrays = {f"0:{k}": v for k, v in state(simulator, np).items()}
    for step in range(1, c["steps"] + 1):
        simulator.advance()
        arrays.update({f"{step}:{k}": v for k, v in state(simulator, np).items()})
    simulator.reset()
    np.savez(out, **arrays)


# ---------------------------------------------------------------- the comparisons


class Layouts3dTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp(prefix="plamn_test_layouts_3d_"))
        cls.runs = {}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def run_case(self, case, layout, allocator):
        key = (case, layout, allocator)
        if key not in self.runs:
            out = self.dir / f"{case}_{layout}_{allocator}.npz"
            done = subprocess.run(
                [sys.executable, __file__, "--run", case, layout, allocator, str(out)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(done.returncode, 0, f"{key}\n{done.stdout}\n{done.stderr}")
            import numpy as np

            self.runs[key] = dict(np.load(out))
        return self.runs[key]

    def compare(self, case):
        if not module_built(*REF):
            self.skipTest(f"no {REF} module (./mkn.py --dim {NDIM} --layout {REF[0]})")
        layouts = [l for l in LAYOUTS if module_built(*l)]
        if not layouts:
            self.skipTest("no module to compare (./mkn.py --dim 3 --layout AoSCMTS ...)")
        import numpy as np

        atol = CASES[case]["atol"]
        kind = {q: k for k, names in FIELDS.items() for q in names}
        ref = self.run_case(case, *REF)
        for layout in layouts:
            with self.subTest(layout=layout):
                cmp = self.run_case(case, *layout)
                self.assertEqual(sorted(ref), sorted(cmp))
                failures = []
                for key in sorted(ref):
                    a, b = ref[key], cmp[key]
                    if a.shape != b.shape:
                        failures.append(f"{key}: shape {a.shape} vs {b.shape}")
                        continue
                    if key.endswith(":levels"):
                        if not np.array_equal(a, b):
                            failures.append(f"{key}: {a} vs {b}")
                        continue
                    q = key.split(":", 1)[1].split("@")[0]
                    tol = atol["particles"] if q == "particles" else atol[kind[q]]
                    diff = np.max(np.abs(a - b)) if a.size else 0.0
                    if diff > tol:
                        failures.append(f"{key}: max |diff| {diff:.3e} > {tol:.0e}")
                self.assertFalse(failures, f"{case} {layout}:\n" + "\n".join(failures))

    def test_init_L0(self):
        self.compare("init_L0")

    def test_init_L1(self):
        self.compare("init_L1")

    def test_advance_L0(self):
        self.compare("advance_L0")

    def test_advance_L1(self):
        self.compare("advance_L1")

    def test_advance_L2(self):
        self.compare("advance_L2")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        run(*sys.argv[2:6])
    else:
        unittest.main()
