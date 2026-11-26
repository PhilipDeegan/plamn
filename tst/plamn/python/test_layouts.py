"""interp 1, 1D 2D 3D: every particle layout (CPU and GPU_UNIFIED) against the reference,
AoSMapped CPU, through the simulator, initialized and advanced, with refinement boxes (no
tagging), on level 0 only, 1 and 2 levels:

  init_L0, init_L1          initialized, fields and particles at t0
  advance_L0, advance_L1    one step
  advance_L2                four steps
  advance_L1_tiles          one step, 12 cells per dimension, tiles of at most 6
                            (DAMN_TILING_MAX_TILE_SIZE): level 0 6 + 6, level 1 (10 fine
                            cells) 6 + 4, uneven tiles

tiling_gpu's model (a periodic gaussian density bump, B and the bulk velocity cos products), its
cells, particles per cell, boxes and tolerances. Fields (B, E, every moment) on every level,
after initialization and after every step, within atol; the level's domain particles too,
sorted (layouts keep particles in their own order).

plamn builds one module per layout (./mkn.py --dim 1 2 3 --layout AoSMapped AoSCMTS SoAVXChunked
AoS4SoAVXChunked; GPU: --layout SoAVXChunked --allocator GPU_UNIFIED -- -x res/mkn/clang_cuda_mpi):
each run is its own process (python3 this_file --run ...), which writes what it holds to an
npz; a layout whose module is not built is skipped.
"""

import importlib.util
import itertools
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

INTERP = 1
REFINED = {1: 2, 2: 4, 3: 6}  # pyphare's default refined particle numbers, interp 1
CELLS = 14
PPC = 15
DL = 0.2
TIME_STEP = 0.001
REF = ("AoSMapped", "CPU")
LAYOUTS = [
    ("AoSCMTS", "CPU"),
    ("SoAVXChunked", "CPU"),
    ("AoS4SoAVXChunked", "CPU"),
    ("SoAVXChunked", "GPU_UNIFIED"),
]

L0_BOX = (1, 12)
L1_BOX = (4, 21)

CASES = {
    "init_L0": dict(
        steps=0, boxes=[], atol=dict(b=1e-15, e=1e-14, moments=1e-14, particles=1e-15)
    ),
    "init_L1": dict(
        steps=0, boxes=[L0_BOX], atol=dict(b=1e-15, e=1e-14, moments=1e-14, particles=1e-15)
    ),
    "advance_L0": dict(
        steps=1, boxes=[], atol=dict(b=1e-14, e=1e-12, moments=1e-14, particles=1e-14)
    ),
    "advance_L1": dict(
        steps=1,
        boxes=[L0_BOX],
        atol=dict(b=1e-15, e=1e-14, moments=1e-14, particles=1e-15),
    ),
    "advance_L2": dict(
        steps=4,
        boxes=[L0_BOX, L1_BOX],
        atol=dict(b=5e-15, e=5e-14, moments=1e-14, particles=1e-15),
    ),
    "advance_L1_tiles": dict(
        steps=1,
        cells=12,
        boxes=[(3, 7)],  # 5 coarse cells: 10 fine
        env=dict(DAMN_TILING_MAX_TILE_SIZE="6"),
        atol=dict(b=1e-15, e=1e-14, moments=1e-14, particles=1e-15),
    ),
}

FIELDS = {
    "b": [f"EM_B_{c}" for c in "xyz"],
    "e": [f"EM_E_{c}" for c in "xyz"],
    "moments": ["protons_density", "protons_charge_density", "ions_charge_density"]
    + [f"protons_flux_{c}" for c in "xyz"]
    + [f"ions_charge_flux_{c}" for c in "xyz"],
}
def particle_columns(ndim):
    return (
        [f"iCell_{c}" for c in "xyz"[:ndim]]
        + [f"delta_{c}" for c in "xyz"[:ndim]]
        + [f"v_{c}" for c in "xyz"]
        + ["weight"]
    )


def module_built(ndim, layout, allocator):
    name = f"plamn.plamn_cpp_{ndim}_{INTERP}_{REFINED[ndim]}_{layout}_{allocator}"
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError as e:
        if e.name != name:  # the plamn package itself does not import: not a missing module
            raise
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


def state(simulator, ndim, np):
    """per level: the fields, and the domain particles sorted by every column"""
    cpp = simulator.cpp_sim
    out = {}
    dw = simulator.data_wrangler()
    for ln in range(cpp.n_levels()):
        for names in FIELDS.values():
            for q in names:
                out[f"{q}@{ln}"] = np.asarray(cpp.level_values(q, ln))
        cols = particle_columns(ndim)
        columns = {c: [] for c in cols}
        live_particles(dw.getPatchLevel(ln).getParticles("protons"), np)
        for pd in dw.getPatchLevel(ln).getParticles("protons"):
            components = pd.data.components()
            for c in cols:
                columns[c].append(np.array(components[c], dtype=float))
        rows = np.stack([np.concatenate(columns[c]) for c in cols], axis=1)
        rows = rows[
            rows[:, cols.index("weight")] != 0
        ]  # SoAVXChunked's gaps
        out[f"particles@{ln}"] = rows[np.lexsort(rows.T[::-1])]
        for c in "xyz":  # B through the wrangler (tiled layouts: a patch data per tile)
            values, dup = wrangler_nodes(dw.getPatchLevel(ln).getB(c), c, ndim, np)
            out[f"wrangler_B_{c}@{ln}"] = values
            out[f"wrangler_dup_B_{c}@{ln}"] = np.array([dup])
    out["levels"] = np.array([cpp.n_levels()])
    return out


def live_particles(patch_datas, np):
    """the wrangler's particles are the simulation's (tiled layouts: each tile's array): a write
    through components() is read back by __getitem__, then undone"""
    for pd in patch_datas:
        if pd.data.size() == 0:
            continue
        weight = pd.data.components()["weight"]
        was = float(weight[0])
        weight[0] = was + 1
        seen = pd.data[0].weight
        weight[0] = was
        if seen != was + 1:
            raise RuntimeError(f"particles of {pd.patchID} not live: {seen} after {was + 1}")
        return


def wrangler_nodes(patch_datas, component, ndim, np):
    """the domain nodes of B's component over the patch datas, rows (index..., value) sorted by
    their AMR index, each once; and the largest difference between two patch datas' copies of
    a node (primal nodes on their shared faces: tiles of a patch, patches of a level)"""
    # B_x primal along x, dual along y and z (faces); B_y, B_z in 1D: dual along x
    primal = ["xyz".index(component)]
    seen, dup = {}, 0.0
    for pd in patch_datas:
        data, g = np.asarray(pd.data), pd.nGhosts
        lower, upper = list(pd.lower), list(pd.upper)
        ranges = [
            range(lower[d], upper[d] + 1 + (1 if d in primal else 0)) for d in range(ndim)
        ]
        for idx in itertools.product(*ranges):
            v = float(data[tuple(i - lower[d] + g for d, i in enumerate(idx))])
            if idx in seen:
                dup = max(dup, abs(seen[idx] - v))
            else:
                seen[idx] = v
    rows = np.array([(*idx, v) for idx, v in sorted(seen.items())], dtype=float)
    return rows, dup


def run(case, ndim, layout, allocator, out):
    import numpy as np
    from pyphare.core.box import nDBox

    import plamn.backend.phare as ph

    ph.startMPI()
    ndim = int(ndim)
    c = CASES[case]
    ph.global_vars.sim = None
    ph.clearDict()
    sim = ph.Simulation(
        time_step_nbr=max(c["steps"], 1),
        time_step=TIME_STEP,
        cells=[c.get("cells", CELLS)] * ndim,
        dl=[DL] * ndim,
        interp_order=INTERP,
        refinement_boxes={
            f"L{i}": {"B0": nDBox(ndim, *box)} for i, box in enumerate(c["boxes"])
        },
        model_layout=layout,
        allocator=allocator,
        hyper_resistivity=0.001,
    )
    setup_model(ph, sim)
    simulator = ph.Simulator(sim, log_to_file=False).setup().initialize()
    arrays = {f"0:{k}": v for k, v in state(simulator, ndim, np).items()}
    for step in range(1, c["steps"] + 1):
        simulator.advance()
        arrays.update({f"{step}:{k}": v for k, v in state(simulator, ndim, np).items()})
    simulator.reset()
    np.savez(out, **arrays)


# ---------------------------------------------------------------- the comparisons


class Layouts:
    """the cases of one dimension, NDIM: a TestCase per dimension below"""

    NDIM = 0

    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp(prefix=f"plamn_test_layouts_{cls.NDIM}d_"))
        cls.runs = {}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def run_case(self, case, layout, allocator):
        key = (case, layout, allocator)
        if key not in self.runs:
            out = self.dir / f"{case}_{layout}_{allocator}.npz"
            done = subprocess.run(
                [sys.executable, __file__, "--run", case, str(self.NDIM), layout, allocator,
                 str(out)],
                capture_output=True,
                text=True,
                env={**os.environ, **CASES[case].get("env", {})},
            )
            self.assertEqual(done.returncode, 0, f"{key}\n{done.stdout}\n{done.stderr}")
            import numpy as np

            self.runs[key] = dict(np.load(out))
        return self.runs[key]

    def compare(self, case):
        ndim = self.NDIM
        if not module_built(ndim, *REF):
            self.skipTest(f"no {REF} module (./mkn.py --dim {ndim} --layout {REF[0]})")
        layouts = [l for l in LAYOUTS if module_built(ndim, *l)]
        if not layouts:
            self.skipTest(f"no module to compare (./mkn.py --dim {ndim} --layout SoAVXChunked)")
        import numpy as np

        atol = CASES[case]["atol"]
        kind = {q: k for k, names in FIELDS.items() for q in names}
        for c in "xyz":
            kind[f"wrangler_B_{c}"] = kind[f"wrangler_dup_B_{c}"] = "b"
        ref = self.run_case(case, *REF)
        for layout in layouts:
            with self.subTest(layout=layout):
                cmp = self.run_case(case, *layout)
                self.assertEqual(sorted(ref), sorted(cmp))
                failures, worst = [], {}
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
                    k = "particles" if q == "particles" else kind[q]
                    diff = np.max(np.abs(a - b)) if a.size else 0.0
                    worst[k] = max(worst.get(k, 0.0), diff)
                    if diff > atol[k]:
                        failures.append(f"{key}: max |diff| {diff:.3e} > {atol[k]:.0e}")
                print(f"{ndim}D {case} {layout}: max |diff|", ", ".join(
                    f"{k} {v:.1e}" for k, v in sorted(worst.items())), file=sys.stderr)
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

    def test_advance_L1_tiles(self):
        self.compare("advance_L1_tiles")


class Layouts1dTest(Layouts, unittest.TestCase):
    NDIM = 1


class Layouts2dTest(Layouts, unittest.TestCase):
    NDIM = 2


class Layouts3dTest(Layouts, unittest.TestCase):
    NDIM = 3


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        run(*sys.argv[2:7])
    else:
        unittest.main()
