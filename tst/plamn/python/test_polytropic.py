"""the polytropic electron closure (#91): Pe = s n^gamma, the entropy s carried by the electron
flow (ds/dt + Ve.grad s = 0, upwind, sub-stepped), its state on whole levels. The cases and
bounds of the upstream polytropic closure's tests, read live from the levels (level_field)
instead of from dumps.

uniform entropy: the ions converge on the middle of the box (Vi = V1 sin(2 pi x / L), B
uniform), Pe = P0 n(x)^gamma: s = Pe / n^gamma stays P0 to round off whatever the density does
(it changes a lot), on one level, with a refined box, and with a tagged level 1 regridded as a
structure in B moves (the new fine nodes' s the coarser level's, not a refined Pe over a noisy
deposit). Seeded from the deposited density it would carry gamma dn / n of its noise, ~10 %.

translation: a pressure balanced structure carried by the ion drift V0 (n = 1, B = Bz(x),
Pe = P0 + dP sin(2 pi x / L), Bz^2 = B0^2 + 2 (P0 + dP - Pe)): s follows s0(x - V0 t), much
better than left in place (0) or carried three times too far (3 V0: a push per PPC stage), on
one level and across a refined box's borders, whose coarse nodes take the fine Pe (the
synchronization) to round off.

layouts: the translation on every other 1D layout / allocator built (tiled, chunked, unified
memory) as on AoSMapped, to round off.
"""

import unittest

import numpy as np

import plamn.backend.phare as ph

GAMMA = 5.0 / 3.0


def domain_nodes(cpp, quantity, ln):
    """x and values of a primal quantity on level ln's domain nodes, shared nodes once"""
    pts, vals, _ = cpp.level_field(quantity, ln)
    x = np.asarray(pts).ravel().astype(float)
    x, idx = np.unique(x, return_index=True)
    return x, np.asarray(vals)[idx]


def entropy(cpp, ln):
    """x, n, s = Pe / n^gamma on level ln's domain nodes (in its cells)"""
    x, pe = domain_nodes(cpp, "Pe", ln)
    xn, n = domain_nodes(cpp, "ions_charge_density", ln)
    np.testing.assert_array_equal(x, xn)
    return x, n, pe / n**GAMMA


def run(config, steps, at=(), **layout):
    """the simulation of config(**layout) over steps, fn(cpp) at each step in at (0: after
    initialization); returns their results by step"""
    ph.global_vars.sim = None
    ph.clearDict()
    config(**layout)
    simulator = ph.Simulator(ph.global_vars.sim, log_to_file=False).setup().initialize()
    cpp, out = simulator.cpp_sim, {}
    for step in range(steps + 1):
        if step > 0:
            simulator.advance()
        if step in at:
            out[step] = at[step](cpp)
    simulator.reset()
    return out


def zero(x):
    return 0.0 * x


def one(x):
    return 1.0 + 0.0 * x


def vth(x):
    return 0.3 + 0.0 * x


class Uniform:
    cells, dl, n1, V1, P0 = 100, 0.5, 0.3, 0.5, 0.5
    dt, steps = 0.01, 100
    L = cells * dl

    @classmethod
    def density(cls, x):
        return 1.0 + cls.n1 * np.sin(2 * np.pi * x / cls.L)

    @classmethod
    def config(cls, bz=None, **layout):
        ph.Simulation(
            time_step=cls.dt,
            time_step_nbr=cls.steps,
            cells=cls.cells,
            dl=cls.dl,
            gamma=GAMMA,
            largest_patch_size=20,
            smallest_patch_size=20,
            **layout,
        )
        ph.MaxwellianFluidModel(
            bx=one,
            by=zero,
            bz=bz or zero,
            protons={
                "charge": 1,
                "density": cls.density,
                "vbulkx": lambda x: cls.V1 * np.sin(2 * np.pi * x / cls.L),
                "vbulky": zero,
                "vbulkz": zero,
                **{f"vth{c}": vth for c in "xyz"},
                "nbr_part_per_cell": 200,
                "init": {"seed": 2026},
            },
        )
        ph.ElectronModel(closure="polytropic", Pe=lambda x: cls.P0 * cls.density(x) ** GAMMA)


class Translation:
    cells, dl, V0, P0, dP, B0 = 200, 0.5, 0.5, 0.5, 0.2, 1.0
    dt, steps = 0.01, 400  # t 4: a translation of 2, 4 cells
    L = cells * dl

    @classmethod
    def Pe(cls, x):
        return cls.P0 + cls.dP * np.sin(2 * np.pi * x / cls.L)

    @classmethod
    def config(cls, **layout):
        ph.Simulation(
            time_step=cls.dt,
            time_step_nbr=cls.steps,
            cells=cls.cells,
            dl=cls.dl,
            gamma=GAMMA,
            largest_patch_size=20,
            smallest_patch_size=20,
            **layout,
        )
        ph.MaxwellianFluidModel(
            bx=zero,
            by=zero,
            bz=lambda x: np.sqrt(cls.B0**2 + 2 * (cls.P0 + cls.dP - cls.Pe(x))),
            protons={
                "charge": 1,
                "density": one,
                "vbulkx": lambda x: cls.V0 + 0.0 * x,
                "vbulky": zero,
                "vbulkz": zero,
                **{f"vth{c}": vth for c in "xyz"},
                "nbr_part_per_cell": 200,
                "init": {"seed": 2026},
            },
        )
        ph.ElectronModel(closure="polytropic", Pe=cls.Pe)


class PolytropicClosureTest(unittest.TestCase):
    def uniform(self, ln, **layout):
        U = Uniform
        level = lambda cpp: entropy(cpp, ln)  # noqa: E731
        out = run(U.config, U.steps, at={0: level, U.steps: level}, **layout)
        (_, n0, s0), (_, n, s) = out[0], out[U.steps]
        compression = np.max(np.abs(n - n0) / n0)
        deviation = [np.max(np.abs(si / U.P0 - 1)) for si in (s0, s)]
        print(f"level {ln}: max dn/n {compression:.2e}, max |s / P0 - 1| {deviation}")
        self.assertGreater(compression, 0.05)  # the flow does compress
        for d in deviation:  # s is the input's, uniform: not the noisy deposit's
            self.assertLess(d, 1e-6)

    def test_uniform_entropy_one_level(self):
        self.uniform(0)

    def test_uniform_entropy_refined_box(self):
        self.uniform(1, refinement_boxes={"L0": {"B0": [(30,), (70,)]}})

    def test_uniform_entropy_regridding(self):
        # level 1 follows a structure in B the flow moves: regridded, its new nodes' entropy
        #  the coarser level's (uniform)
        U = Uniform

        def level1(cpp):
            self.assertEqual(cpp.n_levels(), 2, "no refined level")
            x, _, s = entropy(cpp, 1)
            return x.min(), x.max(), np.max(np.abs(s / U.P0 - 1))

        steps = (0, U.steps // 2, U.steps)
        out = run(
            U.config,
            U.steps,
            at={k: level1 for k in steps},
            bz=lambda x: 1.0 + np.exp(-(((x - 0.35 * U.L) / 3.0) ** 2)),
            refinement="tagging",
            max_nbr_levels=2,
            tagging_threshold=0.01,
        )
        print("level 1 (x min, x max, max |s / P0 - 1|) by step", out)
        self.assertNotEqual(out[0][:2], out[U.steps][:2], "level 1 did not move: no regrid")
        for step in steps:
            self.assertLess(out[step][2], 1e-6, f"entropy not uniform on level 1 at {step}")

    def translation(self, ln, **layout):
        T = Translation

        def at_end(cpp):
            x, _, s = entropy(cpp, ln)
            pe = {l: domain_nodes(cpp, "Pe", l) for l in range(cpp.n_levels())}
            return x * T.dl / 2**ln, s, pe

        x, s, pe = run(T.config, T.steps, at={T.steps: at_end}, **layout)[T.steps]
        t = T.steps * T.dt

        def rms(shift):
            return np.sqrt(np.mean((s - T.Pe(np.mod(x - shift, T.L))) ** 2))

        errors = {k: rms(k * T.V0 * t) for k in (0, 1, 3)}
        print(f"level {ln}: rms s - s0(x - k V0 t)", {k: f"{e:.2e}" for k, e in errors.items()})
        self.assertLess(errors[1], 0.05 * min(errors[0], errors[3]))
        self.assertLess(errors[1], 0.05 * T.dP)

        if ln == 1:  # the coarse nodes under the fine level: the fine Pe (synchronized)
            (xc, pc), (xf, pf) = pe[0], pe[1]
            fine_at = dict(zip(xf / 2, pf))  # fine cells in coarse cells
            inside = [i for i, xi in enumerate(xc) if xf.min() / 2 + 2 < xi < xf.max() / 2 - 2]
            self.assertGreater(len(inside), 10)
            difference = max(abs(pc[i] - fine_at[xc[i]]) for i in inside)
            print(f"max |Pe coarse - Pe fine| {difference:.2e} of {T.dP}")
            self.assertLess(difference, 1e-6 * T.dP)

    def test_translation_one_level(self):
        self.translation(0)

    def test_translation_refined_box(self):
        self.translation(1, refinement_boxes={"L0": {"B0": [(40,), (120,)]}})

    def test_layouts_agree(self):
        # the closure's kernels on every layout built (tiled: on untiled copies; unified memory:
        #  one launch per level): the same as AoSMapped's to round off (deposit order, device
        #  atomics), relative to each quantity's largest value
        import importlib.util

        T = Translation
        quantities = ["Pe", "Se", "EM_B_z", "EM_E_x", "ions_charge_density"]

        def fields(cpp):
            return {
                f"{q}@{ln}": np.asarray(cpp.level_field(q, ln)[1])
                for ln in range(cpp.n_levels())
                for q in quantities
            }

        steps, box = 50, {"L0": {"B0": [(40,), (120,)]}}
        ref = run(T.config, steps, at={steps: fields}, refinement_boxes=box)[steps]
        compared = 0
        for layout, allocator in [
            ("AoSCMTS", "CPU"),
            ("SoAVXChunked", "CPU"),
            ("SoAVXChunked", "GPU_UNIFIED"),
        ]:
            if importlib.util.find_spec(f"plamn.plamn_cpp_1_1_2_{layout}_{allocator}") is None:
                print(f"no 1D {layout} {allocator} module: skipped")
                continue
            got = run(
                T.config,
                steps,
                at={steps: fields},
                refinement_boxes=box,
                model_layout=layout,
                **({"allocator": allocator} if allocator != "CPU" else {}),
            )[steps]
            worst = {k: np.max(np.abs(got[k] - ref[k])) / np.max(np.abs(ref[k])) for k in ref}
            print(layout, allocator, {k: f"{v:.1e}" for k, v in worst.items()})
            for k, v in worst.items():
                self.assertLess(v, 1e-12, f"{layout} {allocator} {k}")
            compared += 1
        if compared == 0:
            self.skipTest("no other 1D layout built (./mkn.py --layout AoSCMTS SoAVXChunked)")


if __name__ == "__main__":
    unittest.main()
