"""Energy budgets on the Yee grid (#81): node averaged dumps hide grid scale B, so the budget
reads each field at its own centring (phareh5). 3D, per dump, over level 0 (periodic, one patch):
  W_B  each B component squared over its own points                       / 2
  K    tr(momentum tensor) over the nodes / 2 (the deposit's node sums are the particles'
       sum w v v exactly)
  D    eta sum J^2 + eta_h sum |grad J|^2, J the discrete curl on the edges (the work of the
       discrete resistive / hyper resistive terms on a periodic grid)
and the balance W_B + K + int D (trapezoid over the dumps: dumps far apart under fast early
dissipation over count it), constant if the scheme conserves energy (Te 0: the electrons hold
none).

    run --case spheromak|harris3d <the case's arguments, --out DIR>
        the case (etc/experiments/spheromak_dipole.py, energy/harris3d.py) with phareh5 dumps,
        one patch on level 0 and the ions' momentum tensor dumped with the fields
    budget DIR <dl> <eta> <eta_h>

From the plamn root:
    PYTHONPATH=.:$PHARE/pyphare python3 etc/experiments/energy/yee_budget.py run --case spheromak \\
        --cells 30 --dl 0.1 --margin -1 --time-step 1e-3 --final-time 0.2 --diag-every 0.05 \\
        --resistivity 0 --hyper-resistivity 0 --out phare_outputs/yee_spheromak
    PYTHONPATH=.:$PHARE/pyphare python3 etc/experiments/energy/yee_budget.py budget \\
        phare_outputs/yee_spheromak 0.1 0 0
A reference build instead of plamn: PYTHONPATH=etc/experiments/energy/reference_backend:
$REF/build:$REF/pyphare (reference_backend/plamn/backend/phare.py)
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASES = {
    "spheromak": HERE.parent / "spheromak_dipole.py",
    "harris3d": HERE / "harris3d.py",
}


def load(case):
    spec = importlib.util.spec_from_file_location(case, CASES[case])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(argv):
    import plamn.backend.phare as ph

    case = argv[argv.index("--case") + 1]
    argv = [
        a for i, a in enumerate(argv) if a != "--case" and argv[i - 1 : i] != ["--case"]
    ]

    Simulation = ph.Simulation

    def h5_simulation(**kw):
        kw["diag_options"] = {**kw["diag_options"], "format": "phareh5"}
        kw.setdefault("largest_patch_size", tuple(kw["cells"]))
        return Simulation(**kw)

    ph.Simulation = h5_simulation
    module = load(case)
    config = module.config

    def with_tensor(a):
        out = config(a)
        times = np.arange(0, a.final_time + 0.5 * a.diag_every, a.diag_every)
        ph.FluidDiagnostics(quantity="momentum_tensor", write_timestamps=times)
        return out

    module.config = with_tensor
    module.main(argv)


def domain(pd):
    """the patch data's domain values: ghosts cut, the last primal point per axis dropped
    (periodic: it is the first one)"""
    g, n = pd.ghosts_nbr, pd.layout.box.shape
    return pd.dataset[tuple(slice(gi, gi + ni) for gi, ni in zip(g, n))].astype(
        np.float64
    )


def level0(hier, t, names):
    patches = hier.level(0, t).patches
    assert len(patches) == 1, "one level 0 patch expected (largest_patch_size = cells)"
    return [domain(patches[0].patch_datas[n]) for n in names]


def back(
    f, ax, dl
):  # primal from dual (edge from face) along ax: periodic backward difference
    return (f - np.roll(f, 1, axis=ax)) / dl


def budget(out, dl, eta, eta_h):
    from pyphare.pharesee.run import Run

    r, rows, dV = Run(out), [], dl**3  # 3D: the curl takes the three directions
    for t in r.times("EM_B"):
        bx, by, bz = level0(r.GetB(t, all_primal=False), t, ("Bx", "By", "Bz"))
        W = 0.5 * dV * sum(np.sum(b**2) for b in (bx, by, bz))
        # Yee: Bx (p, d, d), By (d, p, d), Bz (d, d, p); J on the edges: Jx = dy Bz - dz By ...
        jx = back(bz, 1, dl) - back(by, 2, dl)
        jy = back(bx, 2, dl) - back(bz, 0, dl)
        jz = back(by, 0, dl) - back(bx, 1, dl)
        J2 = sum(np.sum(j**2) for j in (jx, jy, jz))
        gJ2 = sum(np.sum(back(j, ax, dl) ** 2) for j in (jx, jy, jz) for ax in range(3))
        D = (eta * J2 + eta_h * gJ2) * dV
        K = np.nan  # a reference build whose phareh5 has no momentum tensor: unknown
        try:
            M = r._get_hierarchy(t, "ions_momentum_tensor.h5")
            K = 0.5 * dV * sum(np.sum(m) for m in level0(M, t, ("Mxx", "Myy", "Mzz")))
        except (OSError, KeyError, RuntimeError, ValueError) as e:
            print(f"t {t}: no momentum tensor ({type(e).__name__})")
        rows.append((t, W, K, D))
    rows = np.array(rows)
    t = rows[:, 0]
    iD = np.concatenate(
        [[0], np.cumsum(0.5 * (rows[1:, 3] + rows[:-1, 3]) * np.diff(t))]
    )
    W0, tot0 = rows[0, 1], rows[0, 1] + np.nan_to_num(rows[0, 2])
    print(f"{out}: level 0 on the Yee grid (dl {dl}, eta {eta}, eta_h {eta_h})")
    print("     t        W_B          K      int D        sum   resid/W_B0")
    for i, (tt, W, K, D) in enumerate(rows):
        tot = W + np.nan_to_num(K) + iD[i]
        print(
            f"{tt:6.3f} {W:10.5f} {K:10.5f} {iD[i]:10.5f} {tot:10.5f}   {(tot - tot0) / W0:+.2e}"
        )


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2:])
    elif sys.argv[1] == "budget":
        budget(sys.argv[2], *map(float, sys.argv[3:6]))
    else:
        sys.exit(__doc__)
