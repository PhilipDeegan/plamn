"""spheromak_dipole.py outputs (pharevtkhdf: EM_B, ions_mass_density, ions_bulkVelocity), one run
or two (e.g. two resistivities, layouts or resolutions with the same levels)

    python3 etc/experiments/spheromak_compare.py phare_outputs/sph_hybrid
    python3 etc/experiments/spheromak_compare.py phare_outputs/sph_a phare_outputs/sph_b \\
        --plots phare_outputs/sph_plots --times 0 0.5 1

Per run, per dump: magnetic energy (|B|^2 / 2) and bulk kinetic energy (rho V^2 / 2) per level
(level 1: over its box), and over level 0 outside level 1 (the quiet far field): the
mass density's range, max |V|, max |B|. Two runs: per level, the rms and max difference of B, the
mass density and the bulk velocity where both have the same boxes. --plots: the z mid plane of
|B|, the mass density and |V| per level, each run and their difference.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

FILES = {
    "B": "EM_B.vtkhdf",
    "rho": "ions_mass_density.vtkhdf",
    "V": "ions_bulkVelocity.vtkhdf",
}


def read(path):
    """[(time, {level: [(lo, hi, values (z, y, x, components)), ...], spacing}), ...]"""
    out = []
    with h5py.File(path, "r") as h:
        times = h["VTKHDF/Steps/Values"][:]
        levels = sorted(int(k[5:]) for k in h["VTKHDF"] if k.startswith("Level"))
        for i, t in enumerate(times):
            per = {}
            for ln in levels:
                L, S = f"VTKHDF/Level{ln}", f"VTKHDF/Steps/Level{ln}"
                n_boxes = h[f"{S}/NumberOfAMRBox"][i]
                b0 = h[f"{S}/AMRBoxOffset"][i]
                off = h[f"{S}/PointDataOffset/data"][i]
                patches = []
                for box in h[f"{L}/AMRBox"][b0 : b0 + n_boxes]:
                    lo, hi = box[0::2], box[1::2]
                    n = hi - lo + 2  # nodes
                    d = h[f"{L}/PointData/data"][off : off + n.prod()]
                    off += n.prod()
                    patches.append((lo, hi, d.reshape(n[2], n[1], n[0], -1)))
                per[ln] = (patches, h[L].attrs["Spacing"].astype(float))
            out.append((float(t), per))
    return out


def load(run):
    run = Path(run)
    return {k: read(run / f) for k, f in FILES.items()}


def cells(values):
    """a patch's node values on its cells (the last node of each direction dropped): sums over
    patches count each cell once"""
    return values[:-1, :-1, :-1]


def covered(lo, hi, fine):
    """per cell of the level 0 patch [lo, hi], whether a level 1 patch covers it (ratio 2)"""
    mask = np.zeros(tuple((hi - lo + 1)[::-1]), bool)
    for flo, fhi, _ in fine:
        clo, chi = np.maximum(flo // 2, lo), np.minimum(fhi // 2, hi)
        if np.all(chi >= clo):
            mask[
                tuple(
                    slice(a - l, b - l + 1)
                    for a, b, l in zip(clo[::-1], chi[::-1], lo[::-1])
                )
            ] = True
    return mask


def summary(name, d):
    print(f"\n{name}")
    print(
        " t      WB0      KE0      WB1      KE1     | level 0 outside level 1: rho min/max   |V|max   |B|max"
    )
    for i, (t, B) in enumerate(d["B"]):
        rho, V = d["rho"][i][1], d["V"][i][1]
        row = [f"{t:5.2f}"]
        for ln in (0, 1):
            if ln not in B or not B[ln][0]:
                row += ["   -    ", "   -    "]
                continue
            dv = np.prod(B[ln][1])
            wb = sum(0.5 * (cells(b) ** 2).sum() * dv for _, _, b in B[ln][0])
            ke = sum(
                0.5 * (cells(r)[..., 0] * (cells(v) ** 2).sum(-1)).sum() * dv
                for (_, _, r), (_, _, v) in zip(rho[ln][0], V[ln][0])
            )
            row += [f"{wb:8.4f}", f"{ke:8.4f}"]
        fine = B[1][0] if 1 in B else []
        r_out, v_out, b_out = [], [], []
        for (lo, hi, b), (_, _, r), (_, _, v) in zip(B[0][0], rho[0][0], V[0][0]):
            out = ~covered(lo, hi, fine)
            r_out.append(cells(r)[..., 0][out])
            v_out.append(np.linalg.norm(cells(v), axis=-1)[out])
            b_out.append(np.linalg.norm(cells(b), axis=-1)[out])
        r_out, v_out, b_out = (np.concatenate(x) for x in (r_out, v_out, b_out))
        row += [
            f"| {r_out.min():.4f} {r_out.max():.4f}" if r_out.size else "| -",
            f"{v_out.max():.4f}" if v_out.size else "-",
            f"{b_out.max():.4f}" if b_out.size else "-",
        ]
        print(" ".join(row))


def differences(a, b):
    print("\ndifferences (second - first): rms / max per level, where the boxes match")
    for q in FILES:
        for (ta, A), (tb, Bq) in zip(a[q], b[q]):
            if abs(ta - tb) > 1e-9:
                continue
            row = [f"{q:4} t {ta:5.2f}"]
            for ln in sorted(set(A) & set(Bq)):
                pa, pb = A[ln][0], Bq[ln][0]
                same = len(pa) == len(pb) and all(
                    np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1])
                    for x, y in zip(pa, pb)
                )
                if not same or not pa:
                    row.append(f"L{ln} -")
                    continue
                diff = np.concatenate([(y[2] - x[2]).ravel() for x, y in zip(pa, pb)])
                row.append(
                    f"L{ln} {np.sqrt((diff**2).mean()):.3e} / {np.abs(diff).max():.3e}"
                )
            print("  ".join(row))


def plots(runs, out, times):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = list(runs)
    first = runs[names[0]]
    for want in times:
        i = int(np.argmin([abs(t - want) for t, _ in first["B"]]))
        t = first["B"][i][0]
        for ln in sorted(first["B"][i][1]):
            cols = names + (["difference"] if len(names) == 2 else [])
            fig, ax = plt.subplots(
                3, len(cols), figsize=(4.5 * len(cols), 11), squeeze=False
            )
            maps = {}
            for name in names:
                d = runs[name]
                patches, dl = d["B"][i][1][ln]
                if len(patches) != 1:
                    continue  # one patch per level only
                lo, hi, b = patches[0]
                k = b.shape[0] // 2
                maps[name] = (
                    [
                        lo[0] * dl[0],
                        (hi[0] + 1) * dl[0],
                        lo[1] * dl[1],
                        (hi[1] + 1) * dl[1],
                    ],
                    {
                        "|B|": np.linalg.norm(b, axis=-1)[k],
                        "mass density": d["rho"][i][1][ln][0][0][2][..., 0][k],
                        "|V|": np.linalg.norm(d["V"][i][1][ln][0][0][2], axis=-1)[k],
                    },
                )
            if len(maps) == 2:
                (e, m0), (_, m1) = maps[names[0]], maps[names[1]]
                if all(m0[q].shape == m1[q].shape for q in m0):
                    maps["difference"] = (e, {q: m1[q] - m0[q] for q in m0})
            for c, name in enumerate(cols):
                if name not in maps:
                    continue
                extent, fields = maps[name]
                for r, (q, f) in enumerate(fields.items()):
                    im = ax[r, c].imshow(f, origin="lower", extent=extent)
                    ax[r, c].set_title(f"{name}: L{ln} {q}, t {t:.2f}", fontsize=9)
                    fig.colorbar(im, ax=ax[r, c], shrink=0.8)
            fig.tight_layout()
            path = out / f"L{ln}_t{t:.2f}.png"
            fig.savefig(path, dpi=70)
            plt.close(fig)
            print("wrote", path)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("runs", nargs="+", help="one or two output directories")
    p.add_argument("--plots", help="a directory for the mid plane maps")
    p.add_argument("--times", nargs="*", type=float, default=[0.0, 0.5, 1.0])
    a = p.parse_args()
    if len(a.runs) > 2:
        p.error("one or two runs")
    names = [Path(r).name for r in a.runs]
    if len(set(names)) < len(names):  # the same name twice: numbered
        names = [f"{i}:{n}" for i, n in enumerate(names)]
    runs = {n: load(r) for n, r in zip(names, a.runs)}
    for name, d in runs.items():
        summary(name, d)
    if len(runs) == 2:
        differences(*runs.values())
    if a.plots:
        plots(runs, a.plots, a.times)


if __name__ == "__main__":
    main()
