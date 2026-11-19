#!/usr/bin/env python3
"""synthetic frames for view3d (its stdin format, view3d.cpp): a blob turning in a 16^3
level 0, a level 1 box over its middle, one frame per 0.1 s.

    python3 etc/view3d/fake_frames.py | etc/view3d/bin/view3d/plamn.view3d
"""

import struct
import sys
import time

import numpy as np


def frame(title, t, step, xyz, value, radius):
    head = title.encode()
    return b"".join([
        b"PLMF",
        struct.pack("<I", len(head)), head,
        struct.pack("<dQQ", t, step, len(value)),
        np.asarray(xyz, "<f4").tobytes(),
        np.asarray(value, "<f4").tobytes(),
        np.asarray(radius, "<f4").tobytes(),
    ])


def level(n, lo, hi, dx):
    c = [np.arange(l, h) * dx + dx / 2 for l, h in zip(lo, hi)]
    return np.stack(np.meshgrid(*c, indexing="ij"), -1).reshape(-1, 3)


def main():
    l0 = level(16, (0, 0, 0), (16, 16, 16), 1.0)
    l1 = level(32, (8, 8, 8), (24, 24, 24), 0.5)  # level 1 over cells 4..11 of level 0
    inside = np.all((l0 >= 4) & (l0 < 12), axis=1)
    pts = np.concatenate([l0[~inside], l1])
    rad = np.concatenate([np.full((~inside).sum(), 0.5), np.full(len(l1), 0.25)])
    out = sys.stdout.buffer
    for step in range(60):
        a = 0.1 * step
        centre = np.array([8 + 4 * np.cos(a), 8 + 4 * np.sin(a), 8.0])
        value = np.exp(-np.sum((pts - centre) ** 2, axis=1) / 8) - 0.3
        out.write(frame("fake blob", 0.1 * step, step, pts, value, rad))
        out.flush()
        time.sleep(0.1)


if __name__ == "__main__":
    main()
