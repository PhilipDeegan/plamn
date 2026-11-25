"""A reference build standing in for plamn.backend.phare, so the experiments run on it unchanged:
put this directory first on PYTHONPATH, then the reference's build and pyphare, e.g.

    PYTHONPATH=etc/experiments/energy/reference_backend:$REF/build:$REF/pyphare python3 ...

pyphare's names, the Simulation without plamn's own options (model_layout, allocator), one patch
per level by default (the budgets read level 0 as one patch)"""

import pyphare.pharein as _pi
from pyphare.pharein import *
from pyphare.simulator.simulator import Simulator  # noqa: F401


def Simulation(**kw):
    for k in ("model_layout", "allocator"):
        kw.pop(k, None)
    kw.setdefault("largest_patch_size", tuple(kw["cells"]))
    return _pi.Simulation(**kw)
