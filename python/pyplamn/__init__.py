"""pyplamn: run pyphare's python with plamn's C++

    import pyplamn
    pyplamn.patch()  # before using pyphare

patch() points pyphare at plamn:
  - pharein.initialize.dict_populator.dictator() returns the current simulation's
    plamn Dict, a python owned object, instead of the global pybindlibs.dictator module
  - pharein.clearDict() drops it, so the next simulation starts from an empty Dict
  - pyphare.cpp.cpp_etc_lib() returns pyplamn.etc
  - pyphare.cpp.cpp_lib(sim) returns pyplamn's simulator module for sim's permutation,
    built from the Dict (pyplamn.etc.make_hierarchy() hands it to make_simulator)
  - no SAMRAI/MPI start up, no pybindlibs build checks
"""

import atexit

from . import etc

_dict = None


def current_dict():
    """the Dict pyphare is filling, made on first use"""
    global _dict
    if _dict is None:
        _dict = etc.Dict()
    return _dict


def clear_dict():
    """forget the current Dict, releasing the python functions it holds"""
    global _dict
    if _dict is not None:
        _dict.clear()
    _dict = None


def cpp_lib(sim):
    """the plamn simulator module of sim's build permutation, as pyphare.cpp.cpp_lib"""
    import importlib

    # plamn's own id, not pyphare.cpp.simulator_id: PHARE master's has no layout/allocator,
    # tiling_gpu's does. Without them in the simulation, plamn's defaults
    layout = getattr(sim, "model_layout", None) or "AoSMapped"
    allocator = getattr(sim, "allocator", None) or "CPU"
    sim_id = f"{sim.ndim}_{sim.interp_order}_{sim.refined_particle_nbr}_{layout}_{allocator}"
    name = f"pyplamn.plamn_cpp_{sim_id}"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"{name} is not built, see the py_sim profile in plamn's mkn.yaml"
        ) from e


def patch():
    import pyphare.cpp as cpp
    import pyphare.cpp.validate as validate
    import pyphare.pharein as ph
    import pyphare.pharein.initialize.dict_populator as dict_populator
    import pyphare.simulator.simulator as simulator

    dict_populator.dictator = current_dict
    ph.clearDict = clear_dict

    cpp.cpp_etc_lib = lambda: etc
    cpp.cpp_lib = cpp_lib

    simulator.startMPI = lambda: None
    validate.log_runtime_config = lambda *a, **k: None
    validate.check_build_config_is_runtime_compatible = lambda *a, **k: None

    # the Dict holds python functions: release them before the interpreter goes
    atexit.register(clear_dict)
