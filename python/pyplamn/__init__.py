"""pyplamn: run pyphare's python with plamn's C++

    import pyplamn
    pyplamn.patch()  # before using pyphare

patch() points pyphare at plamn:
  - pharein.initialize.dict_populator.dictator() returns the current simulation's
    plamn Dict, a python owned object, instead of the global pybindlibs.dictator module
  - pharein.clearDict() drops it, so the next simulation starts from an empty Dict
  - pyphare.cpp.cpp_etc_lib() returns pyplamn.etc
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


def patch():
    import pyphare.cpp as cpp
    import pyphare.cpp.validate as validate
    import pyphare.pharein as ph
    import pyphare.pharein.initialize.dict_populator as dict_populator
    import pyphare.simulator.simulator as simulator

    dict_populator.dictator = current_dict
    ph.clearDict = clear_dict

    cpp.cpp_etc_lib = lambda: etc

    simulator.startMPI = lambda: None
    validate.log_runtime_config = lambda *a, **k: None
    validate.check_build_config_is_runtime_compatible = lambda *a, **k: None

    # the Dict holds python functions: release them before the interpreter goes
    atexit.register(clear_dict)
