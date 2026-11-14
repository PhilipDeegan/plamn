"""plamn's stand-in for pyphare's pybindlibs.cpp_etc

pyphare.cpp.cpp_etc_lib() returns this module once pyplamn.patch() ran. Serial for now:
the mpi_* functions answer for a single rank.
"""

import socket

import numpy as np

from .plamn_etc import (  # noqa: F401
    AllocatorMode,
    BoundaryType,
    Dict,
    HyperMode,
    LayoutMode,
    TimeStepType,
)


# ---------------------------------------------------------------- mpi, serial


def mpi_size():
    return 1


def mpi_rank():
    return 0


def mpi_barrier():
    pass


def mpi_initialized():
    return False


def mpi_hostname():
    return socket.gethostname()


def mpi_is_root_rank_for_host():
    return True


def make_hierarchy():
    """what pyphare hands to make_simulator: the Dict it filled, plamn builds the
    hierarchy from it"""
    from . import current_dict

    return current_dict()


class SamraiLifeCycle:
    """nothing to start or stop without SAMRAI"""

    @staticmethod
    def reset():
        pass


# ---------------------------------------------------------------- init function arrays


def are_the_same_data(array, other):
    """true if both arrays view the same memory (C++ passes zero-copy views)"""
    return (
        np.asarray(array).__array_interface__["data"][0]
        == np.asarray(other).__array_interface__["data"][0]
    )


def makePyArrayWrapper(array):
    """what a user init function returns to C++: a contiguous float64 array"""
    return np.ascontiguousarray(array, dtype=np.float64)


# ---------------------------------------------------------------- build information


def supported_layouts():
    return [LayoutMode.AoSMapped]


def supported_gpu_layouts():
    return []


def compile_options():
    return {"gpu": False}


def phare_build_config():
    return {}


def phare_deps():
    return {}
