"""plamn's stand-in for pyphare's pybindlibs.cpp_etc

pyphare.cpp.cpp_etc_lib() returns this module once plamn.patch() ran. MPI is plamn_etc's:
started by pyphare's startMPI() (SamraiLifeCycle is MPI's life cycle), finalized at exit.
"""

import socket

import numpy as np

from .plamn_etc import (  # noqa: F401
    AllocatorMode,
    MpiLifeCycle,
    BoundaryType,
    Dict,
    HyperMode,
    LayoutMode,
    ReconstructionType,
    RiemannSolverType,
    SlopeLimiterType,
    TimeIntegratorType,
    TimeStepType,
    mpi_barrier,
    mpi_initialized,
    mpi_is_root_rank_for_host,
    mpi_max,
    mpi_rank,
    mpi_size,
    patch_data_ids,
    restart_path_for_time,
    serialized_simulation_string,
)

# ---------------------------------------------------------------- mpi


def mpi_hostname():
    return socket.gethostname()


def make_hierarchy():
    """what pyphare hands to make_simulator: the Dict it filled, plamn builds the
    hierarchy from it"""
    from . import current_dict

    return current_dict()


# pyphare's startMPI() makes a SamraiLifeCycle: here MPI's, no SAMRAI
SamraiLifeCycle = MpiLifeCycle


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


def restored_time(folder):
    """the time a restart folder (restart_path_for_time's) holds: its restart.h5's /restart
    time (pyphare.pharein.restarts.restored_time, pyphare master's cpp_etc's)"""
    from pathlib import Path

    import h5py

    file = Path(folder) / "restart.h5"
    if not file.exists():
        raise RuntimeError(f"no restart file {file} under {folder}")
    with h5py.File(file, "r") as f:
        return float(f["/restart"].attrs["time"])
