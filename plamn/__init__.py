"""plamn: run pyphare's python with plamn's C++

    import plamn
    plamn.patch()  # before using pyphare

patch() points pyphare at plamn:
  - pharein.initialize.dict_populator.dictator() returns the current simulation's
    plamn Dict, a python owned object, instead of the global pybindlibs.dictator module
  - pharein.clearDict() drops it, so the next simulation starts from an empty Dict
  - pyphare.cpp.cpp_etc_lib() returns plamn.etc
  - pyphare.cpp.cpp_lib(sim) returns plamn's simulator module for sim's permutation,
    built from the Dict (plamn.etc.make_hierarchy() hands it to make_simulator)
  - MPI started by pyphare's startMPI() (plamn_etc's MpiLifeCycle), no SAMRAI, no pybindlibs
    build checks
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
    sim_id = (
        f"{sim.ndim}_{sim.interp_order}_{sim.refined_particle_nbr}_{layout}_{allocator}"
    )
    name = f"plamn.plamn_cpp_{sim_id}"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"{name} is not built: ./mkn.py --dim {sim.ndim} --interp {sim.interp_order}"
            f" --refined {sim.refined_particle_nbr} --layout {layout} --allocator {allocator}"
        ) from e


def _patch_density_background():
    """a population's density_background (default 0): the constant part of its density, added
    to its density function (which then holds what varies, and must not take the density below
    the cut-off where the background is above it); a background at least the population's
    density_cut_off populates every cell, so counting its particles samples nothing. pyphare's
    add_population takes a fixed keyword list: take it here, write it next to density_cut_off"""
    from pyphare.pharein.initialize import hybrid
    from pyphare.pharein.maxwellian_fluid_model import MaxwellianFluidModel

    add_population = MaxwellianFluidModel.add_population

    def add_population_with_background(self, name, density_background=0.0, **kwargs):
        add_population(self, name, **kwargs)
        if not density_background >= 0.0:
            raise ValueError(f"population {name}: density_background must be >= 0")
        self.model_dict[name]["density_background"] = float(density_background)

    MaxwellianFluidModel.add_population = add_population_with_background

    populate = hybrid._populate_populations

    def populate_with_background(dp, sim):
        populate(dp, sim)
        model = sim.model.model_dict
        for i, pop in enumerate(sim.model.populations):
            path = f"simulation/ions/pop{i:d}/particle_initializer/density_background"
            dp.add_double(path, model[pop].get("density_background", 0.0))

    hybrid._populate_populations = populate_with_background


def _patch_layout_for():
    """pharesee's hierarchy_from_sim builds hybrid layouts with PHARE's ghost widths (field 2, 4, 4
    and particles 1, 2, 2 for interp 1, 2, 3, in HybridGridLayoutFor): use the width each patch
    data carries (nGhosts, from plamn's allowed particle cell jump) instead. Both the tiling_gpu
    (no model argument) and PHARE master signatures."""
    try:  # pharesee needs matplotlib, not every plamn user has it
        from pyphare.pharesee.hierarchy import fromsim
    except ImportError:
        return
    from pyphare.core import gridlayout
    from pyphare.core.box import Box

    original = getattr(
        fromsim.make_layout_for, "_plamn_original", fromsim.make_layout_for
    )

    def make_layout_for(simulator, patch_data, qty, dl, *model):
        if model and model[0] == "MHD":
            return original(simulator, patch_data, qty, dl, *model)
        box = Box(patch_data.lower, patch_data.upper)
        return gridlayout.HybridGridLayoutFor(
            box,
            patch_data.origin,
            dl,
            simulator.interp_order(),
            ghosts_nbr=[patch_data.nGhosts] * box.ndim,
            is_particle_layout=(qty == "particles"),
        )

    make_layout_for._plamn_original = original
    fromsim.make_layout_for = make_layout_for


def _finalize_mpi():
    from pyphare.simulator import simulator

    life_cycle = simulator.life_cycles.pop("samrai", None)
    if life_cycle is not None:
        life_cycle.finalize()


_patched = False


def patch():
    """idempotent: test classes call it from setUpClass (nothing runs at import)"""
    global _patched
    if _patched:
        return
    _patched = True

    import pyphare.pharein as ph
    from pyphare import cpp
    from pyphare.cpp import validate
    from pyphare.pharein.initialize import dict_populator
    from pyphare.simulator import simulator

    dict_populator.dictator = current_dict
    ph.clearDict = clear_dict

    cpp.cpp_etc_lib = lambda: etc
    cpp.cpp_lib = cpp_lib

    _patch_layout_for()
    _patch_density_background()

    # restart directories: 10 digits each side of the point, as the C++ restart_path_for_time
    #  (pyphare's 5 decimals cannot tell small steps apart); keep_last deletes by this name
    from pyphare.pharein import restarts

    restarts.format_time = lambda time: f"{time:021.10f}"

    validate.log_runtime_config = lambda *a, **k: None
    validate.check_build_config_is_runtime_compatible = lambda *a, **k: None

    # the Dict holds python functions: release them before the interpreter goes
    atexit.register(clear_dict)
    # MPI, if started (pyphare's startMPI): finalized before the interpreter goes
    atexit.register(_finalize_mpi)
