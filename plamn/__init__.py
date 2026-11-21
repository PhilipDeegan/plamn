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
    if not getattr(sim, "interp_order", None) and getattr(sim, "mhd_timestepper", None):
        # MHD only: pyphare's MHD id (the scheme is a run time choice). No particles: the
        #  layout only picks the grids, untiled for all but the tile set layouts (pyphare branches
        #  differ in an MHD simulation's default: AoS, AoSMapped): plamn's AoS module
        hall = "true" if getattr(sim, "hall", False) else "false"
        if "TS" not in layout:
            layout = "AoS"
        choices = (sim.reconstruction, sim.limiter, sim.riemann)
        sim_id = f"{sim.ndim}_{layout}_{allocator}_" + "_".join(choices) + f"_{hall}"
        build = (
            f"./mkn.py --mhd --dim {sim.ndim} --layout {layout} --allocator {allocator}"
            f" --reconstruction {choices[0]} --limiter {choices[1]} --riemann {choices[2]}"
            f" --hall {hall}"
        )
    else:
        sim_id = (
            f"{sim.ndim}_{sim.interp_order}_{sim.refined_particle_nbr}_{layout}_{allocator}"
        )
        build = (
            f"./mkn.py --dim {sim.ndim} --interp {sim.interp_order}"
            f" --refined {sim.refined_particle_nbr} --layout {layout} --allocator {allocator}"
        )
        if getattr(sim, "mhd_timestepper", None):
            # MHD levels under hybrid ones (AMR/max_mhd_level): the hybrid id, then MHD's
            hall = "true" if getattr(sim, "hall", False) else "false"
            choices = (sim.reconstruction, sim.limiter, sim.riemann)
            sim_id += "_" + "_".join(choices) + f"_{hall}"
            build += (
                f" --coupled --reconstruction {choices[0]} --limiter {choices[1]}"
                f" --riemann {choices[2]} --hall {hall}"
            )
    name = f"plamn.plamn_cpp_{sim_id}"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"{name} is not built: {build}") from e


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


def set_tag_boxes(sim, tag_boxes):
    """levels made at initialization from tag_boxes ({level: boxes}, levels from 1, each in its
    own level's cells: pyphare Boxes or (lower, upper) pairs, e.g. Tagger(sim).tag()) instead of
    from their coarser level's tags (refinement="tagging"); each must be nested in its coarser
    level (the simulator checks)"""
    if sim.refinement != "tagging":
        raise ValueError('tag boxes need refinement="tagging"')
    from plamn.amr import _box

    sim._plamn_tag_boxes = {
        int(ln): [_box(b) for b in boxes] for ln, boxes in tag_boxes.items() if int(ln) > 0
    }


def _patch_tag_boxes():
    from pyphare.pharein.initialize import general

    populate_amr = general.populate_amr

    def populate_amr_with_tag_boxes(dp, sim):
        populate_amr(dp, sim)
        for ln, boxes in sorted((getattr(sim, "_plamn_tag_boxes", None) or {}).items()):
            path = f"simulation/AMR/refinement/tagging/tag_boxes/L{ln}/"
            dp.add_int(path + "nbr_boxes", len(boxes))
            for i, (lower, upper) in enumerate(boxes):
                for d, k in enumerate("xyz"[: len(lower)]):
                    dp.add_int(path + f"B{i}/lower/{k}", lower[d])
                    dp.add_int(path + f"B{i}/upper/{k}", upper[d])

    general.populate_amr = populate_amr_with_tag_boxes


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


def _patch_enum_int():
    """pyphare's DictPopulator.add_enum_int, its enum member names matched without case: pyphare
    branches differ (tiling_gpu: as given, "TVDRK2"; wrangler: lower cased, "tvdrk2"), plamn.etc's
    enums are named as the C++ (TVDRK2, VanLeer, AoSMapped)"""
    from pyphare.pharein.initialize.dict_populator import DictPopulator

    def add_enum_int(self, path, enum_name, member_name):
        enum_cls = getattr(etc, enum_name)
        members = {name.lower(): member for name, member in enum_cls.__members__.items()}
        member = members.get(str(member_name).lower())
        if member is None:
            raise ValueError(
                f"{enum_name}: unknown value '{member_name}',"
                f" expected one of {list(enum_cls.__members__)}"
            )
        self.add_int(path, int(member.value))

    DictPopulator.add_enum_int = add_enum_int


def _patch_live_simulators():
    """plamn's simulators hold no process wide state (PHARE's SAMRAI statics): several may be
    live side by side (the restart tests run one from another's restart, both advanced). A
    pyphare that allows one live simulator per process (wrangler: _replace_live_simulator raises
    while the previous one has not reached its final time, else resets it) keeps its record of
    the last one set up, nothing else"""
    import weakref

    from pyphare.simulator import simulator

    if not hasattr(simulator.Simulator, "_replace_live_simulator"):
        return

    def replace_live_simulator(self):
        from pyphare.simulator import _simulator

        _simulator.obj = weakref.ref(self)

    simulator.Simulator._replace_live_simulator = replace_live_simulator


# plamn's Simulation keywords that PHARE master's pyphare does not know: the module's particle
#  layout and allocator (cpp_lib reads them; default AoSMapped, CPU)
PLAMN_KEYWORDS = ("model_layout", "allocator")


def _patch_keywords():
    from pyphare.pharein import simulation

    if getattr(simulation.check_optional_keywords, "_plamn", False):
        return
    original = simulation.check_optional_keywords

    def check_optional_keywords(**kwargs):
        extra = list(original(**kwargs))
        return extra + [k for k in PLAMN_KEYWORDS if k not in extra]

    check_optional_keywords._plamn = True
    simulation.check_optional_keywords = check_optional_keywords


def _patch_coupled_input():
    """MHD levels under hybrid ones (AMR/max_mhd_level, model_options MHD and hybrid): plamn's
    own input, no pyphare has coupling. One MHDModel holds the profiles and the ion species the
    hybrid levels carry, MHDModel(..., protons={charge, mass, nbr_part_per_cell, init,
    density_cut_off}), with an ElectronModel. Each species' Maxwellian comes from the MHD
    state (one species for now): n = rho / m, V, vth = sqrt(max(P - Pe, 0) / (n m)) with the
    isothermal electrons' Pe = Te q n"""
    import numpy as np
    from pyphare.pharein import global_vars
    from pyphare.pharein.mhd_model import MHDModel

    init = MHDModel.__init__

    def init_with_species(self, density=None, vx=None, vy=None, vz=None, bx=None, by=None,
                          bz=None, p=None, **species):
        init(self, density, vx, vy, vz, bx, by, bz, p)
        self.populations = list(species)
        if len(self.populations) > 1:
            raise ValueError("plamn coupled input: one ion species for now")
        md = self.model_dict
        for name, d in species.items():
            mass, charge = float(d.get("mass", 1.0)), float(d.get("charge", 1.0))
            if name in md:
                raise ValueError(f"population {name}: already registered")

            def n(*x, mass=mass):
                return md["density"](*x) / mass

            def vth(*x, mass=mass, charge=charge):
                te = global_vars.sim.electrons.closure.Te
                ni = n(*x)
                ti = np.maximum(md["p"](*x) - te * charge * ni, 0.0) / ni
                return np.sqrt(ti / mass)

            md[name] = {
                "charge": charge,
                "mass": mass,
                "density": n,
                "vx": md["vx"],
                "vy": md["vy"],
                "vz": md["vz"],
                "vthx": vth,
                "vthy": vth,
                "vthz": vth,
                "nbrParticlesPerCell": int(d.get("nbr_part_per_cell", 100)),
                "init": dict(seed=d.get("init", {}).get("seed")),
                "density_cut_off": float(d.get("density_cut_off", 1e-16)),
            }

    MHDModel.__init__ = init_with_species
    MHDModel.nbr_populations = lambda self: len(getattr(self, "populations", []))


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

    _patch_keywords()
    _patch_layout_for()
    _patch_enum_int()
    _patch_live_simulators()
    _patch_density_background()
    _patch_tag_boxes()
    _patch_coupled_input()

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
