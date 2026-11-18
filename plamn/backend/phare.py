"""pyphare's API on plamn, without naming pyphare:

    import plamn.backend.phare as ph

    sim = ph.Simulation(cells=..., refinement="tagging", ...)
    ph.MaxwellianFluidModel(protons={...})
    ph.ElectronModel(closure="isothermal", Te=0.0)
    ph.Simulator(sim).run()

The names are pyphare's (pharein's, the Simulator, pharesee's Run, the submodules), resolved on
first use: the first attribute read applies plamn.patch() (pyphare's C++ modules swapped for
plamn's, its input extensions such as a population's density_background), so importing this
module does nothing by itself (nothing runs at import: tests import it at module level).
"""

import importlib

# name -> (module, attribute in it, or None for the module itself)
_forward = {
    **{
        name: ("pyphare.pharein", name)
        for name in (
            "ElectromagDiagnostics",
            "ElectronModel",
            "FluidDiagnostics",
            "InfoDiagnostics",
            "LoadBalancer",
            "MHDDiagnostics",
            "MHDModel",
            "MaxwellianFluidModel",
            "MetaDiagnostics",
            "ParticleDiagnostics",
            "Simulation",
            "UniformModel",
            "NO_GUI",
            "clearDict",
            "populateDict",
        )
    },
    "global_vars": ("pyphare.pharein.global_vars", None),
    "Simulator": ("pyphare.simulator.simulator", "Simulator"),
    "startMPI": ("pyphare.simulator.simulator", "startMPI"),
    "Run": ("pyphare.pharesee.run", "Run"),
    "pharein": ("pyphare.pharein", None),
    "pharesee": ("pyphare.pharesee", None),
    "simulator": ("pyphare.simulator", None),
    "cpp": ("pyphare.cpp", None),
}


def __getattr__(name):
    if name not in _forward:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from plamn import patch

    patch()
    module_name, attribute = _forward[name]
    module = importlib.import_module(module_name)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value  # resolved once
    return value


def __dir__():
    return sorted(set(globals()) | set(_forward))
