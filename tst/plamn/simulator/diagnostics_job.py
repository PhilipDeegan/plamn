"""read by test_diagnostics.ipp through its embedded interpreter:

    PLAMN_TEST_DIM         1, 2 or 3
    PLAMN_DIAGS_TEST_DIR   where the diagnostics go (per binary and rank count)

two populations, a refinement box (2 levels), every diagnostic plamn writes (EM, ion totals,
per population, momentum tensors) at t0, as pharevtkhdf. declare() declares it, the C++
populates the Dict; importing does nothing (phlop imports every file under tst/ for tests)
"""

import os

import numpy as np

populations = {"protons": (1.0, 1.0), "alpha": (4.0, 2.0)}  # name: (mass, charge)


def declare():
    import plamn

    plamn.patch()

    import pyphare.pharein as ph

    ndim = int(os.environ["PLAMN_TEST_DIM"])
    out = os.environ["PLAMN_DIAGS_TEST_DIR"]

    cells = {1: 40, 2: 20, 3: 10}[ndim]
    box = {1: ([10], [19]), 2: ([5, 5], [12, 12]), 3: ([3, 3, 3], [6, 6, 6])}[ndim]
    timestep = 0.001

    ph.global_vars.sim = None
    ph.clearDict()
    sim = ph.Simulation(
        time_step_nbr=1,
        time_step=timestep,
        cells=[cells] * ndim,
        dl=[0.3] * ndim,
        hyper_resistivity=0.001,
        refinement_boxes={"L0": {"B0": [box[0], box[1]]}},
        diag_options={
            "format": "pharevtkhdf",
            "options": {"dir": out, "mode": "overwrite"},
        },
    )

    def wave(*xyz):  # a function of every coordinate, so dual averages are not trivial
        L = sim.simulation_domain()
        return np.prod(
            [np.sin(2 * np.pi * x / L[i]) for i, x in enumerate(xyz)], axis=0
        )

    def const(v):
        return lambda *xyz: v + 0 * xyz[0]

    def pop(mass, charge, vx, seed):
        return {
            "mass": mass,
            "charge": charge,
            "density": lambda *xyz: 1.0 + 0.2 * wave(*xyz),
            "vbulkx": const(vx),
            "vbulky": const(0.1),
            "vbulkz": const(0.0),
            **{f"vth{c}": const(0.3) for c in "xyz"},
            "nbr_part_per_cell": 20,
            "init": {"seed": seed},
        }

    ph.MaxwellianFluidModel(
        bx=lambda *xyz: 1.0 + 0.1 * wave(*xyz),
        by=lambda *xyz: 0.2 + 0.1 * wave(*xyz),
        bz=const(0.5),
        **{
            name: pop(m, q, 0.5 * (i + 1), 1234 + i)
            for i, (name, (m, q)) in enumerate(populations.items())
        },
    )
    ph.ElectronModel(closure="isothermal", Te=0.12)

    timestamps = [0.0]
    for q in ["E", "B"]:
        ph.ElectromagDiagnostics(quantity=q, write_timestamps=timestamps)
    for q in [
        "charge_density",
        "mass_density",
        "bulkVelocity",
        "charge_flux",
        "momentum_tensor",
    ]:
        ph.FluidDiagnostics(quantity=q, write_timestamps=timestamps)
    for name in populations:
        for q in ["density", "charge_density", "flux", "momentum_tensor"]:
            ph.FluidDiagnostics(
                quantity=q, write_timestamps=timestamps, population_name=name
            )
