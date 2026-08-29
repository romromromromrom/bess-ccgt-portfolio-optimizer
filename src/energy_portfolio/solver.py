"""Solver selection. HiGHS is used through Pyomo's APPSI interface.

HiGHS is fully open-source (MIT) and ships as a prebuilt wheel via `highspy`,
so the whole stack installs with pip and needs no commercial licence — which
matters for a demonstrator that has to run in CI and in a free-tier container.
"""

from __future__ import annotations

import pyomo.environ as pyo

_CANDIDATES = ("appsi_highs", "highs", "cbc", "glpk")


class SolverNotAvailable(RuntimeError):
    pass


def get_solver():
    for name in _CANDIDATES:
        try:
            solver = pyo.SolverFactory(name)
        except Exception:
            continue
        if solver is not None and solver.available(exception_flag=False):
            return solver
    raise SolverNotAvailable(
        "No MILP solver available. Install highspy (`pip install highspy`)."
    )


def solve(model: pyo.ConcreteModel, tee: bool = False) -> None:
    """Solve in place and raise if the model is not proven optimal."""
    solver = get_solver()
    results = solver.solve(model, tee=tee)
    status = results.solver.termination_condition
    if status not in (
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible,
        pyo.TerminationCondition.locallyOptimal,
    ):
        raise RuntimeError(f"Optimization failed with termination condition: {status}")
