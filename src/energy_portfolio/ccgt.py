"""MILP: CCGT unit commitment co-optimized with reserve capacity sales.

The interesting behaviour this model reproduces — and the one worth being able
to explain — is that a CCGT will sometimes **start up and sit at Pmin while
the spot price is below its marginal cost**, losing money on energy, purely to
be able to sell the Pmax - Pmin headroom as upward aFRR. Nothing hard-codes
that: it falls out of the objective once energy and reserve are optimized
jointly. Solving the two markets separately cannot find it.
"""

from __future__ import annotations

import math

import pandas as pd
import pyomo.environ as pyo

from energy_portfolio.config import CCGTConfig, ReserveProductConfig
from energy_portfolio.solver import solve
from energy_portfolio.valuation import (
    ccgt_opportunity_cost,
    reserve_unit_value_eur_per_mw_h,
)


def _as_series(value, index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.astype(float)
    return pd.Series(float(value), index=index)


def build_ccgt_model(
    prices: pd.Series,
    gas_price,
    co2_price,
    config: CCGTConfig,
    reserve_products: list[ReserveProductConfig] | None = None,
    dt: float = 1.0,
) -> pyo.ConcreteModel:
    reserve_products = reserve_products or []
    periods = list(range(len(prices)))
    gas = _as_series(gas_price, prices.index)
    co2 = _as_series(co2_price, prices.index)

    price_by_t = {t: float(prices.iloc[t]) for t in periods}
    marginal_by_t = {
        t: config.marginal_cost_eur_mwh(float(gas.iloc[t]), float(co2.iloc[t]))
        for t in periods
    }

    m = pyo.ConcreteModel(name="ccgt_unit_commitment_and_reserve")
    m.T = pyo.Set(initialize=periods, ordered=True)
    m.P = pyo.Set(initialize=[r.name for r in reserve_products], ordered=True)
    products = {r.name: r for r in reserve_products}

    m.on = pyo.Var(m.T, domain=pyo.Binary)
    m.start = pyo.Var(m.T, domain=pyo.Binary)
    m.stop = pyo.Var(m.T, domain=pyo.Binary)
    m.p = pyo.Var(m.T, bounds=(0.0, config.p_max_mw))
    m.reserve = pyo.Var(
        m.P, m.T, bounds=lambda _m, p, _t: (0.0, products[p].max_volume_mw)
    )

    # ---- generation limits, conditional on being committed ----------------
    m.p_upper = pyo.Constraint(
        m.T, rule=lambda _m, t: _m.p[t] <= config.p_max_mw * _m.on[t]
    )
    m.p_lower = pyo.Constraint(
        m.T, rule=lambda _m, t: _m.p[t] >= config.p_min_mw * _m.on[t]
    )

    # ---- commitment logic -------------------------------------------------
    initial_on = 1 if config.initially_on else 0

    def _state_transition(_m, t):
        previous = initial_on if t == 0 else _m.on[t - 1]
        return _m.start[t] - _m.stop[t] == _m.on[t] - previous

    m.state_transition = pyo.Constraint(m.T, rule=_state_transition)
    m.no_simultaneous_start_stop = pyo.Constraint(
        m.T, rule=lambda _m, t: _m.start[t] + _m.stop[t] <= 1
    )

    # ---- ramping ----------------------------------------------------------
    # Relaxed on the start/stop periods: modelling the real startup trajectory
    # (purge, ignition, loading to Pmin) would need a multi-stage formulation.
    # Documented simplification — the plant is allowed to reach Pmin instantly.
    initial_power = config.p_min_mw if config.initially_on else 0.0

    def _ramp_up(_m, t):
        previous = initial_power if t == 0 else _m.p[t - 1]
        return _m.p[t] - previous <= config.ramp_up_mw_per_h * dt + config.p_max_mw * _m.start[t]

    def _ramp_down(_m, t):
        previous = initial_power if t == 0 else _m.p[t - 1]
        return previous - _m.p[t] <= config.ramp_down_mw_per_h * dt + config.p_max_mw * _m.stop[t]

    m.ramp_up = pyo.Constraint(m.T, rule=_ramp_up)
    m.ramp_down = pyo.Constraint(m.T, rule=_ramp_down)

    # ---- minimum up / down time ------------------------------------------
    # Sliding-window form: once started at t, the unit must stay on for the
    # whole window. The window is truncated at the horizon end and the
    # right-hand side uses the truncated length, so a start late in the
    # horizon is allowed rather than made artificially infeasible.
    min_up = max(1, int(math.ceil(config.min_up_time_h / dt)))
    min_down = max(1, int(math.ceil(config.min_down_time_h / dt)))

    def _min_up_rule(_m, t):
        window = [k for k in periods if t <= k < t + min_up]
        return sum(_m.on[k] for k in window) >= len(window) * _m.start[t]

    def _min_down_rule(_m, t):
        window = [k for k in periods if t <= k < t + min_down]
        return sum(1 - _m.on[k] for k in window) >= len(window) * _m.stop[t]

    m.min_up_time = pyo.Constraint(m.T, rule=_min_up_rule)
    m.min_down_time = pyo.Constraint(m.T, rule=_min_down_rule)

    # ---- reserve headroom -------------------------------------------------
    # Same logic as the battery: upward reserve is only credible if the unit
    # can actually climb that far, which for a thermal unit means it must be
    # committed AND running below Pmax by at least the reserved volume.
    up = [r.name for r in reserve_products if r.provides_up]
    down = [r.name for r in reserve_products if r.provides_down]

    if up:
        m.reserve_headroom_up = pyo.Constraint(
            m.T,
            rule=lambda _m, t: _m.p[t] + sum(_m.reserve[p, t] for p in up)
            <= config.p_max_mw * _m.on[t],
        )
    if down:
        m.reserve_headroom_down = pyo.Constraint(
            m.T,
            rule=lambda _m, t: _m.p[t] - sum(_m.reserve[p, t] for p in down)
            >= config.p_min_mw * _m.on[t],
        )

    # ---- objective --------------------------------------------------------
    # Upward activation burns gas the unit would not otherwise have burnt, so
    # activation is valued at its margin over SRMC, not at the gross
    # activation price. Same helper as the battery and the backtest.
    marginal_cost_series = pd.Series(
        [marginal_by_t[t] for t in periods], index=prices.index
    )
    unit_value = {
        r.name: reserve_unit_value_eur_per_mw_h(
            r, ccgt_opportunity_cost(marginal_cost_series)
        ).to_numpy()
        for r in reserve_products
    }

    def _objective(_m):
        gross_margin = sum(
            (price_by_t[t] - marginal_by_t[t]) * _m.p[t] * dt for t in _m.T
        )
        startup = sum(config.startup_cost_eur * _m.start[t] for t in _m.T)
        reserve_revenue = sum(
            float(unit_value[p][t]) * _m.reserve[p, t] * dt
            for p in _m.P
            for t in _m.T
        )
        return gross_margin - startup + reserve_revenue

    m.objective = pyo.Objective(rule=_objective, sense=pyo.maximize)
    return m


def solve_ccgt(
    prices: pd.Series,
    gas_price,
    co2_price,
    config: CCGTConfig,
    reserve_products: list[ReserveProductConfig] | None = None,
    dt: float = 1.0,
) -> pd.DataFrame:
    """Optimize the CCGT and return its schedule indexed like `prices`."""
    reserve_products = reserve_products or []
    model = build_ccgt_model(prices, gas_price, co2_price, config, reserve_products, dt)
    solve(model)

    periods = list(range(len(prices)))
    data = {
        "p_mw": [pyo.value(model.p[t]) for t in periods],
        "on": [round(pyo.value(model.on[t])) for t in periods],
        "start": [round(pyo.value(model.start[t])) for t in periods],
        "stop": [round(pyo.value(model.stop[t])) for t in periods],
    }
    for r in reserve_products:
        data[f"reserve_{r.name}_mw"] = [
            pyo.value(model.reserve[r.name, t]) for t in periods
        ]

    schedule = pd.DataFrame(data, index=prices.index)
    for col in schedule.columns:
        if col not in ("on", "start", "stop"):
            schedule[col] = schedule[col].round(9).clip(lower=0.0)

    gas = _as_series(gas_price, prices.index)
    co2 = _as_series(co2_price, prices.index)
    schedule["price_eur_mwh"] = prices.values
    schedule["marginal_cost_eur_mwh"] = [
        config.marginal_cost_eur_mwh(float(gas.iloc[t]), float(co2.iloc[t]))
        for t in periods
    ]
    return schedule
