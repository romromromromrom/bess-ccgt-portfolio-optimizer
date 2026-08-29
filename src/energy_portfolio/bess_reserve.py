"""MILP: BESS energy arbitrage co-optimized with reserve capacity sales.

The economic point of this model is the **opportunity cost of reserve**: every
MW the battery holds back for FCR/aFRR is a MW it cannot use for arbitrage,
and every MWh of state-of-charge it must keep available to sustain that
reserve is an MWh it cannot sell. Those two couplings are expressed as the
power-headroom and energy-headroom constraints below, and they are the reason
the problem has to be co-optimized rather than solved market by market.
"""

from __future__ import annotations

import pandas as pd
import pyomo.environ as pyo

from energy_portfolio.bess_config import BESSReserveConfig
from energy_portfolio.config import ReserveProductConfig
from energy_portfolio.solver import solve
from energy_portfolio.valuation import (
    bess_opportunity_cost,
    reserve_unit_value_eur_per_mw_h,
)


def build_bess_model(
    prices: pd.Series,
    config: BESSReserveConfig,
    reserve_products: list[ReserveProductConfig] | None = None,
    dt: float = 1.0,
) -> pyo.ConcreteModel:
    reserve_products = reserve_products or []
    periods = list(range(len(prices)))
    price_by_t = {t: float(prices.iloc[t]) for t in periods}

    m = pyo.ConcreteModel(name="bess_energy_and_reserve")
    m.T = pyo.Set(initialize=periods, ordered=True)
    m.P = pyo.Set(initialize=[r.name for r in reserve_products], ordered=True)
    products = {r.name: r for r in reserve_products}

    m.charge = pyo.Var(m.T, bounds=(0.0, config.max_charge_mw))
    m.discharge = pyo.Var(m.T, bounds=(0.0, config.max_discharge_mw))
    # Binary mode selector: forbids charging and discharging in the same
    # period. Efficiency losses already make it unprofitable at positive
    # prices, but it becomes exploitable at negative prices, so it is enforced.
    m.is_charging = pyo.Var(m.T, domain=pyo.Binary)
    m.soc = pyo.Var(m.T, bounds=(config.min_soc_mwh, config.max_soc_mwh))

    def _reserve_bounds(_m, p, _t):
        return (0.0, products[p].max_volume_mw)

    m.reserve = pyo.Var(m.P, m.T, bounds=_reserve_bounds)

    # ---- mode exclusivity -------------------------------------------------
    m.charge_mode = pyo.Constraint(
        m.T, rule=lambda _m, t: _m.charge[t] <= config.max_charge_mw * _m.is_charging[t]
    )
    m.discharge_mode = pyo.Constraint(
        m.T,
        rule=lambda _m, t: _m.discharge[t]
        <= config.max_discharge_mw * (1 - _m.is_charging[t]),
    )

    # ---- state of charge --------------------------------------------------
    def _soc_balance(_m, t):
        previous = config.initial_soc_mwh if t == 0 else _m.soc[t - 1]
        return _m.soc[t] == previous + (
            _m.charge[t] * config.charge_efficiency
            - _m.discharge[t] / config.discharge_efficiency
        ) * dt

    m.soc_balance = pyo.Constraint(m.T, rule=_soc_balance)

    if config.final_soc_mwh is not None:
        m.final_soc = pyo.Constraint(
            expr=m.soc[periods[-1]] >= config.final_soc_mwh
        )

    # ---- power headroom ---------------------------------------------------
    # Net power convention: p_net = discharge - charge (positive = exporting).
    # Selling upward reserve means promising to move p_net UP by that many MW,
    # so the promise is only credible if the headroom to max_discharge exists.
    up = [r.name for r in reserve_products if r.provides_up]
    down = [r.name for r in reserve_products if r.provides_down]

    def _net(_m, t):
        return _m.discharge[t] - _m.charge[t]

    if up:
        m.power_headroom_up = pyo.Constraint(
            m.T,
            rule=lambda _m, t: _net(_m, t) + sum(_m.reserve[p, t] for p in up)
            <= config.max_discharge_mw,
        )
    if down:
        m.power_headroom_down = pyo.Constraint(
            m.T,
            rule=lambda _m, t: _net(_m, t) - sum(_m.reserve[p, t] for p in down)
            >= -config.max_charge_mw,
        )

    # ---- energy (SOC) headroom -------------------------------------------
    # A MW of upward reserve is only deliverable if the battery holds enough
    # energy to sustain it for `sustain_duration_h`; symmetrically, downward
    # reserve needs empty room to absorb the energy. Enforced on both the
    # opening and closing SOC of the period, so the promise holds throughout.
    #
    # Simplification (documented): one sustain duration per product, instead
    # of the exact per-product/per-TSO prequalification rules.
    def _sustain(name: str) -> float:
        override = products[name].sustain_duration_h
        return config.reserve_sustain_duration_h if override is None else override

    def _opening_soc(_m, t):
        return config.initial_soc_mwh if t == 0 else _m.soc[t - 1]

    if up:
        m.energy_headroom_up = pyo.Constraint(
            m.T,
            rule=lambda _m, t: sum(_m.reserve[p, t] * _sustain(p) for p in up)
            <= _m.soc[t] - config.min_soc_mwh,
        )
        m.energy_headroom_up_open = pyo.Constraint(
            m.T,
            rule=lambda _m, t: sum(_m.reserve[p, t] * _sustain(p) for p in up)
            <= _opening_soc(_m, t) - config.min_soc_mwh,
        )
    if down:
        m.energy_headroom_down = pyo.Constraint(
            m.T,
            rule=lambda _m, t: sum(_m.reserve[p, t] * _sustain(p) for p in down)
            <= config.max_soc_mwh - _m.soc[t],
        )
        m.energy_headroom_down_open = pyo.Constraint(
            m.T,
            rule=lambda _m, t: sum(_m.reserve[p, t] * _sustain(p) for p in down)
            <= config.max_soc_mwh - _opening_soc(_m, t),
        )

    # ---- objective --------------------------------------------------------
    # Holding 1 MW of reserve for 1 h is worth its capacity payment plus the
    # expected *margin* on activation energy - never the gross activation
    # price, since delivering it costs the battery a spot sale plus wear.
    # See energy_portfolio.valuation, which the backtest reuses verbatim.
    unit_value = {
        r.name: reserve_unit_value_eur_per_mw_h(
            r, bess_opportunity_cost(r, prices, config)
        ).to_numpy()
        for r in reserve_products
    }

    def _objective(_m):
        energy = sum(price_by_t[t] * _net(_m, t) * dt for t in _m.T)
        degradation = sum(
            config.degradation_cost_eur_mwh * _m.discharge[t] * dt for t in _m.T
        )
        reserve_revenue = sum(
            float(unit_value[p][t]) * _m.reserve[p, t] * dt
            for p in _m.P
            for t in _m.T
        )
        return energy - degradation + reserve_revenue

    m.objective = pyo.Objective(rule=_objective, sense=pyo.maximize)
    return m


def solve_bess_reserve(
    prices: pd.Series,
    config: BESSReserveConfig,
    reserve_products: list[ReserveProductConfig] | None = None,
    dt: float = 1.0,
) -> pd.DataFrame:
    """Optimize the battery and return its schedule indexed like `prices`."""
    reserve_products = reserve_products or []
    model = build_bess_model(prices, config, reserve_products, dt)
    solve(model)

    periods = list(range(len(prices)))
    data = {
        "charge_mw": [pyo.value(model.charge[t]) for t in periods],
        "discharge_mw": [pyo.value(model.discharge[t]) for t in periods],
        "soc_mwh": [pyo.value(model.soc[t]) for t in periods],
    }
    for r in reserve_products:
        data[f"reserve_{r.name}_mw"] = [
            pyo.value(model.reserve[r.name, t]) for t in periods
        ]

    schedule = pd.DataFrame(data, index=prices.index)
    # Clean solver noise so downstream ">1e-6" filters and assertions behave.
    schedule = schedule.round(9).clip(lower=0.0)
    schedule["net_mw"] = schedule["discharge_mw"] - schedule["charge_mw"]
    schedule["price_eur_mwh"] = prices.values
    return schedule
