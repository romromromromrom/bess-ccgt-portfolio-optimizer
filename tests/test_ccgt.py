from __future__ import annotations

import pandas as pd
import pytest

from energy_portfolio.ccgt import solve_ccgt
from energy_portfolio.config import CCGTConfig, ReserveProductConfig

TOL = 1e-4

GAS = 30.0        # EUR/MWh_gas
CO2 = 70.0        # EUR/tCO2
# With the default CCGTConfig (heat rate 2.0, 0.202 tCO2/MWh_gas, 3 EUR/MWh VOM)
# this gives a short-run marginal cost of 2*30 + 2*0.202*70 + 3 = 91.28 EUR/MWh.
MARGINAL_COST = 91.28


@pytest.fixture
def idx() -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01", periods=24, freq="h")


@pytest.fixture
def cfg() -> CCGTConfig:
    return CCGTConfig()


def _on_run_lengths(on: pd.Series) -> list[int]:
    runs, current = [], 0
    for value in on:
        if value == 1:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def test_marginal_cost_matches_hand_calculation(cfg):
    assert cfg.marginal_cost_eur_mwh(GAS, CO2) == pytest.approx(MARGINAL_COST, abs=1e-6)


def test_stays_off_when_spot_below_marginal_cost(idx, cfg):
    """Energy-only market, spot under SRMC: running can only destroy value."""
    prices = pd.Series(MARGINAL_COST - 10, index=idx)
    schedule = solve_ccgt(prices, GAS, CO2, cfg)
    assert schedule["on"].sum() == 0
    assert schedule["p_mw"].max() <= TOL


def test_runs_at_pmax_when_spot_well_above_marginal_cost(idx, cfg):
    prices = pd.Series(MARGINAL_COST + 60, index=idx)
    schedule = solve_ccgt(prices, GAS, CO2, cfg)
    assert schedule["on"].sum() == len(idx)
    assert schedule["p_mw"].min() >= cfg.p_max_mw - TOL


def test_output_respects_pmin_pmax_when_committed(idx, cfg):
    prices = pd.Series([60, 200] * 12, index=idx)
    schedule = solve_ccgt(prices, GAS, CO2, cfg)
    running = schedule[schedule["on"] == 1]
    assert (running["p_mw"] >= cfg.p_min_mw - TOL).all()
    assert (running["p_mw"] <= cfg.p_max_mw + TOL).all()
    assert (schedule[schedule["on"] == 0]["p_mw"] <= TOL).all()


def test_min_up_time_is_respected(idx, cfg):
    """A 2h price spike forces a start; the unit must then stay on >= min_up_time."""
    prices = pd.Series(50.0, index=idx)
    prices.iloc[10:12] = 400.0
    schedule = solve_ccgt(prices, GAS, CO2, cfg)
    runs = _on_run_lengths(schedule["on"])
    assert runs, "expected the unit to start for the price spike"
    assert min(runs) >= cfg.min_up_time_h


def test_reserve_never_exceeds_thermal_headroom(idx, cfg):
    prices = pd.Series(120.0, index=idx)
    up = ReserveProductConfig(name="afrr_up", direction="up", capacity_price_eur_mw_h=20.0)
    down = ReserveProductConfig(name="afrr_down", direction="down", capacity_price_eur_mw_h=10.0)
    schedule = solve_ccgt(prices, GAS, CO2, cfg, [up, down])
    assert (
        schedule["p_mw"] + schedule["reserve_afrr_up_mw"]
        <= cfg.p_max_mw * schedule["on"] + TOL
    ).all()
    assert (
        schedule["p_mw"] - schedule["reserve_afrr_down_mw"]
        >= cfg.p_min_mw * schedule["on"] - TOL
    ).all()


def test_starts_at_pmin_below_marginal_cost_to_sell_reserve(idx, cfg):
    """The headline result: reserve revenue can justify running at a loss.

    Spot sits 11 EUR/MWh below SRMC, so energy-only the plant stays off. Add an
    aFRR-up capacity price of 20 EUR/MW/h and it becomes rational to commit at
    Pmin and monetise the full Pmax - Pmin headroom as reserve.
    """
    prices = pd.Series(MARGINAL_COST - 11, index=idx)

    energy_only = solve_ccgt(prices, GAS, CO2, cfg)
    assert energy_only["on"].sum() == 0

    afrr_up = ReserveProductConfig(
        name="afrr_up", direction="up", capacity_price_eur_mw_h=20.0
    )
    with_reserve = solve_ccgt(prices, GAS, CO2, cfg, [afrr_up])

    assert with_reserve["on"].sum() == len(idx), "reserve revenue should force a start"
    running = with_reserve[with_reserve["on"] == 1]
    assert (running["p_mw"] <= cfg.p_min_mw + TOL).all(), "should sit at Pmin"
    assert (
        running["reserve_afrr_up_mw"] >= cfg.p_max_mw - cfg.p_min_mw - TOL
    ).all(), "should sell the whole headroom"


def test_startup_cost_deters_a_single_short_spike(idx, cfg):
    """One hour marginally above SRMC does not repay a 15 kEUR start."""
    prices = pd.Series(50.0, index=idx)
    prices.iloc[12] = MARGINAL_COST + 5
    schedule = solve_ccgt(prices, GAS, CO2, cfg)
    assert schedule["start"].sum() == 0
