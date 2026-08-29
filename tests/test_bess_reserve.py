from __future__ import annotations

import pandas as pd
import pytest

from energy_portfolio.bess_config import BESSReserveConfig
from energy_portfolio.bess_reserve import solve_bess_reserve
from energy_portfolio.config import ReserveProductConfig

TOL = 1e-4


@pytest.fixture
def idx() -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01", periods=12, freq="h")


def test_soc_within_bounds(idx):
    prices = pd.Series([20, 80, 20, 80, 20, 80, 20, 80, 20, 80, 20, 80], index=idx)
    cfg = BESSReserveConfig(max_charge_mw=50, max_discharge_mw=50, energy_capacity_mwh=100,
                             min_soc_mwh=10, max_soc_mwh=90, initial_soc_mwh=50, final_soc_mwh=None)
    schedule = solve_bess_reserve(prices, cfg)
    assert (schedule["soc_mwh"] >= cfg.min_soc_mwh - TOL).all()
    assert (schedule["soc_mwh"] <= cfg.max_soc_mwh + TOL).all()


def test_no_simultaneous_charge_discharge(idx):
    prices = pd.Series([20, 80] * 6, index=idx)
    cfg = BESSReserveConfig(max_charge_mw=50, max_discharge_mw=50, energy_capacity_mwh=100)
    schedule = solve_bess_reserve(prices, cfg)
    assert ((schedule["charge_mw"] > TOL) & (schedule["discharge_mw"] > TOL)).sum() == 0


def test_reserve_never_exceeds_power_headroom(idx):
    prices = pd.Series(50.0, index=idx)
    cfg = BESSReserveConfig(max_charge_mw=50, max_discharge_mw=50, energy_capacity_mwh=100)
    fcr = ReserveProductConfig(name="fcr", direction="symmetric", capacity_price_eur_mw_h=20.0)
    schedule = solve_bess_reserve(prices, cfg, [fcr])
    net = schedule["discharge_mw"] - schedule["charge_mw"]
    assert (net + schedule["reserve_fcr_mw"] <= cfg.max_discharge_mw + TOL).all()
    assert (net - schedule["reserve_fcr_mw"] >= -cfg.max_charge_mw - TOL).all()


def test_reserve_never_exceeds_energy_headroom(idx):
    prices = pd.Series(50.0, index=idx)
    cfg = BESSReserveConfig(
        max_charge_mw=200, max_discharge_mw=200, energy_capacity_mwh=100,
        min_soc_mwh=0, max_soc_mwh=100, initial_soc_mwh=5, final_soc_mwh=None,
        reserve_sustain_duration_h=1.0,
    )
    afrr_up = ReserveProductConfig(name="afrr_up", direction="up", capacity_price_eur_mw_h=50.0)
    schedule = solve_bess_reserve(prices, cfg, [afrr_up])
    # reserve_up * sustain_duration must not exceed usable SOC above min_soc
    assert (
        schedule["reserve_afrr_up_mw"] * cfg.reserve_sustain_duration_h
        <= schedule["soc_mwh"] - cfg.min_soc_mwh + TOL
    ).all()


def test_analytical_scenario_prefers_fcr_when_energy_flat(idx):
    """Constant energy price + high FCR price -> battery should reserve near-max FCR."""
    prices = pd.Series(40.0, index=idx)
    cfg = BESSReserveConfig(max_charge_mw=50, max_discharge_mw=50, energy_capacity_mwh=100,
                             initial_soc_mwh=50, final_soc_mwh=50, reserve_sustain_duration_h=0.1)
    fcr = ReserveProductConfig(name="fcr", direction="symmetric", capacity_price_eur_mw_h=25.0)
    schedule = solve_bess_reserve(prices, cfg, [fcr])
    assert schedule["reserve_fcr_mw"].mean() > 0.9 * cfg.max_discharge_mw
