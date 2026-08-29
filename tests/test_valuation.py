from __future__ import annotations

import pandas as pd
import pytest

from energy_portfolio.config import ReserveProductConfig
from energy_portfolio.valuation import (
    activation_margin_eur_per_mw_h,
    reserve_unit_value_eur_per_mw_h,
)


@pytest.fixture
def opportunity() -> pd.Series:
    return pd.Series([50.0, 80.0, 110.0], index=pd.date_range("2026-01-01", periods=3, freq="h"))


def test_symmetric_product_has_no_activation_margin(opportunity):
    """FCR is called up and down about equally: the energy legs cancel."""
    fcr = ReserveProductConfig(
        name="fcr", direction="symmetric", capacity_price_eur_mw_h=20.0,
        activation_price_eur_mwh=500.0, expected_activation_ratio=0.9,
    )
    assert (activation_margin_eur_per_mw_h(fcr, opportunity) == 0.0).all()
    assert (reserve_unit_value_eur_per_mw_h(fcr, opportunity) == 20.0).all()


def test_upward_activation_is_valued_at_margin_over_opportunity_cost(opportunity):
    """Delivering upward energy costs fuel/SOC: only the spread is revenue."""
    up = ReserveProductConfig(
        name="afrr_up", direction="up", capacity_price_eur_mw_h=10.0,
        activation_price_eur_mwh=100.0, expected_activation_ratio=0.5,
    )
    margin = activation_margin_eur_per_mw_h(up, opportunity)
    assert margin.tolist() == pytest.approx([0.5 * 50.0, 0.5 * 20.0, 0.5 * -10.0])


def test_upward_activation_can_be_value_destroying(opportunity):
    """When the activation price sits below SRMC, being called loses money."""
    up = ReserveProductConfig(
        name="afrr_up", direction="up", capacity_price_eur_mw_h=10.0,
        activation_price_eur_mwh=60.0, expected_activation_ratio=1.0,
    )
    assert reserve_unit_value_eur_per_mw_h(up, opportunity).iloc[-1] < 10.0


def test_downward_activation_reverses_the_sign(opportunity):
    """Providing downward reserve means not producing: you save the SRMC and
    pay back the (lower) activation price."""
    down = ReserveProductConfig(
        name="afrr_down", direction="down", capacity_price_eur_mw_h=5.0,
        activation_price_eur_mwh=40.0, expected_activation_ratio=0.25,
    )
    margin = activation_margin_eur_per_mw_h(down, opportunity)
    assert margin.tolist() == pytest.approx([0.25 * 10.0, 0.25 * 40.0, 0.25 * 70.0])


def test_realized_overrides_replace_both_ratio_and_price(opportunity):
    up = ReserveProductConfig(
        name="afrr_up", direction="up", activation_price_eur_mwh=100.0,
        expected_activation_ratio=0.1,
    )
    realized = activation_margin_eur_per_mw_h(
        up, opportunity, activation_ratio=0.4, activation_price=200.0
    )
    assert realized.iloc[0] == pytest.approx(0.4 * (200.0 - 50.0))
