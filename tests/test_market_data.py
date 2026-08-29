from __future__ import annotations

import pandas as pd

from energy_portfolio.market_data import (
    REQUIRED_COLUMNS,
    make_realized,
    realized_activation_ratios,
    synthetic_market_data,
)


def test_generated_frame_carries_every_required_column():
    market = synthetic_market_data(periods=48)
    assert set(REQUIRED_COLUMNS) <= set(market.columns)
    assert isinstance(market.index, pd.DatetimeIndex)
    assert len(market) == 48


def test_generation_is_reproducible_from_the_seed():
    a = synthetic_market_data(periods=24, seed=123)
    b = synthetic_market_data(periods=24, seed=123)
    c = synthetic_market_data(periods=24, seed=124)
    pd.testing.assert_frame_equal(a, b)
    assert not a["spot_price_eur_mwh"].equals(c["spot_price_eur_mwh"])


def test_prices_have_a_plausible_daily_shape():
    """Evening peak hours must clear above the small hours of the night."""
    market = synthetic_market_data(periods=336, seed=5)
    by_hour = market.groupby(market.index.hour)["spot_price_eur_mwh"].mean()
    assert by_hour.loc[18:21].mean() > by_hour.loc[1:4].mean()


def test_realized_differs_from_forecast_but_keeps_the_schema():
    forecast = synthetic_market_data(periods=48)
    realized = make_realized(forecast, seed=7)
    assert list(realized.columns) == list(forecast.columns)
    assert not realized["spot_price_eur_mwh"].equals(forecast["spot_price_eur_mwh"])


def test_activation_ratios_stay_within_zero_and_one():
    index = pd.date_range("2026-01-01", periods=48, freq="h")
    ratios = realized_activation_ratios(index, ["afrr_up"], {"afrr_up": 0.12})
    series = ratios["afrr_up"]
    assert ((series >= 0.0) & (series <= 1.0)).all()
    assert len(series) == len(index)
