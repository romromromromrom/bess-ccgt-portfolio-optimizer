"""SYNTHETIC market data generator.

None of this is real market data. It is a deterministic, seeded stand-in whose
*shape* is plausible for a French/continental market — double daily peak, solar
depression at midday, cheaper weekends, occasional scarcity spikes, reserve
capacity prices that firm up when the system is tight — so that the optimizer
is exercised on something economically meaningful.

Replacing it with real data is a drop-in: any DataFrame carrying the same
columns (see `REQUIRED_COLUMNS`) works everywhere in the codebase. RTE's
Éco2mix / Services Système publications and ENTSO-E's Transparency Platform are
the natural sources.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "spot_price_eur_mwh",
    "gas_price_eur_mwh",
    "co2_price_eur_t",
    "fcr_capacity_price_eur_mw_h",
    "afrr_up_capacity_price_eur_mw_h",
    "afrr_down_capacity_price_eur_mw_h",
    "afrr_up_activation_price_eur_mwh",
    "afrr_down_activation_price_eur_mwh",
]


def _diurnal_shape(hours: np.ndarray) -> np.ndarray:
    """Two demand peaks (morning ~8h, evening ~19h) minus a midday solar dip."""
    morning = 18.0 * np.exp(-0.5 * ((hours - 8.0) / 2.0) ** 2)
    evening = 30.0 * np.exp(-0.5 * ((hours - 19.5) / 2.2) ** 2)
    solar_dip = -26.0 * np.exp(-0.5 * ((hours - 13.5) / 2.8) ** 2)
    night = -12.0 * np.exp(-0.5 * ((hours - 3.0) / 3.0) ** 2)
    return morning + evening + solar_dip + night


def synthetic_market_data(
    start: str = "2026-01-05",
    periods: int = 168,
    freq: str = "h",
    seed: int = 42,
    base_spot_eur_mwh: float = 78.0,
    volatility: float = 9.0,
) -> pd.DataFrame:
    """Generate a seeded synthetic market dataset. Default: one week, hourly."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=periods, freq=freq)
    hours = index.hour + index.minute / 60.0
    is_weekend = index.dayofweek >= 5

    # AR(1) residual: consecutive hours are correlated, unlike white noise.
    residual = np.zeros(periods)
    for t in range(1, periods):
        residual[t] = 0.75 * residual[t - 1] + rng.normal(0.0, volatility)

    spot = base_spot_eur_mwh + _diurnal_shape(hours) + residual
    spot = np.where(is_weekend, spot * 0.86, spot)

    # A handful of scarcity spikes and one negative-price episode, so the
    # battery has something worth arbitraging and the CCGT something worth
    # starting for.
    for _ in range(max(1, periods // 60)):
        t = int(rng.integers(0, periods))
        spot[t : t + 2] += rng.uniform(120.0, 260.0)
    if periods > 30:
        t = int(rng.integers(10, periods - 4))
        spot[t : t + 3] -= rng.uniform(70.0, 110.0)

    # Fuel and carbon drift slowly relative to power: a shallow random walk.
    gas = 30.0 + np.cumsum(rng.normal(0.0, 0.12, periods))
    co2 = 70.0 + np.cumsum(rng.normal(0.0, 0.10, periods))

    # Reserve capacity prices: partly driven by system tightness (proxied by
    # the diurnal shape), partly idiosyncratic — they are cleared in separate
    # auctions, so they are correlated with, not a function of, the spot.
    tightness = (_diurnal_shape(hours) - _diurnal_shape(hours).min()) / (
        np.ptp(_diurnal_shape(hours)) + 1e-9
    )
    fcr = np.clip(12.0 + 14.0 * tightness + rng.normal(0.0, 2.5, periods), 2.0, None)
    afrr_up = np.clip(9.0 + 16.0 * tightness + rng.normal(0.0, 3.0, periods), 1.0, None)
    afrr_down = np.clip(7.0 + 6.0 * (1.0 - tightness) + rng.normal(0.0, 2.0, periods), 1.0, None)

    # Activation energy is priced off the spot: upward activation earns a
    # scarcity premium, downward activation is bought back slightly below
    # spot, so providing it is worth only a thin margin.
    afrr_up_activation = np.clip(spot * 1.25 + 15.0, 0.0, None)
    afrr_down_activation = np.clip(spot * 0.85, 0.0, None)

    return pd.DataFrame(
        {
            "spot_price_eur_mwh": np.round(spot, 2),
            "gas_price_eur_mwh": np.round(gas, 3),
            "co2_price_eur_t": np.round(co2, 3),
            "fcr_capacity_price_eur_mw_h": np.round(fcr, 2),
            "afrr_up_capacity_price_eur_mw_h": np.round(afrr_up, 2),
            "afrr_down_capacity_price_eur_mw_h": np.round(afrr_down, 2),
            "afrr_up_activation_price_eur_mwh": np.round(afrr_up_activation, 2),
            "afrr_down_activation_price_eur_mwh": np.round(afrr_down_activation, 2),
        },
        index=index,
    )


def make_realized(
    forecast: pd.DataFrame,
    seed: int = 7,
    price_error_std_eur_mwh: float = 12.0,
    spike_probability: float = 0.03,
) -> pd.DataFrame:
    """Perturb a forecast into a plausible 'what actually happened' dataset.

    Used by the backtest: the schedule stays the one decided on the forecast,
    only the valuation data changes. The error is autocorrelated rather than
    i.i.d., because forecast errors persist for hours in practice.
    """
    rng = np.random.default_rng(seed)
    realized = forecast.copy()
    n = len(forecast)

    error = np.zeros(n)
    for t in range(1, n):
        error[t] = 0.8 * error[t - 1] + rng.normal(0.0, price_error_std_eur_mwh)

    spikes = rng.random(n) < spike_probability
    error = error + spikes * rng.uniform(60.0, 200.0, n)

    realized["spot_price_eur_mwh"] = np.round(forecast["spot_price_eur_mwh"] + error, 2)
    realized["gas_price_eur_mwh"] = np.round(
        forecast["gas_price_eur_mwh"] * rng.uniform(0.96, 1.06), 3
    )
    realized["afrr_up_activation_price_eur_mwh"] = np.clip(
        np.round(realized["spot_price_eur_mwh"] * 1.25 + 15.0, 2), 0.0, None
    )
    realized["afrr_down_activation_price_eur_mwh"] = np.clip(
        np.round(realized["spot_price_eur_mwh"] * 0.85, 2), 0.0, None
    )
    return realized


def realized_activation_ratios(
    index: pd.DatetimeIndex,
    product_names: list[str],
    expected_ratios: dict[str, float] | None = None,
    seed: int = 11,
) -> dict[str, pd.Series]:
    """Draw the share of each reserved MW that the TSO actually called.

    Activation is bursty: most periods see almost nothing, a few see heavy
    calls. A Beta draw reproduces that better than Gaussian noise around the
    mean, and it is what makes the backtest's forecast-error cost non-trivial.
    """
    rng = np.random.default_rng(seed)
    expected_ratios = expected_ratios or {}
    ratios: dict[str, pd.Series] = {}
    for name in product_names:
        mean = max(expected_ratios.get(name, 0.1), 1e-3)
        concentration = 3.0
        alpha = mean * concentration
        beta = (1.0 - mean) * concentration
        draws = rng.beta(max(alpha, 1e-3), max(beta, 1e-3), len(index))
        ratios[name] = pd.Series(np.round(draws, 4), index=index)
    return ratios
