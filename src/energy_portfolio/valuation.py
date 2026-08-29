"""What one MW of reserved capacity is actually worth, per period.

This module exists so that the MILP objective, the P&L attribution and the
backtest all value reserve *the same way*. If they disagree, the "expected"
P&L reported by the optimizer stops matching the "expected" leg of the
backtest, and the forecast-error cost becomes meaningless.

The key modelling point — and the one worth defending in an interview — is
that **activation energy is not free revenue**. When a CCGT is called upward
it burns gas it would not otherwise have burnt; when a battery is called
upward it discharges energy it could have sold on the spot market. So
activation is valued at its *margin* over the opportunity cost of the energy,
not at the gross activation price:

    up   :  ratio x (activation_price - opportunity_cost)
    down :  ratio x (opportunity_cost - activation_price)

A symmetric product (FCR) is treated as energy-neutral in expectation: it is
called up about as often as down, so the energy legs cancel and only the
capacity payment remains. That is also why FCR in France is remunerated on
capacity only.
"""

from __future__ import annotations

import pandas as pd

from energy_portfolio.bess_config import BESSReserveConfig
from energy_portfolio.config import ReserveProductConfig


def _as_series(value, index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.astype(float)
    return pd.Series(float(value), index=index)


def activation_margin_eur_per_mw_h(
    product: ReserveProductConfig,
    opportunity_cost: pd.Series,
    activation_ratio: pd.Series | float | None = None,
    activation_price: pd.Series | float | None = None,
) -> pd.Series:
    """Expected EUR earned per MW held, per hour, from being activated.

    `activation_ratio` and `activation_price` default to the product's
    assumptions; the backtest overrides both with realized values.
    """
    index = opportunity_cost.index
    ratio = _as_series(
        product.expected_activation_ratio if activation_ratio is None else activation_ratio,
        index,
    )
    if product.direction == "symmetric":
        return pd.Series(0.0, index=index)
    price = _as_series(
        product.activation_price_eur_mwh if activation_price is None else activation_price,
        index,
    )
    if product.direction == "up":
        return ratio * (price - opportunity_cost)
    return ratio * (opportunity_cost - price)


def reserve_unit_value_eur_per_mw_h(
    product: ReserveProductConfig,
    opportunity_cost: pd.Series,
    activation_ratio: pd.Series | float | None = None,
    activation_price: pd.Series | float | None = None,
) -> pd.Series:
    """Total value of holding 1 MW of this product: capacity + activation."""
    return product.capacity_price_eur_mw_h + activation_margin_eur_per_mw_h(
        product, opportunity_cost, activation_ratio, activation_price
    )


def bess_opportunity_cost(
    product: ReserveProductConfig, prices: pd.Series, config: BESSReserveConfig
) -> pd.Series:
    """Cost to the battery of the energy an activation would move.

    Discharging on an upward call forgoes a spot sale *and* wears the cells,
    so degradation is added. Charging on a downward call only forgoes the
    opportunity to buy that energy later, valued at spot.
    """
    if product.direction == "up":
        return prices + config.degradation_cost_eur_mwh
    return prices.copy()


def ccgt_opportunity_cost(marginal_cost: pd.Series) -> pd.Series:
    """For a thermal unit the opportunity cost of energy is its SRMC."""
    return marginal_cost.copy()


def bess_reserve_revenue(
    schedule: pd.DataFrame,
    prices: pd.Series,
    config: BESSReserveConfig,
    reserve_products: list[ReserveProductConfig],
    activation_ratios: dict[str, pd.Series] | None = None,
    dt: float = 1.0,
    activation_prices: dict[str, pd.Series] | None = None,
) -> tuple[float, float]:
    """(capacity revenue, activation margin) in EUR over the horizon."""
    activation_ratios = activation_ratios or {}
    activation_prices = activation_prices or {}
    capacity = activation = 0.0
    for product in reserve_products:
        column = f"reserve_{product.name}_mw"
        if column not in schedule:
            continue
        volume = schedule[column]
        opportunity = bess_opportunity_cost(product, prices, config)
        capacity += float((product.capacity_price_eur_mw_h * volume * dt).sum())
        activation += float(
            (
                activation_margin_eur_per_mw_h(
                    product,
                    opportunity,
                    activation_ratios.get(product.name),
                    activation_prices.get(product.name),
                )
                * volume
                * dt
            ).sum()
        )
    return capacity, activation


def ccgt_reserve_revenue(
    schedule: pd.DataFrame,
    marginal_cost: pd.Series,
    reserve_products: list[ReserveProductConfig],
    activation_ratios: dict[str, pd.Series] | None = None,
    dt: float = 1.0,
    activation_prices: dict[str, pd.Series] | None = None,
) -> tuple[float, float]:
    activation_ratios = activation_ratios or {}
    activation_prices = activation_prices or {}
    capacity = activation = 0.0
    opportunity = ccgt_opportunity_cost(marginal_cost)
    for product in reserve_products:
        column = f"reserve_{product.name}_mw"
        if column not in schedule:
            continue
        volume = schedule[column]
        capacity += float((product.capacity_price_eur_mw_h * volume * dt).sum())
        activation += float(
            (
                activation_margin_eur_per_mw_h(
                    product,
                    opportunity,
                    activation_ratios.get(product.name),
                    activation_prices.get(product.name),
                )
                * volume
                * dt
            ).sum()
        )
    return capacity, activation
