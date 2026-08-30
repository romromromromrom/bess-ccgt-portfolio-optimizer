"""Backtest: re-value a committed decision against realized data.

Two-phase discipline, and the whole point of the module:

1. **Optimization stage** — solve on FORECAST prices and assumed activation
   ratios. The output is a *decision*: charge/discharge/p_mw/reserve MW.
2. **Backtest stage** — take that SAME decision, unchanged, and re-value it
   with REALIZED prices, realized fuel costs and realized activation. The
   schedule is never re-optimized: it was committed before the outcome was
   known, and re-solving with hindsight would silently manufacture skill that
   the strategy never had. This is the single most common way a backtest
   flatters itself.

`forecast_error_cost = expected_pnl - realized_pnl` is therefore the money lost
purely to being wrong about the future, holding the strategy fixed.

Reserve is valued through `energy_portfolio.valuation`, the same helper the
MILP objective uses, so "expected" here is exactly the optimizer's own number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from energy_portfolio.bess_config import BESSReserveConfig
from energy_portfolio.config import CCGTConfig, ReserveProductConfig
from energy_portfolio.valuation import (
    activation_margin_eur_per_mw_h,
    bess_opportunity_cost,
    ccgt_opportunity_cost,
)


@dataclass
class BacktestResult:
    expected_pnl: float
    realized_pnl: float
    forecast_error_cost: float
    reserve_revenue_realized: float
    energy_revenue_realized: float
    detail: pd.DataFrame
    reserve_capacity_revenue_realized: float = 0.0
    reserve_activation_margin_realized: float = 0.0

    def summary(self) -> pd.Series:
        return pd.Series(
            {
                "expected_pnl_eur": round(self.expected_pnl, 2),
                "realized_pnl_eur": round(self.realized_pnl, 2),
                "forecast_error_cost_eur": round(self.forecast_error_cost, 2),
                "energy_revenue_realized_eur": round(self.energy_revenue_realized, 2),
                "reserve_revenue_realized_eur": round(self.reserve_revenue_realized, 2),
            }
        )


def _as_series(value, index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.astype(float)
    return pd.Series(float(value), index=index)


def _reserve_series(
    schedule: pd.DataFrame,
    reserve_products: list[ReserveProductConfig],
    opportunity_cost_for: Callable[[ReserveProductConfig], pd.Series],
    activation_ratio: dict[str, pd.Series],
    activation_price: dict[str, pd.Series],
    dt: float,
) -> tuple[pd.Series, pd.Series]:
    """Per-period (capacity revenue, activation margin) for a fixed schedule."""
    capacity = pd.Series(0.0, index=schedule.index)
    activation = pd.Series(0.0, index=schedule.index)
    for product in reserve_products:
        column = f"reserve_{product.name}_mw"
        if column not in schedule:
            continue
        volume = schedule[column]
        capacity = capacity + product.capacity_price_eur_mw_h * volume * dt
        activation = activation + (
            activation_margin_eur_per_mw_h(
                product,
                opportunity_cost_for(product),
                activation_ratio.get(product.name),
                activation_price.get(product.name),
            )
            * volume
            * dt
        )
    return capacity, activation


def _assemble(
    expected: pd.Series,
    realized: pd.Series,
    energy_realized: pd.Series,
    capacity_realized: pd.Series,
    activation_realized: pd.Series,
) -> BacktestResult:
    detail = pd.DataFrame({"expected_pnl": expected, "realized_pnl": realized})
    detail["forecast_error_cost"] = detail["expected_pnl"] - detail["realized_pnl"]
    detail["cumulative_expected_pnl"] = detail["expected_pnl"].cumsum()
    detail["cumulative_realized_pnl"] = detail["realized_pnl"].cumsum()
    return BacktestResult(
        expected_pnl=float(expected.sum()),
        realized_pnl=float(realized.sum()),
        forecast_error_cost=float(detail["forecast_error_cost"].sum()),
        reserve_revenue_realized=float(capacity_realized.sum() + activation_realized.sum()),
        energy_revenue_realized=float(energy_realized.sum()),
        detail=detail,
        reserve_capacity_revenue_realized=float(capacity_realized.sum()),
        reserve_activation_margin_realized=float(activation_realized.sum()),
    )


def backtest_ccgt(
    schedule: pd.DataFrame,
    forecast_price: pd.Series,
    realized_price: pd.Series,
    forecast_gas: pd.Series,
    realized_gas: pd.Series,
    co2_price: pd.Series,
    config: CCGTConfig,
    reserve_products: list[ReserveProductConfig],
    realized_activation_ratio: dict[str, pd.Series] | None = None,
    dt: float = 1.0,
    realized_activation_price: dict[str, pd.Series] | None = None,
) -> BacktestResult:
    """Re-value a committed CCGT schedule. Dispatch is fixed; data is swapped."""
    index = schedule.index
    realized_activation_ratio = realized_activation_ratio or {}
    realized_activation_price = realized_activation_price or {}
    co2 = _as_series(co2_price, index)
    power = schedule["p_mw"]

    def leg(price: pd.Series, gas: pd.Series, ratios, prices_override):
        price = _as_series(price, index)
        gas = _as_series(gas, index)
        marginal = (
            config.heat_rate_mwh_gas_per_mwh_e * gas
            + config.heat_rate_mwh_gas_per_mwh_e * config.co2_ton_per_mwh_gas * co2
            + config.variable_om_cost_eur_mwh
        )
        energy_margin = (price - marginal) * power * dt
        startup = config.startup_cost_eur * schedule["start"]
        capacity, activation = _reserve_series(
            schedule,
            reserve_products,
            lambda _p: ccgt_opportunity_cost(marginal),
            ratios,
            prices_override,
            dt,
        )
        energy_revenue = price * power * dt
        return energy_margin - startup + capacity + activation, energy_revenue, capacity, activation

    expected, _, _, _ = leg(forecast_price, forecast_gas, {}, {})
    realized, energy_realized, capacity_realized, activation_realized = leg(
        realized_price, realized_gas, realized_activation_ratio, realized_activation_price
    )
    return _assemble(
        expected, realized, energy_realized, capacity_realized, activation_realized
    )


def backtest_bess(
    schedule: pd.DataFrame,
    forecast_price: pd.Series,
    realized_price: pd.Series,
    config: BESSReserveConfig,
    reserve_products: list[ReserveProductConfig],
    realized_activation_ratio: dict[str, pd.Series] | None = None,
    dt: float = 1.0,
    realized_activation_price: dict[str, pd.Series] | None = None,
) -> BacktestResult:
    """Re-value a committed BESS schedule. Dispatch is fixed; data is swapped."""
    index = schedule.index
    realized_activation_ratio = realized_activation_ratio or {}
    realized_activation_price = realized_activation_price or {}
    net = schedule["discharge_mw"] - schedule["charge_mw"]
    degradation = config.degradation_cost_eur_mwh * schedule["discharge_mw"] * dt

    def leg(price: pd.Series, ratios, prices_override):
        price = _as_series(price, index)
        energy_revenue = price * net * dt
        capacity, activation = _reserve_series(
            schedule,
            reserve_products,
            lambda product: bess_opportunity_cost(product, price, config),
            ratios,
            prices_override,
            dt,
        )
        return energy_revenue - degradation + capacity + activation, energy_revenue, capacity, activation

    expected, _, _, _ = leg(forecast_price, {}, {})
    realized, energy_realized, capacity_realized, activation_realized = leg(
        realized_price, realized_activation_ratio, realized_activation_price
    )
    return _assemble(
        expected, realized, energy_realized, capacity_realized, activation_realized
    )


@dataclass
class PortfolioBacktestResult:
    bess: BacktestResult
    ccgt: BacktestResult

    @property
    def expected_pnl(self) -> float:
        return self.bess.expected_pnl + self.ccgt.expected_pnl

    @property
    def realized_pnl(self) -> float:
        return self.bess.realized_pnl + self.ccgt.realized_pnl

    @property
    def forecast_error_cost(self) -> float:
        return self.expected_pnl - self.realized_pnl

    def detail(self) -> pd.DataFrame:
        combined = self.bess.detail[["expected_pnl", "realized_pnl"]].add(
            self.ccgt.detail[["expected_pnl", "realized_pnl"]], fill_value=0.0
        )
        combined["forecast_error_cost"] = (
            combined["expected_pnl"] - combined["realized_pnl"]
        )
        combined["cumulative_expected_pnl"] = combined["expected_pnl"].cumsum()
        combined["cumulative_realized_pnl"] = combined["realized_pnl"].cumsum()
        return combined

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "BESS": self.bess.summary(),
                "CCGT": self.ccgt.summary(),
                "Portfolio": self.bess.summary() + self.ccgt.summary(),
            }
        )
