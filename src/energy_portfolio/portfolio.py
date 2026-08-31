"""Portfolio layer: solve both assets, value them, and quantify the uplift.

Scope, stated honestly: BESS and CCGT are optimized **independently** and their
schedules are then aggregated. That is the right level for a demonstrator and
it is already enough to show the energy-vs-reserve arbitrage inside each asset.
It is *not* a joint portfolio optimization: there is no shared constraint
coupling the two units (a single aggregated reserve obligation to deliver
collectively, a shared grid connection limit, or a portfolio VaR budget would
each introduce one). Adding such a constraint is the natural next step, and the
model structure here — one Pyomo block per asset, one objective — extends to it
directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from energy_portfolio.bess_config import BESSReserveConfig
from energy_portfolio.bess_reserve import solve_bess_reserve
from energy_portfolio.ccgt import solve_ccgt
from energy_portfolio.config import CCGTConfig, ReserveProductConfig
from energy_portfolio.valuation import bess_reserve_revenue, ccgt_reserve_revenue


@dataclass
class AssetPnL:
    """EUR over the whole optimization horizon, positive = revenue."""

    energy_revenue: float = 0.0
    reserve_capacity_revenue: float = 0.0
    reserve_activation_revenue: float = 0.0
    fuel_cost: float = 0.0
    co2_cost: float = 0.0
    vom_cost: float = 0.0
    startup_cost: float = 0.0
    degradation_cost: float = 0.0

    @property
    def reserve_revenue(self) -> float:
        return self.reserve_capacity_revenue + self.reserve_activation_revenue

    @property
    def costs(self) -> float:
        return (
            self.fuel_cost
            + self.co2_cost
            + self.vom_cost
            + self.startup_cost
            + self.degradation_cost
        )

    @property
    def total(self) -> float:
        return self.energy_revenue + self.reserve_revenue - self.costs

    def as_dict(self) -> dict[str, float]:
        return {
            "energy_revenue": self.energy_revenue,
            "reserve_capacity_revenue": self.reserve_capacity_revenue,
            "reserve_activation_revenue": self.reserve_activation_revenue,
            "fuel_cost": self.fuel_cost,
            "co2_cost": self.co2_cost,
            "vom_cost": self.vom_cost,
            "startup_cost": self.startup_cost,
            "degradation_cost": self.degradation_cost,
            "total": self.total,
        }


def value_bess(
    schedule: pd.DataFrame,
    prices: pd.Series,
    config: BESSReserveConfig,
    reserve_products: list[ReserveProductConfig] | None = None,
    dt: float = 1.0,
) -> AssetPnL:
    reserve_products = reserve_products or []
    net = schedule["discharge_mw"] - schedule["charge_mw"]
    capacity, activation = bess_reserve_revenue(
        schedule, prices, config, reserve_products, dt=dt
    )
    return AssetPnL(
        energy_revenue=float((prices * net * dt).sum()),
        reserve_capacity_revenue=capacity,
        reserve_activation_revenue=activation,
        degradation_cost=float(
            (config.degradation_cost_eur_mwh * schedule["discharge_mw"] * dt).sum()
        ),
    )


def value_ccgt(
    schedule: pd.DataFrame,
    prices: pd.Series,
    gas_price: pd.Series,
    co2_price: pd.Series,
    config: CCGTConfig,
    reserve_products: list[ReserveProductConfig] | None = None,
    dt: float = 1.0,
) -> AssetPnL:
    reserve_products = reserve_products or []
    power = schedule["p_mw"]
    gas_burn = config.heat_rate_mwh_gas_per_mwh_e * power * dt
    marginal_cost = schedule.get(
        "marginal_cost_eur_mwh",
        config.heat_rate_mwh_gas_per_mwh_e * gas_price
        + config.heat_rate_mwh_gas_per_mwh_e * config.co2_ton_per_mwh_gas * co2_price
        + config.variable_om_cost_eur_mwh,
    )
    capacity, activation = ccgt_reserve_revenue(
        schedule, marginal_cost, reserve_products, dt=dt
    )
    return AssetPnL(
        energy_revenue=float((prices * power * dt).sum()),
        reserve_capacity_revenue=capacity,
        reserve_activation_revenue=activation,
        fuel_cost=float((gas_burn * gas_price).sum()),
        co2_cost=float((gas_burn * config.co2_ton_per_mwh_gas * co2_price).sum()),
        vom_cost=float((config.variable_om_cost_eur_mwh * power * dt).sum()),
        startup_cost=float((config.startup_cost_eur * schedule["start"]).sum()),
    )


@dataclass
class PortfolioResult:
    bess_schedule: pd.DataFrame
    ccgt_schedule: pd.DataFrame
    bess_pnl: AssetPnL
    ccgt_pnl: AssetPnL
    timeseries: pd.DataFrame
    reserve_products: list[ReserveProductConfig] = field(default_factory=list)

    @property
    def total_pnl(self) -> float:
        return self.bess_pnl.total + self.ccgt_pnl.total

    def pnl_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"BESS": self.bess_pnl.as_dict(), "CCGT": self.ccgt_pnl.as_dict()}
        ).assign(Portfolio=lambda df: df["BESS"] + df["CCGT"])


def _aggregate_timeseries(
    bess_schedule: pd.DataFrame,
    ccgt_schedule: pd.DataFrame,
    reserve_products: list[ReserveProductConfig],
) -> pd.DataFrame:
    ts = pd.DataFrame(index=bess_schedule.index)
    ts["bess_net_mw"] = bess_schedule["discharge_mw"] - bess_schedule["charge_mw"]
    ts["bess_soc_mwh"] = bess_schedule["soc_mwh"]
    ts["ccgt_p_mw"] = ccgt_schedule["p_mw"]
    ts["ccgt_on"] = ccgt_schedule["on"]
    ts["portfolio_net_mw"] = ts["bess_net_mw"] + ts["ccgt_p_mw"]
    for product in reserve_products:
        column = f"reserve_{product.name}_mw"
        bess_volume = bess_schedule.get(column, pd.Series(0.0, index=ts.index))
        ccgt_volume = ccgt_schedule.get(column, pd.Series(0.0, index=ts.index))
        ts[f"bess_{column}"] = bess_volume
        ts[f"ccgt_{column}"] = ccgt_volume
        ts[f"portfolio_{column}"] = bess_volume + ccgt_volume
    return ts


def run_portfolio(
    market: pd.DataFrame,
    bess_config: BESSReserveConfig,
    ccgt_config: CCGTConfig,
    reserve_products: list[ReserveProductConfig] | None = None,
    dt: float = 1.0,
    bess_reserve_products: list[ReserveProductConfig] | None = None,
    ccgt_reserve_products: list[ReserveProductConfig] | None = None,
) -> PortfolioResult:
    """Optimize both assets over the same horizon and value the result.

    `bess_reserve_products` / `ccgt_reserve_products` allow different
    prequalified product sets per asset (a battery can sell FCR, a CCGT
    typically cannot at the same terms); both default to `reserve_products`.
    """
    reserve_products = reserve_products or []
    bess_products = (
        reserve_products if bess_reserve_products is None else bess_reserve_products
    )
    ccgt_products = (
        reserve_products if ccgt_reserve_products is None else ccgt_reserve_products
    )

    prices = market["spot_price_eur_mwh"]
    gas = market["gas_price_eur_mwh"]
    co2 = market["co2_price_eur_t"]

    bess_schedule = solve_bess_reserve(prices, bess_config, bess_products, dt)
    ccgt_schedule = solve_ccgt(prices, gas, co2, ccgt_config, ccgt_products, dt)

    all_products = {p.name: p for p in [*bess_products, *ccgt_products]}
    return PortfolioResult(
        bess_schedule=bess_schedule,
        ccgt_schedule=ccgt_schedule,
        bess_pnl=value_bess(bess_schedule, prices, bess_config, bess_products, dt),
        ccgt_pnl=value_ccgt(
            ccgt_schedule, prices, gas, co2, ccgt_config, ccgt_products, dt
        ),
        timeseries=_aggregate_timeseries(
            bess_schedule, ccgt_schedule, list(all_products.values())
        ),
        reserve_products=list(all_products.values()),
    )


def compare_energy_only_vs_reserves(
    market: pd.DataFrame,
    bess_config: BESSReserveConfig,
    ccgt_config: CCGTConfig,
    reserve_products: list[ReserveProductConfig],
    dt: float = 1.0,
    **kwargs,
) -> tuple[pd.DataFrame, PortfolioResult, PortfolioResult]:
    """Quantify what co-optimizing energy and reserves is worth.

    Both runs use the *same* assets, horizon and prices. The only difference is
    whether the reserve markets are available to the optimizer — so the uplift
    is attributable to co-optimization, not to a change of assumptions.
    """
    energy_only = run_portfolio(market, bess_config, ccgt_config, [], dt)
    with_reserves = run_portfolio(
        market, bess_config, ccgt_config, reserve_products, dt, **kwargs
    )

    def row(label: str, result: PortfolioResult) -> dict[str, float | str]:
        energy = result.bess_pnl.energy_revenue + result.ccgt_pnl.energy_revenue
        reserve = result.bess_pnl.reserve_revenue + result.ccgt_pnl.reserve_revenue
        costs = result.bess_pnl.costs + result.ccgt_pnl.costs
        return {
            "scenario": label,
            "energy_revenue_eur": round(energy, 2),
            "reserve_revenue_eur": round(reserve, 2),
            "costs_eur": round(costs, 2),
            "bess_pnl_eur": round(result.bess_pnl.total, 2),
            "ccgt_pnl_eur": round(result.ccgt_pnl.total, 2),
            "total_pnl_eur": round(result.total_pnl, 2),
        }

    table = pd.DataFrame(
        [row("Energy only", energy_only), row("Energy + Reserves", with_reserves)]
    ).set_index("scenario")

    baseline = energy_only.total_pnl
    uplift = with_reserves.total_pnl - baseline
    table["uplift_eur"] = [0.0, round(uplift, 2)]
    table["uplift_pct"] = [
        0.0,
        round(100.0 * uplift / abs(baseline), 2) if abs(baseline) > 1e-9 else float("nan"),
    ]
    return table, energy_only, with_reserves


DEFAULT_VOLUME_CAPS_MW = {"fcr": 25.0, "afrr_up": 100.0, "afrr_down": 100.0}


def reserve_products_from_market(
    market: pd.DataFrame,
    expected_activation_up: float = 0.12,
    expected_activation_down: float = 0.10,
    volume_caps_mw: dict[str, float] | None = None,
    afrr_sustain_duration_h: float = 1.0,
) -> list[ReserveProductConfig]:
    """Build product configs whose prices are the horizon average of the data.

    Reserve auctions clear per block rather than per hour, so a single cleared
    price per product over the optimization horizon is a defensible first
    approximation. Carrying a full hourly capacity-price curve into the MILP is
    a small change (the objective is already linear in the reserve volume) and
    is listed as a next step.

    `volume_caps_mw` stands in for market depth and prequalification: a single
    BSP cannot clear the whole national reserve requirement (French FCR is
    ~600 MW system-wide), so without a cap the model happily sells every
    available MW into a market that could not absorb it. The caps are a blunt
    proxy for a proper bid-into-a-supply-curve formulation.
    """
    caps = DEFAULT_VOLUME_CAPS_MW if volume_caps_mw is None else volume_caps_mw
    # FCR's 15-minute full-activation requirement is a prequalification rule,
    # not a modelling choice, so it stays fixed. The aFRR sustain duration is
    # the assumption worth exploring, and is exposed.
    return [
        ReserveProductConfig(
            name="fcr",
            max_volume_mw=caps.get("fcr"),
            direction="symmetric",
            capacity_price_eur_mw_h=float(market["fcr_capacity_price_eur_mw_h"].mean()),
            activation_price_eur_mwh=0.0,
            expected_activation_ratio=0.0,
            sustain_duration_h=0.25,
        ),
        ReserveProductConfig(
            name="afrr_up",
            max_volume_mw=caps.get("afrr_up"),
            direction="up",
            capacity_price_eur_mw_h=float(
                market["afrr_up_capacity_price_eur_mw_h"].mean()
            ),
            activation_price_eur_mwh=float(
                market["afrr_up_activation_price_eur_mwh"].mean()
            ),
            expected_activation_ratio=expected_activation_up,
            sustain_duration_h=afrr_sustain_duration_h,
        ),
        ReserveProductConfig(
            name="afrr_down",
            max_volume_mw=caps.get("afrr_down"),
            direction="down",
            capacity_price_eur_mw_h=float(
                market["afrr_down_capacity_price_eur_mw_h"].mean()
            ),
            activation_price_eur_mwh=float(
                market["afrr_down_activation_price_eur_mwh"].mean()
            ),
            expected_activation_ratio=expected_activation_down,
            sustain_duration_h=afrr_sustain_duration_h,
        ),
    ]
