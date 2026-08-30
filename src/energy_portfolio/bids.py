"""Turn optimized schedules into a bid-proposal table.

This is a demonstrator table, not a reproduction of an operational bidding
system: no gate-closure handling, no block/linked orders, no portfolio
aggregation rules. What it does show is the translation from a MILP decision
to something an operator would actually submit — one row per market, per
direction, per delivery period.

Two deliberate choices worth defending:

- **A CCGT bids its short-run marginal cost into the energy market**, not the
  spot price. Bidding SRMC is what makes the unit dispatch exactly when the
  market clears above its cost. Bidding the spot price back at the market is
  meaningless.
- **Bids below `min_volume_mw` are dropped.** MILP solutions carry numerical
  residue (volumes of 1e-9 MW), and real markets have a minimum bid increment
  anyway, so anything under the threshold is noise rather than an order.

Not modelled: opportunity-cost adders on the bid price — a battery should bid
above SRMC-equivalent by the marginal value of its stored energy (the SOC
shadow price, available from the MILP duals), and a hydro plant by its water
value. That is the natural next iteration.
"""

from __future__ import annotations

import pandas as pd

from energy_portfolio.config import CCGTConfig, ReserveProductConfig

COLUMNS = ["timestamp", "asset", "market", "direction", "volume_mw", "price"]

# French aFRR/mFRR standard bid increment is 1 MW; kept looser here so small
# demonstrator assets still produce a visible bid book.
DEFAULT_MIN_VOLUME_MW = 0.1


def _reserve_rows(
    asset: str,
    timestamp,
    row: pd.Series,
    schedule: pd.DataFrame,
    reserve_products: list[ReserveProductConfig],
    min_volume_mw: float,
) -> list[dict]:
    rows = []
    for product in reserve_products:
        column = f"reserve_{product.name}_mw"
        if column not in schedule:
            continue
        volume = round(float(row[column]), 2)
        if volume < min_volume_mw:
            continue
        rows.append(
            {
                "timestamp": timestamp,
                "asset": asset,
                "market": product.name.upper(),
                "direction": product.direction.upper(),
                "volume_mw": volume,
                "price": round(product.capacity_price_eur_mw_h, 2),
            }
        )
    return rows


def bess_bids(
    schedule: pd.DataFrame,
    prices: pd.Series,
    reserve_products: list[ReserveProductConfig] | None = None,
    min_volume_mw: float = DEFAULT_MIN_VOLUME_MW,
) -> pd.DataFrame:
    reserve_products = reserve_products or []
    rows: list[dict] = []
    for timestamp, row in schedule.iterrows():
        discharge = round(float(row["discharge_mw"]), 2)
        charge = round(float(row["charge_mw"]), 2)
        if discharge >= min_volume_mw:
            rows.append(
                {
                    "timestamp": timestamp, "asset": "BESS", "market": "Energy",
                    "direction": "SELL", "volume_mw": discharge,
                    "price": round(float(prices.loc[timestamp]), 2),
                }
            )
        if charge >= min_volume_mw:
            rows.append(
                {
                    "timestamp": timestamp, "asset": "BESS", "market": "Energy",
                    "direction": "BUY", "volume_mw": charge,
                    "price": round(float(prices.loc[timestamp]), 2),
                }
            )
        rows.extend(
            _reserve_rows("BESS", timestamp, row, schedule, reserve_products, min_volume_mw)
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def ccgt_bids(
    schedule: pd.DataFrame,
    prices: pd.Series,
    gas_price: pd.Series,
    co2_price: pd.Series,
    config: CCGTConfig,
    reserve_products: list[ReserveProductConfig] | None = None,
    min_volume_mw: float = DEFAULT_MIN_VOLUME_MW,
) -> pd.DataFrame:
    reserve_products = reserve_products or []
    marginal_cost = (
        config.heat_rate_mwh_gas_per_mwh_e * gas_price
        + config.heat_rate_mwh_gas_per_mwh_e * config.co2_ton_per_mwh_gas * co2_price
        + config.variable_om_cost_eur_mwh
    )
    rows: list[dict] = []
    for timestamp, row in schedule.iterrows():
        power = round(float(row["p_mw"]), 2)
        if power >= min_volume_mw:
            rows.append(
                {
                    "timestamp": timestamp, "asset": "CCGT", "market": "Energy",
                    "direction": "SELL", "volume_mw": power,
                    "price": round(float(marginal_cost.loc[timestamp]), 2),
                }
            )
        rows.extend(
            _reserve_rows("CCGT", timestamp, row, schedule, reserve_products, min_volume_mw)
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def portfolio_bids(
    bess_schedule: pd.DataFrame,
    ccgt_schedule: pd.DataFrame,
    prices: pd.Series,
    gas_price: pd.Series,
    co2_price: pd.Series,
    ccgt_config: CCGTConfig,
    bess_reserve_products: list[ReserveProductConfig] | None = None,
    ccgt_reserve_products: list[ReserveProductConfig] | None = None,
    min_volume_mw: float = DEFAULT_MIN_VOLUME_MW,
) -> pd.DataFrame:
    """The full bid book both assets would submit, sorted for reading."""
    bess = bess_bids(bess_schedule, prices, bess_reserve_products, min_volume_mw)
    ccgt = ccgt_bids(
        ccgt_schedule, prices, gas_price, co2_price, ccgt_config,
        ccgt_reserve_products, min_volume_mw,
    )
    book = pd.concat([bess, ccgt], ignore_index=True)
    if book.empty:
        return book
    return book.sort_values(["timestamp", "asset", "market"]).reset_index(drop=True)
