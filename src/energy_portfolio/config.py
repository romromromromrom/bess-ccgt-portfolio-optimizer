"""Asset and market-product configuration objects.

Two ideas are kept strictly separate throughout the codebase, because
conflating them is the classic modelling error on ancillary services:

- **Capacity remuneration** (`capacity_price_eur_mw_h`, EUR/MW/h) pays for
  *availability*: the MW you promise to hold in reserve for the whole
  settlement period, whether or not the TSO ever calls you.
- **Activation remuneration** (`activation_price_eur_mwh`, EUR/MWh) pays for
  *energy actually delivered* when the TSO does call you. Because activation
  is uncertain at bidding time, the optimizer values it through an assumed
  `expected_activation_ratio`, and the backtest stage re-values the same
  committed schedule against the realized ratio.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Direction = Literal["up", "down", "symmetric"]


class ReserveProductConfig(BaseModel):
    """One ancillary-service product (FCR, aFRR up, aFRR down, mFRR up...)."""

    name: str
    direction: Direction = "symmetric"

    capacity_price_eur_mw_h: float = 0.0
    activation_price_eur_mwh: float = 0.0

    # Share of the reserved MW assumed to be actually activated over the
    # period. A simplification: real activation is stochastic and asymmetric.
    expected_activation_ratio: float = Field(default=0.0, ge=0.0, le=1.0)

    # Market depth / prequalified volume cap for this asset, if any.
    max_volume_mw: float | None = None

    # Per-product override of how long the reserve must be sustainable.
    # FCR in continental Europe is prequalified on a 15-min full-activation
    # basis; aFRR requirements differ. Falls back to the asset-level value.
    sustain_duration_h: float | None = None

    @property
    def provides_up(self) -> bool:
        return self.direction in ("up", "symmetric")

    @property
    def provides_down(self) -> bool:
        return self.direction in ("down", "symmetric")


class CCGTConfig(BaseModel):
    """Combined-cycle gas turbine: thermal + unit-commitment parameters."""

    p_min_mw: float = 100.0
    p_max_mw: float = 400.0

    # MWh of gas burned per MWh of electricity produced. 2.0 == 50% efficiency.
    # Modelled as a constant: a real plant has a convex, load-dependent heat
    # rate, which would need a piecewise-linear formulation.
    heat_rate_mwh_gas_per_mwh_e: float = 2.0
    co2_ton_per_mwh_gas: float = 0.202  # standard natural-gas emission factor

    variable_om_cost_eur_mwh: float = 3.0
    startup_cost_eur: float = 15_000.0

    ramp_up_mw_per_h: float = 200.0
    ramp_down_mw_per_h: float = 200.0

    min_up_time_h: int = 4
    min_down_time_h: int = 3

    initially_on: bool = False

    @model_validator(mode="after")
    def _check_bounds(self) -> "CCGTConfig":
        if self.p_min_mw < 0 or self.p_max_mw <= 0:
            raise ValueError("CCGT power limits must be positive")
        if self.p_min_mw > self.p_max_mw:
            raise ValueError("p_min_mw cannot exceed p_max_mw")
        return self

    def marginal_cost_eur_mwh(self, gas_price: float, co2_price: float) -> float:
        """Short-run marginal cost: fuel + carbon + variable O&M.

        Startup cost is deliberately excluded — it is not a per-MWh cost and
        enters the MILP objective separately, on the `start` binary.
        """
        return (
            self.heat_rate_mwh_gas_per_mwh_e * gas_price
            + self.heat_rate_mwh_gas_per_mwh_e * self.co2_ton_per_mwh_gas * co2_price
            + self.variable_om_cost_eur_mwh
        )


# Convenience presets used by the Streamlit app and the tests. Prices are
# order-of-magnitude representative of the French market, not sourced data.
def default_reserve_products() -> list[ReserveProductConfig]:
    return [
        ReserveProductConfig(
            name="fcr",
            direction="symmetric",
            capacity_price_eur_mw_h=18.0,
            activation_price_eur_mwh=0.0,  # FCR is capacity-only in France
            expected_activation_ratio=0.0,
            sustain_duration_h=0.25,
        ),
        ReserveProductConfig(
            name="afrr_up",
            direction="up",
            capacity_price_eur_mw_h=12.0,
            activation_price_eur_mwh=90.0,
            expected_activation_ratio=0.12,
            sustain_duration_h=1.0,
        ),
        ReserveProductConfig(
            name="afrr_down",
            direction="down",
            capacity_price_eur_mw_h=8.0,
            activation_price_eur_mwh=25.0,
            expected_activation_ratio=0.10,
            sustain_duration_h=1.0,
        ),
    ]
