"""Battery (BESS) configuration, including reserve-provision parameters."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class BESSReserveConfig(BaseModel):
    """A grid-scale battery able to arbitrage energy and sell reserve capacity.

    SOC is tracked in MWh at the *end* of each period. Round-trip efficiency is
    split into a charge and a discharge leg, so the energy actually stored is
    `charge_mw * charge_efficiency * dt` and delivering `discharge_mw` drains
    `discharge_mw / discharge_efficiency * dt` from the battery.
    """

    max_charge_mw: float
    max_discharge_mw: float
    energy_capacity_mwh: float

    min_soc_mwh: float = 0.0
    # Default to the full nameplate capacity / a half-full battery.
    max_soc_mwh: float | None = None
    initial_soc_mwh: float | None = None
    # None => the optimizer is free to end the horizon at any SOC. Setting it
    # (typically to the initial SOC) prevents the model from "cheating" by
    # emptying the battery on the last period with no obligation to refill.
    final_soc_mwh: float | None = None

    charge_efficiency: float = Field(default=0.94, gt=0.0, le=1.0)
    discharge_efficiency: float = Field(default=0.94, gt=0.0, le=1.0)

    # Marginal cost of cycling the battery, charged on discharged MWh. Stands
    # in for calendar+cycle degradation; without it the model over-cycles on
    # tiny price spreads that a real operator would never chase.
    degradation_cost_eur_mwh: float = 2.0

    # How long a reserved MW must be deliverable for, in hours. Drives the
    # SOC (energy) headroom constraint. Asset-level default; a product may
    # override it via ReserveProductConfig.sustain_duration_h.
    reserve_sustain_duration_h: float = 0.25

    @model_validator(mode="after")
    def _fill_defaults(self) -> "BESSReserveConfig":
        if self.max_soc_mwh is None:
            self.max_soc_mwh = self.energy_capacity_mwh
        if self.initial_soc_mwh is None:
            self.initial_soc_mwh = 0.5 * self.energy_capacity_mwh
        if self.min_soc_mwh > self.max_soc_mwh:
            raise ValueError("min_soc_mwh cannot exceed max_soc_mwh")
        if not (self.min_soc_mwh <= self.initial_soc_mwh <= self.max_soc_mwh):
            raise ValueError("initial_soc_mwh must lie within the SOC bounds")
        if self.final_soc_mwh is not None and not (
            self.min_soc_mwh <= self.final_soc_mwh <= self.max_soc_mwh
        ):
            raise ValueError("final_soc_mwh must lie within the SOC bounds")
        return self

    @property
    def round_trip_efficiency(self) -> float:
        return self.charge_efficiency * self.discharge_efficiency

    @property
    def usable_energy_mwh(self) -> float:
        return self.max_soc_mwh - self.min_soc_mwh
