from __future__ import annotations

import pytest

from energy_portfolio.bess_config import BESSReserveConfig
from energy_portfolio.config import CCGTConfig
from energy_portfolio.market_data import synthetic_market_data
from energy_portfolio.portfolio import reserve_products_from_market


@pytest.fixture(scope="session")
def market():
    """A short synthetic horizon: long enough to be interesting, fast to solve."""
    return synthetic_market_data(periods=48, seed=42)


@pytest.fixture
def bess_config() -> BESSReserveConfig:
    return BESSReserveConfig(
        max_charge_mw=50,
        max_discharge_mw=50,
        energy_capacity_mwh=100,
        initial_soc_mwh=50,
        final_soc_mwh=50,
    )


@pytest.fixture
def ccgt_config() -> CCGTConfig:
    return CCGTConfig()


@pytest.fixture
def reserve_products(market):
    return reserve_products_from_market(market)
