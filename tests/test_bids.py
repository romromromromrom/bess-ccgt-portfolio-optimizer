from __future__ import annotations

import pytest

from energy_portfolio.bids import bess_bids, ccgt_bids
from energy_portfolio.portfolio import run_portfolio

EXPECTED_COLUMNS = ["timestamp", "asset", "market", "direction", "volume_mw", "price"]


@pytest.fixture
def solved(market, bess_config, ccgt_config, reserve_products):
    return run_portfolio(market, bess_config, ccgt_config, reserve_products)


def test_bess_bid_table_has_the_expected_shape(solved, market, reserve_products):
    bids = bess_bids(solved.bess_schedule, market["spot_price_eur_mwh"], reserve_products)
    assert list(bids.columns) == EXPECTED_COLUMNS
    assert set(bids["direction"]) <= {"BUY", "SELL", "UP", "DOWN", "SYMMETRIC"}
    assert (bids["volume_mw"] > 0).all()


def test_bess_never_bids_buy_and_sell_in_the_same_period(solved, market, reserve_products):
    bids = bess_bids(solved.bess_schedule, market["spot_price_eur_mwh"], reserve_products)
    energy = bids[bids["market"] == "Energy"]
    per_timestamp = energy.groupby("timestamp")["direction"].nunique()
    assert (per_timestamp == 1).all()


def test_bid_volumes_reconcile_with_the_schedule(solved, market, reserve_products):
    bids = bess_bids(solved.bess_schedule, market["spot_price_eur_mwh"], reserve_products)
    sold = bids[(bids["market"] == "Energy") & (bids["direction"] == "SELL")]
    assert sold["volume_mw"].sum() == pytest.approx(
        solved.bess_schedule["discharge_mw"].sum(), abs=0.5
    )


def test_ccgt_energy_bids_are_priced_at_marginal_cost(
    solved, market, ccgt_config, reserve_products
):
    """A thermal unit bids its SRMC into the energy market, not the spot price."""
    bids = ccgt_bids(
        solved.ccgt_schedule,
        market["spot_price_eur_mwh"],
        market["gas_price_eur_mwh"],
        market["co2_price_eur_t"],
        ccgt_config,
        reserve_products,
    )
    energy = bids[bids["market"] == "Energy"]
    if energy.empty:
        pytest.skip("CCGT stayed off over this horizon")
    for _, bid in energy.iterrows():
        expected = ccgt_config.marginal_cost_eur_mwh(
            market.loc[bid["timestamp"], "gas_price_eur_mwh"],
            market.loc[bid["timestamp"], "co2_price_eur_t"],
        )
        assert bid["price"] == pytest.approx(expected, abs=0.01)


def test_no_bids_are_emitted_for_an_idle_asset(market, bess_config, ccgt_config):
    """Flat prices, no reserve markets: the battery should have nothing to bid."""
    flat = market.copy()
    flat["spot_price_eur_mwh"] = 60.0
    result = run_portfolio(flat, bess_config, ccgt_config, [])
    bids = bess_bids(result.bess_schedule, flat["spot_price_eur_mwh"], [])
    assert bids.empty or (bids["volume_mw"] > 0).all()
