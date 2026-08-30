from __future__ import annotations

import pandas as pd
import pytest

from energy_portfolio.backtest import (
    PortfolioBacktestResult,
    backtest_bess,
    backtest_ccgt,
)
from energy_portfolio.market_data import make_realized, realized_activation_ratios
from energy_portfolio.portfolio import run_portfolio


@pytest.fixture
def solved(market, bess_config, ccgt_config, reserve_products):
    return run_portfolio(market, bess_config, ccgt_config, reserve_products)


@pytest.fixture
def realized(market):
    return make_realized(market, seed=7)


def _bt_bess(schedule, market, realized_market, bess_config, products, **kwargs):
    return backtest_bess(
        schedule,
        market["spot_price_eur_mwh"],
        realized_market["spot_price_eur_mwh"],
        bess_config,
        products,
        **kwargs,
    )


def _bt_ccgt(schedule, market, realized_market, ccgt_config, products, **kwargs):
    return backtest_ccgt(
        schedule,
        market["spot_price_eur_mwh"],
        realized_market["spot_price_eur_mwh"],
        market["gas_price_eur_mwh"],
        realized_market["gas_price_eur_mwh"],
        market["co2_price_eur_t"],
        ccgt_config,
        products,
        **kwargs,
    )


def test_expected_leg_reproduces_the_optimizer_pnl(
    solved, market, bess_config, ccgt_config, reserve_products
):
    """The backtest's 'expected' must BE the optimizer's own number.

    If these two drift apart, forecast_error_cost stops measuring forecast
    error and starts measuring an inconsistency between two valuation models.
    """
    bess = _bt_bess(solved.bess_schedule, market, market, bess_config, reserve_products)
    ccgt = _bt_ccgt(solved.ccgt_schedule, market, market, ccgt_config, reserve_products)
    assert bess.expected_pnl == pytest.approx(solved.bess_pnl.total, rel=1e-9)
    assert ccgt.expected_pnl == pytest.approx(solved.ccgt_pnl.total, rel=1e-9)


def test_perfect_foresight_leaves_no_forecast_error(
    solved, market, bess_config, ccgt_config, reserve_products
):
    """Realized == forecast => the cost of being wrong is exactly zero."""
    bess = _bt_bess(solved.bess_schedule, market, market, bess_config, reserve_products)
    ccgt = _bt_ccgt(solved.ccgt_schedule, market, market, ccgt_config, reserve_products)
    assert bess.forecast_error_cost == pytest.approx(0.0, abs=1e-6)
    assert ccgt.forecast_error_cost == pytest.approx(0.0, abs=1e-6)


def test_forecast_error_cost_is_the_difference_of_the_two_legs(
    solved, market, realized, bess_config, reserve_products
):
    result = _bt_bess(
        solved.bess_schedule, market, realized, bess_config, reserve_products
    )
    assert result.forecast_error_cost == pytest.approx(
        result.expected_pnl - result.realized_pnl
    )
    assert result.detail["cumulative_realized_pnl"].iloc[-1] == pytest.approx(
        result.realized_pnl
    )


def test_backtest_does_not_re_optimize_the_schedule(
    solved, market, realized, bess_config, reserve_products
):
    """The committed decision must survive the backtest untouched."""
    before = solved.bess_schedule.copy(deep=True)
    _bt_bess(solved.bess_schedule, market, realized, bess_config, reserve_products)
    pd.testing.assert_frame_equal(before, solved.bess_schedule)


def test_realized_activation_changes_only_the_realized_leg(
    solved, market, realized, ccgt_config, reserve_products
):
    ratios = realized_activation_ratios(
        market.index,
        [p.name for p in reserve_products],
        {p.name: p.expected_activation_ratio for p in reserve_products},
        seed=11,
    )
    baseline = _bt_ccgt(
        solved.ccgt_schedule, market, realized, ccgt_config, reserve_products
    )
    with_activation = _bt_ccgt(
        solved.ccgt_schedule,
        market,
        realized,
        ccgt_config,
        reserve_products,
        realized_activation_ratio=ratios,
    )
    assert with_activation.expected_pnl == pytest.approx(baseline.expected_pnl)
    assert with_activation.realized_pnl != pytest.approx(baseline.realized_pnl)


def test_portfolio_backtest_aggregates_both_assets(
    solved, market, realized, bess_config, ccgt_config, reserve_products
):
    bess = _bt_bess(solved.bess_schedule, market, realized, bess_config, reserve_products)
    ccgt = _bt_ccgt(solved.ccgt_schedule, market, realized, ccgt_config, reserve_products)
    combined = PortfolioBacktestResult(bess=bess, ccgt=ccgt)
    assert combined.expected_pnl == pytest.approx(bess.expected_pnl + ccgt.expected_pnl)
    assert combined.forecast_error_cost == pytest.approx(
        combined.expected_pnl - combined.realized_pnl
    )
    assert list(combined.summary().columns) == ["BESS", "CCGT", "Portfolio"]
