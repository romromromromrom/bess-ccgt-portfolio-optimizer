"""End-to-end checks on the Streamlit app and its chart builders.

AppTest actually executes the app script, so an exception in any tab — a
missing column, a bad Altair encoding, a stale keyword argument — fails here
rather than in front of an interviewer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

import charts  # noqa: E402

from energy_portfolio.portfolio import run_portfolio  # noqa: E402


@pytest.fixture(scope="module")
def solved_pair():
    from energy_portfolio.bess_config import BESSReserveConfig
    from energy_portfolio.config import CCGTConfig
    from energy_portfolio.market_data import synthetic_market_data
    from energy_portfolio.portfolio import reserve_products_from_market

    market = synthetic_market_data(periods=48, seed=42)
    products = reserve_products_from_market(market)
    result = run_portfolio(
        market,
        BESSReserveConfig(
            max_charge_mw=50, max_discharge_mw=50, energy_capacity_mwh=100,
            initial_soc_mwh=50, final_soc_mwh=50,
        ),
        CCGTConfig(),
        products,
    )
    return market, result


def test_every_chart_builder_produces_a_spec(solved_pair):
    market, result = solved_pair
    specs = [
        charts.price_chart(market),
        charts.reserve_price_chart(market),
        charts.bess_dispatch_chart(result.bess_schedule, market["spot_price_eur_mwh"]),
        charts.ccgt_dispatch_chart(result.ccgt_schedule),
        charts.reserve_stack_chart(result.bess_schedule, "BESS"),
    ]
    for spec in specs:
        assert spec is not None
        assert spec.to_dict()  # raises if the encoding is invalid


def test_reserve_stack_returns_none_without_reserve_columns(solved_pair):
    market, result = solved_pair
    stripped = result.bess_schedule.drop(
        columns=[c for c in result.bess_schedule.columns if c.startswith("reserve_")]
    )
    assert charts.reserve_stack_chart(stripped, "empty") is None


def test_comparison_and_backtest_charts_render(solved_pair):
    from energy_portfolio.bess_config import BESSReserveConfig
    from energy_portfolio.config import CCGTConfig
    from energy_portfolio.backtest import backtest_bess
    from energy_portfolio.market_data import make_realized
    from energy_portfolio.portfolio import (
        compare_energy_only_vs_reserves,
        reserve_products_from_market,
    )

    market, result = solved_pair
    products = reserve_products_from_market(market)
    bess_config = BESSReserveConfig(
        max_charge_mw=50, max_discharge_mw=50, energy_capacity_mwh=100,
        initial_soc_mwh=50, final_soc_mwh=50,
    )
    table, _, _ = compare_energy_only_vs_reserves(
        market, bess_config, CCGTConfig(), products
    )
    assert charts.comparison_bar_chart(table).to_dict()

    realized = make_realized(market)
    bt = backtest_bess(
        result.bess_schedule,
        market["spot_price_eur_mwh"],
        realized["spot_price_eur_mwh"],
        bess_config,
        products,
    )
    assert charts.backtest_chart(bt.detail).to_dict()


@pytest.mark.slow
def test_streamlit_app_runs_without_exception():
    """Boot the real app, press Run, and walk every tab."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP_DIR / "streamlit_app.py"), default_timeout=300)

    # First render: nothing solved yet, the app should invite a run and stop.
    app.run()
    assert not app.exception, [str(e) for e in app.exception]
    assert not app.metric, "no results should be shown before the first run"

    # Press "Run optimization" and re-render every tab with real results.
    app.button[0].click().run()
    assert not app.exception, [str(e) for e in app.exception]
    assert len(app.metric) == 4, "expected the four headline metrics"
    assert len(app.tabs) >= 6, "expected all result tabs to render"
    assert app.session_state["result"]["with_reserves"].total_pnl != 0.0
