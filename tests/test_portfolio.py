from __future__ import annotations

import pytest

from energy_portfolio.portfolio import (
    compare_energy_only_vs_reserves,
    run_portfolio,
    value_bess,
    value_ccgt,
)

TOL = 1e-6


def test_access_to_reserve_markets_can_never_destroy_value(
    market, bess_config, ccgt_config, reserve_products
):
    """Adding a market only adds options: reserve=0 is always still feasible.

    A violation here would mean the co-optimized model is mis-specified — the
    cheapest possible sanity check on the whole formulation.
    """
    energy_only = run_portfolio(market, bess_config, ccgt_config, [])
    with_reserves = run_portfolio(market, bess_config, ccgt_config, reserve_products)
    assert with_reserves.total_pnl >= energy_only.total_pnl - TOL


def test_comparison_table_uplift_is_self_consistent(
    market, bess_config, ccgt_config, reserve_products
):
    table, energy_only, with_reserves = compare_energy_only_vs_reserves(
        market, bess_config, ccgt_config, reserve_products
    )
    assert list(table.index) == ["Energy only", "Energy + Reserves"]
    assert table.loc["Energy only", "reserve_revenue_eur"] == 0.0
    uplift = (
        table.loc["Energy + Reserves", "total_pnl_eur"]
        - table.loc["Energy only", "total_pnl_eur"]
    )
    assert table.loc["Energy + Reserves", "uplift_eur"] == pytest.approx(uplift, abs=0.05)
    assert with_reserves.total_pnl > energy_only.total_pnl


def test_pnl_attribution_reconciles_with_the_valuation_helpers(
    market, bess_config, ccgt_config, reserve_products
):
    """Re-valuing the returned schedules must reproduce the reported P&L."""
    result = run_portfolio(market, bess_config, ccgt_config, reserve_products)
    prices = market["spot_price_eur_mwh"]

    bess_again = value_bess(
        result.bess_schedule, prices, bess_config, reserve_products
    )
    ccgt_again = value_ccgt(
        result.ccgt_schedule,
        prices,
        market["gas_price_eur_mwh"],
        market["co2_price_eur_t"],
        ccgt_config,
        reserve_products,
    )
    assert bess_again.total == pytest.approx(result.bess_pnl.total)
    assert ccgt_again.total == pytest.approx(result.ccgt_pnl.total)


def test_pnl_table_columns_add_up(market, bess_config, ccgt_config, reserve_products):
    result = run_portfolio(market, bess_config, ccgt_config, reserve_products)
    table = result.pnl_table()
    assert table["Portfolio"].loc["total"] == pytest.approx(result.total_pnl)
    assert (
        table["Portfolio"] == table["BESS"] + table["CCGT"]
    ).all()


def test_reserve_volume_caps_are_respected(
    market, bess_config, ccgt_config, reserve_products
):
    result = run_portfolio(market, bess_config, ccgt_config, reserve_products)
    for product in reserve_products:
        if product.max_volume_mw is None:
            continue
        column = f"reserve_{product.name}_mw"
        assert result.bess_schedule[column].max() <= product.max_volume_mw + 1e-6
        assert result.ccgt_schedule[column].max() <= product.max_volume_mw + 1e-6


def test_aggregated_timeseries_sums_both_assets(
    market, bess_config, ccgt_config, reserve_products
):
    result = run_portfolio(market, bess_config, ccgt_config, reserve_products)
    ts = result.timeseries
    assert ts["portfolio_net_mw"].equals(ts["bess_net_mw"] + ts["ccgt_p_mw"])
    for product in reserve_products:
        column = f"reserve_{product.name}_mw"
        assert ts[f"portfolio_{column}"].equals(
            ts[f"bess_{column}"] + ts[f"ccgt_{column}"]
        )


def test_assets_can_hold_different_prequalified_products(
    market, bess_config, ccgt_config, reserve_products
):
    """A CCGT is typically not prequalified for FCR on battery-like terms."""
    ccgt_products = [p for p in reserve_products if p.name != "fcr"]
    result = run_portfolio(
        market,
        bess_config,
        ccgt_config,
        bess_reserve_products=reserve_products,
        ccgt_reserve_products=ccgt_products,
    )
    assert "reserve_fcr_mw" in result.bess_schedule
    assert "reserve_fcr_mw" not in result.ccgt_schedule
