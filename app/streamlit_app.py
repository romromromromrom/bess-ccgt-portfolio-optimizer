"""Multi-Asset Energy & Ancillary Services Optimizer — Streamlit demonstrator.

Narrative the UI is built around:
  1. here is the market,
  2. here is what the MILP decides to do in it,
  3. here is what co-optimizing energy and reserves is worth versus energy alone,
  4. here is how that decision held up against data it had never seen,
  5. here are the bids it would actually submit.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make both the app directory and the src layout importable. This lets the app
# run from a bare checkout on deployment targets that install dependencies but
# do not build the project itself (Streamlit Community Cloud). Locally, in
# Docker and in CI the package is pip-installed and these are simply redundant.
_APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_APP_DIR))
sys.path.insert(0, str(_APP_DIR.parent / "src"))

import charts  # noqa: E402

from energy_portfolio.backtest import (  # noqa: E402
    PortfolioBacktestResult,
    backtest_bess,
    backtest_ccgt,
)
from energy_portfolio.bess_config import BESSReserveConfig  # noqa: E402
from energy_portfolio.bids import portfolio_bids  # noqa: E402
from energy_portfolio.config import CCGTConfig  # noqa: E402
from energy_portfolio.market_data import (  # noqa: E402
    make_realized,
    realized_activation_ratios,
    synthetic_market_data,
)
from energy_portfolio.portfolio import (  # noqa: E402
    compare_energy_only_vs_reserves,
    reserve_products_from_market,
    run_portfolio,
)

st.set_page_config(
    page_title="Energy & Ancillary Services Optimizer",
    page_icon="⚡",
    layout="wide",
)

DATA_DISCLAIMER = (
    "**Synthetic market data.** Prices are generated from a seeded model with a "
    "plausible French-market shape (double daily peak, midday solar depression, "
    "cheaper weekends, occasional scarcity spikes). They are *not* sourced from "
    "RTE or ENTSO-E. Swap in a real CSV with the same columns and everything "
    "downstream works unchanged."
)


# --------------------------------------------------------------------------- #
# Solve
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def compute(
    periods: int,
    seed: int,
    bess_power_mw: float,
    bess_capacity_mwh: float,
    bess_sustain_h: float,
    bess_degradation: float,
    ccgt_p_min: float,
    ccgt_p_max: float,
    ccgt_heat_rate: float,
    ccgt_startup: float,
    ccgt_min_up: int,
    ccgt_min_down: int,
    fcr_cap_mw: float,
    afrr_up_cap_mw: float,
    afrr_down_cap_mw: float,
    activation_up: float,
    activation_down: float,
    ccgt_sells_fcr: bool,
    forecast_error_std: float,
    realized_seed: int,
) -> dict:
    market = synthetic_market_data(periods=periods, seed=seed)

    bess_config = BESSReserveConfig(
        max_charge_mw=bess_power_mw,
        max_discharge_mw=bess_power_mw,
        energy_capacity_mwh=bess_capacity_mwh,
        initial_soc_mwh=0.5 * bess_capacity_mwh,
        final_soc_mwh=0.5 * bess_capacity_mwh,
        reserve_sustain_duration_h=bess_sustain_h,
        degradation_cost_eur_mwh=bess_degradation,
    )
    ccgt_config = CCGTConfig(
        p_min_mw=ccgt_p_min,
        p_max_mw=ccgt_p_max,
        heat_rate_mwh_gas_per_mwh_e=ccgt_heat_rate,
        startup_cost_eur=ccgt_startup,
        min_up_time_h=ccgt_min_up,
        min_down_time_h=ccgt_min_down,
    )

    products = reserve_products_from_market(
        market,
        expected_activation_up=activation_up,
        expected_activation_down=activation_down,
        volume_caps_mw={
            "fcr": fcr_cap_mw,
            "afrr_up": afrr_up_cap_mw,
            "afrr_down": afrr_down_cap_mw,
        },
        afrr_sustain_duration_h=bess_sustain_h,
    )
    # A CCGT is not normally prequalified for FCR on the same terms as a
    # battery — FCR needs full activation in 30 s, which a thermal unit cannot
    # deliver from its ramp rate alone.
    ccgt_products = products if ccgt_sells_fcr else [p for p in products if p.name != "fcr"]

    comparison, energy_only, with_reserves = compare_energy_only_vs_reserves(
        market,
        bess_config,
        ccgt_config,
        products,
        bess_reserve_products=products,
        ccgt_reserve_products=ccgt_products,
    )

    realized_market = make_realized(
        market, seed=realized_seed, price_error_std_eur_mwh=forecast_error_std
    )
    ratios = realized_activation_ratios(
        market.index,
        [p.name for p in products],
        {p.name: p.expected_activation_ratio for p in products},
        seed=realized_seed,
    )
    activation_prices = {
        "afrr_up": realized_market["afrr_up_activation_price_eur_mwh"],
        "afrr_down": realized_market["afrr_down_activation_price_eur_mwh"],
    }

    bess_bt = backtest_bess(
        with_reserves.bess_schedule,
        market["spot_price_eur_mwh"],
        realized_market["spot_price_eur_mwh"],
        bess_config,
        products,
        realized_activation_ratio=ratios,
        realized_activation_price=activation_prices,
    )
    ccgt_bt = backtest_ccgt(
        with_reserves.ccgt_schedule,
        market["spot_price_eur_mwh"],
        realized_market["spot_price_eur_mwh"],
        market["gas_price_eur_mwh"],
        realized_market["gas_price_eur_mwh"],
        market["co2_price_eur_t"],
        ccgt_config,
        ccgt_products,
        realized_activation_ratio=ratios,
        realized_activation_price=activation_prices,
    )
    backtest = PortfolioBacktestResult(bess=bess_bt, ccgt=ccgt_bt)

    book = portfolio_bids(
        with_reserves.bess_schedule,
        with_reserves.ccgt_schedule,
        market["spot_price_eur_mwh"],
        market["gas_price_eur_mwh"],
        market["co2_price_eur_t"],
        ccgt_config,
        bess_reserve_products=products,
        ccgt_reserve_products=ccgt_products,
    )

    return {
        "market": market,
        "realized_market": realized_market,
        "comparison": comparison,
        "energy_only": energy_only,
        "with_reserves": with_reserves,
        "backtest": backtest,
        "bids": book,
        "products": products,
        "ccgt_config": ccgt_config,
        "bess_config": bess_config,
    }


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("⚡ Configuration")

with st.sidebar.expander("Horizon & data", expanded=True):
    days = st.slider("Horizon (days)", 1, 14, 7)
    seed = st.number_input("Market seed", value=42, step=1)

with st.sidebar.expander("BESS", expanded=True):
    bess_power = st.slider("Power (MW)", 10.0, 200.0, 50.0, step=5.0)
    bess_capacity = st.slider("Energy capacity (MWh)", 20.0, 800.0, 100.0, step=10.0)
    bess_sustain = st.select_slider(
        "aFRR sustain duration (h)", options=[0.25, 0.5, 1.0, 2.0, 4.0], value=1.0,
        help=(
            "How long a reserved aFRR MW must be deliverable for. Drives the SOC "
            "headroom constraint, so it converts reserve MW into locked-up MWh. "
            "FCR is fixed at 0.25 h by its prequalification rule."
        ),
    )
    bess_degradation = st.slider("Degradation cost (EUR/MWh discharged)", 0.0, 15.0, 2.0, step=0.5)

with st.sidebar.expander("CCGT", expanded=False):
    ccgt_p_max = st.slider("Pmax (MW)", 100.0, 800.0, 400.0, step=25.0)
    ccgt_p_min = st.slider("Pmin (MW)", 20.0, 400.0, 100.0, step=10.0)
    ccgt_heat_rate = st.slider(
        "Heat rate (MWh gas / MWh e)", 1.6, 3.0, 2.0, step=0.05,
        help="2.0 = 50% efficiency.",
    )
    ccgt_startup = st.slider("Startup cost (EUR)", 0.0, 60_000.0, 15_000.0, step=1_000.0)
    ccgt_min_up = st.slider("Min up time (h)", 1, 12, 4)
    ccgt_min_down = st.slider("Min down time (h)", 1, 12, 3)

with st.sidebar.expander("Reserve markets", expanded=False):
    fcr_cap = st.slider("FCR volume cap (MW)", 0.0, 200.0, 25.0, step=5.0)
    afrr_up_cap = st.slider("aFRR up volume cap (MW)", 0.0, 400.0, 100.0, step=10.0)
    afrr_down_cap = st.slider("aFRR down volume cap (MW)", 0.0, 400.0, 100.0, step=10.0)
    activation_up = st.slider("Assumed aFRR up activation ratio", 0.0, 1.0, 0.12, step=0.01)
    activation_down = st.slider("Assumed aFRR down activation ratio", 0.0, 1.0, 0.10, step=0.01)
    ccgt_sells_fcr = st.checkbox(
        "CCGT prequalified for FCR", value=False,
        help="Off by default: FCR requires full activation in 30 s.",
    )

with st.sidebar.expander("Backtest", expanded=False):
    forecast_error = st.slider("Forecast error std (EUR/MWh)", 0.0, 40.0, 12.0, step=1.0)
    realized_seed = st.number_input("Realized-outcome seed", value=7, step=1)

if ccgt_p_min > ccgt_p_max:
    st.sidebar.error("Pmin cannot exceed Pmax.")
    st.stop()

run = st.sidebar.button("▶️  Run optimization", type="primary", width="stretch")

st.title("Multi-Asset Energy & Ancillary Services Optimizer")
st.caption(
    "MILP co-optimization of a BESS and a CCGT across energy, FCR and aFRR — "
    "Pyomo + HiGHS. Technical demonstrator."
)

if run or "result" in st.session_state:
    with st.spinner("Solving the MILP…"):
        st.session_state["result"] = compute(
            periods=int(days * 24),
            seed=int(seed),
            bess_power_mw=float(bess_power),
            bess_capacity_mwh=float(bess_capacity),
            bess_sustain_h=float(bess_sustain),
            bess_degradation=float(bess_degradation),
            ccgt_p_min=float(ccgt_p_min),
            ccgt_p_max=float(ccgt_p_max),
            ccgt_heat_rate=float(ccgt_heat_rate),
            ccgt_startup=float(ccgt_startup),
            ccgt_min_up=int(ccgt_min_up),
            ccgt_min_down=int(ccgt_min_down),
            fcr_cap_mw=float(fcr_cap),
            afrr_up_cap_mw=float(afrr_up_cap),
            afrr_down_cap_mw=float(afrr_down_cap),
            activation_up=float(activation_up),
            activation_down=float(activation_down),
            ccgt_sells_fcr=bool(ccgt_sells_fcr),
            forecast_error_std=float(forecast_error),
            realized_seed=int(realized_seed),
        )
else:
    st.info("Set the assets and markets in the sidebar, then press **Run optimization**.")
    st.markdown(DATA_DISCLAIMER)
    st.stop()

result = st.session_state["result"]
market = result["market"]
comparison = result["comparison"]
with_reserves = result["with_reserves"]
energy_only = result["energy_only"]
backtest = result["backtest"]

# --------------------------------------------------------------------------- #
# Headline metrics
# --------------------------------------------------------------------------- #
uplift_eur = float(comparison.loc["Energy + Reserves", "uplift_eur"])
uplift_pct = float(comparison.loc["Energy + Reserves", "uplift_pct"])
col1, col2, col3, col4 = st.columns(4)
col1.metric("Portfolio P&L (energy only)", f"{energy_only.total_pnl:,.0f} EUR")
col2.metric(
    "Portfolio P&L (energy + reserves)",
    f"{with_reserves.total_pnl:,.0f} EUR",
    delta=f"{uplift_eur:,.0f} EUR ({uplift_pct:.0f}%)",
)
col3.metric("Realized P&L (backtest)", f"{backtest.realized_pnl:,.0f} EUR")
col4.metric(
    "Forecast error cost",
    f"{backtest.forecast_error_cost:,.0f} EUR",
    delta=f"{-backtest.forecast_error_cost:,.0f} EUR",
    delta_color="normal",
)

tabs = st.tabs(
    ["📈 Market", "⚙️ Dispatch", "💰 Energy vs Energy+Reserves", "🔄 Backtest", "📋 Bid book", "☁️ Cloud"]
)

# --------------------------------------------------------------------------- #
with tabs[0]:
    st.warning(DATA_DISCLAIMER)
    st.altair_chart(charts.price_chart(market), width="stretch")
    st.altair_chart(charts.reserve_price_chart(market), width="stretch")
    st.dataframe(market.describe().T.round(2), width="stretch")
    st.download_button(
        "Download market data (CSV)",
        market.to_csv().encode(),
        "market_data.csv",
        "text/csv",
    )

# --------------------------------------------------------------------------- #
with tabs[1]:
    st.subheader("BESS")
    st.altair_chart(
        charts.bess_dispatch_chart(with_reserves.bess_schedule, market["spot_price_eur_mwh"]),
        width="stretch",
    )
    bess_reserves = charts.reserve_stack_chart(
        with_reserves.bess_schedule, "Reserve capacity held — BESS"
    )
    if bess_reserves is not None:
        st.altair_chart(bess_reserves, width="stretch")

    st.subheader("CCGT")
    st.altair_chart(
        charts.ccgt_dispatch_chart(with_reserves.ccgt_schedule), width="stretch"
    )
    ccgt_reserves = charts.reserve_stack_chart(
        with_reserves.ccgt_schedule, "Reserve capacity held — CCGT"
    )
    if ccgt_reserves is not None:
        st.altair_chart(ccgt_reserves, width="stretch")

    schedule = with_reserves.ccgt_schedule
    below_srmc = schedule[
        (schedule["on"] == 1) & (schedule["price_eur_mwh"] < schedule["marginal_cost_eur_mwh"])
    ]
    if not below_srmc.empty:
        st.success(
            f"**The co-optimization result worth explaining:** the CCGT runs for "
            f"**{len(below_srmc)} h** with the spot price *below* its short-run marginal "
            f"cost — losing money on energy — because holding the Pmax−Pmin headroom as "
            f"aFRR capacity more than covers the loss. An energy-only model never finds this."
        )

    st.dataframe(with_reserves.timeseries.round(2), width="stretch", height=280)

# --------------------------------------------------------------------------- #
with tabs[2]:
    st.subheader("What is co-optimization worth?")
    st.markdown(
        "Same assets, same horizon, same prices. The **only** difference is whether "
        "the reserve markets are available to the optimizer, so the uplift is "
        "attributable to co-optimization rather than to a change of assumptions."
    )
    st.altair_chart(charts.comparison_bar_chart(comparison), width="stretch")
    st.dataframe(comparison.T.round(2), width="stretch")
    st.subheader("P&L attribution — energy + reserves")
    st.dataframe(with_reserves.pnl_table().round(0), width="stretch")

# --------------------------------------------------------------------------- #
with tabs[3]:
    st.subheader("Expected vs realized")
    st.markdown(
        "The schedule is **not re-optimized**. It was committed on forecast data; "
        "here it is re-valued against realized prices, realized fuel costs and the "
        "activation the TSO actually called. Re-solving with hindsight would "
        "manufacture skill the strategy never had."
    )
    st.altair_chart(charts.backtest_chart(backtest.detail()), width="stretch")
    st.dataframe(backtest.summary().round(0), width="stretch")
    st.caption(
        f"Forecast error cost = expected − realized = "
        f"**{backtest.forecast_error_cost:,.0f} EUR** over the horizon."
    )

# --------------------------------------------------------------------------- #
with tabs[4]:
    book = result["bids"]
    st.subheader("Bids the optimizer would submit")
    st.caption(
        "CCGT energy bids are priced at short-run marginal cost; reserve bids at the "
        "product's capacity price. Opportunity-cost adders (SOC shadow price, water "
        "value) are a documented next step, not implemented."
    )
    left, right = st.columns([1, 2])
    with left:
        st.dataframe(
            book.groupby(["asset", "market", "direction"])
            .agg(bids=("volume_mw", "size"), total_mw=("volume_mw", "sum"))
            .round(1),
            width="stretch",
        )
    with right:
        st.dataframe(book.head(400), width="stretch", height=400)
    st.download_button(
        "Download bid book (CSV)", book.to_csv(index=False).encode(), "bids.csv", "text/csv"
    )

# --------------------------------------------------------------------------- #
with tabs[5]:
    st.subheader("S3 persistence")
    bucket = os.environ.get("S3_BUCKET", "")
    region = os.environ.get("AWS_REGION", "eu-west-3")
    st.markdown(
        "Market data in, results out. Credentials are never read from the code: "
        "boto3 resolves them from the environment locally and from the **EC2 instance "
        "role** once deployed, so no static key ever exists as a file on the server."
    )
    if not bucket:
        st.info(
            "`S3_BUCKET` is not set, so persistence is disabled. Set it (plus AWS "
            "credentials or an instance role) to enable upload."
        )
    else:
        st.write(f"Target bucket: `{bucket}` in `{region}`")
        if st.button("Upload results to S3"):
            try:
                from energy_portfolio.s3_store import upload_dataframe

                stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
                upload_dataframe(market, bucket, f"market-data/{stamp}.csv", region)
                upload_dataframe(
                    with_reserves.timeseries, bucket, f"results/{stamp}-dispatch.csv", region
                )
                upload_dataframe(result["bids"], bucket, f"results/{stamp}-bids.csv", region)
                st.success(f"Uploaded three objects under prefix `{stamp}`.")
            except Exception as error:  # noqa: BLE001 - surfaced to the user
                st.error(f"Upload failed: {error}")
