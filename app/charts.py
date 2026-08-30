"""Altair chart builders for the Streamlit app.

Kept out of streamlit_app.py so the plotting logic stays testable and the app
file stays readable as a narrative.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

TIME = alt.X("timestamp:T", title=None)


def _long(df: pd.DataFrame, columns: list[str], value_name: str) -> pd.DataFrame:
    present = [c for c in columns if c in df.columns]
    out = df[present].reset_index()
    out = out.rename(columns={out.columns[0]: "timestamp"})
    return out.melt("timestamp", var_name="series", value_name=value_name)


def price_chart(market: pd.DataFrame) -> alt.Chart:
    data = _long(
        market,
        ["spot_price_eur_mwh", "afrr_up_activation_price_eur_mwh"],
        "eur_mwh",
    )
    labels = {
        "spot_price_eur_mwh": "Spot",
        "afrr_up_activation_price_eur_mwh": "aFRR up activation",
    }
    data["series"] = data["series"].map(labels)
    return (
        alt.Chart(data)
        .mark_line(interpolate="step-after")
        .encode(
            x=TIME,
            y=alt.Y("eur_mwh:Q", title="EUR/MWh"),
            color=alt.Color("series:N", title=None),
            tooltip=["timestamp:T", "series:N", alt.Tooltip("eur_mwh:Q", format=".1f")],
        )
        .properties(height=260)
    )


def reserve_price_chart(market: pd.DataFrame) -> alt.Chart:
    data = _long(
        market,
        [
            "fcr_capacity_price_eur_mw_h",
            "afrr_up_capacity_price_eur_mw_h",
            "afrr_down_capacity_price_eur_mw_h",
        ],
        "eur_mw_h",
    )
    data["series"] = data["series"].str.replace("_capacity_price_eur_mw_h", "", regex=False).str.upper()
    return (
        alt.Chart(data)
        .mark_line(interpolate="step-after")
        .encode(
            x=TIME,
            y=alt.Y("eur_mw_h:Q", title="EUR/MW/h"),
            color=alt.Color("series:N", title="Capacity price"),
            tooltip=["timestamp:T", "series:N", alt.Tooltip("eur_mw_h:Q", format=".1f")],
        )
        .properties(height=220)
    )


def bess_dispatch_chart(schedule: pd.DataFrame, price: pd.Series) -> alt.LayerChart:
    frame = schedule.reset_index().rename(columns={schedule.index.name or "index": "timestamp"})
    frame.columns = ["timestamp", *frame.columns[1:]]
    frame["price"] = price.values

    power = (
        alt.Chart(frame)
        .mark_area(interpolate="step-after", opacity=0.75)
        .encode(
            x=TIME,
            y=alt.Y("net_mw:Q", title="Net power (MW)"),
            color=alt.value("#4c78a8"),
            tooltip=["timestamp:T", alt.Tooltip("net_mw:Q", format=".1f")],
        )
        .properties(height=200)
    )
    soc = (
        alt.Chart(frame)
        .mark_line(color="#e45756", strokeWidth=2)
        .encode(
            x=TIME,
            y=alt.Y("soc_mwh:Q", title="SOC (MWh)", axis=alt.Axis(titleColor="#e45756")),
            tooltip=["timestamp:T", alt.Tooltip("soc_mwh:Q", format=".1f")],
        )
    )
    return alt.layer(power, soc).resolve_scale(y="independent").properties(height=240)


def reserve_stack_chart(schedule: pd.DataFrame, title: str) -> alt.Chart | None:
    columns = [c for c in schedule.columns if c.startswith("reserve_")]
    if not columns:
        return None
    data = _long(schedule, columns, "mw")
    data["series"] = (
        data["series"].str.replace("reserve_", "", regex=False)
        .str.replace("_mw", "", regex=False).str.upper()
    )
    return (
        alt.Chart(data)
        .mark_area(interpolate="step-after")
        .encode(
            x=TIME,
            y=alt.Y("mw:Q", title="Reserve held (MW)", stack="zero"),
            color=alt.Color("series:N", title="Product"),
            tooltip=["timestamp:T", "series:N", alt.Tooltip("mw:Q", format=".1f")],
        )
        .properties(height=220, title=title)
    )


def ccgt_dispatch_chart(schedule: pd.DataFrame) -> alt.LayerChart:
    frame = schedule.reset_index()
    frame.columns = ["timestamp", *frame.columns[1:]]

    power = (
        alt.Chart(frame)
        .mark_area(interpolate="step-after", opacity=0.75, color="#54a24b")
        .encode(
            x=TIME,
            y=alt.Y("p_mw:Q", title="Output (MW)"),
            tooltip=["timestamp:T", alt.Tooltip("p_mw:Q", format=".1f")],
        )
    )
    prices = (
        alt.Chart(
            frame.melt(
                "timestamp",
                value_vars=["price_eur_mwh", "marginal_cost_eur_mwh"],
                var_name="series",
                value_name="eur_mwh",
            ).assign(
                series=lambda d: d["series"].map(
                    {"price_eur_mwh": "Spot", "marginal_cost_eur_mwh": "SRMC"}
                )
            )
        )
        .mark_line(interpolate="step-after", strokeWidth=2)
        .encode(
            x=TIME,
            y=alt.Y("eur_mwh:Q", title="EUR/MWh"),
            color=alt.Color(
                "series:N", title=None,
                scale=alt.Scale(domain=["Spot", "SRMC"], range=["#4c78a8", "#e45756"]),
            ),
            tooltip=["timestamp:T", "series:N", alt.Tooltip("eur_mwh:Q", format=".1f")],
        )
    )
    return alt.layer(power, prices).resolve_scale(y="independent").properties(height=260)


SCENARIO_ORDER = ["Energy only", "Energy + Reserves"]


def comparison_bar_chart(table: pd.DataFrame) -> alt.LayerChart:
    """Decompose each scenario into revenues (up) and costs (down).

    The black tick is the net P&L: the whole point of the chart is that the
    right-hand bar's tick sits higher than the left-hand one, and by how much.
    """
    labels = {
        "energy_revenue_eur": "Energy revenue",
        "reserve_revenue_eur": "Reserve revenue",
        "costs_eur": "Costs",
    }
    data = table.reset_index().melt(
        "scenario", value_vars=list(labels), var_name="component", value_name="eur"
    )
    data["component"] = data["component"].map(labels)
    data.loc[data["component"] == "Costs", "eur"] *= -1

    x = alt.X(
        "scenario:N", sort=SCENARIO_ORDER, title=None, axis=alt.Axis(labelAngle=0)
    )
    bars = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=x,
            y=alt.Y("eur:Q", title="EUR over the horizon"),
            color=alt.Color(
                "component:N",
                title=None,
                scale=alt.Scale(
                    domain=["Energy revenue", "Reserve revenue", "Costs"],
                    range=["#4c78a8", "#54a24b", "#d0d3d8"],
                ),
            ),
            tooltip=["scenario:N", "component:N", alt.Tooltip("eur:Q", format=",.0f")],
        )
    )

    totals = table.reset_index()[["scenario", "total_pnl_eur"]]
    tick = (
        alt.Chart(totals)
        .mark_tick(color="black", thickness=3, size=90)
        .encode(x=x, y=alt.Y("total_pnl_eur:Q"))
    )
    label = (
        alt.Chart(totals)
        .mark_text(dy=-14, fontWeight="bold", fontSize=13)
        .encode(
            x=x,
            y=alt.Y("total_pnl_eur:Q"),
            text=alt.Text("total_pnl_eur:Q", format=",.0f"),
        )
    )
    return alt.layer(bars, tick, label).properties(height=340)


def backtest_chart(detail: pd.DataFrame) -> alt.Chart:
    data = _long(
        detail,
        ["cumulative_expected_pnl", "cumulative_realized_pnl"],
        "eur",
    )
    data["series"] = data["series"].map(
        {
            "cumulative_expected_pnl": "Expected (forecast data)",
            "cumulative_realized_pnl": "Realized (actual data)",
        }
    )
    return (
        alt.Chart(data)
        .mark_line(strokeWidth=2)
        .encode(
            x=TIME,
            y=alt.Y("eur:Q", title="Cumulative P&L (EUR)"),
            color=alt.Color(
                "series:N", title=None,
                scale=alt.Scale(
                    domain=["Expected (forecast data)", "Realized (actual data)"],
                    range=["#4c78a8", "#e45756"],
                ),
            ),
            tooltip=["timestamp:T", "series:N", alt.Tooltip("eur:Q", format=",.0f")],
        )
        .properties(height=320)
    )
