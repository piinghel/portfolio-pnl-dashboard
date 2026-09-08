"""Follow a selected portfolio period through its reconciled stock contributions."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st

import attribution_dashboard.accounting.realized as realized


def period_totals(daily: pl.DataFrame, frequency: str) -> pl.DataFrame:
    """Keep actual first/last sessions, including partial boundary months."""
    bucket = (
        pl.col("date").dt.truncate("1mo")
        if frequency == "Month"
        else pl.col("date")
        if frequency == "Day"
        else pl.lit(daily["date"][0])
    )
    return (
        daily.lazy()
        .group_by(bucket.alias("period"))
        .agg(
            pl.col("date").min().alias("start"),
            pl.col("date").max().alias("end"),
            pl.col("long_short_net").sum().alias("net"),
            pl.col("cost_pnl").sum().alias("costs"),
            pl.col("long_pnl").sum().alias("long"),
            pl.col("short_pnl").sum().alias("short"),
        )
        .sort("period")
        .collect()
    )


def stock_totals(assets: pl.DataFrame, start: dt.date, end: dt.date) -> pl.DataFrame:
    """Combine each security's signed long and short P&L without allocating costs."""
    return (
        assets.lazy()
        .filter(pl.col("date").is_between(start, end))
        .group_by("asset_id")
        .agg(
            pl.col("label").first(),
            pl.col("asset_pnl").sum().alias("pnl"),
            pl.col("asset_pnl").filter(pl.col("side") == "long").sum().alias("long"),
            pl.col("asset_pnl").filter(pl.col("side") == "short").sum().alias("short"),
        )
        .sort("pnl", "asset_id")
        .collect()
    )


def breakdown_figure(
    stocks: pl.DataFrame, costs: float, scale: float, unit: str
) -> go.Figure:
    """Show the biggest absolute contributors, preserving the rest and costs."""
    shown = (
        stocks.lazy()
        .sort(pl.col("pnl").abs(), descending=True)
        .head(8)
        .sort("pnl")
        .collect()
    )
    total = float(stocks["pnl"].sum())
    other = total - float(shown["pnl"].sum())
    names = shown["label"].to_list() + ["Other stocks", "Trading costs", "Net P&L"]
    values = [*shown["pnl"].to_list(), other, costs, 0]
    figure = go.Figure(
        go.Waterfall(
            orientation="h",
            y=list(range(len(names))),
            x=[v * scale for v in values],
            customdata=[
                [
                    r["asset_id"],
                    r["label"],
                    r["long"] * scale,
                    r["short"] * scale,
                    f"{r['pnl'] * scale:+.3f}",
                    f"Long {r['long'] * scale:+.3f} · Short {r['short'] * scale:+.3f}<br>Click to inspect stock",
                ]
                for r in shown.iter_rows(named=True)
            ]
            + [
                ["", name, None, None, f"{value * scale:+.3f}", ""]
                for name, value in zip(
                    names[-3:], [other, costs, total + costs], strict=True
                )
            ],
            measure=["relative"] * (len(names) - 1) + ["total"],
            decreasing={"marker": {"color": "#c65d36"}},
            increasing={"marker": {"color": "#20766b"}},
            totals={"marker": {"color": "#182f42"}},
            connector={"line": {"color": "#b9c0c6", "width": 1}},
            hovertemplate="%{customdata[1]}<br>%{customdata[4]} "
            + unit
            + "<br>%{customdata[5]}<extra></extra>",
            textposition="outside",
            text=[f"{v * scale:+.3f}" for v in values[:-1]]
            + [f"{(total + costs) * scale:+.3f}"],
        )
    )
    figure.update_layout(
        height=36 * len(names) + 55,
        template="plotly_white",
        showlegend=False,
        margin={"l": 165, "r": 75, "t": 5, "b": 45},
        clickmode="event+select",
        font={"size": 12, "color": "#37424a"},
    )
    figure.update_yaxes(
        autorange="reversed",
        tickmode="array",
        tickvals=list(range(len(names))),
        ticktext=[name if len(name) <= 25 else name[:22] + "…" for name in names],
    )
    figure.update_xaxes(
        title_text=f"Contribution ({unit})", zeroline=True, zerolinecolor="#a5adb3"
    )
    return figure


def queue_stock(
    directory: Path,
    stock: str,
    start: dt.date,
    end: dt.date,
    origin: tuple[dt.date, dt.date],
) -> None:
    """Carry the selected stock and exact period into the existing detail view."""
    st.session_state["pnl_drilldown_navigation"] = {
        "directory": str(directory),
        "stock": stock,
        "start": start,
        "end": end,
        "page": "Stock detail",
        "origin": origin,
    }


def render(
    report: realized.RealizedPnlReport,
    figure: go.Figure,
    directory: Path,
    scale: float,
    unit: str,
) -> None:
    """Link a portfolio chart, selected day/month, and its stock drivers."""
    daily = report.daily
    origin = (daily["date"][0], daily["date"][-1])
    context = f"{directory}_{origin[0]}_{origin[1]}"
    frequency_key = f"explain_frequency_{context}"
    saved_focus = st.session_state.get("pnl_drilldown_focus", {})
    if saved_focus.get("context") == context:
        st.session_state.setdefault(frequency_key, saved_focus["frequency"])
        st.session_state.setdefault(
            f"explain_period_{context}_{saved_focus['frequency']}",
            saved_focus["chosen"],
        )
    with st.container(horizontal=True, vertical_alignment="bottom"):
        frequency = st.segmented_control(
            "Explain P&L for",
            ["Whole period", "Month", "Day"],
            default=None if frequency_key in st.session_state else "Whole period",
            required=True,
            key=frequency_key,
        )
        assert frequency is not None
        periods = period_totals(daily, frequency)
        period_key = f"explain_period_{context}_{frequency}"
        period_labels = {
            r["period"]: (
                r["period"].strftime("%b %Y" if frequency == "Month" else "%d %b %Y")
                + f" · {r['net'] * scale:+.3f} {unit}"
            )
            for r in periods.iter_rows(named=True)
        }
        if frequency != "Whole period":
            chosen = st.selectbox(
                "Period to explain",
                periods["period"].to_list(),
                format_func=lambda value: period_labels[value],
                key=period_key,
            )
        else:
            chosen = periods["period"][0]
    selection = periods.filter(pl.col("period") == chosen).row(0, named=True)
    start, end = selection["start"], selection["end"]
    st.session_state["pnl_drilldown_focus"] = {
        "context": context,
        "frequency": frequency,
        "chosen": chosen,
    }
    if frequency != "Whole period":
        # Half-day padding makes a selected single session visible too.
        for row in (1, 2):
            figure.add_vrect(
                x0=dt.datetime.combine(start, dt.time()) - dt.timedelta(hours=12),
                x1=dt.datetime.combine(end, dt.time()) + dt.timedelta(hours=12),
                fillcolor="#3275a8",
                opacity=0.10,
                line_width=0,
                row=row,
                col=1,
            )
    chart_key = f"overview_lines_{context}_{frequency}"

    def choose_period() -> None:
        points = st.session_state[chart_key].get("selection", {}).get("points", [])
        if not points or "x" not in points[-1]:
            return
        date = dt.date.fromisoformat(str(points[-1]["x"])[:10])
        target_frequency = "Day" if frequency == "Whole period" else frequency
        candidates = period_totals(daily, target_frequency)
        match = candidates.filter(pl.col("start").le(date) & pl.col("end").ge(date))
        if not match.is_empty():
            st.session_state[frequency_key] = target_frequency
            st.session_state[f"explain_period_{context}_{target_frequency}"] = match[
                "period"
            ][0]

    figure.update_layout(clickmode="event+select", hovermode="closest")
    for trace in figure.data:
        trace.update(mode="lines+markers", marker={"size": 4, "opacity": 0.35})
        trace.hovertemplate = "%{x|%d %b %Y}<br>" + (trace.hovertemplate or "")
    st.plotly_chart(
        figure,
        theme=None,
        key=chart_key,
        on_select=choose_period,
        selection_mode="points",
        config={"displaylogo": False},
    )
    title = (
        start.strftime("%d %b %Y")
        if start == end
        else f"{start:%d %b %Y} – {end:%d %b %Y}"
    )
    st.markdown(
        f"**What drove {title}?**  Net P&L **{selection['net'] * scale:+.3f} {unit}**"
    )
    stocks = stock_totals(report.assets, start, end)
    stock_key = f"pnl_driver_stock_{context}_{start}_{end}"
    labels = {
        r["asset_id"]: f"{r['label']} · {r['pnl'] * scale:+.3f} {unit}"
        for r in stocks.iter_rows(named=True)
    }

    def inspect_stock() -> None:
        stock = st.session_state.get(stock_key)
        if stock is not None and stock in labels:
            queue_stock(directory, stock, start, end, origin)

    st.selectbox(
        "Open a stock",
        stocks["asset_id"].to_list(),
        index=None,
        format_func=lambda value: labels[value],
        key=stock_key,
        on_change=inspect_stock,
        placeholder="Search stocks in this period…",
    )
    driver_key = f"pnl_drivers_{context}_{start}_{end}"

    def click_stock() -> None:
        points = st.session_state[driver_key].get("selection", {}).get("points", [])
        if points:
            custom = points[-1].get("customdata", [])
            if custom and custom[0] in labels:
                queue_stock(directory, custom[0], start, end, origin)

    st.plotly_chart(
        breakdown_figure(stocks, selection["costs"], scale, unit),
        theme=None,
        key=driver_key,
        on_select=click_stock,
        selection_mode="points",
        config={"displayModeBar": False},
    )
    with st.expander("Breakdown details"):
        st.caption(
            "Click the portfolio chart to choose a day, or a month in Month mode. "
            "Click a stock bar to inspect it. The eight largest absolute stock contributions "
            "are shown; Other stocks retains the rest. Stock P&L is gross; trading costs are separate. "
            "These identify where the P&L came from, not the economic cause of a price move."
        )
        display = stocks.select(
            "asset_id",
            "label",
            *[(pl.col(c) * scale).alias(c) for c in ("pnl", "long", "short")],
        )
        st.dataframe(
            display,
            hide_index=True,
            column_config={
                c: st.column_config.NumberColumn(f"{c.title()} ({unit})", format="%.3f")
                for c in ("pnl", "long", "short")
            },
        )
        st.download_button(
            "Download stock breakdown",
            display.write_csv(),
            "stock-breakdown.csv",
            "text/csv",
        )
        monthly = period_totals(daily, "Month")
        st.download_button(
            "Download monthly P&L",
            monthly.with_columns(
                pl.col("net", "costs", "long", "short") * scale
            ).write_csv(),
            "monthly-pnl.csv",
            "text/csv",
        )
