"""Views for the realized ledger; calculations live in the accounting package."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import streamlit as st

import attribution_dashboard.accounting.realized as realized
import attribution_dashboard.accounting.realized_risk as risk
import attribution_dashboard.chart_settings as visual
import attribution_dashboard.ledger_charts as charts
import attribution_dashboard.pnl_drilldown as drilldown


def overview(
    report: realized.RealizedPnlReport,
    scale: float,
    unit: str,
    *,
    history: pl.DataFrame,
    directory: Path,
    benchmark_daily: pl.DataFrame | None = None,
    benchmark_label: str | None = None,
    opening_date: dt.date | None = None,
    settings: visual.ChartSettings = visual.DEFAULT_CHARTS,
) -> None:
    """Render cumulative fixed-notional P&L, drawdown and monthly totals."""
    daily = report.daily
    st.subheader("Portfolio performance")
    frame = (
        daily.lazy()
        .select(
            "date",
            pl.col("long_pnl").cum_sum().alias("Long"),
            pl.col("short_pnl").cum_sum().alias("Short"),
            pl.col("cost_pnl").cum_sum().alias("Costs"),
            pl.col("long_short_net").cum_sum().alias("Net"),
        )
        .unpivot(index="date", variable_name="series", value_name="value")
        .with_columns(pl.col("value") * scale)
        .collect()
    )
    frame = _benchmark_reference(frame, benchmark_daily, benchmark_label, scale)
    path = realized.fixed_notional_path(history)
    visible = (
        path.lazy()
        .filter(
            pl.col("date").is_between(
                opening_date or daily["date"][0], daily["date"][-1]
            )
        )
        .select("date", (pl.col("drawdown") * scale).alias("value"))
        .collect()
    )
    figure = charts.pnl_drawdown(
        frame,
        visible,
        unit=unit,
        key="overview_lines",
        opening_date=opening_date,
        settings=settings,
    )
    drilldown.render(report, figure, directory, scale, unit, settings=settings)


def risk_reward(
    report: realized.RealizedPnlReport,
    scale: float,
    unit: str,
    *,
    settings: visual.ChartSettings = visual.DEFAULT_CHARTS,
) -> None:
    """Render complete grouped contributions and trailing covariance risk."""
    st.subheader("P&L and risk contributions")
    grouping_control, side_control, risk_control = st.columns(3)
    with grouping_control:
        group = st.segmented_control(
            "Group by",
            ["Sector", "Stock"],
            default="Sector",
            key="group",
            required=True,
        )
    with side_control:
        side = st.selectbox(
            "Positions",
            ["Total", "Long and short", "long", "short"],
            key="side",
            format_func=lambda value: {
                "Total": "Combined",
                "Long and short": "Long / short separately",
                "long": "Long only",
                "short": "Short only",
            }[value],
        )
    with risk_control:
        risk_metric = st.selectbox(
            "Risk measure",
            ["Volatility contribution", "Share of variance"],
            key="risk_measure",
        )
    keys = ("sector",) if group == "Sector" else ("asset_id", "label", "sector")
    grouping = keys if side == "Total" else ("side", *keys)
    values = risk.summarize_realized_risk(report, group_by=grouping)
    if side == "Total":
        values = values.with_columns(
            pl.when(pl.col("sector") == "Costs")
            .then(pl.lit("costs"))
            .otherwise(pl.lit("total"))
            .alias("side")
        )
    elif side in {"long", "short"}:
        values = values.lazy().filter(pl.col("side") == side).collect()
    display = (
        values.lazy()
        .with_columns(
            (pl.col("additive_pnl") * scale).alias("P&L"),
            (pl.col("risk_vol") * 100).alias("Risk contribution (vol pp)"),
            (pl.col("variance_share") * 100).alias("Share of variance (%)"),
            pl.when(pl.col("side") == "costs")
            .then(pl.lit("Costs"))
            .when(pl.col("side") == "total")
            .then(pl.col("sector"))
            .otherwise(pl.concat_str([pl.col("sector"), pl.lit(" · "), pl.col("side")]))
            .alias("name")
            if group == "Sector"
            else pl.col("label").alias("name"),
        )
        .collect()
    )
    if group == "Stock":
        display = charts.stock_labels(display)
    if group == "Stock":
        ranked = display.sort("P&L")
        bars = pl.concat([ranked.head(8), ranked.tail(8)]).unique().sort("P&L")
        st.caption(
            "Up to eight from each end; the download retains every stock and costs."
        )
    else:
        bars = display
    risk_value = (
        "Risk contribution (vol pp)"
        if risk_metric == "Volatility contribution"
        else "Share of variance (%)"
    )
    charts.paired_bars(
        bars,
        unit=unit,
        risk_value=risk_value,
        risk_title="Risk (vol pp)"
        if risk_metric == "Volatility contribution"
        else "Variance share (%)",
        settings=settings,
    )
    st.download_button(
        "Download this breakdown", display.write_csv(), "attribution.csv", "text/csv"
    )
    st.caption(
        "P&L is gross, with costs separate. Risk uses covariance with the net portfolio; negative contributions diversify. Filtered rows need not sum to the portfolio total."
    )
    _risk_history(report, risk_metric, settings)


def _risk_history(
    report: realized.RealizedPnlReport, risk_metric: str, settings: visual.ChartSettings
) -> None:
    """Compare trailing risk by book side or sector."""
    window = st.selectbox(
        "Trailing risk window (trading days)",
        [21, 63, 126, 252],
        index=1,
        key="risk_window",
    )
    if report.daily.height < window:
        st.info(
            f"Select at least {window} trading days to see trailing risk, or choose a shorter window."
        )
    else:
        trend_group = st.selectbox(
            "Risk over time by", ["Book side", "Sector"], key="trend_group"
        )
        keys = ("side",) if trend_group == "Book side" else ("sector",)
        rolling = risk.rolling_realized_risk(report, group_by=keys, window=window)
        column = (
            "risk_vol" if risk_metric == "Volatility contribution" else "variance_share"
        )
        charts.lines(
            rolling.lazy()
            .with_columns(
                (pl.col(column) * 100).alias("value"),
                pl.col(keys[0]).alias("series"),
            )
            .collect(),
            value="value",
            title=f"{window}-day risk (vol pp)"
            if column == "risk_vol"
            else f"{window}-day variance share (%)",
            trim_warmup=True,
            key=f"realized_risk_lines_{trend_group}",
            settings=settings,
        )
        st.caption(
            "All groups including costs add to trailing net volatility. Full windows within the selection only."
            if column == "risk_vol"
            else "Signed shares sum to 100%; negative shares diversify. Full windows within the selection only."
        )


def _benchmark_reference(
    frame: pl.DataFrame,
    daily: pl.DataFrame | None,
    label: str | None,
    scale: float,
) -> pl.DataFrame:
    """Add an optional reference without touching the strategy partition."""
    if label is None or not st.toggle(
        f"Show {label}",
        key="show_benchmark",
        help="Buy-and-hold on initial notional, using saved index returns. Strategy P&L uses fixed notional; its risk and drawdown remain unchanged. The label identifies the index return convention.",
    ):
        return frame
    if daily is None:
        st.info("No saved benchmark returns are available for this book.")
        return frame
    try:
        benchmark = realized.benchmark_path(daily)
    except ValueError as error:
        st.error(f"Cannot show the benchmark: {error}")
        return frame
    return pl.concat(
        [
            frame,
            benchmark.select(
                "date",
                pl.lit(label).alias("series"),
                (pl.col("value") * scale).alias("value"),
            ),
        ]
    )
