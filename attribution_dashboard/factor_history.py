"""A single signed composition of factor rewards and risk through time."""

from __future__ import annotations

import datetime as dt
import typing
from pathlib import Path

import polars as pl
import streamlit as st

import attribution_dashboard.accounting.factor_risk as risk
import attribution_dashboard.accounting.source_schema as source_schema
import attribution_dashboard.chart_period as chart_period
import attribution_dashboard.chart_settings as visual
import attribution_dashboard.contribution_chart as contribution_chart
import attribution_dashboard.factor_data as data


def render(
    folder: Path,
    daily: pl.DataFrame,
    scale: float,
    unit: str,
    *,
    history: pl.DataFrame | None,
    settings: visual.ChartSettings,
    columns: source_schema.SourceColumns = source_schema.DEFAULT_COLUMNS,
) -> None:
    """Show gains/losses and risk shares on one shared date axis."""
    start, end = daily["date"].min(), daily["date"].max()
    if not isinstance(start, dt.date) or not isinstance(end, dt.date):
        raise ValueError("Factor history requires a nonempty Date calendar")
    source = data.read_period(
        folder / "daily.parquet",
        dt.date.min,
        end,
        data.stamp(folder / "daily.parquet"),
        columns=columns,
    )
    parent = (
        history
        if history is not None
        else data.read_period(
            folder.parent / "daily.parquet",
            dt.date.min,
            end,
            data.stamp(folder.parent / "daily.parquet"),
            columns=columns,
        )
    )
    try:
        risk.validate_factor_partition(source, parent.filter(pl.col("date") <= end))
    except ValueError as error:
        st.error(f"Cannot show historical drivers: {error}.")
        return
    grouped = data.chart_series(
        source.select("date", "factor", pl.col("pnl").alias("value")),
    ).rename({"series": "factor", "value": "pnl"})
    left, right = st.columns(2)
    with left:
        focus = st.multiselect(
            "Focus factors",
            sorted(grouped["factor"].unique().to_list()),
            key="factor_focus_selection",
            placeholder="Overview — choose factors to combine",
            help="Choose one component or add several to view their summed P&L and combined variance share against the rest of the book. Clear the selection to return to Overview. Intercept enters the focused group only when you select it.",
            wrap=True,
        )
    with right:
        frequency = st.selectbox(
            "Bar frequency",
            ["Automatic", "Weekly", "Monthly", "Quarterly", "Yearly"],
            key="factor_frequency",
        )
    components = _components(grouped, focus)
    interval, frequency = _frequency(
        frequency, daily["date"], settings.contribution_max_periods
    )
    periods = risk.period_factor_pnl(
        components, start=start, end=end, interval=interval
    )
    cumulative = risk.cumulative_factor_pnl(components, start=start, end=end)
    trailing = risk.rolling_factor_risk(
        components, window=settings.contribution_risk_window
    ).filter(pl.col("date").is_between(start, end))
    figure = contribution_chart.figure(
        periods,
        cumulative,
        trailing,
        scale=scale,
        unit=unit,
        frequency=frequency,
        settings=settings,
        opening_date=typing.cast(
            dt.date | None, parent.filter(pl.col("date") < start)["date"].max()
        ),
    )
    chart_period.plot(figure, width="stretch", theme=None, key="factor_composition")
    st.caption(
        "One date axis: cumulative P&L, period gains/losses and trailing realized risk. Negative risk shares diversify; partial periods are retained."
    )
    _performance_comparison(components, start, end, scale, unit)
    st.download_button(
        "Download factor history",
        components.filter(pl.col("date").is_between(start, end)).write_csv(),
        "factor-history.csv",
        "text/csv",
        help="Daily P&L for the displayed groups in raw decimal units of notional, independent of the display unit.",
    )


def _components(grouped: pl.DataFrame, focus: list[str]) -> pl.DataFrame:
    """Keep the overview compact; use consistent groups across all three panels."""
    if not focus:
        name = (
            pl.when(pl.col("factor").is_in(["Residual", "Costs"]))
            .then(pl.col("factor"))
            .when(pl.col("factor").is_in(["Unmodeled", "Reconciliation"]))
            .then(pl.lit("Unmodeled"))
            .otherwise(pl.lit("Modeled factors"))
        )
    else:
        label = (
            " + ".join(sorted(focus))
            if len(focus) <= 3
            else f"Selected components ({len(focus)})"
        )
        name = (
            pl.when(pl.col("factor").is_in(focus))
            .then(pl.lit(label))
            .otherwise(pl.lit("Other components combined"))
        )
    return (
        grouped.lazy()
        .with_columns(name.alias("factor"))
        .group_by("date", "factor")
        .agg(pl.col("pnl").sum())
        .sort("date", "factor")
        .collect()
    )


def _frequency(choice: str, dates: pl.Series, maximum: int) -> tuple[str, str]:
    """Honor explicit periods, or select readable monthly/quarterly/yearly buckets."""
    intervals = {"Weekly": "1w", "Monthly": "1mo", "Quarterly": "1q", "Yearly": "1y"}
    if choice == "Automatic":
        choice = "Yearly"
        for candidate in ["Monthly", "Quarterly"]:
            if dates.dt.truncate(intervals[candidate]).n_unique() <= maximum:
                choice = candidate
                break
    return intervals[choice], choice


def _performance_comparison(
    components: pl.DataFrame, start: dt.date, end: dt.date, scale: float, unit: str
) -> None:
    """Keep exact period comparisons available without another default chart."""
    with st.expander("Compare performance with the preceding period"):
        summary = risk.compare_factor_performance(components, start=start, end=end)
        spans = (
            summary.select("period", "start", "end", "sessions").unique().sort("start")
        )
        st.caption(
            " · ".join(
                f"{r['period']}: {r['start']} → {r['end']} ({r['sessions']} sessions)"
                for r in spans.iter_rows(named=True)
            )
        )
        if spans.height == 1:
            st.caption(
                "Preceding period unavailable: saved history does not cover an equally long window."
            )
        table = (
            summary.with_columns(
                (pl.col("annual_pnl") * scale).alias("annual_pnl"),
                (pl.col("variance_share") * 100).alias("variance_share"),
                pl.when(pl.col("factor") != "Costs")
                .then(pl.col("return_risk"))
                .alias("return_risk"),
            )
            .sort(
                [
                    pl.col("factor") != "Net",
                    pl.col("factor") == "Costs",
                    pl.col("factor"),
                    pl.col("start"),
                ]
            )
            .select(
                pl.col("factor").alias("Source"),
                pl.col("period").alias("Period"),
                pl.col("annual_pnl").alias("P&L / year"),
                pl.col("variance_share").alias("Variance share (%)"),
                pl.col("return_risk").alias("Own return / risk"),
                pl.col("ratio_contribution").alias("Contribution to net ratio"),
            )
        )
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "P&L / year": st.column_config.NumberColumn(
                    f"P&L / year ({unit})", format="%.3f"
                ),
                "Variance share (%)": st.column_config.NumberColumn(format="%.2f"),
                "Own return / risk": st.column_config.NumberColumn(format="%.2f"),
                "Contribution to net ratio": st.column_config.NumberColumn(
                    format="%.2f"
                ),
            },
        )
        st.caption(
            "Return/risk = annualized mean ÷ volatility, with a zero hurdle. Own ratios do not add; contributions use net volatility and do add. Residual return/risk is model-dependent idiosyncratic IR. Unmodeled includes uncovered P&L and reconciliation. Risk is realized covariance contribution."
        )
