"""Follow a selected portfolio period through reconciled stock, classification and factor contributions."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st

import attribution_dashboard.accounting.realized as realized
import attribution_dashboard.accounting.realized_io as realized_io
import attribution_dashboard.accounting.source_schema as source_schema
import attribution_dashboard.chart_period as chart_period
import attribution_dashboard.chart_settings as visual
import attribution_dashboard.factor_data as factor_data
import attribution_dashboard.pnl_breakdown as breakdown


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
    *,
    settings: visual.ChartSettings = visual.DEFAULT_CHARTS,
    columns: source_schema.SourceColumns = source_schema.DEFAULT_COLUMNS,
) -> None:
    """Link a portfolio chart and every breakdown to one selected period."""
    daily = report.daily
    origin = (daily["date"][0], daily["date"][-1])
    context = f"{directory}_{origin[0]}_{origin[1]}"
    frequency_key = f"explain_frequency_{context}"
    with st.container(horizontal=True, vertical_alignment="bottom"):
        frequency = st.segmented_control(
            "Select a period",
            ["Whole period", "Month", "Day"],
            default="Whole period",
            required=True,
            key=frequency_key,
        )
        periods = period_totals(daily, frequency)
        if frequency != "Whole period":
            period_key = f"explain_period_{context}_{frequency}"
            period_labels = {
                r["period"]: (
                    r["period"].strftime(
                        "%b %Y" if frequency == "Month" else "%d %b %Y"
                    )
                    + f" · {r['net'] * scale:+.3f} {unit}"
                )
                for r in periods.iter_rows(named=True)
            }

            def select_period() -> None:
                chosen = st.session_state.get(period_key)
                match = periods.filter(pl.col("period") == chosen)
                if not match.is_empty():
                    row = match.row(0, named=True)
                    chart_period.queue(row["start"], row["end"])

            st.selectbox(
                "Period to analyse",
                periods["period"].to_list(),
                index=None,
                format_func=lambda value: period_labels[value],
                key=period_key,
                on_change=select_period,
                placeholder="Choose a period…",
            )
    start, end = origin

    def choose_period(points: list[dict]) -> None:
        if not points or "x" not in points[-1]:
            return
        date = dt.date.fromisoformat(str(points[-1]["x"])[:10])
        target_frequency = "Day" if frequency == "Whole period" else frequency
        candidates = period_totals(daily, target_frequency)
        match = candidates.filter(pl.col("start").le(date) & pl.col("end").ge(date))
        if not match.is_empty():
            row = match.row(0, named=True)
            chart_period.queue(row["start"], row["end"])

    figure.update_layout(clickmode="event+select", hovermode="closest")
    for trace in figure.data:
        trace.update(mode="lines+markers", marker={"size": 4, "opacity": 0.35})
        trace.hovertemplate = "%{x|%d %b %Y}<br>" + (trace.hovertemplate or "")
    chart_period.plot(
        figure,
        theme=None,
        key=f"overview_lines_{context}_{frequency}",
        on_points=choose_period if frequency != "Whole period" else None,
        config={"displaylogo": False},
    )
    title = (
        start.strftime("%d %b %Y")
        if start == end
        else f"{start:%d %b %Y} – {end:%d %b %Y}"
    )
    st.markdown(
        f"**What drove {title}?**  Net P&L **{float(daily['long_short_net'].sum()) * scale:+.3f} {unit}**"
    )
    _breakdown(
        report, directory, scale, unit, context, origin, settings, columns=columns
    )


def _breakdown(
    report: realized.RealizedPnlReport,
    directory: Path,
    scale: float,
    unit: str,
    context: str,
    origin: tuple[dt.date, dt.date],
    settings: visual.ChartSettings,
    *,
    columns: source_schema.SourceColumns = source_schema.DEFAULT_COLUMNS,
) -> None:
    """Keep grouping and book-side choices together above the common waterfall."""
    start, end = origin
    with st.container(horizontal=True, vertical_alignment="bottom"):
        group = st.segmented_control(
            "Break down by",
            ["Stocks", "Sectors", "Industries", "Factors"],
            default="Stocks",
            required=True,
            key="breakdown_group",
            persist_state="session",
        )
        side = st.segmented_control(
            "Portfolio",
            ["Combined", "Long", "Short"],
            default="Combined",
            required=True,
            key="breakdown_side",
            persist_state="session",
            help="Long and short show signed gross contributions on the same fixed notional. Combined includes the separately recorded trading costs.",
        )
        count = st.selectbox(
            "Contributors",
            ["Top 10", "Top 20", "All"],
            key="breakdown_count",
            persist_state="session",
            help="Ranked by absolute P&L. Other retains every omitted contribution.",
        )
    try:
        values = _drivers(report, directory, group, side, start, end, columns=columns)
    except (OSError, ValueError, pl.exceptions.PolarsError) as error:
        st.error(f"Cannot show this breakdown: {error}")
        return
    if values is None:
        return
    costs = float(report.daily["cost_pnl"].sum()) if side == "Combined" else None
    total = float(values["pnl"].sum()) + (costs or 0.0)
    if side != "Combined":
        st.caption(
            f"{side} gross P&L {total * scale:+,.{settings.pnl_decimals}f} {unit}"
        )
    driver_key = f"pnl_drivers_{context}_{group}_{side}_{count}"
    interaction = (
        _stock_navigation(
            values, directory, context, side, origin, scale, unit, settings, driver_key
        )
        if group == "Stocks"
        else {}
    )
    st.plotly_chart(
        breakdown.figure(
            values,
            costs,
            scale,
            unit,
            group=group,
            side=side,
            limit=None if count == "All" else int(count.split()[-1]),
            settings=settings,
        ),
        theme=None,
        key=driver_key,
        selection_mode="points",
        config={"displayModeBar": False},
        **interaction,
    )
    with st.expander("Breakdown details"):
        st.caption(
            "Every view uses the selected dates. Drag a range in Whole period mode, or click a date in Day/Month mode. "
            "Reset period restores the wider view. The largest absolute contributors are shown; Other retains the rest. "
            "Long and short are gross contributions to the same fixed notional, not standalone portfolio returns. "
            "Whole-book costs are separate and are not allocated to stocks or sides."
        )
        if group == "Factors":
            st.caption(
                "This is model-based attribution, not a proof of causality. Sector effects combine the model's sector terms. "
                "Residual means unexplained by this specification; it is not necessarily alpha. Reconciliation is the "
                "difference between model and ledger price bases. Unmodeled retains uncovered P&L. "
                "See Factors for the model specification and individual component history."
            )
        elif group in {"Sectors", "Industries"}:
            st.caption(
                f"Classification: {realized_io.read_metadata(directory).classification or 'saved labels'}. "
                "Retrospective snapshots are not point-in-time classifications. Missing labels remain in Unclassified."
            )
        else:
            st.caption(
                "Click a stock bar to open its price, positions and model inputs for this period."
            )
        display = (
            values.lazy()
            .select(
                pl.col("label").alias("Contributor"),
                *[
                    (pl.col(c) * scale).alias(c)
                    for c in ("pnl", "long", "short")
                    if c in values.columns
                ],
            )
            .collect()
        )
        st.dataframe(
            display,
            hide_index=True,
            column_config={
                c: st.column_config.NumberColumn(f"{c.title()} ({unit})", format="%.3f")
                for c in ("pnl", "long", "short")
                if c in display.columns
            },
        )
        st.download_button(
            "Download this breakdown",
            display.write_csv(),
            f"{group.lower()}-{side.lower()}-breakdown.csv",
            "text/csv",
        )
        monthly = period_totals(report.daily, "Month")
        st.download_button(
            "Download monthly P&L",
            monthly.with_columns(
                pl.col("net", "costs", "long", "short") * scale
            ).write_csv(),
            "monthly-pnl.csv",
            "text/csv",
        )


def _drivers(
    report: realized.RealizedPnlReport,
    directory: Path,
    group: str,
    side: str,
    start: dt.date,
    end: dt.date,
    *,
    columns: source_schema.SourceColumns = source_schema.DEFAULT_COLUMNS,
) -> pl.DataFrame | None:
    """Load only optional metadata/model files needed for the selected grouping."""
    if group != "Factors":
        assets = report.assets
        if group == "Industries" and "industry" not in assets.columns:
            source = directory / "classifications.parquet"
            if not source.is_file():
                st.info("Industry classifications are not available for this strategy.")
                return None
            assets = (
                assets.lazy()
                .join(
                    source_schema.scan_parquet(source, columns=columns).select(
                        "asset_id", "industry"
                    ),
                    on="asset_id",
                    how="left",
                    validate="m:1",
                )
                .collect()
            )
        return breakdown.group_totals(assets, group, side)
    folder = directory / "factors"
    if not (folder / "daily.parquet").is_file():
        st.info("Factor attribution is not available for this strategy.")
        return None
    daily = factor_data.read_period(
        folder / "daily.parquet",
        start,
        end,
        factor_data.stamp(folder / "daily.parquet"),
        columns=columns,
    )
    conventions = factor_data.read_conventions(folder)
    values = breakdown.factor_totals(daily, report.daily, conventions=conventions)
    if side == "Combined":
        return values
    if not all(
        (folder / name).is_file()
        for name in ("stocks.parquet", "asset_factors.parquet")
    ):
        st.info("The saved factor bundle has no long/short decomposition.")
        return None
    daily = factor_data.side_partition(
        folder,
        start,
        end,
        side.lower(),
        factor_data.stamp(folder / "stocks.parquet"),
        factor_data.stamp(folder / "asset_factors.parquet"),
        report.daily.select("date"),
        columns=columns,
    )
    parent = report.daily.select(
        "date", pl.col(f"{side.lower()}_pnl").alias("long_short_net")
    )
    return breakdown.factor_totals(daily, parent, conventions=conventions)


def _stock_navigation(
    values: pl.DataFrame,
    directory: Path,
    context: str,
    side: str,
    origin: tuple[dt.date, dt.date],
    scale: float,
    unit: str,
    settings: visual.ChartSettings,
    driver_key: str,
) -> dict:
    """Open a stock from either the search control or its contribution bar."""
    start, end = origin
    stock_key = f"pnl_driver_stock_{context}_{side}"
    labels = {
        r["id"]: f"{r['label']} · {r['pnl'] * scale:+,.{settings.pnl_decimals}f} {unit}"
        for r in values.iter_rows(named=True)
    }

    def inspect_stock() -> None:
        stock = st.session_state.get(stock_key)
        if stock is not None and stock in labels:
            queue_stock(directory, stock, start, end, origin)

    st.selectbox(
        "Open a stock",
        values["id"].to_list(),
        index=None,
        format_func=lambda value: labels[value],
        key=stock_key,
        on_change=inspect_stock,
        placeholder="Search stocks in this period…",
    )

    def click_stock() -> None:
        points = st.session_state[driver_key].get("selection", {}).get("points", [])
        if points:
            custom = points[-1].get("customdata", [])
            if custom and custom[0] in labels:
                queue_stock(directory, custom[0], start, end, origin)

    return {"on_select": click_stock}
