"""A small P&L-first explorer over the reconciled security ledger."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl
import streamlit as st

import attribution_dashboard.accounting.realized as realized
import attribution_dashboard.accounting.realized_io as realized_io
import attribution_dashboard.chart_period as chart_period
import attribution_dashboard.chart_settings as visual
import attribution_dashboard.config as config_mod
import attribution_dashboard.factor_panels as factor_panels
import attribution_dashboard.realized_panels as panels
import attribution_dashboard.stock_panels as stock_panels


@st.cache_data(
    max_entries=2, ttl=3600, show_spinner="Recalculating the selected period…"
)
def load_period(
    directory: str, start: dt.date, end: dt.date, revision: tuple[int, ...]
) -> realized.RealizedPnlReport:
    """Read only the requested dates; recompute linking from the raw P&L."""
    return realized_io.load_period(directory, start, end)


def render_config(path: Path | str) -> None:
    """Open a strategy selected from an external instance configuration."""
    try:
        config = config_mod.load_config(path)
    except (OSError, ValueError) as error:
        st.error(f"Cannot open dashboard configuration: {error}")
        return
    if not config.books:
        st.error("The dashboard configuration must list at least one books entry.")
        return
    if len(config.books) == 1:
        book = config.books[0]
        st.sidebar.caption(book.display_label)
    else:
        chosen = st.sidebar.selectbox(
            "Portfolio", [book.display_label for book in config.books], key="strategy"
        )
        book = next(book for book in config.books if book.display_label == chosen)
    render(
        book.directory,
        label=book.display_label,
        default_start=book.default_start,
        default_end=book.default_end,
        benchmark_label=book.benchmark_label,
        settings=config.charts,
    )


def render(
    directory: Path,
    *,
    label: str | None = None,
    default_start: dt.date | None = None,
    default_end: dt.date | None = None,
    benchmark_label: str | None = None,
    settings: visual.ChartSettings = visual.DEFAULT_CHARTS,
) -> None:
    """Render the local ledger explorer; all accounting and risk math is shared."""
    st.set_page_config(
        page_title="Portfolio P&L", page_icon=":material/monitoring:", layout="wide"
    )
    st.html(
        "<style>[data-testid=stMainBlockContainer]{padding-left:1.5rem;padding-right:1.5rem;}</style>"
    )
    st.title("Portfolio P&L")
    files = [
        directory / name
        for name in ("assets.parquet", "daily.parquet", "manifest.json")
    ]
    if not all(path.exists() for path in files):
        st.error(
            f"The configured ledger is incomplete: {directory}. Expected assets.parquet, daily.parquet and manifest.json."
        )
        return
    try:
        metadata = realized_io.read_metadata(directory)
    except (OSError, ValueError) as error:
        st.error(f"Cannot open this ledger: {error}")
        return
    st.caption(metadata.description or label or metadata.name)
    history = (
        pl.scan_parquet(files[1])
        .select("date", "long_short_net")
        .sort("date")
        .collect()
    )
    calendar = history["date"]
    if calendar.is_empty():
        st.error("The configured ledger has no trading days.")
        return
    first, last = calendar[0], calendar[-1]
    chart_period.apply_pending(str(directory))
    navigation = st.session_state.pop("pnl_drilldown_navigation", None)
    if navigation and navigation["directory"] == str(directory):
        st.session_state["preset"] = "Custom"
        st.session_state[f"dates_{directory}_Custom"] = (
            navigation["start"],
            navigation["end"],
        )
        st.session_state["page"] = navigation["page"]
        st.session_state["pnl_drilldown_origin"] = (
            str(directory),
            *navigation["origin"],
        )
    else:
        navigation = None

    with st.sidebar:
        st.subheader("Period")
        presets = [
            "Full history",
            "Latest year",
            "Year to date (YTD)",
            "Month to date (MTD)",
            "Last 5 years",
            "Last 10 years",
            "Worst drawdown",
            "Custom",
        ]
        if default_start is not None and default_end is not None:
            presets.insert(0, "Saved period")
        preset = st.selectbox(
            "Start with",
            presets,
            key="preset",
            on_change=chart_period.clear_origin,
        )
        if (
            preset == "Saved period"
            and default_start is not None
            and default_end is not None
        ):
            start, end = (
                max(first, default_start),
                min(last, default_end),
            )
            if start > end:
                st.info(
                    "The saved period does not overlap this ledger. Choose another period preset."
                )
                return
        elif preset == "Worst drawdown":
            start, end = realized.worst_drawdown_period(history)
        elif preset in {"Latest year", "Last 5 years", "Last 10 years"}:
            years = {"Latest year": 1, "Last 5 years": 5, "Last 10 years": 10}[preset]
            start, end = (
                max(first, pl.Series([last]).dt.offset_by(f"-{years}y")[0]),
                last,
            )
        elif preset == "Year to date (YTD)":
            start, end = max(first, last.replace(month=1, day=1)), last
        elif preset == "Month to date (MTD)":
            start, end = max(first, last.replace(day=1)), last
        else:
            start, end = first, last
        selected = st.date_input(
            "Date range",
            value=(start, end),
            min_value=first,
            max_value=last,
            key=f"dates_{directory}_{preset}",
            on_change=chart_period.clear_origin,
        )
        units = st.selectbox(
            "P&L units", ["% of fixed notional", "Money (millions)"], key="units"
        )
        origin = st.session_state.get("analysis_origin")
        if origin and origin["directory"] == str(directory):
            st.button("Reset period", on_click=chart_period.reset, width="stretch")
        st.caption("Drag across a time chart to analyse that period in every view.")
        st.caption(f"Available: {first} → {last}")
    if len(selected) != 2:
        st.info("Select both the start and end dates.")
        return
    start, end = selected
    if not calendar.filter(calendar.is_between(start, end)).len():
        st.info("There are no trading days in this date range.")
        return
    st.session_state["analysis_context"] = {
        "directory": str(directory),
        "start": start,
        "end": end,
        "calendar": calendar.filter(calendar.is_between(start, end)).to_list(),
    }
    revision = tuple(path.stat().st_mtime_ns for path in files)
    try:
        report = load_period(str(directory), start, end, revision)
    except (OSError, ValueError, pl.exceptions.PolarsError) as error:
        st.error(f"Cannot calculate this period: {error}")
        return
    context = (str(directory), start, end)
    if st.session_state.get("detail_context") != context:
        previous = st.session_state.get("detail_context")
        different_book = previous is None or previous[0] != str(directory)
        st.session_state["reset_stock_detail"] = different_book
        st.session_state["reset_factor_stock_detail"] = different_book
        st.session_state["detail_context"] = context
    if navigation and navigation.get("stock"):
        st.session_state["stock"] = navigation["stock"]
        st.session_state["reset_stock_detail"] = False
    notional = metadata.notional
    scale, unit = (
        (100.0, "% notional")
        if units == "% of fixed notional"
        else (notional / 1e6, f"{metadata.currency} m")
    )
    daily = report.daily
    preceding = calendar.filter(calendar < daily["date"][0])
    opening_date = preceding[-1] if len(preceding) else None
    st.caption(
        f"{daily['date'][0]} → {daily['date'][-1]} · {daily.height:,} trading days · fixed notional {notional:,.0f} {metadata.currency}"
    )
    with st.container(horizontal=True):
        st.metric(
            f"Net P&L ({unit})",
            f"{float(daily['long_short_net'].sum()) * scale:+.3f}",
            help=unit,
            border=True,
        )
        st.metric(
            f"Trading costs ({unit})",
            f"{float(daily['cost_pnl'].sum()) * scale:+.3f}",
            help=unit,
            border=True,
        )
        vol = daily["long_short_net"].std(ddof=1)
        st.metric(
            "Realized volatility",
            f"{vol * 252**0.5:.2%}" if vol is not None else "—",
            help="Annualized daily net P&L volatility, normalized by fixed notional.",
            border=True,
        )
    unclassified = (
        report.assets.lazy()
        .filter(pl.col("sector") == "Unclassified")
        .select(pl.col("asset_id").n_unique())
        .collect()
        .item()
    )
    if unclassified:
        st.caption(
            f"{unclassified:,} stocks have no sector classification; their P&L remains in Unclassified."
        )
    st.session_state.setdefault("page", "Overview")
    origin = st.session_state.get("pnl_drilldown_origin")
    if (
        st.session_state["page"] == "Stock detail"
        and origin
        and origin[0] == str(directory)
    ):

        def back_to_breakdown() -> None:
            st.session_state["pnl_drilldown_navigation"] = {
                "directory": str(directory),
                "start": origin[1],
                "end": origin[2],
                "page": "Overview",
                "origin": (origin[1], origin[2]),
            }

        st.button("Back to P&L breakdown", on_click=back_to_breakdown)
    page = st.segmented_control(
        "Explore",
        ["Overview", "Risk and reward", "Factors", "Stock detail"],
        key="page",
        required=True,
    )
    if page == "Factors":
        factor_panels.render(
            directory,
            start,
            end,
            scale,
            unit,
            net_daily=report.daily,
            history=history,
            opening_date=opening_date,
            settings=settings,
        )
    elif page == "Risk and reward":
        panels.risk_reward(report, scale, unit, settings=settings)
    elif page == "Stock detail":
        stock_panels.render(
            report,
            scale,
            unit,
            directory=directory,
            opening_date=opening_date,
            settings=settings,
        )
    else:
        panels.overview(
            report,
            scale,
            unit,
            history=history,
            directory=directory,
            benchmark_label=benchmark_label,
            benchmark_daily=(
                pl.scan_parquet(files[1])
                .select("date", "benchmark")
                .filter(pl.col("date").is_between(start, end))
                .collect()
                if benchmark_label is not None
                and "benchmark" in pl.scan_parquet(files[1]).collect_schema()
                else None
            ),
            opening_date=opening_date,
            settings=settings,
        )
    with st.expander("Data and definitions"):
        st.caption(
            "Drag to zoom; double-click to reset. Click a legend to hide a series."
        )
        if metadata.pnl_method:
            st.markdown(metadata.pnl_method)
        if metadata.classification:
            st.caption(f"Sector labels: {metadata.classification}.")
        source_text = files[2].read_text()
        omitted_costs = (
            json.loads(source_text).get("conventions", {}).get("omitted_costs")
        )
        if omitted_costs:
            st.markdown(f"**Costs not included:** {omitted_costs}.")
        st.markdown(
            "P&L adds daily contributions on fixed notional; it does not assume reinvestment. Drawdown retains peaks before the selected period."
        )
        st.markdown(
            "Risk contributions use covariance with the net portfolio, including flat days. Sector groupings are distinct from factor attribution."
        )
        st.markdown(
            f"Maximum daily reconciliation error: {daily['reconciliation_error'].abs().max():.2e}."
        )
        st.download_button(
            "Download data metadata",
            source_text,
            "manifest.json",
            "application/json",
        )
