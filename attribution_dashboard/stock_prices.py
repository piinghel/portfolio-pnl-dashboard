"""Original prices and recorded holding boundaries for the stock drilldown."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import streamlit as st

import attribution_dashboard.accounting.stock_history as stock_history
import attribution_dashboard.chart_period as chart_period
import attribution_dashboard.chart_settings as visual
import attribution_dashboard.factor_data as data
import attribution_dashboard.linear_history as linear_history
import attribution_dashboard.prediction_detail as predictions
import attribution_dashboard.prediction_history as prediction_history
import attribution_dashboard.stock_price_chart as stock_price_chart


@st.cache_data(max_entries=2, ttl=300, show_spinner=False)
def _calendar(path: Path, stamp: tuple[int, int]) -> pl.DataFrame:
    del stamp
    return pl.scan_parquet(path).select("date").unique().sort("date").collect()


@st.cache_data(max_entries=8, ttl=300, show_spinner=False)
def _read_stock(
    directory: Path, security: str, stamps: tuple[tuple[int, int], ...]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    del stamps
    quotes = (
        pl.scan_parquet(directory / "prices.parquet")
        .filter(pl.col("asset_id") == security)
        .sort("date")
        .collect()
    )
    holdings = (
        pl.scan_parquet(directory / "positions.parquet")
        .filter(pl.col("asset_id") == security)
        .collect()
    )
    calendar = _calendar(
        directory / "positions.parquet", data.stamp(directory / "positions.parquet")
    )
    return quotes, stock_history.position_events(holdings, calendar)


def render(
    directory: Path,
    security: str,
    start: dt.date,
    end: dt.date,
    *,
    contributions: pl.DataFrame,
    label: str,
    scale: float,
    unit: str,
    opening_date: dt.date | None,
    settings: visual.ChartSettings = visual.DEFAULT_CHARTS,
) -> bool:
    """Align adjusted prices, cumulative gross P&L and marked position sizes."""
    paths = [directory / f"{name}.parquet" for name in ("prices", "positions")]
    if not all(path.is_file() for path in paths):
        st.info(
            "Original prices and recorded holdings were not supplied for this ledger."
        )
        return False
    quotes, events = _read_stock(
        directory, security, tuple(data.stamp(path) for path in paths)
    )
    with st.container(horizontal=True):
        basis = st.segmented_control(
            "Stock price",
            ["Adjusted close", "Original close"],
            default="Adjusted close",
            required=True,
            key="stock_price_basis",
        )
        price_scale = st.segmented_control(
            "Price scale",
            ["Linear", "Log"],
            default="Log",
            required=True,
            key="stock_price_scale",
            help="Log scale gives equal percentage moves equal vertical spacing.",
        )
    column = "px_last_unadjusted" if basis == "Original close" else "px_last"
    visible = (
        quotes.lazy()
        .filter(pl.col("date").is_between(opening_date or start, end))
        .collect()
    )
    if visible.is_empty():
        st.info("No saved stock prices in this selection.")
        return False
    log_price = price_scale == "Log"
    if log_price and (visible[column].drop_nulls() <= 0).any():
        st.info(
            "Log scale requires positive prices. Showing linear scale for this selection."
        )
        log_price = False
    currencies = visible["price_currency"].drop_nulls().unique().to_list()
    currency = "/".join(sorted(currencies)) or "source currency"
    selected = (
        events.lazy()
        .filter(pl.col("date").is_between(start, end))
        .join(
            quotes.lazy().select("date", pl.col(column).alias("price")),
            on="date",
            how="left",
            validate="m:1",
        )
        .collect()
    )
    try:
        prediction_bundle = predictions.load(directory, security)
    except (OSError, ValueError, pl.exceptions.PolarsError) as error:
        st.warning(f"Prediction explanation unavailable: {error}")
        prediction_bundle = None
    explanations = (
        {
            (
                r["date"].isoformat(),
                r["side"],
            ): f"Score {r['score']:+.3f} · rank {r['rank']}/{r['universe_size']}<br>Click to explain this decision"
            for r in prediction_bundle[0].iter_rows(named=True)
        }
        if prediction_bundle is not None
        else {}
    )
    figure, axis = stock_price_chart.build(
        visible,
        selected,
        contributions,
        security=security,
        label=label,
        basis=basis,
        column=column,
        currency=currency,
        start=start,
        end=end,
        scale=scale,
        unit=unit,
        opening_date=opening_date,
        log_price=log_price,
        settings=settings,
        explanations=explanations if prediction_bundle is not None else None,
    )
    revision = st.session_state.get("prediction_selection_revision", 0)
    chart_key = f"stock_price_{security}_{start}_{end}_{revision}"

    def open_decision(points: list[dict]) -> None:
        choice = predictions.clicked_decision(points, set(explanations))
        if choice:
            st.session_state["open_prediction"] = (security, *choice)

    chart_options: dict = {
        "width": "stretch",
        "key": chart_key,
        "config": {"displaylogo": False},
    }
    if prediction_bundle is not None:
        action = st.segmented_control(
            "Chart action",
            ["Select period", "Inspect signals"],
            default="Select period",
            required=True,
            key="stock_chart_action",
            help="Drag to analyse dates, or switch to Inspect signals to click holding markers.",
        )
        if action == "Inspect signals":
            # Dense daily prices otherwise win hover/click hit-testing over the
            # larger holding markers. Price hover remains in Select period mode.
            figure.data[0].hovertemplate = None
            figure.data[0].hoverinfo = "skip"
        chart_options["on_points"] = (
            open_decision if action == "Inspect signals" else None
        )
        predictions.controls(
            prediction_bundle,
            security,
            label,
            start,
            end,
            xaxis=axis,
            stock_figure=figure,
            chart_options=chart_options,
        )
    else:
        try:
            rows = linear_history.load(directory, security, start, end)
            if rows is None:
                chart_period.plot(figure, theme=None, **chart_options)
            else:
                prediction_history.render(
                    rows,
                    security,
                    "model",
                    start,
                    end,
                    xaxis=axis,
                    trading_dates=contributions["date"].to_list(),
                    stock_figure=figure,
                    chart_options=chart_options,
                )
        except (OSError, ValueError, pl.exceptions.PolarsError) as error:
            chart_period.plot(figure, theme=None, **chart_options)
            st.warning(f"Predictor history unavailable: {error}")
    with st.expander("Holding dates and price definitions"):
        st.caption(
            "Markers show holding boundaries, not execution fills. Resizing is not an entry; exits mark the first flat session. Positions already open at the start are not shown as new entries."
        )
        st.caption(
            "Exposure is closing position value as a share of fixed notional. Averages include flat days; End is the final selected session. Adjusted close is the ledger's price basis; original close can jump at corporate actions. Price changes alone do not equal P&L."
        )
        missing = selected["price"].null_count()
        if missing:
            st.caption(
                f"{missing} boundaries lack a source price; the download retains them without an invented marker."
            )
        st.download_button(
            "Download holding boundaries",
            selected.write_csv(),
            f"{security}-holding-boundaries.csv",
            "text/csv",
        )
    return True
