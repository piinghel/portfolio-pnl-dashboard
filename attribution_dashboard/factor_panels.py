"""Display saved factor ledgers without importing model estimation into the UI."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl
import streamlit as st

import attribution_dashboard.accounting.factor_risk as factor_risk
import attribution_dashboard.chart_settings as visual
import attribution_dashboard.factor_data as factor_data
import attribution_dashboard.factor_exposures as factor_exposures
import attribution_dashboard.factor_history as factor_history
import attribution_dashboard.ledger_charts as charts
import attribution_dashboard.stock_panels as stock_panels


def render(
    directory: Path,
    start: dt.date,
    end: dt.date,
    scale: float,
    unit: str,
    *,
    net_daily: pl.DataFrame,
    history: pl.DataFrame | None = None,
    opening_date: dt.date | None = None,
    settings: visual.ChartSettings = visual.DEFAULT_CHARTS,
) -> None:
    """Explore a saved factor ledger over the dashboard's selected period.

    Parameters
    ----------
    directory
        Parent ledger directory, containing an optional ``factors`` bundle.
    start, end
        Inclusive selected period, shared with the other dashboard views.
    scale, unit
        Display multiplier and label for P&L only.
    net_daily
        Selected parent ledger, used to reject stale factor decompositions.
    """
    folder = directory / "factors"
    required = [
        folder / f"{name}.parquet"
        for name in ("daily", "stocks", "asset_factors", "risk", "coverage")
    ]
    required.append(folder / "manifest.json")
    if not all(path.is_file() for path in required):
        st.info("Factor analysis has not been built for this strategy yet.")
        return
    daily = factor_data.read_period(
        folder / "daily.parquet",
        start,
        end,
        factor_data.stamp(folder / "daily.parquet"),
    )
    if daily.is_empty() and net_daily.is_empty():
        st.info("No factor observations in this period. Choose another date range.")
        return
    try:
        factor_risk.validate_factor_partition(daily, net_daily)
    except ValueError as error:
        st.error(
            f"Cannot show these factors: {error}. Rebuild the factor bundle for this ledger."
        )
        return
    metadata = json.loads((folder / "manifest.json").read_text())
    st.subheader("What exposures explain the P&L?")
    model = metadata.get("model", {})
    scope = model.get(
        "scope_note",
        "Residual means unexplained by this specification; it does not establish stock-specific alpha.",
    )
    with st.expander("Model scope and limits"):
        if metadata.get("description"):
            st.caption(str(metadata["description"]))
        st.caption(scope)
        st.caption(
            "Reconciliation compares ledger prices/P&L with model prices/returns; it is not an investment factor. It can reflect source rounding or a real price-basis mismatch. Original amounts remain in the download."
        )
        st.caption(
            "This is descriptive attribution, not causal signal attribution. Estimation uncertainty is not shown; omitted factors can remain in residual P&L."
        )
        st.caption(
            "Intercept reflects the common baseline times net dollars; it is not portfolio market beta. Read it together with the beta contribution."
            if "beta" in metadata.get("factor_names", [])
            else "Intercept reflects the common baseline times net dollars; it is not benchmark beta."
        )
    st.session_state.setdefault("factor_view", "P&L")
    view = st.segmented_control(
        "Factor analysis",
        ["P&L", "Exposure and risk", "Stock drivers"],
        required=True,
        key="factor_view",
    )
    if view == "P&L":
        factor_history.render(
            folder,
            daily,
            scale,
            unit,
            history=history,
            settings=settings,
        )
    elif view == "Exposure and risk":
        _exposure_risk(
            folder,
            daily,
            start,
            end,
            exposure_units=model.get("exposure_units", "weight × rank score"),
            settings=settings,
        )
    else:
        stock_panels.factor_drivers(
            folder,
            daily,
            start,
            end,
            scale,
            unit,
            opening_date=opening_date,
            settings=settings,
        )


def _exposure_risk(
    folder: Path,
    daily: pl.DataFrame,
    start: dt.date,
    end: dt.date,
    *,
    settings: visual.ChartSettings = visual.DEFAULT_CHARTS,
    exposure_units: str = "weight × rank score",
) -> None:
    names = factor_data.names(daily["factor"].unique().to_list())
    calendar = daily.lazy().select("date").unique().sort("date")
    coverage = factor_data.read_period(
        folder / "coverage.parquet",
        start,
        end,
        factor_data.stamp(folder / "coverage.parquet"),
    )
    factor_exposures.render(daily, coverage, units=exposure_units, settings=settings)
    risk = factor_data.read_period(
        folder / "risk.parquet", start, end, factor_data.stamp(folder / "risk.parquet")
    )
    if not risk.is_empty():
        st.subheader("Forecast risk")
        # Preserve all selected sessions. Nulls must break paths, never bridge missing estimates.
        dense = (
            calendar.join(risk.lazy().select("factor").unique(), how="cross")
            .join(risk.lazy(), on=["date", "factor"], how="left")
            .with_columns(
                (pl.col("risk_vol") * 100).alias("value"),
                pl.col("factor").replace(names).alias("series"),
            )
            .sort("date", "factor")
            .collect()
        )
        charts.lines(
            factor_data.chart_series(dense),
            title="Forecast risk (vol pp)",
            key="factor_forecast_lines",
            points=True,
            settings=settings,
        )
    if not risk.is_empty():
        st.caption(
            "Forecast risk uses prior-session inputs and beginning positions; missing estimates remain gaps."
        )
