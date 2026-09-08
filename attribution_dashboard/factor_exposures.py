"""Explicit complete-book and covered-holding exposure displays."""

from __future__ import annotations

import polars as pl
import streamlit as st

import attribution_dashboard.chart_settings as visual
import attribution_dashboard.factor_data as factor_data
import attribution_dashboard.ledger_charts as charts


def render(
    daily: pl.DataFrame,
    coverage: pl.DataFrame,
    *,
    units: str,
    settings: visual.ChartSettings = visual.DEFAULT_CHARTS,
    conventions: factor_data.FactorConventions = factor_data.DEFAULT_CONVENTIONS,
) -> None:
    """Show unscaled known exposures or the stricter complete-portfolio series."""
    covered = "covered_exposure" in daily.columns
    column = "covered_exposure" if covered else "exposure"
    short_units = (
        "z-score" if "z-score" in units else "rank" if "rank" in units else units
    )
    names = factor_data.names(
        daily["factor"].unique().to_list(), conventions=conventions
    )
    exposures = daily.lazy().filter(
        ~pl.col("factor").is_in(
            ["idio_pnl", "price_basis_gap", "unmodeled_pnl", "costs"]
        )
    )
    calendar = daily.lazy().select("date").unique().sort("date")
    frame = (
        calendar.join(exposures.select("factor").unique(), how="cross")
        .join(exposures, on=["date", "factor"], how="left")
        .with_columns(
            pl.col(column).alias("value"),
            pl.col("factor").replace(names).alias("series"),
        )
        .sort("date", "factor")
        .collect()
    )
    if covered:
        if not coverage.is_empty():
            share = coverage["exposure_coverage"].drop_nulls()
            if len(share):
                st.caption(
                    f"Covered share of gross starting exposure: average {share.mean():.1%}; minimum {share.min():.1%}. Missing holdings are omitted; known weights are not scaled up."
                )
    if frame.is_empty() or frame["value"].null_count() == frame.height:
        st.info("Modeled exposures are unavailable in this period.")
    else:
        _plots(
            frame,
            title=f"{'Covered' if covered else 'Complete'} exposure ({short_units})",
            settings=settings,
            conventions=conventions,
        )
    note = (
        " The intercept is signed invested weight in this scope, not market beta."
        if conventions.intercept_factor in daily["factor"].to_list()
        else ""
    )
    st.caption(f"Style units: {units}.{note} Missing estimates leave gaps.")


def _plots(
    frame: pl.DataFrame,
    *,
    title: str,
    settings: visual.ChartSettings,
    conventions: factor_data.FactorConventions,
) -> None:
    sectors = conventions.sector_mask()
    intercept = (
        pl.col("factor") == conventions.intercept_factor
        if conventions.intercept_factor is not None
        else pl.lit(False)
    )
    charts.lines(
        frame.filter(~sectors & ~intercept),
        title=title,
        key="factor_exposure_lines",
        settings=settings,
    )
    net_dollars = frame.filter(intercept)
    if not net_dollars.is_empty():
        with st.expander("Net dollar exposure"):
            st.caption(
                "Signed invested weight in the selected exposure scope. A beta-neutral portfolio can hold unequal long and short dollars."
            )
            charts.lines(
                net_dollars.with_columns(
                    pl.col("value") * 100, pl.lit("Net dollars").alias("series")
                ),
                title="Net dollars (% notional)",
                key="factor_net_dollar_lines",
                settings=settings,
            )
    sector_frame = frame.filter(sectors)
    if not sector_frame.is_empty():
        with st.expander("Sector exposures"):
            st.caption(
                "Signed sector weights in the same selected exposure scope; these are percentages of fixed notional, not style z-scores."
            )
            charts.lines(
                sector_frame.with_columns(pl.col("value") * 100),
                title="Sector exposure (% notional)",
                key="factor_sector_exposure_lines",
                settings=settings,
            )
