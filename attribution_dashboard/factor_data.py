"""Saved factor bundle reads and labels, isolated from dashboard layout."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import streamlit as st

import attribution_dashboard.accounting.source_schema as source_schema


def stamp(path: Path) -> tuple[int, int]:
    """Identify a local artifact revision for the bounded read cache."""
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


@st.cache_data(max_entries=8, ttl=300, show_spinner=False)
def read_period(
    path: Path,
    start: dt.date,
    end: dt.date,
    stamp: tuple[int, int],
    *,
    columns: source_schema.SourceColumns = source_schema.DEFAULT_COLUMNS,
) -> pl.DataFrame:
    """Read one inclusive date range from a saved factor artifact."""
    del stamp  # File identity participates in the cache key.
    return (
        source_schema.scan_parquet(path, columns=columns)
        .filter(pl.col("date").is_between(start, end))
        .collect()
    )


def names(factors: list[str]) -> dict[str, str]:
    """Keep factor IDs in data and short readable names in displays."""
    return {
        **{name: name.removeprefix("sector:") for name in factors},
        "beta": "Beta",
        "reversal": "Reversal",
        "market": "Intercept",
        "size": "Size",
        "momentum": "Momentum",
        "volatility": "Volatility",
        "idio_pnl": "Residual",
        "idio": "Residual",
        "price_basis_gap": "Reconciliation",
        "unmodeled_pnl": "Unmodeled",
        "costs": "Costs",
    }


@st.cache_data(max_entries=4, ttl=300, show_spinner=False)
def side_partition(
    folder: Path,
    start: dt.date,
    end: dt.date,
    side: str,
    stocks_stamp: tuple[int, int],
    factors_stamp: tuple[int, int],
    calendar: pl.DataFrame,
    *,
    columns: source_schema.SourceColumns = source_schema.DEFAULT_COLUMNS,
) -> pl.DataFrame:
    """Read a gross long/short factor partition, retaining all residual components.

    Values already carry the signed portfolio contribution. No side normalization
    or allocation of aggregate trading costs takes place here. The caller must
    reconcile this partition to the parent ledger before showing it. The supplied
    ledger calendar participates in the cache key and retains fully flat sessions.
    """
    del stocks_stamp, factors_stamp
    modeled = (
        source_schema.scan_parquet(folder / "asset_factors.parquet", columns=columns)
        .filter(pl.col("date").is_between(start, end) & (pl.col("view") == side))
        .select("date", "factor", "pnl")
    )
    residual = (
        source_schema.scan_parquet(folder / "stocks.parquet", columns=columns)
        .filter(pl.col("date").is_between(start, end) & (pl.col("side") == side))
        .select("date", "idio_pnl", "price_basis_gap", "unmodeled_pnl")
        .unpivot(index="date", variable_name="factor", value_name="pnl")
    )
    rows = pl.concat([modeled, residual]).collect()
    if rows.filter(pl.col("pnl").is_null() | ~pl.col("pnl").is_finite()).height:
        raise ValueError("The side factor partition contains missing or nonfinite P&L")
    # A flat side still has a genuine zero contribution on each parent session.
    flat_sessions = (
        calendar.lazy()
        .filter(pl.col("date").is_between(start, end))
        .select(
            "date", pl.lit("unmodeled_pnl").alias("factor"), pl.lit(0.0).alias("pnl")
        )
    )
    return (
        pl.concat([rows.lazy(), flat_sessions])
        .group_by("date", "factor")
        .agg(pl.col("pnl").sum())
        .sort("date", "factor")
        .collect()
    )


@st.cache_data(max_entries=4, ttl=300, show_spinner=False)
def driver_rows(
    folder: Path,
    start: dt.date,
    end: dt.date,
    chosen: str,
    stocks_stamp: tuple[int, int],
    factors_stamp: tuple[int, int],
    *,
    columns: source_schema.SourceColumns = source_schema.DEFAULT_COLUMNS,
) -> pl.DataFrame:
    """Read security contributions for one selected factor or residual component."""
    del stocks_stamp, factors_stamp
    stocks = source_schema.scan_parquet(
        folder / "stocks.parquet", columns=columns
    ).filter(pl.col("date").is_between(start, end))
    keys = ["date", "asset_id", "side"]
    if chosen in {"idio_pnl", "price_basis_gap", "unmodeled_pnl"}:
        return stocks.select(
            *keys, "label", "sector", pl.col(chosen).alias("pnl")
        ).collect()
    values = (
        source_schema.scan_parquet(folder / "asset_factors.parquet", columns=columns)
        .filter(pl.col("date").is_between(start, end) & (pl.col("factor") == chosen))
        .rename({"view": "side"})
    )
    return (
        values.select(*keys, "pnl")
        .join(
            stocks.select(*keys, "label", "sector"), on=keys, how="left", validate="1:1"
        )
        .collect()
    )


def chart_series(
    frame: pl.DataFrame,
    *,
    separate_sectors: bool = False,
    include_net: bool = False,
    combine_unexplained: bool = False,
) -> pl.DataFrame:
    """Group display contributions without changing the source attribution ledger.

    Missing estimates remain missing. The optional Net overlay is the full sum,
    independent of whether sector components are grouped or displayed separately.
    """
    labels = names(frame["factor"].unique().to_list())
    values = frame.lazy().select("date", "factor", "value")
    total = (
        pl.when(pl.col("value").is_not_null().all())
        .then(pl.col("value").sum())
        .otherwise(None)
    )
    grouped = (
        values.with_columns(
            pl.when(
                pl.col("factor").str.starts_with("sector:")
                & pl.lit(not separate_sectors)
            )
            .then(pl.lit("Sectors"))
            .when(
                pl.col("factor").is_in(["idio_pnl", "unmodeled_pnl", "price_basis_gap"])
                & pl.lit(combine_unexplained)
            )
            .then(pl.lit("Unexplained"))
            .otherwise(pl.col("factor").replace(labels))
            .alias("series")
        )
        .group_by("date", "series")
        .agg(total.alias("value"))
        .select("date", "series", "value")
    )
    if include_net:
        net = (
            values.group_by("date")
            .agg(total.alias("value"))
            .with_columns(pl.lit("Net").alias("series"))
            .select("date", "series", "value")
        )
        grouped = pl.concat([grouped, net])
    return grouped.sort("date", "series").collect()
