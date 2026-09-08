"""Saved factor bundle reads and labels, isolated from dashboard layout."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import streamlit as st

import attribution_dashboard.accounting.source_schema as source_schema


@dataclass(frozen=True)
class FactorConventions:
    """Saved model roles; an intercept has unit loading and sectors are indicators."""

    intercept_factor: str | None = "market"
    sector_prefix: str | None = "sector:"
    factor_labels: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        accounting = {"idio_pnl", "idio", "price_basis_gap", "unmodeled_pnl", "costs"}
        for name in ("intercept_factor", "sector_prefix"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"model.{name} must be a nonempty string or null")
        if self.intercept_factor in accounting or (
            self.sector_prefix is not None
            and any(key.startswith(self.sector_prefix) for key in accounting)
        ):
            raise ValueError("Model factor roles cannot include accounting components")
        if (
            self.intercept_factor is not None
            and self.sector_prefix is not None
            and self.intercept_factor.startswith(self.sector_prefix)
        ):
            raise ValueError("The intercept factor cannot also be a sector factor")
        for key, label in self.factor_labels:
            if not all(
                isinstance(value, str) and value.strip() for value in (key, label)
            ):
                raise ValueError(
                    "model.factor_labels must map nonempty strings to labels"
                )
            if key in accounting:
                raise ValueError("Accounting component labels cannot be overridden")
            if label in {
                "Residual",
                "Reconciliation",
                "Unmodeled",
                "Costs",
                "Net",
                "Sectors",
                "Sector effects",
            }:
                raise ValueError(
                    f"Factor label {label!r} is reserved for accounting groups"
                )

    def sector_mask(self) -> pl.Expr:
        """Select categorical sector factors without guessing from their labels."""
        return (
            pl.col("factor").str.starts_with(self.sector_prefix)
            if self.sector_prefix is not None
            else pl.lit(False)
        )


DEFAULT_CONVENTIONS = FactorConventions()


def conventions(
    metadata: dict, *, defaults: FactorConventions = DEFAULT_CONVENTIONS
) -> FactorConventions:
    """Validate optional model conventions in a factor manifest."""
    if not isinstance(metadata, dict) or not isinstance(
        metadata.get("model", {}), dict
    ):
        raise ValueError("Factor manifest and model must be mappings")
    model = metadata.get("model", {})
    labels = model.get("factor_labels", {})
    if not isinstance(labels, dict):
        raise ValueError("model.factor_labels must be a mapping")
    return FactorConventions(
        intercept_factor=model.get("intercept_factor", defaults.intercept_factor),
        sector_prefix=model.get("sector_prefix", defaults.sector_prefix),
        factor_labels=tuple(labels.items()),
    )


def read_conventions(folder: Path) -> FactorConventions:
    """Read model roles afresh so metadata edits immediately update displays."""
    return conventions(json.loads((folder / "manifest.json").read_text()))


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


def names(
    factors: list[str], *, conventions: FactorConventions = DEFAULT_CONVENTIONS
) -> dict[str, str]:
    """Keep factor IDs in data and short readable names in displays."""
    labels = {
        **{
            name: name.removeprefix(conventions.sector_prefix or "") for name in factors
        },
        "beta": "Beta",
        "reversal": "Reversal",
        "size": "Size",
        "momentum": "Momentum",
        "volatility": "Volatility",
        "idio_pnl": "Residual",
        "idio": "Residual",
        "price_basis_gap": "Reconciliation",
        "unmodeled_pnl": "Unmodeled",
        "costs": "Costs",
    }
    if conventions.intercept_factor is not None:
        labels[conventions.intercept_factor] = "Intercept"
    labels.update(conventions.factor_labels)
    selected = {name: labels[name] for name in factors}
    if len(set(selected.values())) != len(selected):
        raise ValueError("Factor display labels must distinguish every saved component")
    return selected


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
    conventions: FactorConventions = DEFAULT_CONVENTIONS,
) -> pl.DataFrame:
    """Group display contributions without changing the source attribution ledger.

    Missing estimates remain missing. The optional Net overlay is the full sum,
    independent of whether sector components are grouped or displayed separately.
    """
    labels = names(frame["factor"].unique().to_list(), conventions=conventions)
    values = frame.lazy().select("date", "factor", "value")
    total = (
        pl.when(pl.col("value").is_not_null().all())
        .then(pl.col("value").sum())
        .otherwise(None)
    )
    grouped = (
        values.with_columns(
            pl.when(conventions.sector_mask() & pl.lit(not separate_sectors))
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
