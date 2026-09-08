"""Read optional local linear-model history without requiring trade explanations."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl
import streamlit as st

import attribution_dashboard.factor_data as data


@st.cache_data(max_entries=12, ttl=300, show_spinner="Loading stock predictor history…")
def _read(
    source: str, security: str, start: dt.date, end: dt.date, stamps: tuple
) -> pl.DataFrame:
    del stamps
    config = json.loads(source)
    coefficients = pl.scan_parquet(config["coefficients"])
    features = (
        coefficients.select("predictor").unique().collect()["predictor"].to_list()
    )
    folds = coefficients.select("model_date").unique().sort("model_date")
    forecasts = (
        pl.scan_parquet(config["forecasts"])
        .filter(
            (pl.col(config["asset_column"]) == security)
            & pl.col("date").is_between(start, end)
            & (pl.col("model_name") == config["model_name"])
        )
        .select("date")
    )
    inputs = (
        pl.scan_parquet(config["inputs"])
        .filter(
            (pl.col(config["asset_column"]) == security)
            & pl.col("date").is_between(start, end)
        )
        .select("date", *features)
        .join(forecasts, on="date", how="inner", validate="1:1")
        .sort("date")
        .join_asof(folds, left_on="date", right_on="model_date", strategy="backward")
        .unpivot(
            index=["date", "model_date"],
            on=features,
            variable_name="predictor",
            value_name="input_value",
        )
        .join(coefficients, on=["model_date", "predictor"], how="left", validate="m:1")
        .with_columns(
            (pl.col("input_value") * pl.col("coefficient")).alias("contribution"),
            pl.lit(security).alias("asset_id"),
            pl.lit("model").alias("side"),
        )
        .collect()
    )
    if inputs.filter(
        pl.any_horizontal(
            [
                pl.col(c).is_null() | ~pl.col(c).is_finite()
                for c in ["input_value", "coefficient", "contribution"]
            ]
        )
    ).height:
        raise ValueError("Saved inputs or matching model coefficients are missing.")
    return inputs


def load(
    directory: Path, security: str, start: dt.date, end: dt.date
) -> pl.DataFrame | None:
    """Load one stock's point-in-time contributions from an optional local manifest."""
    manifest = directory / "linear_history.json"
    if not manifest.exists():
        return None
    source = manifest.read_text()
    config = json.loads(source)
    if not isinstance(config, dict) or any(
        not isinstance(config.get(key), str) or not config[key].strip()
        for key in ["inputs", "coefficients", "forecasts", "asset_column", "model_name"]
    ):
        raise ValueError("The linear history manifest is incomplete.")
    stamps = tuple(
        data.stamp(Path(config[key])) for key in ["inputs", "coefficients", "forecasts"]
    )
    return _read(source, security, start, end, stamps)
