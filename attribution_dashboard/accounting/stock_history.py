"""Recorded holding lifecycles, independent of plots and execution-fill assumptions."""

from __future__ import annotations

import polars as pl


def position_events(holdings: pl.DataFrame, calendar: pl.DataFrame) -> pl.DataFrame:
    """Identify aggregate side entries/exits from complete saved holding history.

    Quantities are positive magnitudes on both sides. All schedules and tranches
    aggregate by security/side before detecting runs. Resizing an active position
    is not a new entry. A run starting at the source boundary is left-censored
    (``Already held``); its unknown entry is not fabricated. Exit is the first
    saved session with no quantity, not an execution-fill date. An active run at
    the final source session has no exit. Filter events only AFTER this function.

    ``calendar`` must contain every recorded holdings session, including those
    on which the selected stock was absent. Dates are unique calendar dates.
    """
    keys = ["asset_id", "side"]
    if (
        calendar.is_empty()
        or calendar["date"].is_duplicated().any()
        or calendar["date"].null_count()
    ):
        raise ValueError("calendar must have nonempty unique nonnull dates")
    if holdings.filter(
        pl.col("holding_qty").is_null()
        | ~pl.col("holding_qty").is_finite()
        | (pl.col("holding_qty") < 0)
    ).height:
        raise ValueError("holding_qty must be finite nonnegative side magnitudes")
    if (
        holdings.select("date")
        .join(calendar.select("date"), on="date", how="anti")
        .height
    ):
        raise ValueError("holdings dates must belong to the full calendar")
    sessions = calendar.lazy().select("date").sort("date").with_row_index("session")
    active = (
        holdings.lazy()
        .group_by("date", *keys)
        .agg(pl.col("holding_qty").sum())
        .filter(pl.col("holding_qty") > 0)
        .join(sessions, on="date", validate="m:1")
        .sort(*keys, "date")
    )
    runs = (
        active.with_columns(
            (
                pl.col("session").cast(pl.Int64).diff().over(keys).fill_null(2) != 1
            ).alias("new_run")
        )
        .with_columns(pl.col("new_run").cum_sum().over(keys).alias("run"))
        .group_by(*keys, "run")
        .agg(
            pl.col("date").min().alias("first_held"),
            pl.col("date").max().alias("last_held"),
            pl.col("session").min().alias("first_session"),
            pl.col("session").max().alias("last_session"),
        )
    )
    starts = runs.select(
        *keys,
        pl.col("first_held").alias("date"),
        "last_held",
        pl.when(pl.col("first_session") == 0)
        .then(pl.lit("Already held"))
        .otherwise(pl.lit("Entry"))
        .alias("event"),
    )
    ends = (
        runs.with_columns((pl.col("last_session") + 1).alias("session"))
        .join(sessions, on="session", how="inner", validate="m:1")
        .select(*keys, "date", "last_held", pl.lit("Exit").alias("event"))
    )
    return pl.concat([starts, ends]).sort("date", *keys, "event").collect()


def summarize_stocks(assets: pl.DataFrame, calendar: pl.DataFrame) -> pl.DataFrame:
    """Aggregate gross rewards and marked weights over the full selected calendar.

    Stock P&L combines both sides and excludes separately recorded portfolio
    costs. Average exposures include flat sessions; end exposures refer to the
    last selected saved session, not the last day on which the stock was held.
    Missing weight columns remain null, distinct from a known flat position.
    """
    if calendar.is_empty() or calendar["date"].is_duplicated().any():
        raise ValueError("calendar must have nonempty unique dates")
    last = calendar["date"].max()
    days = calendar.height
    frame = assets.lazy()
    if "gross_weight" not in assets.columns:
        frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("gross_weight"))
    weights = "gross_weight" in assets.columns
    return (
        frame.group_by("asset_id")
        .agg(
            pl.col("label").first(),
            pl.col("sector").first(),
            pl.col("asset_pnl").sum().alias("pnl"),
            *[
                (
                    pl.col("gross_weight").filter(pl.col("side") == side).sum()
                    / days
                    * sign
                    if weights
                    else pl.lit(None, dtype=pl.Float64)
                ).alias(f"average_{side}")
                for side, sign in [("long", 1), ("short", -1)]
            ],
            *[
                (
                    pl.col("gross_weight")
                    .filter((pl.col("side") == side) & (pl.col("date") == last))
                    .sum()
                    * sign
                    if weights
                    else pl.lit(None, dtype=pl.Float64)
                ).alias(f"ending_{side}")
                for side, sign in [("long", 1), ("short", -1)]
            ],
        )
        .sort("pnl", "asset_id")
        .collect()
    )
