"""Reconcile security P&L to a saved book and link an episode's contributions.

This accounting layer does not require a factor model. Inputs are signed gross
security contributions per unit of strategy notional; costs remain separate.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import cast

import polars as pl


@dataclass
class RealizedPnlReport:
    """Daily audit, linked security ledger, and additive/linked summary tables."""

    daily: pl.DataFrame
    assets: pl.DataFrame
    components: pl.DataFrame
    stocks: pl.DataFrame
    sectors: pl.DataFrame


def build_realized_pnl_report(
    asset_pnl: pl.DataFrame,
    returns: pl.DataFrame,
    *,
    tolerance: float = 1e-10,
) -> RealizedPnlReport:
    """Reconcile an explicit episode and link all contributions to its net index.

    Parameters
    ----------
    asset_pnl
        Unique date/asset_id/side rows with signed ``asset_pnl``.
        Side is long or short; a short loss is negative. Optional ``sector``
        and ``label`` describe the classification available on that date.
        No model-support filtering is allowed upstream of this accounting.
    returns
        Complete daily calendar for the episode, with engine columns
        long_gross, short_gross, long_short_gross, long_net, short_net,
        long_short_net. Engine short columns are unsigned-leg returns;
        this function negates them. Dates must be unique and non-null.
    tolerance
        Maximum absolute daily reconciliation error, in return units.

    Returns
    -------
    RealizedPnlReport
        Contributions in decimal return units. ``linked_pnl`` uses the same
        prior net index level for every security and cost on each day:
        C_k = sum_t V_(t-1) c_(k,t), V_0 = 1. Linked components sum to the
        episode's compounded net return; additive components sum daily P&L.
        This is index linking, not a claim that fixed notional was reinvested.
        Stock rows remain split by side and sector; classification changes
        can therefore produce multiple rows for the same security.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    return_cols = [
        "long_gross",
        "short_gross",
        "long_short_gross",
        "long_net",
        "short_net",
        "long_short_net",
    ]
    _validate_frame(returns, keys=["date"], values=return_cols, name="returns")
    _validate_frame(
        asset_pnl,
        keys=["date", "asset_id", "side"],
        values=["asset_pnl"],
        name="asset_pnl",
    )
    if returns.is_empty():
        raise ValueError("returns must contain the complete episode calendar")
    if returns.schema["date"] != pl.Date or asset_pnl.schema["date"] != pl.Date:
        raise ValueError("date must have Date dtype in both inputs")
    if cast(float, returns.get_column("long_short_net").min()) <= -1:
        raise ValueError("Net returns <= -100% cannot be linked as a positive index")
    if not set(asset_pnl.get_column("side").unique()).issubset({"long", "short"}):
        raise ValueError("side must contain only long or short")
    if (
        asset_pnl.lazy()
        .join(returns.lazy().select("date"), on="date", how="anti")
        .collect()
        .height
    ):
        raise ValueError("asset_pnl contains dates outside the episode calendar")
    labeled = asset_pnl.lazy()
    if "sector" not in asset_pnl.columns:
        labeled = labeled.with_columns(pl.lit(None, pl.String).alias("sector"))
    if "label" not in asset_pnl.columns:
        labeled = labeled.with_columns(pl.lit(None, pl.String).alias("label"))
    labeled = labeled.with_columns(
        pl.col("sector").fill_null("Unclassified").replace("", "Unclassified"),
        pl.col("label").fill_null(pl.col("asset_id")),
    )
    book = labeled.group_by("date").agg(
        pl.col("asset_pnl").filter(pl.col("side") == "long").sum().alias("long_pnl"),
        pl.col("asset_pnl").filter(pl.col("side") == "short").sum().alias("short_pnl"),
    )
    # Sort before any cumulative operation; caller row order has no meaning.
    daily = (
        returns.lazy()
        .sort("date")
        .join(book, on="date", how="left", validate="1:1")
        .with_columns(pl.col("long_pnl", "short_pnl").fill_null(0.0))
        .with_columns(
            (pl.col("long_short_net") - pl.col("long_short_gross")).alias("cost_pnl"),
            (pl.col("long_pnl") - pl.col("long_gross")).alias("long_error"),
            (pl.col("short_pnl") + pl.col("short_gross")).alias("short_error"),
            (
                pl.col("long_gross")
                - pl.col("short_gross")
                - pl.col("long_short_gross")
            ).alias("gross_identity_error"),
            (pl.col("long_net") - pl.col("short_net") - pl.col("long_short_net")).alias(
                "net_identity_error"
            ),
        )
        .sort("date")
        .with_columns(
            (
                pl.col("long_pnl")
                + pl.col("short_pnl")
                + pl.col("cost_pnl")
                - pl.col("long_short_net")
            ).alias("reconciliation_error"),
            (1 + pl.col("long_short_net"))
            .cum_prod()
            .shift(1)
            .fill_null(1.0)
            .alias("prior_net_index"),
        )
        .collect()
    )
    errors = [c for c in daily.columns if c.endswith("error")]
    max_error = (
        daily.lazy()
        .select(pl.max_horizontal(pl.col(errors).abs()).max())
        .collect()
        .item()
    )
    if max_error > tolerance:
        raise ValueError(
            f"Security P&L does not reconcile to saved returns: {max_error:.12g} > {tolerance:.12g}"
        )
    if cast(float, daily.get_column("cost_pnl").max()) > tolerance:
        raise ValueError(
            "Net exceeds gross: expected a nonpositive trading-cost contribution"
        )
    assets = (
        labeled.join(
            daily.lazy().select("date", "prior_net_index"),
            on="date",
            how="left",
            validate="m:1",
        )
        .with_columns(
            (pl.col("asset_pnl") * pl.col("prior_net_index")).alias("linked_pnl")
        )
        .sort("date", "side", "asset_id")
        .collect()
    )
    components = (
        daily.lazy()
        .select("date", "prior_net_index", "long_pnl", "short_pnl", "cost_pnl")
        .unpivot(
            index=["date", "prior_net_index"],
            variable_name="component",
            value_name="asset_pnl",
        )
        .with_columns(
            (pl.col("asset_pnl") * pl.col("prior_net_index")).alias("linked_pnl")
        )
    )
    return RealizedPnlReport(
        daily=daily,
        assets=assets,
        components=_summarize(components, ["component"]),
        stocks=_summarize(assets.lazy(), ["side", "asset_id", "label", "sector"]),
        sectors=_summarize(assets.lazy(), ["side", "sector"]),
    )


def fixed_notional_path(daily: pl.DataFrame) -> pl.DataFrame:
    """Calculate cumulative net P&L and loss from its previous high-water mark.

    Parameters
    ----------
    daily
        Complete dated history with ``long_short_net`` in decimal fixed-notional
        units. Pass history before a selected view to preserve earlier peaks.

    Returns
    -------
    pl.DataFrame
        Date, cumulative_pnl, peak_pnl and drawdown in the same decimal units.
        Initial cumulative P&L is zero. Drawdown is an additive loss divided by
        fixed notional, not a percentage loss of a compounded or funded NAV.
        Filter dates AFTER this calculation to retain an ongoing drawdown.
    """
    _validate_frame(daily, keys=["date"], values=["long_short_net"], name="daily")
    if daily.schema["date"] != pl.Date:
        raise ValueError("daily date must have Date dtype")
    return (
        daily.lazy()
        .sort("date")
        .select("date", pl.col("long_short_net").cum_sum().alias("cumulative_pnl"))
        .with_columns(
            pl.col("cumulative_pnl").cum_max().clip(lower_bound=0).alias("peak_pnl")
        )
        .with_columns((pl.col("cumulative_pnl") - pl.col("peak_pnl")).alias("drawdown"))
        .collect()
    )


def _summarize(frame: pl.LazyFrame, keys: list[str]) -> pl.DataFrame:
    return (
        frame.group_by(keys)
        .agg(
            pl.col("asset_pnl").sum().alias("additive_pnl"),
            pl.col("linked_pnl").sum(),
            pl.col("date").n_unique().alias("n_days"),
        )
        .sort("linked_pnl", *keys)
        .collect()
    )


def _validate_frame(
    frame: pl.DataFrame, *, keys: list[str], values: list[str], name: str
) -> None:
    missing = set(keys + values) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")
    if frame.select(keys).is_duplicated().any():
        raise ValueError(f"{name} has duplicate keys: {keys}")
    if any(frame.get_column(k).null_count() for k in keys):
        raise ValueError(f"{name} has null keys")
    if any(not frame.schema[v].is_numeric() for v in values):
        raise ValueError(f"{name} values must be numeric")
    if (
        frame.lazy()
        .filter(
            pl.any_horizontal(pl.col(values).is_null() | ~pl.col(values).is_finite())
        )
        .collect()
        .height
    ):
        raise ValueError(f"{name} has missing or nonfinite values")


def worst_drawdown_period(daily: pl.DataFrame) -> tuple[dt.date, dt.date]:
    """Return loss sessions after the last peak through the worst net trough.

    The peak session is excluded: its return belongs to the preceding ascent.
    If the worst loss starts from the initial zero baseline, include the first
    session. A book with no drawdown returns its first session as a neutral view.
    """
    path = fixed_notional_path(daily)
    if path.is_empty():
        raise ValueError("daily must contain at least one session")
    trough = path.sort("drawdown", "date").row(0, named=True)
    if trough["drawdown"] == 0:
        return path["date"][0], path["date"][0]
    peaks = path.filter((pl.col("date") < trough["date"]) & (pl.col("drawdown") == 0))
    start = (
        path["date"][0]
        if peaks.is_empty()
        else path.filter(pl.col("date") > peaks["date"][-1])["date"][0]
    )
    return start, trough["date"]


def benchmark_path(daily: pl.DataFrame) -> pl.DataFrame:
    """Compound a selected benchmark return stream relative to its opening notional.

    Parameters
    ----------
    daily
        Unique Date rows with finite decimal ``benchmark`` returns, including
        the selected first session's return. Returns below -100% are invalid.
        Exactly -100% is valid and leaves wealth at zero thereafter. Missing
        dates or returns are never fabricated; the caller supplies the intended
        complete selection and adds a zero display anchor separately.

    Returns
    -------
    pl.DataFrame
        Sorted ``date`` and ``value = cumprod(1 + benchmark) - 1`` in decimal
        units of opening notional. This is a buy-and-hold reference path, not an
        additive strategy P&L component. Dividend and cost treatment follow the
        supplied benchmark series; compounding does not add either.
    """
    _validate_frame(daily, keys=["date"], values=["benchmark"], name="benchmark")
    if daily.is_empty() or daily.schema["date"] != pl.Date:
        raise ValueError("benchmark requires nonempty Date observations")
    if daily.filter(pl.col("benchmark") < -1).height:
        raise ValueError("benchmark returns must be at least -100%")
    return (
        daily.lazy()
        .sort("date")
        .select(
            "date",
            ((1 + pl.col("benchmark").cast(pl.Float64)).cum_prod() - 1).alias("value"),
        )
        .collect()
    )
