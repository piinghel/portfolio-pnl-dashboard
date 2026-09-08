"""Ex-post covariance attribution of a reconciled daily P&L ledger."""

from __future__ import annotations

import math
from typing import cast

import polars as pl

import attribution_dashboard.accounting.realized as realized


def contribution_panel(
    report: realized.RealizedPnlReport, *, group_by: tuple[str, ...]
) -> pl.DataFrame:
    """Group gross security contributions and retain a separate costs row.

    Parameters
    ----------
    report
        A report returned by ``build_realized_pnl_report`` for the selected period.
    group_by
        Any nonempty subset of side, sector, asset_id and label.
        Costs use side=costs, sector=Costs, label=Trading costs, asset ID=Costs.

    Returns
    -------
    pl.DataFrame
        Sparse daily contributions with additive and linked P&L. Missing
        group/dates mean zero, including dates when a position was not held.
        This is a complete partition of NET P&L; no groups are filtered out.
    """
    allowed = {"side", "sector", "asset_id", "label"}
    if not group_by or len(set(group_by)) != len(group_by) or set(group_by) - allowed:
        raise ValueError(
            "group_by must be a nonempty unique subset of security label columns"
        )
    assets = report.assets.lazy().select("date", *group_by, "asset_pnl", "linked_pnl")
    names = {
        "side": "costs",
        "sector": "Costs",
        "asset_id": "Costs",
        "label": "Trading costs",
    }
    costs = report.daily.lazy().select(
        "date",
        *[pl.lit(names[key]).alias(key) for key in group_by],
        pl.col("cost_pnl").alias("asset_pnl"),
        (pl.col("cost_pnl") * pl.col("prior_net_index")).alias("linked_pnl"),
    )
    return (
        pl.concat([assets, costs])
        .group_by("date", *group_by)
        .agg(pl.col("asset_pnl", "linked_pnl").sum())
        .sort(*group_by, "date")
        .collect()
    )


def summarize_realized_risk(
    report: realized.RealizedPnlReport,
    *,
    group_by: tuple[str, ...] = ("side", "asset_id", "label", "sector"),
    annualization: int = 252,
) -> pl.DataFrame:
    """Pair period P&L with each group's covariance contribution to realized risk.

    Risk contribution is sqrt(annualization) * Cov(c_i, net) / Std(net).
    Variance share is Cov(c_i, net) / Var(net). Covariance uses the COMPLETE
    selected trading calendar with absent positions set to zero (sample ddof=1).
    The contributions sum to portfolio annualized volatility; negative values
    describe diversification. These are historical P&L streams with changing
    positions, not a forecast for today's holdings or evidence of skill.
    For fewer than two dates or zero net variance, contributions are null.
    """
    if annualization <= 0:
        raise ValueError("annualization must be positive")
    panel = contribution_panel(report, group_by=group_by)
    net = report.daily.get_column("long_short_net")
    n = len(net)
    variance = cast(float, net.var(ddof=1)) if n > 1 else None
    valid = variance is not None and variance > 0
    scale = math.sqrt(annualization / variance) if valid else None
    return (
        panel.lazy()
        .join(
            report.daily.lazy().select("date", "long_short_net"),
            on="date",
            validate="m:1",
        )
        .group_by(*group_by)
        .agg(
            pl.col("asset_pnl").sum().alias("additive_pnl"),
            pl.col("linked_pnl").sum(),
            pl.col("date").n_unique().alias("days_present"),
            (
                (pl.col("asset_pnl") * (pl.col("long_short_net") - net.mean())).sum()
                / max(n - 1, 1)
            ).alias("covariance"),
        )
        .with_columns(
            (pl.col("covariance") * pl.lit(scale, pl.Float64)).alias("risk_vol"),
            (
                pl.col("covariance") / pl.lit(variance if valid else None, pl.Float64)
            ).alias("variance_share"),
        )
        .sort("additive_pnl", *group_by)
        .collect()
    )


def rolling_realized_risk(
    report: realized.RealizedPnlReport,
    *,
    group_by: tuple[str, ...] = ("side",),
    window: int = 63,
    annualization: int = 252,
) -> pl.DataFrame:
    """Return trailing risk contributions, including zero days for absent groups.

    Only full windows within the selected period are estimated. Risk is null
    until the window is complete and when the portfolio's variance is zero.
    Keep grouping coarse (side/sector); the complete group/calendar grid is
    intentionally constructed to make the zero-position convention explicit.
    """
    if window < 2 or annualization <= 0:
        raise ValueError("window must be at least 2 and annualization positive")
    panel = contribution_panel(report, group_by=group_by).lazy()
    net = report.daily.lazy().select("date", "long_short_net").sort("date")
    dense = (
        panel.select(*group_by)
        .unique()
        .join(net, how="cross")
        .join(panel, on=["date", *group_by], how="left", validate="1:1")
        .with_columns(pl.col("asset_pnl", "linked_pnl").fill_null(0.0))
        .sort(*group_by, "date")
    )
    return (
        dense.with_columns(
            pl.rolling_cov("asset_pnl", "long_short_net", window_size=window)
            .over(*group_by)
            .alias("covariance"),
            pl.col("long_short_net")
            .rolling_var(window, ddof=1)
            .over(*group_by)
            .alias("net_variance"),
        )
        .with_columns(
            pl.when(pl.col("net_variance") > 0)
            .then(
                pl.col("covariance")
                / pl.col("net_variance").sqrt()
                * math.sqrt(annualization)
            )
            .otherwise(None)
            .alias("risk_vol"),
            pl.when(pl.col("net_variance") > 0)
            .then(pl.col("covariance") / pl.col("net_variance"))
            .otherwise(None)
            .alias("variance_share"),
        )
        .select("date", *group_by, "risk_vol", "variance_share")
        .sort("date", *group_by)
        .collect()
    )
