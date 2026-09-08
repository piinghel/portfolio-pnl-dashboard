"""Historical covariance risk of a complete daily factor P&L partition."""

from __future__ import annotations

import datetime as dt
import math

import polars as pl


def _validate_partition(daily: pl.DataFrame) -> None:
    """Require finite contributions with unique date/factor keys."""
    if not {"date", "factor", "pnl"} <= set(daily.columns):
        raise ValueError("daily requires date, factor and pnl")
    if daily.schema["date"] != pl.Date or daily.schema["factor"] != pl.String:
        raise ValueError("date must be Date and factor must be String")
    if daily.is_empty() or daily.select("date", "factor").is_duplicated().any():
        raise ValueError("daily must be nonempty with unique date/factor keys")
    if daily.filter(
        pl.col("date").is_null()
        | pl.col("factor").is_null()
        | pl.col("pnl").is_null()
        | ~pl.col("pnl").is_finite()
    ).height:
        raise ValueError("daily has missing keys or nonfinite P&L")


def _dense_partition(daily: pl.DataFrame, annualization: int) -> pl.LazyFrame:
    """Validate the net P&L partition and make absent factor days explicit zeros."""
    if annualization <= 0 or not math.isfinite(annualization):
        raise ValueError("annualization must be finite and positive")
    _validate_partition(daily)
    panel = daily.lazy().select("date", "factor", "pnl")
    net = panel.group_by("date").agg(pl.col("pnl").sum().alias("net_pnl"))
    return (
        panel.select("factor")
        .unique()
        .join(net, how="cross")
        .join(panel, on=["date", "factor"], how="left", validate="1:1")
        .with_columns(pl.col("pnl").fill_null(0.0))
        .sort("factor", "date")
    )


def validate_factor_partition(
    daily: pl.DataFrame, net_daily: pl.DataFrame, *, tolerance: float = 1e-10
) -> None:
    """Reject a factor partition that does not explain the selected parent ledger.

    Parameters
    ----------
    daily
        Complete date/factor/pnl partition, including costs and unmodeled P&L.
        Sparse factor rows mean zero, but every parent trading date must occur.
    net_daily
        Parent ledger with unique date/long_short_net rows in decimal units.
    tolerance
        Absolute daily reconciliation tolerance in those same units.

    Raises
    ------
    ValueError
        If keys, observations, date coverage or daily totals disagree. Agreement
        checks the partition against the parent; it does not validate the model
        specification or independently verify individual factor estimates.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    _validate_partition(daily)
    if not {"date", "long_short_net"} <= set(net_daily.columns):
        raise ValueError("Parent ledger requires date and long_short_net")
    if net_daily.schema["date"] != pl.Date or net_daily.is_empty():
        raise ValueError("Parent ledger requires nonempty Date observations")
    if (
        net_daily["date"].is_duplicated().any()
        or net_daily.filter(
            pl.col("date").is_null()
            | pl.col("long_short_net").is_null()
            | ~pl.col("long_short_net").is_finite()
        ).height
    ):
        raise ValueError("Parent ledger requires unique dates and finite net P&L")
    aligned = (
        daily.lazy()
        .group_by("date")
        .agg(pl.col("pnl").sum())
        .join(
            net_daily.lazy().select("date", "long_short_net"),
            on="date",
            how="full",
            coalesce=True,
            validate="1:1",
        )
        .collect()
    )
    if aligned["pnl"].null_count() or aligned["long_short_net"].null_count():
        raise ValueError("Factor dates do not match the selected parent ledger")
    if aligned.filter(
        ~pl.col("pnl").is_finite()
        | ((pl.col("pnl") - pl.col("long_short_net")).abs() > tolerance)
    ).height:
        raise ValueError("Daily factor P&L does not reconcile to the parent ledger")


def rolling_factor_risk(
    daily: pl.DataFrame, *, window: int = 63, annualization: int = 252
) -> pl.DataFrame:
    """Return each factor's contribution to trailing realized net volatility.

    Parameters
    ----------
    daily
        Complete NET P&L partition with date/factor/pnl, in decimal return units.
        Include costs, price-basis gaps and unmodeled P&L; never prefilter factors.
        All trading dates must occur; absent factor/date rows mean zero P&L.
    window, annualization
        Trailing trading observations and observations per year. Only full
        windows are estimated. Zero net variance produces null contributions.

    Returns
    -------
    pl.DataFrame
        Date, factor and risk_vol = sqrt(annualization)*Cov(factor,net)/Std(net).
        Signed contributions sum to annualized net volatility; negatives reflect
        diversification. This describes historical P&L, not forecast risk or skill.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    panel = _dense_partition(daily, annualization)
    return (
        panel.with_columns(
            pl.rolling_cov("pnl", "net_pnl", window_size=window)
            .over("factor")
            .alias("covariance"),
            pl.col("net_pnl")
            .rolling_var(window, ddof=1)
            .over("factor")
            .alias("variance"),
        )
        .with_columns(
            pl.when(pl.col("variance") > 0)
            .then(
                pl.col("covariance")
                / pl.col("variance").sqrt()
                * math.sqrt(annualization)
            )
            .otherwise(None)
            .alias("risk_vol")
        )
        .with_columns(
            pl.when(pl.col("variance") > 0)
            .then(pl.col("covariance") / pl.col("variance"))
            .otherwise(None)
            .alias("variance_share")
        )
        .select("date", "factor", "risk_vol", "variance_share")
        .sort("date", "factor")
        .collect()
    )


def summarize_factor_risk(
    daily: pl.DataFrame, *, annualization: int = 252
) -> pl.DataFrame:
    """Pair selected-period additive P&L and realized covariance risk by factor.

    The same complete-net-partition and missing-factor-zero convention applies
    as in ``rolling_factor_risk``. Sample covariance uses ddof=1. Fewer than two
    dates or zero net variance return null risk_vol and variance_share.
    """
    panel = _dense_partition(daily, annualization)
    return (
        panel.group_by("factor")
        .agg(
            pl.col("pnl").sum().alias("additive_pnl"),
            pl.cov("pnl", "net_pnl", ddof=1).alias("covariance"),
            pl.col("net_pnl").var(ddof=1).alias("variance"),
        )
        .with_columns(
            pl.when(pl.col("variance") > 0)
            .then(
                pl.col("covariance")
                / pl.col("variance").sqrt()
                * math.sqrt(annualization)
            )
            .otherwise(None)
            .alias("risk_vol"),
            pl.when(pl.col("variance") > 0)
            .then(pl.col("covariance") / pl.col("variance"))
            .otherwise(None)
            .alias("variance_share"),
        )
        .select("factor", "additive_pnl", "risk_vol", "variance_share")
        .sort("additive_pnl", "factor")
        .collect()
    )


def period_factor_pnl(
    daily: pl.DataFrame,
    *,
    start: dt.date,
    end: dt.date,
    interval: str,
) -> pl.DataFrame:
    """Sum selected contributions in calendar buckets, retaining partial bounds.

    Weekly buckets start on Monday. Absent factor rows are zero contributions;
    the first and last observed session delimit each selected bucket.
    """
    if interval not in {"1w", "1mo", "1q", "1y"} or start > end:
        raise ValueError(
            "Require ordered dates and a weekly, monthly, quarterly or yearly interval"
        )
    return (
        _dense_partition(daily, 252)
        .filter(pl.col("date").is_between(start, end))
        .with_columns(pl.col("date").dt.truncate(interval).alias("period"))
        .group_by("period", "factor")
        .agg(
            pl.col("date").min().alias("start"),
            pl.col("date").max().alias("date"),
            pl.len().alias("sessions"),
            pl.col("pnl").sum(),
        )
        .sort("date", "factor")
        .collect()
    )


def cumulative_factor_pnl(
    daily: pl.DataFrame, *, start: dt.date, end: dt.date
) -> pl.DataFrame:
    """Accumulate selected daily contributions on a complete factor/date grid.

    Supply full preceding history to retain factors inactive in the selection.
    Accumulation resets at start and includes its first daily P&L. Absent factor
    rows are zero contributions; invalid/null P&L is rejected.
    """
    if start > end:
        raise ValueError("Require ordered dates")
    return (
        _dense_partition(daily, 252)
        .filter(pl.col("date").is_between(start, end))
        .with_columns(pl.col("pnl").cum_sum().over("factor").alias("pnl"))
        .select("date", "factor", "pnl")
        .collect()
    )


def compare_factor_performance(
    daily: pl.DataFrame, *, start: dt.date, end: dt.date, annualization: int = 252
) -> pl.DataFrame:
    """Compare selected and preceding equal-calendar-span component performance.

    Return/risk is sqrt(A)*mean/std with a zero hurdle, not benchmark-relative
    IR. Component ratios are standalone and do not add. Contributions use the
    common net volatility denominator and sum to net return/risk. Residual IR
    is conditional on the supplied model; uncovered P&L must remain separate.
    The preceding window is omitted unless the supplied history covers it.
    """
    if start > end:
        raise ValueError("Require ordered dates")
    panel = _dense_partition(daily, annualization).collect()
    previous = start - (end - start + dt.timedelta(days=1))
    windows = [("Selected", start, end)]
    first_date = panel["date"].min()
    if isinstance(first_date, dt.date) and first_date <= previous:
        windows.append(("Preceding", previous, start - dt.timedelta(days=1)))
    results = []
    for label, first, last in windows:
        selected = panel.filter(pl.col("date").is_between(first, last))
        if selected.is_empty():
            continue
        net = selected.select("date", pl.col("net_pnl").alias("pnl")).unique()
        net = net.with_columns(
            pl.lit("Net").alias("factor"), pl.col("pnl").alias("net_pnl")
        )
        summary = (
            pl.concat([selected, net.select(selected.columns)])
            .sort("factor", "date")
            .group_by("factor")
            .agg(
                pl.col("date").min().alias("start"),
                pl.col("date").max().alias("end"),
                pl.len().alias("sessions"),
                pl.col("pnl").mean().alias("mean"),
                pl.col("pnl").std(ddof=1).alias("vol"),
                pl.col("net_pnl").std(ddof=1).alias("net_vol"),
                pl.cov("pnl", "net_pnl", ddof=1).alias("covariance"),
            )
            .with_columns(
                pl.lit(label).alias("period"),
                (pl.col("mean") * annualization).alias("annual_pnl"),
                (pl.col("vol") * math.sqrt(annualization)).alias("annual_vol"),
                pl.when(pl.col("net_vol") > 0)
                .then(pl.col("covariance") / pl.col("net_vol").pow(2))
                .alias("variance_share"),
                pl.when(pl.col("vol") > 0)
                .then(pl.col("mean") / pl.col("vol") * math.sqrt(annualization))
                .alias("return_risk"),
                pl.when(pl.col("net_vol") > 0)
                .then(pl.col("mean") / pl.col("net_vol") * math.sqrt(annualization))
                .alias("ratio_contribution"),
            )
        )
        results.append(
            summary.select(
                "period",
                "factor",
                "start",
                "end",
                "sessions",
                "annual_pnl",
                "annual_vol",
                "variance_share",
                "return_risk",
                "ratio_contribution",
            )
        )
    if not results:
        raise ValueError("No observations in selected period")
    return pl.concat(results).sort("period", "factor")
