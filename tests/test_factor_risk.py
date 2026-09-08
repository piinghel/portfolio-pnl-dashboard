"""Complete-calendar factor covariance accounting, including sparse contributors."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import polars as pl
import pytest

import attribution_dashboard.accounting.factor_risk as factor_risk


def test_sparse_factors_reconcile_to_net_rolling_risk_with_negative_hedge() -> None:
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(8)]
    values = [0.01, -0.02, 0.03, 0.015, -0.025, 0.012, 0.02, -0.01]
    rows = [
        (date, factor, value)
        for date, x in zip(dates, values, strict=True)
        for factor, value in [("style", x), ("hedge", -0.25 * x), ("costs", -0.001)]
    ]
    rows += [(dates[2], "unmodeled", 0.002), (dates[5], "unmodeled", 0.001)]
    daily = pl.DataFrame(rows, schema=["date", "factor", "pnl"], orient="row")
    result = factor_risk.rolling_factor_risk(daily, window=3)
    assert result.filter(pl.col("date") < dates[2])["risk_vol"].null_count() == 8
    net = [
        0.75 * x - 0.001 + (0.002 if i == 2 else 0.001 if i == 5 else 0.0)
        for i, x in enumerate(values)
    ]
    for i in range(2, 8):
        day = result.filter(pl.col("date") == dates[i])
        assert day["risk_vol"].null_count() == 0
        assert day["risk_vol"].sum() == pytest.approx(
            np.std(net[i - 2 : i + 1], ddof=1) * math.sqrt(252)
        )
        assert day.filter(pl.col("factor") == "hedge")["risk_vol"].item() < 0
        sparse = [
            (0.002 if j == 2 else 0.001 if j == 5 else 0.0) for j in range(i - 2, i + 1)
        ]
        expected = (
            np.cov(sparse, net[i - 2 : i + 1], ddof=1)[0, 1]
            / np.std(net[i - 2 : i + 1], ddof=1)
            * math.sqrt(252)
        )
        assert day.filter(pl.col("factor") == "unmodeled")[
            "risk_vol"
        ].item() == pytest.approx(expected, abs=1e-12)
    summary = factor_risk.summarize_factor_risk(daily)
    assert summary["additive_pnl"].sum() == pytest.approx(sum(net))
    assert summary["risk_vol"].sum() == pytest.approx(
        np.std(net, ddof=1) * math.sqrt(252)
    )
    assert summary["variance_share"].sum() == pytest.approx(1.0)


def test_constant_net_or_single_day_risk_is_undefined() -> None:
    daily = pl.DataFrame(
        {
            "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
            "factor": ["style", "style"],
            "pnl": [0.01, 0.01],
        }
    )
    assert (
        factor_risk.rolling_factor_risk(daily, window=2)["risk_vol"].null_count() == 2
    )
    for frame in (daily, daily.head(1)):
        summary = factor_risk.summarize_factor_risk(frame)
        assert summary["risk_vol"].null_count() == 1
        assert summary["variance_share"].null_count() == 1
    with pytest.raises(ValueError, match="unique"):
        factor_risk.rolling_factor_risk(pl.concat([daily, daily.head(1)]), window=2)
    with pytest.raises(ValueError, match="nonfinite"):
        factor_risk.summarize_factor_risk(
            daily.with_columns(pl.lit(float("nan")).alias("pnl"))
        )


def test_factor_partition_must_match_parent_dates_and_daily_net() -> None:
    dates = [dt.date(2024, 1, 2), dt.date(2024, 1, 3)]
    daily = pl.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1]],
            "factor": ["style", "costs", "style"],
            "pnl": [0.01, -0.001, -0.005],
        }
    )
    parent = pl.DataFrame({"date": dates, "long_short_net": [0.009, -0.005]})
    factor_risk.validate_factor_partition(daily, parent)
    with pytest.raises(ValueError, match="does not reconcile"):
        factor_risk.validate_factor_partition(
            daily, parent.with_columns(pl.col("long_short_net") + 0.001)
        )
    for incomplete, expected in [(daily.head(2), parent), (daily, parent.head(1))]:
        with pytest.raises(ValueError, match="dates do not match"):
            factor_risk.validate_factor_partition(incomplete, expected)
    with pytest.raises(ValueError, match="unique"):
        factor_risk.validate_factor_partition(pl.concat([daily, daily.head(1)]), parent)


def test_period_and_cumulative_factor_pnl_keep_selected_cash():
    dates = [dt.date(2024, 1, 29) + dt.timedelta(days=i) for i in range(8)]
    daily = pl.DataFrame(
        [
            (d, n, v * k)
            for i, d in enumerate(dates)
            for n, k in [("style", 1.0), ("hedge", -0.25)]
            for v in [float(i % 3 - 1)]
        ],
        schema=["date", "factor", "pnl"],
        orient="row",
    )
    result = factor_risk.period_factor_pnl(
        daily, start=dates[2], end=dates[-2], interval="1mo"
    )
    expected = daily.filter(pl.col("date").is_between(dates[2], dates[-2]))["pnl"].sum()
    assert result["pnl"].sum() == pytest.approx(expected)
    assert result["date"].unique().sort().to_list() == [dates[2], dates[-2]]
    assert result["sessions"].to_list() == [1, 1, 4, 4]
    future = daily.with_columns(
        pl.when(pl.col("date") > dates[-2])
        .then(1000.0)
        .otherwise(pl.col("pnl"))
        .alias("pnl")
    )
    assert factor_risk.period_factor_pnl(
        future, start=dates[2], end=dates[-2], interval="1mo"
    ).equals(result)
    for interval in ["1w", "1q", "1y"]:
        coarse = factor_risk.period_factor_pnl(
            daily, start=dates[2], end=dates[-2], interval=interval
        )
        assert coarse["pnl"].sum() == pytest.approx(expected)
    selected = daily.filter(pl.col("date").is_between(dates[2], dates[-2]))
    cumulative = factor_risk.cumulative_factor_pnl(daily, start=dates[2], end=dates[-2])
    assert cumulative.group_by("factor").agg(pl.col("pnl").last())[
        "pnl"
    ].sum() == pytest.approx(expected)
    assert cumulative.filter(pl.col("date") == dates[2])["pnl"].sum() == pytest.approx(
        selected.filter(pl.col("date") == dates[2])["pnl"].sum()
    )


def test_inactive_selected_factor_keeps_its_cumulative_zero_and_trailing_risk():
    dates = [dt.date(2024, 1, day) for day in (2, 3, 4)]
    daily = pl.DataFrame(
        {
            "date": [*dates, dates[0], dates[1]],
            "factor": ["costs"] * 3 + ["style"] * 2,
            "pnl": [-0.001, -0.002, -0.001, 0.01, -0.02],
        }
    )
    cumulative = factor_risk.cumulative_factor_pnl(
        daily, start=dates[-1], end=dates[-1]
    )
    assert cumulative.filter(pl.col("factor") == "style")["pnl"].to_list() == [0.0]
    periods = factor_risk.period_factor_pnl(
        daily, start=dates[-1], end=dates[-1], interval="1w"
    )
    assert periods.filter(pl.col("factor") == "style")["pnl"].to_list() == [0.0]
    trailing = factor_risk.rolling_factor_risk(daily, window=3).filter(
        pl.col("date") == dates[-1]
    )
    assert trailing.filter(pl.col("factor") == "style")["variance_share"].item() > 0.9
    assert set(cumulative["factor"]) == set(trailing["factor"])


def test_period_comparison_keeps_ratio_and_variance_accounting() -> None:
    dates = [dt.date(2024, 1, day) for day in range(1, 6)]
    daily = pl.DataFrame(
        {
            "date": dates * 2,
            "factor": ["Factor"] * 5 + ["Residual"] * 5,
            "pnl": [0.01, 0.03, 0.02, 0.06, 100.0, 0.02, -0.01, -0.01, -0.02, -100.0],
        }
    )
    summary = factor_risk.compare_factor_performance(
        daily, start=dates[2], end=dates[3]
    )
    for period in ["Selected", "Preceding"]:
        rows = summary.filter(pl.col("period") == period)
        parts, net = (
            rows.filter(pl.col("factor") != "Net"),
            rows.filter(pl.col("factor") == "Net").row(0, named=True),
        )
        assert parts["annual_pnl"].sum() == pytest.approx(net["annual_pnl"])
        assert parts["variance_share"].sum() == pytest.approx(1.0)
        assert parts["ratio_contribution"].sum() == pytest.approx(net["return_risk"])
    chosen = summary.filter(
        (pl.col("period") == "Selected") & (pl.col("factor") == "Factor")
    ).row(0, named=True)
    assert chosen["annual_pnl"] == pytest.approx(0.04 * 252)
    assert chosen["return_risk"] == pytest.approx(0.04 / (0.04 / 2**0.5) * 252**0.5)
    assert summary.equals(
        factor_risk.compare_factor_performance(
            daily.filter(pl.col("date") <= dates[3]), start=dates[2], end=dates[3]
        )
    )
    flat = daily.with_columns(pl.lit(0.0).alias("pnl"))
    assert (
        factor_risk.compare_factor_performance(flat, start=dates[0], end=dates[1])[
            "return_risk"
        ].null_count()
        == 3
    )
