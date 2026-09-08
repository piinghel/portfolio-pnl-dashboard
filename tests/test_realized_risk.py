"""Portfolio risk must reconcile, including hedges and zero-position days."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

import attribution_dashboard.accounting.realized as realized
import attribution_dashboard.accounting.realized_risk as risk


def sample() -> realized.RealizedPnlReport:
    dates = [dt.date(2024, 1, i) for i in range(1, 6)]
    a = np.array([0.03, -0.01, 0.02, 0, -0.02])
    b = np.array([-0.01, 0.004, -0.008, 0, 0.008])
    cost = np.array([-0.001, 0, -0.002, 0, -0.001])
    rows = [
        (d, name, side, pnl)
        for name, side, values in [("A", "long", a), ("B", "short", b)]
        for d, pnl in zip(dates, values, strict=True)
        if pnl != 0
    ]
    assets = pl.DataFrame(
        rows, schema=["date", "asset_id", "side", "asset_pnl"], orient="row"
    )
    returns = pl.DataFrame(
        {
            "date": dates,
            "long_gross": a,
            "short_gross": -b,
            "long_short_gross": a + b,
            "long_net": a + cost,
            "short_net": -b,
            "long_short_net": a + b + cost,
        }
    )
    return realized.build_realized_pnl_report(assets, returns)


def test_covariance_includes_flat_days_and_negative_hedges() -> None:
    report = sample()
    result = risk.summarize_realized_risk(report, group_by=("side",))
    expected = np.std(report.daily["long_short_net"].to_numpy(), ddof=1) * np.sqrt(252)
    assert result["risk_vol"].sum() == pytest.approx(expected)
    assert result["variance_share"].sum() == pytest.approx(1)
    hedge = result.filter(pl.col("side") == "short")["risk_vol"].item()
    assert hedge < 0
    b = np.array([-0.01, 0.004, -0.008, 0, 0.008])
    net = report.daily["long_short_net"].to_numpy()
    assert hedge == pytest.approx(
        np.cov(b, net)[0, 1] / np.std(net, ddof=1) * np.sqrt(252)
    )
    assert result["additive_pnl"].sum() == pytest.approx(net.sum())
    assert result["linked_pnl"].sum() == pytest.approx(np.prod(1 + net) - 1)
    rolled = risk.rolling_realized_risk(report, window=3)
    for i, date in enumerate(report.daily["date"]):
        values = rolled.filter(pl.col("date") == date)["risk_vol"]
        if i < 2:
            assert values.null_count() == len(values)
        else:
            assert values.sum() == pytest.approx(
                np.std(net[i - 2 : i + 1], ddof=1) * np.sqrt(252)
            )
    # A selected subperiod must reset linking and re-estimate covariance.
    subset = realized.build_realized_pnl_report(
        report.assets.drop("prior_net_index", "linked_pnl").filter(
            pl.col("date") >= dt.date(2024, 1, 3)
        ),
        report.daily.select(
            "date",
            "long_gross",
            "short_gross",
            "long_short_gross",
            "long_net",
            "short_net",
            "long_short_net",
        ).tail(3),
    )
    window = risk.summarize_realized_risk(subset)
    assert window["linked_pnl"].sum() == pytest.approx(np.prod(1 + net[2:]) - 1)
    assert window["risk_vol"].sum() == pytest.approx(
        np.std(net[2:], ddof=1) * np.sqrt(252)
    )


def test_one_day_and_constant_net_risk_are_undefined() -> None:
    report = sample()
    for n in [1, 5]:
        returns = (
            report.daily.head(n)
            .select("date")
            .with_columns(
                *[
                    pl.lit(0.0).alias(c)
                    for c in (
                        "long_gross",
                        "short_gross",
                        "long_short_gross",
                        "long_net",
                        "short_net",
                        "long_short_net",
                    )
                ]
            )
        )
        flat = realized.build_realized_pnl_report(
            report.assets.head(0).drop("prior_net_index", "linked_pnl"), returns
        )
        result = risk.summarize_realized_risk(flat)
        assert result["risk_vol"].null_count() == result.height
        assert result["variance_share"].null_count() == result.height
