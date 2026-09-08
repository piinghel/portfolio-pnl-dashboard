"""Accounting invariants independent of factor-model estimation."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

import attribution_dashboard.accounting.realized as realized


def inputs() -> tuple[pl.DataFrame, pl.DataFrame]:
    dates = [dt.date(2023, 1, 2), dt.date(2023, 1, 3)]
    assets = pl.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "asset_id": ["A", "B", "A", "B"],
            "side": ["long", "short", "long", "short"],
            "asset_pnl": [0.1, -0.2, -0.05, 0.1],
            "sector": ["Tech", None, "Tech", None],
        }
    )
    returns = pl.DataFrame(
        {
            "date": dates,
            "long_gross": [0.1, -0.05],
            "short_gross": [0.2, -0.1],
            "long_short_gross": [-0.1, 0.05],
            "long_net": [0.09, -0.055],
            "short_net": [0.2, -0.095],
            "long_short_net": [-0.11, 0.04],
        }
    )
    return assets, returns


def test_signed_legs_costs_and_linked_drawdown_reconcile() -> None:
    assets, returns = inputs()
    report = realized.build_realized_pnl_report(assets.reverse(), returns.reverse())
    expected = {
        "long_pnl": 0.1 - 0.89 * 0.05,
        "short_pnl": -0.2 + 0.89 * 0.1,
        "cost_pnl": -0.01 - 0.89 * 0.01,
    }
    for row in report.components.iter_rows(named=True):
        assert row["linked_pnl"] == pytest.approx(expected[row["component"]])
    assert report.components["linked_pnl"].sum() == pytest.approx(0.89 * 1.04 - 1)
    assert report.components["additive_pnl"].sum() == pytest.approx(-0.07)
    assert report.stocks["linked_pnl"].sum() == pytest.approx(
        report.sectors["linked_pnl"].sum()
    )
    assert "Unclassified" in report.sectors["sector"].to_list()


@pytest.mark.parametrize(
    "problem",
    [
        "missing_stock",
        "duplicate_stock",
        "duplicate_date",
        "nonfinite",
        "bad_side",
        "net_identity",
        "extra_date",
    ],
)
def test_corrupt_inputs_cannot_make_a_reconciled_report(problem: str) -> None:
    assets, returns = inputs()
    if problem == "missing_stock":
        assets = assets.slice(1)
    elif problem == "duplicate_stock":
        assets = pl.concat([assets, assets.head(1)])
    elif problem == "duplicate_date":
        returns = pl.concat([returns, returns.head(1)])
    elif problem == "nonfinite":
        assets = assets.with_columns(pl.lit(float("nan")).alias("asset_pnl"))
    elif problem == "bad_side":
        assets = assets.with_columns(pl.lit("other").alias("side"))
    elif problem == "net_identity":
        returns = returns.with_columns((pl.col("long_net") + 0.1).alias("long_net"))
    else:
        assets = assets.with_columns(pl.col("date") + dt.timedelta(days=10))
    with pytest.raises(ValueError):
        realized.build_realized_pnl_report(assets, returns)


def test_flat_calendar_day_preserves_linking_and_cost() -> None:
    assets, returns = inputs()
    flat = pl.DataFrame(
        {
            "date": [dt.date(2023, 1, 4)],
            **{c: [0.0] for c in returns.columns if c != "date"},
        }
    )
    report = realized.build_realized_pnl_report(assets, pl.concat([returns, flat]))
    assert report.daily.height == 3
    assert report.daily["reconciliation_error"].abs().max() < 1e-15


def test_fixed_notional_drawdown_retains_prior_peak_and_initial_loss():
    dates = [dt.date(2024, 1, day) for day in [2, 3, 4, 5, 8]]
    daily = pl.DataFrame(
        {"date": dates, "long_short_net": [-0.1, 0.3, -0.15, 0.1, 0.1]}
    )
    path = realized.fixed_notional_path(daily.reverse())
    assert path["cumulative_pnl"].to_list() == pytest.approx(
        [-0.1, 0.2, 0.05, 0.15, 0.25]
    )
    assert path["drawdown"].to_list() == pytest.approx([-0.1, 0, -0.15, -0.05, 0])
    # Selecting inside a drawdown must retain the peak before the view.
    assert path.filter(pl.col("date") == dates[3])["drawdown"].item() == pytest.approx(
        -0.05
    )
    with pytest.raises(ValueError, match="duplicate"):
        realized.fixed_notional_path(pl.concat([daily, daily.head(1)]))


def test_worst_drawdown_window_excludes_peak_return():
    dates = [dt.date(2024, 1, d) for d in range(2, 7)]
    daily = pl.DataFrame(
        {"date": dates, "long_short_net": [0.1, -0.01, 0.03, -0.06, -0.02]}
    )
    start, end = realized.worst_drawdown_period(daily)
    assert (start, end) == (dates[3], dates[4])
    assert daily.filter(pl.col("date").is_between(start, end))[
        "long_short_net"
    ].sum() == pytest.approx(-0.08)


def test_benchmark_path_rebases_without_dropping_first_return_or_filling_bad_inputs():
    dates = [dt.date(2024, 1, day) for day in [2, 3, 4]]
    daily = pl.DataFrame({"date": dates, "benchmark": [0.1, -0.1, 0.2]})
    result = realized.benchmark_path(daily.reverse())
    assert result["date"].to_list() == dates
    assert result["value"].to_list() == pytest.approx([0.1, -0.01, 0.188])
    assert realized.benchmark_path(daily.tail(2))["value"].to_list() == pytest.approx(
        [-0.1, 0.08]
    )
    exhausted = daily.with_columns(pl.Series("benchmark", [0.1, -1.0, 0.2]))
    assert realized.benchmark_path(exhausted)["value"].to_list() == pytest.approx(
        [0.1, -1.0, -1.0]
    )
    for invalid in [
        daily.with_columns(pl.lit(None, pl.Float64).alias("benchmark")),
        daily.with_columns(pl.lit(float("nan")).alias("benchmark")),
        daily.with_columns(pl.lit(float("inf")).alias("benchmark")),
        daily.with_columns(pl.lit(-1.01).alias("benchmark")),
        daily.with_columns(pl.lit(None, pl.Date).alias("date")),
        daily.with_columns(pl.col("date").cast(pl.String)),
        pl.concat([daily, daily.head(1)]),
        daily.head(0),
    ]:
        with pytest.raises(ValueError):
            realized.benchmark_path(invalid)
