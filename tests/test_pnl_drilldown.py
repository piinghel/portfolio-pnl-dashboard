"""Selected-period attribution reconciles and navigation keeps stock/date context."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl
import pytest
import streamlit.testing.v1 as testing

import attribution_dashboard.accounting.realized_io as realized_io
import attribution_dashboard.factor_data as factor_data
import attribution_dashboard.pnl_breakdown as breakdown
import attribution_dashboard.pnl_drilldown as drilldown


def test_partial_month_and_stock_cost_reconciliation():
    dates = [dt.date(2024, 1, 30), dt.date(2024, 1, 31), dt.date(2024, 2, 1)]
    daily = pl.DataFrame(
        {
            "date": dates,
            "long_short_net": [-0.03, 0.01, 0.2],
            "cost_pnl": [-0.01, -0.01, 0.0],
            "long_pnl": [0.02, 0.02, 0.0],
            "short_pnl": [-0.04, 0.0, 0.2],
        }
    )
    assets = pl.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[2]],
            "asset_id": ["A", "B", "A", "B"],
            "label": ["Alpha", "Beta", "Alpha", "Beta"],
            "side": ["long", "short", "long", "short"],
            "asset_pnl": [0.02, -0.04, 0.02, 0.2],
        }
    )
    periods = drilldown.period_totals(daily, "Month")
    first = periods.row(0, named=True)
    assert (first["start"], first["end"]) == (dates[0], dates[1])
    stocks = breakdown.group_totals(
        assets.filter(pl.col("date").is_between(first["start"], first["end"])),
        "Stocks",
        "Combined",
    )
    assert stocks["pnl"].sum() + first["costs"] == pytest.approx(first["net"])
    figure = breakdown.figure(
        stocks, first["costs"], 100, "% notional", group="Stocks", side="Combined"
    )
    assert sum(figure.data[0].x) == pytest.approx(first["net"] * 100)
    day = breakdown.group_totals(
        assets.filter(pl.col("date") == dates[0]), "Stocks", "Combined"
    )
    assert day["pnl"].sum() == pytest.approx(-0.02)


def test_month_to_stock_and_back_keeps_the_selected_period():
    root = Path(__file__).resolve().parents[1]
    app = testing.AppTest.from_file(str(root / "app.py"), default_timeout=30).run()
    assert not app.exception
    mode = next(
        w for w in app.segmented_control if w.key.startswith("explain_frequency_")
    )
    mode.set_value("Month").run()
    assert not app.exception
    original_dates = app.date_input[0].value
    period = next(w for w in app.selectbox if w.key.startswith("explain_period_"))
    period.select_index(len(period.options) - 1).run()
    selected_dates = app.date_input[0].value
    assert selected_dates != original_dates
    selected_month = selected_dates[0].replace(day=1)
    daily = pl.read_parquet(root / "data/demo/daily.parquet").filter(
        pl.col("date").is_between(*selected_dates)
    )
    assert float(app.metric[0].value) == pytest.approx(
        daily["long_short_net"].sum() * 100, abs=0.0005
    )
    app.segmented_control(key="breakdown_side").set_value("Short").run()
    app.selectbox(key="breakdown_count").set_value("All").run()
    target = next(w for w in app.selectbox if w.key.startswith("pnl_driver_stock_"))
    target.select_index(min(3, len(target.options) - 1))
    expected_stock = target.value.split(" · ")[0]
    app.run()
    assert not app.exception
    assert app.segmented_control(key="page").value == "Stock detail"
    detail = app.selectbox(key="stock")
    assert any(
        label.startswith(expected_stock) and label.endswith(detail.value)
        for label in detail.options
    )
    assert all(
        date.replace(day=1) == selected_month for date in app.date_input[0].value
    )
    next(b for b in app.button if b.label == "Back to P&L breakdown").click().run()
    assert not app.exception
    assert app.segmented_control(key="page").value == "Overview"
    assert app.date_input[0].value == selected_dates
    assert app.segmented_control(key="breakdown_side").value == "Short"
    assert app.selectbox(key="breakdown_count").value == "All"
    app.segmented_control(key="breakdown_group").set_value("Industries").run()
    app.segmented_control(key="page").set_value("Stock detail").run()
    next(b for b in app.button if b.label == "Back to P&L breakdown").click().run()
    assert not app.exception
    assert app.segmented_control(key="breakdown_group").value == "Industries"
    next(b for b in app.button if b.label == "Reset period").click().run()
    assert not app.exception
    assert app.date_input[0].value == original_dates
    assert any(
        json.loads(c.proto.spec)["data"][0]["type"] == "waterfall"
        for c in app.get("plotly_chart")
    )


def test_groupings_sides_and_factor_partitions_reconcile():
    root = Path(__file__).resolve().parents[1] / "data/demo"
    start, end = dt.date(2023, 1, 3), dt.date(2023, 2, 2)
    report = realized_io.load_period(root, start, end)
    assets = (
        report.assets.lazy()
        .join(
            pl.scan_parquet(root / "classifications.parquet"),
            on="asset_id",
            validate="m:1",
        )
        .collect()
    )
    daily = factor_data.read_period(
        root / "factors/daily.parquet",
        start,
        end,
        factor_data.stamp(root / "factors/daily.parquet"),
    )
    for side in ["Combined", "Long", "Short"]:
        expected = (
            assets["asset_pnl"].sum()
            if side == "Combined"
            else assets.filter(pl.col("side") == side.lower())["asset_pnl"].sum()
        )
        for group in ["Stocks", "Sectors", "Industries"]:
            rows = breakdown.group_totals(assets, group, side)
            assert rows["pnl"].sum() == pytest.approx(expected)
            costs = report.daily["cost_pnl"].sum() if side == "Combined" else None
            fig = breakdown.figure(
                rows, costs, 100, "% notional", group=group, side=side, limit=2
            )
            assert sum(fig.data[0].x) == pytest.approx((expected + (costs or 0)) * 100)
        if side == "Combined":
            factors = breakdown.factor_totals(daily, report.daily)
        else:
            folder = root / "factors"
            partition = factor_data.side_partition(
                folder,
                start,
                end,
                side.lower(),
                factor_data.stamp(folder / "stocks.parquet"),
                factor_data.stamp(folder / "asset_factors.parquet"),
                report.daily.select("date"),
            )
            parent = report.daily.select(
                "date", pl.col(f"{side.lower()}_pnl").alias("long_short_net")
            )
            factors = breakdown.factor_totals(partition, parent)
            with pytest.raises(ValueError, match="reconcile"):
                breakdown.factor_totals(
                    partition.with_columns(pl.col("pnl") * -1), parent
                )
        assert factors["pnl"].sum() == pytest.approx(expected)
    # A missing classification and a stock changing side must preserve both P&Ls.
    changed = pl.DataFrame(
        {
            "asset_id": ["A", "A"],
            "label": ["Alpha", "Alpha"],
            "sector": [None, ""],
            "industry": [None, "Banks"],
            "side": ["long", "short"],
            "asset_pnl": [0.02, -0.03],
        }
    )
    rows = breakdown.group_totals(changed, "Sectors", "Combined")
    assert rows["label"].to_list() == ["Unclassified"]
    assert rows["pnl"].sum() == pytest.approx(-0.01)
    assert breakdown.group_totals(changed, "Stocks", "Short")[
        "pnl"
    ].sum() == pytest.approx(-0.03)


def test_side_cache_follows_the_supplied_ledger_calendar(tmp_path):
    dates = [dt.date(2024, 1, day) for day in [2, 3, 4]]
    folder = tmp_path / "factors"
    folder.mkdir()
    pl.DataFrame(
        {"date": [dates[0]], "view": ["long"], "factor": ["size"], "pnl": [0.01]}
    ).write_parquet(folder / "asset_factors.parquet")
    pl.DataFrame(
        {
            "date": [dates[0]],
            "side": ["long"],
            "idio_pnl": [0.02],
            "price_basis_gap": [0.0],
            "unmodeled_pnl": [0.0],
        }
    ).write_parquet(folder / "stocks.parquet")
    for calendar in [dates[:2], dates]:
        rows = factor_data.side_partition(
            folder,
            dates[0],
            dates[-1],
            "long",
            factor_data.stamp(folder / "stocks.parquet"),
            factor_data.stamp(folder / "asset_factors.parquet"),
            pl.DataFrame({"date": calendar}),
        )
        assert sorted(rows["date"].unique()) == calendar
        assert rows["pnl"].sum() == pytest.approx(0.03)
