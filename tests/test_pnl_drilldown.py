"""Selected-period attribution reconciles and navigation keeps stock/date context."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl
import pytest
import streamlit.testing.v1 as testing

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
    stocks = drilldown.stock_totals(assets, first["start"], first["end"])
    assert stocks["pnl"].sum() + first["costs"] == pytest.approx(first["net"])
    figure = drilldown.breakdown_figure(stocks, first["costs"], 100, "% notional")
    assert sum(figure.data[0].x) == pytest.approx(first["net"] * 100)
    day = drilldown.stock_totals(assets, dates[0], dates[0])
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
    selected_month = next(
        w for w in app.selectbox if w.key.startswith("explain_period_")
    ).value
    target = next(w for w in app.selectbox if w.key.startswith("pnl_driver_stock_"))
    target.select_index(min(3, len(target.options) - 1))
    expected_stock = target.value
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
    assert app.date_input[0].value == original_dates
    assert (
        next(
            w for w in app.segmented_control if w.key.startswith("explain_frequency_")
        ).value
        == "Month"
    )
    assert (
        next(w for w in app.selectbox if w.key.startswith("explain_period_")).value
        == selected_month
    )
    assert any(
        json.loads(c.proto.spec)["data"][0]["type"] == "waterfall"
        for c in app.get("plotly_chart")
    )
