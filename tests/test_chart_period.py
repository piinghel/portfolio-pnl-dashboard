"""A chart range means saved sessions, and never changes stock identity."""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

import polars as pl
import pytest
import streamlit.testing.v1 as testing

import attribution_dashboard.chart_period as periods


def test_selection_uses_calendar_not_marker_dates():
    calendar = [dt.date(2023, 1, d) for d in [13, 17, 18, 19]]
    selection = {
        "box": [{"xref": "x3", "x": ["2023-01-13 09:00", "2023-01-18 12:00"]}],
        "points": [{"x": "2023-01-17"}],
    }
    assert periods.selected_dates(selection, calendar) == (calendar[0], calendar[2])
    selection["box"][0]["x"] = ["2023-01-16", "2023-01-16"]
    assert periods.selected_dates(selection, calendar) is None
    selection["box"][0]["x"] = ["2023-01-17", "2023-01-17"]
    assert periods.selected_dates(selection, calendar) == (calendar[1], calendar[1])
    assert periods.selected_dates({"box": []}, calendar) is None
    selection["box"][0]["x"] = [1.5, 2.5]
    assert periods.selected_dates(selection, calendar) is None


def test_stock_range_recalculates_and_preserves_a_flat_stock(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    directory = tmp_path / "demo"
    shutil.copytree(root / "data/demo", directory)
    assets = pl.read_parquet(directory / "assets.parquet")
    # Sparse ledgers omit zero holdings. Remove only a stock/session whose
    # aggregate exposure and P&L are both zero, preserving reconciliation.
    zero = (
        assets.group_by("asset_id", "date")
        .agg(pl.col("gross_weight").sum(), pl.col("asset_pnl").sum())
        .filter((pl.col("gross_weight") == 0) & (pl.col("asset_pnl") == 0))
    )
    stock, flat = zero.select("asset_id", "date").row(0)
    assets.filter(
        ~((pl.col("asset_id") == stock) & (pl.col("date") == flat))
    ).write_parquet(directory / "assets.parquet")
    app = testing.AppTest.from_string(
        "from pathlib import Path\n"
        "import attribution_dashboard.realized_page as page\n"
        f"page.render(Path({str(directory)!r}))\n",
        default_timeout=30,
    ).run()
    app.segmented_control(key="page").set_value("Stock detail").run()
    app.selectbox(key="stock").set_value(stock).run()
    assert not app.exception
    app.session_state["analysis_pending"] = {
        "directory": str(directory),
        "start": flat,
        "end": flat,
    }
    app.run()
    assert not app.exception
    assert app.date_input[0].value == (flat, flat)
    assert app.selectbox(key="stock").value == stock
    figure = json.loads(app.get("plotly_chart")[0].proto.spec)
    stock_pnl = next(t for t in figure["data"] if t.get("name") == "Gross stock P&L")
    assert stock_pnl["y"][-1] == pytest.approx(0)
    assert any("No portfolio holdings" in c.value for c in app.caption)
    app.segmented_control(key="page").set_value("Overview").run()
    assert not app.exception
    assert app.date_input[0].value == (flat, flat)
