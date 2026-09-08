"""Original price basis, recorded boundaries and window censoring stay distinct."""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import polars as pl
import streamlit.testing.v1 as testing


def test_stock_prices_keep_original_values_and_do_not_invent_window_entries(
    tmp_path: Path,
):
    dates = [dt.date(2024, 1, d) for d in range(2, 7)]
    pl.DataFrame(
        {
            "date": dates,
            "asset_id": ["A"] * 5,
            "px_last_unadjusted": [100.0, 102.0, 103.0, 105.0, 106.0],
            "px_last": [50.0, 51.0, 51.5, 52.5, 53.0],
            "price_currency": ["USD"] * 5,
        }
    ).write_parquet(tmp_path / "prices.parquet")
    rows = [(d, "B", "long", 1.0) for d in dates] + [
        (dates[i], "A", "short", 2.0) for i in [1, 2, 3]
    ]
    pl.DataFrame(
        rows, schema=["date", "asset_id", "side", "holding_qty"], orient="row"
    ).write_parquet(tmp_path / "positions.parquet")
    app = testing.AppTest.from_string(
        "from pathlib import Path\nimport datetime as dt\nimport streamlit as st\nimport attribution_dashboard.stock_prices as prices\nimport polars as pl\n"
        "start = st.date_input('Start',dt.date(2024,1,2),key='start')\n"
        "frame = pl.DataFrame({'date': [dt.date(2024,1,d) for d in range(2,7)], 'asset_pnl':[0.,-.01,.02,-.01,0.], 'Long exposure':[0.]*5, 'Short exposure':[0.,-.1,-.1,-.1,0.] }).filter(pl.col('date') >= start)\n"
        f"prices.render(Path({str(tmp_path)!r}), 'A', start, dt.date(2024,1,6), contributions=frame,label='Stock A',scale=100.,unit='% notional',opening_date=None)",
        default_timeout=30,
    ).run()
    assert not app.exception
    spec = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert spec["data"][0]["y"] == [50, 51, 51.5, 52.5, 53]
    assert {t["name"]: t["x"] for t in spec["data"] if " · " in t["name"]} == {
        "Short · entry": ["2024-01-03"],
        "Short · exit": ["2024-01-06"],
    }
    labels = {
        a["text"]: a["x"] for a in spec["layout"]["annotations"] if a.get("showarrow")
    }
    assert labels == {"Short entry": "2024-01-03", "Short exit": "2024-01-06"}
    app.segmented_control(key="stock_price_basis").set_value("Original close").run()
    assert json.loads(app.get("plotly_chart")[0].proto.spec)["data"][0]["y"] == [
        100,
        102,
        103,
        105,
        106,
    ]
    assert spec["layout"]["xaxis"]["matches"] == "x3"
    assert spec["layout"]["xaxis2"]["matches"] == "x3"
    assert next(t for t in spec["data"] if t["name"] == "Short position")["y"] == [
        0,
        -10,
        -10,
        -10,
        0,
    ]
    # Plotly.js 3.7 drops leading-sign formats and displays unrounded values.
    short = next(t for t in spec["data"] if t["name"] == "Short position")
    assert "%{y:.3f}" in short["hovertemplate"]
    # Log scaling must retain actual prices and correctly anchor event labels.
    app.segmented_control(key="stock_price_scale").set_value("Log").run()
    assert not app.exception
    log_spec = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert log_spec["layout"]["yaxis"]["type"] == "log"
    for axis in ["yaxis2", "yaxis3"]:
        assert log_spec["layout"][axis].get("type", "linear") == "linear"
    assert log_spec["data"][0]["y"] == [100, 102, 103, 105, 106]
    entry = next(
        a for a in log_spec["layout"]["annotations"] if a["text"] == "Short entry"
    )
    assert entry["y"] == math.log10(102)
    app.segmented_control(key="stock_price_basis").set_value("Adjusted close").run()
    adjusted = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert adjusted["layout"]["yaxis"]["type"] == "log"
    entry = next(
        a for a in adjusted["layout"]["annotations"] if a["text"] == "Short entry"
    )
    assert entry["y"] == math.log10(51)
    app.segmented_control(key="stock_price_scale").set_value("Linear").run()
    linear = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert linear["layout"]["yaxis"]["type"] == "linear"
    entry = next(
        a for a in linear["layout"]["annotations"] if a["text"] == "Short entry"
    )
    assert entry["y"] == 51
    app.date_input(key="start").set_value(dt.date(2024, 1, 4)).run()
    assert not app.exception
    traces = json.loads(app.get("plotly_chart")[0].proto.spec)["data"]
    assert [t["name"] for t in traces if " · " in t["name"]] == ["Short · exit"]
