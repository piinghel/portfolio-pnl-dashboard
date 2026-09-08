"""Saved fold coefficients stay point in time in the stock heatmap."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl
import streamlit.testing.v1 as testing

import attribution_dashboard.linear_history as history
import attribution_dashboard.prediction_history as charts


def test_dated_coefficients_forecast_coverage_and_heatmap_controls(
    tmp_path: Path,
) -> None:
    dates = [dt.date(2024, 1, day) for day in [2, 3, 4, 5]]
    pl.DataFrame(
        {
            "date": dates,
            "security": ["A"] * 4,
            "quality": [2.0, 3.0, 4.0, 5.0],
        }
    ).write_parquet(tmp_path / "inputs.parquet")
    pl.DataFrame(
        {
            "model_date": [dates[0], dates[2]],
            "predictor": ["quality", "quality"],
            "coefficient": [0.5, -0.25],
        }
    ).write_parquet(tmp_path / "coefficients.parquet")
    pl.DataFrame(
        {
            "date": [dates[0], dates[2], dates[3]],
            "security": ["A"] * 3,
            "model_name": ["ridge"] * 3,
        }
    ).write_parquet(tmp_path / "forecasts.parquet")
    (tmp_path / "linear_history.json").write_text(
        json.dumps(
            {
                **{
                    name: str(tmp_path / f"{name}.parquet")
                    for name in ["inputs", "coefficients", "forecasts"]
                },
                "asset_column": "security",
                "model_name": "ridge",
            }
        )
    )
    rows = history.load(tmp_path, "A", dates[0], dates[3])
    assert rows is not None
    assert rows["date"].to_list() == [dates[0], dates[2], dates[3]]
    assert rows["contribution"].to_list() == [1.0, -1.0, -1.25]
    earlier = history.load(tmp_path, "A", dates[0], dates[1])
    assert earlier is not None and earlier["contribution"].to_list() == [1.0]
    chart = charts.heatmap(
        rows, "contribution", calendar_axis=True, trading_dates=dates
    )
    assert list(chart.data[0].z[0]) == [1.0, None, -1.0, -1.25]
    assert chart.layout.xaxis.type == "date"
    assert "2024-01-01" not in chart.data[0].x  # Closed session is not invented.
    app = testing.AppTest.from_string(
        "from pathlib import Path\nimport datetime as dt\n"
        "import attribution_dashboard.linear_history as history\n"
        "import attribution_dashboard.prediction_history as charts\n"
        f"rows = history.load(Path({str(tmp_path)!r}), 'A', dt.date(2024,1,2), dt.date(2024,1,5))\n"
        "charts.render(rows, 'A', 'model', dt.date(2024,1,2), dt.date(2024,1,5), "
        "trading_dates=[dt.date(2024,1,d) for d in [2,3,4,5]])\n",
        default_timeout=30,
    ).run()
    assert not app.exception
    app.segmented_control(key="prediction_history_metric").set_value(
        "Model input"
    ).run()
    assert not app.exception
    spec = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert spec["data"][0]["z"] == [[2.0, None, 4.0, 5.0]]
