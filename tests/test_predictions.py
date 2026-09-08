"""Prediction explanations must match both the model score and held book."""

import datetime as dt
import json
from pathlib import Path

import polars as pl
import pytest
import streamlit.testing.v1 as testing

from attribution_dashboard import prediction_detail as predictions
from attribution_dashboard import prediction_history as history

ROOT = Path(__file__).resolve().parents[1]


def test_saved_predictions_reconcile_and_select_actual_holdings():
    folder = ROOT / "data/demo"
    decisions = pl.scan_parquet(folder / "predictions/decisions.parquet").collect()
    contributions = pl.scan_parquet(
        folder / "predictions/contributions.parquet"
    ).collect()
    predictions.validate(decisions, contributions)
    held = (
        decisions.lazy()
        .join(
            pl.scan_parquet(folder / "positions.parquet"),
            on=["date", "asset_id", "side"],
            validate="1:1",
        )
        .collect()
    )
    assert held.height == decisions.height
    assert (held["selected"] == (held["holding_qty"] > 0)).all()
    assert (
        decisions["selected"] == (decisions["rank"] <= decisions["selection_count"])
    ).all()
    for (_, side), rows in decisions.partition_by("date", "side", as_dict=True).items():
        ordered = rows.sort("score", descending=side == "long")
        assert ordered["rank"].to_list() == list(range(1, 19))
        assert (ordered["cutoff"] == ordered["score"][11]).all()
    with pytest.raises(ValueError, match="reconcile"):
        predictions.validate(
            decisions.with_columns(pl.col("intercept") + 0.01), contributions
        )
    with pytest.raises(ValueError, match="cutoff"):
        predictions.validate(
            decisions.with_columns(pl.col("cutoff") + 10), contributions
        )
    with pytest.raises(ValueError, match="input"):
        predictions.validate(
            decisions, contributions.with_columns(pl.col("coefficient") + 0.1)
        )
    with pytest.raises(ValueError, match="Duplicate"):
        predictions.validate(pl.concat([decisions, decisions.head(1)]), contributions)


def test_clicks_require_an_exact_saved_marker_snapshot():
    available = {("2024-01-02", "short")}
    assert predictions.clicked_decision(
        [{"customdata": ["2024-01-02", "short", "Entry"]}], available
    ) == ("2024-01-02", "short")
    for point in [
        {"x": "2024-01-02"},
        {"customdata": ["2024-01-03", "short", "Entry"]},
        {"customdata": ["2024-01-02", "long", "Entry"]},
    ]:
        assert predictions.clicked_decision([point], available) is None


def test_prediction_dialog_and_top_five_remainder():
    app = testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.segmented_control(key="page").set_value("Stock detail").run()
    assert not app.exception
    next(
        button for button in app.button if button.label == "Explain decision"
    ).click().run()
    assert not app.exception
    charts = [json.loads(chart.proto.spec) for chart in app.get("plotly_chart")]
    bars = next(
        chart["data"][0] for chart in charts if chart["data"][0]["type"] == "waterfall"
    )
    assert "Other 5 predictors" in bars["y"]
    assert sum(bars["x"][:-1]) == pytest.approx(float(bars["text"][-1]), abs=0.00051)
    # AppTest does not simulate dialog fragment reruns. Check the full chart
    # independently; exercise the selector in the browser.
    decisions, rows, _ = predictions.load(ROOT / "data/demo", "DEMO001")
    decision = decisions.row(0, named=True)
    rows = rows.filter(pl.col("date") == decision["date"])
    bars = predictions.waterfall(decision, rows, 10).to_plotly_json()["data"][0]
    assert len(bars["y"]) == 12
    assert not any(name.startswith("Other") for name in bars["y"])
    assert sum(bars["x"][:-1]) == pytest.approx(decision["score"])


def test_history_preserves_signed_values_missing_cells_and_row_order():
    first, last = dt.date(2024, 1, 2), dt.date(2024, 3, 4)
    rows = pl.DataFrame(
        {
            "date": [last, first, first],
            "predictor": ["Momentum", "Momentum", "Value"],
            "input_value": [2.0, -4.0, 0.5],
            "coefficient": [0.5, 0.5, 1.0],
            "contribution": [1.0, -2.0, 0.5],
        }
    )
    chart = history.heatmap(rows, "contribution").data[0]
    assert list(chart.x) == [first.isoformat(), last.isoformat()]
    assert list(chart.y) == ["Momentum", "Value"]
    assert [list(row) for row in chart.z] == [[-2.0, 1.0], [0.5, None]]
    assert chart.zmin == -2.0 and chart.zmax == 2.0
    inputs = history.heatmap(rows, "input_value").data[0]
    assert list(inputs.y) == list(chart.y)
    assert [list(row) for row in inputs.z] == [[-4.0, 2.0], [0.5, None]]


def test_history_switch():
    app = testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.segmented_control(key="page").set_value("Stock detail").run()
    # Both panels use the stock calendar, including its opening baseline.
    figures = [json.loads(c.proto.spec) for c in app.get("plotly_chart")]
    stock = figures[0]
    heatmaps = [f for f in figures if f["data"][0]["type"] == "heatmap"]
    assert len(heatmaps) == 2
    assert heatmaps[0]["data"][0]["y"] == heatmaps[1]["data"][0]["y"]
    for figure in heatmaps:
        assert figure["layout"]["xaxis"]["type"] == "date"
        assert figure["layout"]["xaxis"]["range"] == stock["layout"]["xaxis"]["range"]
        for edge in ("l", "r"):
            assert figure["layout"]["margin"][edge] == stock["layout"]["margin"][edge]
    assert not next(
        e for e in app.expander if e.label == "About these charts"
    ).proto.expanded
    app.segmented_control(key="prediction_history_metric").set_value(
        "Model input"
    ).run()
    assert not app.exception
    chart = next(
        json.loads(c.proto.spec)["data"][0]
        for c in app.get("plotly_chart")
        if json.loads(c.proto.spec)["data"][0]["type"] == "heatmap"
    )
    assert len(chart["y"]) == 5
    assert chart["colorbar"]["title"]["text"] == "Model input"
    app.selectbox(key="prediction_history_selection").set_value("Top 10").run()
    chart = next(
        json.loads(c.proto.spec)["data"][0]
        for c in app.get("plotly_chart")
        if json.loads(c.proto.spec)["data"][0]["type"] == "heatmap"
    )
    assert len(chart["y"]) == 10
    app.selectbox(key="prediction_history_selection").set_value(
        "Choose predictors"
    ).run()
    chosen = app.multiselect[0].value[:2][::-1]
    app.multiselect[0].set_value(chosen).run()
    app.segmented_control(key="prediction_history_metric").set_value(
        "Score contribution"
    ).run()
    chart = next(
        json.loads(c.proto.spec)["data"][0]
        for c in app.get("plotly_chart")
        if json.loads(c.proto.spec)["data"][0]["type"] == "heatmap"
    )
    assert chart["y"] == chosen
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any("Choose at least one predictor" in item.value for item in app.info)
