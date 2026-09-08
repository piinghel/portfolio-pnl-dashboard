"""Prediction explanations must match both the model score and held book."""

import json
from pathlib import Path

import polars as pl
import pytest
import streamlit.testing.v1 as testing

from attribution_dashboard import prediction_detail as predictions

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
            decisions.with_columns(pl.col("score") + 0.01), contributions
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
