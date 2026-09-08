"""Check shipped data accounting and the complete public demo."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import streamlit.testing.v1 as testing

import attribution_dashboard.accounting.factor_risk as factor_risk
import attribution_dashboard.accounting.realized_io as realized_io


def test_demo_reconciles_prices_holdings_and_factor_partition() -> None:
    root = Path(__file__).resolve().parents[1] / "data/demo"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["synthetic"] is True
    prices = pl.scan_parquet(root / "prices.parquet")
    calendar = prices.select("date").unique().sort("date").collect()["date"]
    report = realized_io.load_period(root, calendar[0], calendar[-1])
    factors = pl.scan_parquet(root / "factors/daily.parquet").collect()
    factor_risk.validate_factor_partition(factors, report.daily)
    pnl = (
        prices.join(
            pl.scan_parquet(root / "positions.parquet"),
            on=["date", "asset_id"],
            validate="1:1",
        )
        .sort("asset_id", "date")
        .with_columns(
            (
                pl.col("holding_qty").shift(1).over("asset_id").fill_null(0)
                * pl.col("px_last").diff().over("asset_id").fill_null(0)
                * pl.when(pl.col("side") == "long").then(1).otherwise(-1)
                / manifest["portfolio"]["notional"]
            ).alias("expected")
        )
        .join(report.assets.lazy(), on=["date", "asset_id", "side"], validate="1:1")
        .select((pl.col("expected") - pl.col("asset_pnl")).abs().max())
        .collect()
        .item()
    )
    assert pnl < 1e-12
    assert report.assets["asset_id"].str.starts_with("DEMO").all()


def test_public_demo_pages_and_price_scale() -> None:
    app = testing.AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30
    ).run()
    assert not app.exception
    assert not app.error
    assert any("Synthetic demo" in item.value for item in app.caption)
    assert any(
        json.loads(c.proto.spec)["data"][0]["type"] == "waterfall"
        for c in app.get("plotly_chart")
    )
    for page in ["Risk and reward", "Factors", "Stock detail"]:
        app.segmented_control(key="page").set_value(page).run()
        assert not app.exception
        assert not app.error
    app.segmented_control(key="stock_price_scale").set_value("Log").run()
    assert not app.exception
    spec = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert spec["layout"]["yaxis"]["type"] == "log"
    app.segmented_control(key="page").set_value("Factors").run()
    for view in ["Exposure and risk", "Stock drivers"]:
        app.segmented_control(key="factor_view").set_value(view).run()
        assert not app.exception
        assert not app.error
