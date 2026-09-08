"""Holdings boundaries must not turn into invented executions."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

import attribution_dashboard.accounting.stock_history as history


def test_aggregate_lifecycles_keep_side_overlap_and_censored_boundaries():
    dates = [dt.date(2024, 1, d) for d in [5, 8, 9, 10, 11]]
    rows = [
        (dates[i], "A", side, qty)
        for i, side, qty in [
            (0, "long", 1.0),
            (1, "long", 2.0),
            (1, "long", 3.0),
            (1, "short", 4.0),
            (2, "short", 1.0),
            (3, "short", 2.0),
            (4, "short", 2.0),
            (3, "long", 5.0),
            (4, "long", 0.0),
        ]
    ]
    held = pl.DataFrame(
        rows, schema=["date", "asset_id", "side", "holding_qty"], orient="row"
    )
    events = history.position_events(held, pl.DataFrame({"date": dates}))
    assert events.select("date", "side", "event").rows() == [
        (dates[0], "long", "Already held"),
        (dates[1], "short", "Entry"),
        (dates[2], "long", "Exit"),
        (dates[3], "long", "Entry"),
        (dates[4], "long", "Exit"),
    ]
    assert events.filter(pl.col("event") == "Exit")["last_held"].to_list() == [
        dates[1],
        dates[3],
    ]
    assert (
        events.filter(pl.col("date") >= dates[2])
        .filter(pl.col("side") == "short")
        .is_empty()
    )
    with pytest.raises(ValueError, match="full calendar"):
        history.position_events(held, pl.DataFrame({"date": dates[1:]}))
    with pytest.raises(ValueError, match="nonnegative"):
        history.position_events(
            held.with_columns(-pl.col("holding_qty")), pl.DataFrame({"date": dates})
        )


def test_stock_summary_combines_sides_and_keeps_flat_dates():
    dates = [dt.date(2024, 1, day) for day in range(2, 5)]
    assets = pl.DataFrame(
        {
            "date": dates[:2] * 2,
            "asset_id": ["A"] * 4,
            "label": ["Stock A"] * 4,
            "sector": ["Tech"] * 4,
            "side": ["long", "long", "short", "short"],
            "asset_pnl": [1.0, 2.0, -1.0, -1.0],
            "gross_weight": [0.1, 0.2, 0.05, 0.05],
        }
    )
    result = history.summarize_stocks(assets, pl.DataFrame({"date": dates})).row(
        0, named=True
    )
    assert result["pnl"] == 1.0
    assert result["average_long"] == pytest.approx(0.1)
    assert result["average_short"] == pytest.approx(-0.1 / 3)
    assert result["ending_long"] == result["ending_short"] == 0.0
    missing = history.summarize_stocks(
        assets.drop("gross_weight"), pl.DataFrame({"date": dates})
    )
    assert missing["ending_long"].item() is None
