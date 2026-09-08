"""Public bundle boundary: arbitrary windows rebase, metadata fixes money units."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl
import pytest

import attribution_dashboard.accounting.realized_io as realized_io


def test_load_period_rebases_saved_linked_data(tmp_path: Path) -> None:
    dates = [dt.date(2023, 1, 3), dt.date(2023, 1, 4)]
    pl.DataFrame(
        {
            "date": dates,
            "asset_id": ["A", "A"],
            "side": ["long", "long"],
            "asset_pnl": [0.1, 0.2],
            "prior_net_index": [1.0, 1.1],
            "linked_pnl": [0.1, 0.22],
        }
    ).write_parquet(tmp_path / "assets.parquet")
    pl.DataFrame(
        {
            "date": dates,
            "long_gross": [0.1, 0.2],
            "short_gross": [0.0, 0.0],
            "long_short_gross": [0.1, 0.2],
            "long_net": [0.1, 0.2],
            "short_net": [0.0, 0.0],
            "long_short_net": [0.1, 0.2],
        }
    ).write_parquet(tmp_path / "daily.parquet")
    report = realized_io.load_period(tmp_path, dates[-1], dates[-1])
    assert report.assets["linked_pnl"].sum() == pytest.approx(0.2)
    assert report.assets["prior_net_index"].to_list() == [1.0]
    assert report.assets["label"].to_list() == ["A"]
    with pytest.raises(ValueError, match="calendar"):
        realized_io.load_period(tmp_path, dt.date(2024, 1, 1), dt.date(2024, 1, 2))


@pytest.mark.parametrize("notional", [0, -1, float("nan"), True, "100"])
def test_metadata_rejects_invalid_money_denominator(
    tmp_path: Path, notional: object
) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"portfolio": {"name": "Book", "notional": notional}})
    )
    with pytest.raises(ValueError, match="notional"):
        realized_io.read_metadata(tmp_path)
