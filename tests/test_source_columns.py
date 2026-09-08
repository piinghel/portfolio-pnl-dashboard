"""A different source schema preserves accounting and the connected dashboard."""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

import polars as pl
import pytest
import streamlit.testing.v1 as testing
import yaml

import attribution_dashboard.accounting.realized_io as realized_io
import attribution_dashboard.accounting.source_schema as source_schema
import attribution_dashboard.realized_page as page


def test_configured_columns_preserve_all_dashboard_views(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "data" / "demo"
    folder = tmp_path / "strategy"
    shutil.copytree(source, folder)
    mapping = {
        "date": "session",
        "asset_id": "security",
        "label": "name",
        "sector": "sector_name",
        "industry": "industry_name",
    }
    for path in folder.rglob("*.parquet"):
        frame = pl.scan_parquet(path)
        names = frame.collect_schema().names()
        frame.rename(
            {key: value for key, value in mapping.items() if key in names}
        ).collect().write_parquet(path)
    # Exercise the classification sidecar rather than only embedded industries.
    assets_path = folder / "assets.parquet"
    pl.scan_parquet(assets_path).drop(
        "industry_name", strict=False
    ).collect().write_parquet(assets_path)
    configuration = {
        "books": [{"label": "Alternate schema", "dir": "strategy", "columns": mapping}]
    }
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(configuration))
    app = testing.AppTest.from_string(
        f"import attribution_dashboard.realized_page as page\npage.render_config({str(config)!r})",
        default_timeout=30,
    ).run()
    assert not app.exception and not app.error
    start, end = app.date_input[0].value
    expected = realized_io.load_period(source, start, end)
    assert float(app.metric[0].value) == pytest.approx(
        expected.daily["long_short_net"].sum() * 100, abs=0.00051
    )
    for group in ("Stocks", "Sectors", "Industries", "Factors"):
        app.segmented_control(key="breakdown_group").set_value(group)
        for side in ("Combined", "Long", "Short"):
            app.segmented_control(key="breakdown_side").set_value(side).run()
            assert not app.exception and not app.error
    app.segmented_control(key="page").set_value("Stock detail").run()
    assert not app.exception and not app.error and not app.warning
    stock = app.selectbox(key="stock").value
    spec = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert spec["layout"]["yaxis"]["type"] == "log"
    app.segmented_control(key="page").set_value("Factors").run()
    for view in ("P&L", "Exposure and risk", "Stock drivers"):
        app.segmented_control(key="factor_view").set_value(view).run()
        assert not app.exception and not app.error
    # Reconfiguring the same directory must discard stale dates/stock navigation.
    for path in folder.rglob("*.parquet"):
        frame = pl.scan_parquet(path)
        if "security" in frame.collect_schema():
            frame.with_columns(
                pl.concat_str(pl.lit("other-"), "security").alias("other_security")
            ).collect().write_parquet(path)
    configuration["books"][0]["columns"]["asset_id"] = "other_security"
    config.write_text(yaml.safe_dump(configuration))
    app.segmented_control(key="page").set_value("Stock detail").run()
    assert not app.exception and not app.error
    assert app.selectbox(key="stock").value != stock
    assert app.selectbox(key="stock").value.startswith("other-")


def test_mapping_is_part_of_cached_read_identity(tmp_path: Path) -> None:
    dates = [dt.date(2024, 1, 2)]
    pl.DataFrame(
        {
            "date": dates,
            "first_id": ["A"],
            "second_id": ["B"],
            "side": ["long"],
            "asset_pnl": [0.01],
        }
    ).write_parquet(tmp_path / "assets.parquet")
    pl.DataFrame(
        {
            "date": dates,
            "long_gross": [0.01],
            "short_gross": [0.0],
            "long_short_gross": [0.01],
            "long_net": [0.01],
            "short_net": [0.0],
            "long_short_net": [0.01],
        }
    ).write_parquet(tmp_path / "daily.parquet")
    first = page.load_period(
        str(tmp_path),
        dates[0],
        dates[0],
        (1,),
        columns=source_schema.SourceColumns(asset_id="first_id"),
    )
    second = page.load_period(
        str(tmp_path),
        dates[0],
        dates[0],
        (1,),
        columns=source_schema.SourceColumns(asset_id="second_id"),
    )
    assert first.assets["asset_id"].to_list() == ["A"]
    assert second.assets["asset_id"].to_list() == ["B"]


@pytest.mark.parametrize(
    "values",
    [
        {"date": ""},
        {"date": "same", "asset_id": "same"},
        {"industryy": "x"},
        {"date": 4},
    ],
)
def test_invalid_source_mapping(values: dict) -> None:
    with pytest.raises(ValueError):
        source_schema.parse_columns(values)


def test_mapping_rejects_ambiguous_or_missing_configured_column(tmp_path: Path) -> None:
    path = tmp_path / "assets.parquet"
    pl.DataFrame({"asset_id": ["A"], "security": ["B"]}).write_parquet(path)
    with pytest.raises(ValueError, match="duplicate"):
        source_schema.scan_parquet(
            path, columns=source_schema.SourceColumns(asset_id="security")
        )
    with pytest.raises(ValueError, match="missing"):
        source_schema.scan_parquet(
            path, columns=source_schema.SourceColumns(asset_id="missing")
        )
