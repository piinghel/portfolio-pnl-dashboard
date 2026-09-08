"""Exercise actual date, unit and drilldown controls over a reconciled ledger."""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import polars as pl
import pytest
import streamlit.testing.v1 as testing

import attribution_dashboard.accounting.realized as realized


def test_period_units_stock_and_risk_navigation(tmp_path: Path) -> None:
    dates = pl.date_range(dt.date(2022, 12, 1), dt.date(2023, 3, 31), eager=True)
    dates = dates.filter(dates.dt.weekday() <= 5).to_list()
    a = [math.sin(i) * 0.01 for i in range(len(dates))]
    b = [-x * 0.4 for x in a]
    rows = [
        (date, name, side, value, 0.1)
        for name, side, values in [("A", "long", a), ("B", "short", b)]
        for date, value in zip(dates, values, strict=True)
    ]
    assets = pl.DataFrame(
        rows,
        schema=["date", "asset_id", "side", "asset_pnl", "gross_weight"],
        orient="row",
    )
    returns = pl.DataFrame(
        {
            "date": dates,
            "long_gross": a,
            "short_gross": [-x for x in b],
            "long_short_gross": [x + y for x, y in zip(a, b, strict=True)],
            "long_net": [x - 0.001 for x in a],
            "short_net": [-x for x in b],
            "long_short_net": [x + y - 0.001 for x, y in zip(a, b, strict=True)],
        }
    )
    result = realized.build_realized_pnl_report(assets, returns)
    result.assets.write_parquet(tmp_path / "assets.parquet")
    result.daily.with_columns(pl.lit(0.001).alias("benchmark")).write_parquet(
        tmp_path / "daily.parquet"
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "portfolio": {
                    "name": "Independent test strategy",
                    "notional": 15e6,
                    "currency": "USD",
                    "classification": "2026-08-04 snapshot",
                },
                "conventions": {"omitted_costs": "Borrow and financing"},
            }
        )
    )
    app = testing.AppTest.from_string(
        f"from pathlib import Path\nimport attribution_dashboard.realized_page as page\npage.render(Path({str(tmp_path)!r}), benchmark_label="
        + repr("Test price index")
        + ")",
        default_timeout=30,
    ).run()
    assert not app.exception
    assert any("Borrow and financing" in entry.value for entry in app.markdown)
    app.selectbox(key="preset").set_value("Full history").run()
    assert not app.exception
    for preset, first_day in [
        ("Year to date (YTD)", dt.date(2023, 1, 1)),
        ("Month to date (MTD)", dt.date(2023, 3, 1)),
        ("Last 5 years", dates[0]),
        ("Last 10 years", dates[0]),
    ]:
        app.selectbox(key="preset").set_value(preset).run()
        assert not app.exception
        selected = returns.filter(pl.col("date") >= first_day)
        assert float(app.metric[0].value) == pytest.approx(
            selected["long_short_net"].sum() * 100, abs=0.00051
        )
        assert app.date_input[0].value == (first_day, dates[-1])
    app.selectbox(key="preset").set_value("Full history").run()
    total = float(returns["long_short_net"].sum())
    assert float(app.metric[0].value) == pytest.approx(total * 100, abs=0.00051)
    before = json.loads(app.get("plotly_chart")[0].proto.spec)["data"]
    app.toggle(key="show_benchmark").set_value(True).run()
    assert not app.exception
    after = json.loads(app.get("plotly_chart")[0].proto.spec)["data"]
    benchmark = next(t for t in after if t["name"] == "Test price index")
    assert benchmark["y"][-1] == pytest.approx((1.001 ** len(dates) - 1) * 100)
    assert [t for t in after if t["name"] != "Test price index"] == before
    saved = pl.read_parquet(tmp_path / "daily.parquet")
    saved.drop("benchmark").write_parquet(tmp_path / "daily.parquet")
    app.run()
    assert not app.exception
    assert any("No saved benchmark" in item.value for item in app.info)
    assert float(app.metric[0].value) == pytest.approx(total * 100, abs=0.00051)
    saved.write_parquet(tmp_path / "daily.parquet")
    app.toggle(key="show_benchmark").set_value(False).run()
    app.selectbox(key="units").set_value("Money (millions)").run()
    assert float(app.metric[0].value) == pytest.approx(total * 15, abs=0.00051)
    app.segmented_control(key="page").set_value("Risk and reward").run()
    app.selectbox(key="trend_group").set_value("Sector").run()
    assert not app.exception
    app.segmented_control(key="group").set_value("Stock").run()
    app.selectbox(key="side").set_value("short").run()
    assert not app.exception
    reward = json.loads(app.get("plotly_chart")[0].proto.spec)["data"][0]
    assert reward["customdata"] == ["B · short"]
    assert reward["x"] == pytest.approx([sum(b) * 15])
    assert "%{x:,.3f}" in reward["hovertemplate"]
    app.segmented_control(key="page").set_value("Stock detail").run()
    app.selectbox(key="stock").set_value("B").run()
    assert not app.exception
    stock_path = json.loads(app.get("plotly_chart")[0].proto.spec)["data"][0]
    assert stock_path["y"][-1] == pytest.approx(sum(b) * 15)
    assert "%{y:,.3f}" in stock_path["hovertemplate"]
    app.selectbox(key="preset").set_value("Custom").run()
    app.date_input(key=f"dates_{tmp_path}_Custom").set_value(
        (dt.date(2023, 1, 7), dt.date(2023, 1, 8))
    ).run()
    assert not app.exception and app.info
    app.date_input(key=f"dates_{tmp_path}_Custom").set_value(
        (dt.date(2023, 1, 9), dt.date(2023, 1, 9))
    ).run()
    assert not app.exception
    assert app.metric[2].value == "—"
    app.segmented_control(key="page").set_value("Stock detail").run()
    app.selectbox(key="stock").set_value("B").run()
    traces = json.loads(app.get("plotly_chart")[0].proto.spec)["data"]
    # Monday selection opens on the actual preceding Friday; first P&L is retained.
    assert all(trace["x"] == ["2023-01-06", "2023-01-09"] for trace in traces)
    assert all(trace["y"][0] == 0.0 for trace in traces)
    assert traces[0]["y"][-1] == pytest.approx(
        (a if app.selectbox(key="stock").value == "A" else b)[
            dates.index(dt.date(2023, 1, 9))
        ]
        * 15
    )
    app.segmented_control(key="page").set_value("Overview").run()
    spec = json.loads(app.get("plotly_chart")[0].proto.spec)
    net = next(trace for trace in spec["data"] if trace["name"] == "Net")
    assert net["y"] == pytest.approx(
        [
            0,
            returns.filter(pl.col("date") == dt.date(2023, 1, 9))[
                "long_short_net"
            ].item()
            * 15,
        ]
    )
    drawdown = next(t for t in spec["data"] if t["name"] == "Total net drawdown")
    assert spec["layout"]["xaxis"]["matches"] == "x2"
    assert drawdown["yaxis"] == "y2"
    expected = realized.fixed_notional_path(returns).filter(
        pl.col("date").is_in([dt.date(2023, 1, 6), dt.date(2023, 1, 9)])
    )
    assert drawdown["y"] == pytest.approx((expected["drawdown"] * 15).to_list())
    # Two differently configured strategies must not share dates or money scaling.
    other = tmp_path / "other"
    other.mkdir()
    result.assets.drop("gross_weight").with_columns(
        pl.col("date").dt.offset_by("2y"),
        pl.col("asset_id").replace({"A": "C", "B": "D"}),
    ).write_parquet(other / "assets.parquet")
    result.daily.with_columns(pl.col("date").dt.offset_by("2y")).write_parquet(
        other / "daily.parquet"
    )
    (other / "manifest.json").write_text(
        json.dumps({"portfolio": {"name": "Second strategy", "notional": 2e6}})
    )
    config = tmp_path / "config.toml"
    config.write_text(
        '[[book]]\nlabel = "First"\nkind = "historical"\ndir = "."\n'
        "default_start = 2023-01-09\ndefault_end = 2023-01-09\n"
        '[[book]]\nlabel = "Second"\nkind = "historical"\ndir = "other"\n'
    )
    app = testing.AppTest.from_string(
        f"import attribution_dashboard.realized_page as page\npage.render_config({str(config)!r})",
        default_timeout=30,
    ).run()
    assert not app.exception
    assert app.selectbox(key="preset").value == "Saved period"
    assert app.metric[2].value == "—"
    app.selectbox(key="units").set_value("Money (millions)").run()
    app.segmented_control(key="page").set_value("Stock detail").run()
    app.selectbox(key="stock").set_value("B").run()
    app.selectbox(key="strategy").set_value("Second").run()
    assert not app.exception
    assert app.selectbox(key="preset").value == "Latest year"
    assert float(app.metric[0].value) == pytest.approx(total * 2, abs=0.00051)
    assert app.selectbox(key="stock").value in {"C", "D"}
    assert app.date_input[0].value[0] >= dt.date(2024, 12, 1)
    assert any("weights were not supplied" in element.value for element in app.caption)
    stock_path = json.loads(app.get("plotly_chart")[0].proto.spec)["data"][0]
    assert stock_path["y"][-1] == pytest.approx(
        (sum(a) if app.selectbox(key="stock").value == "C" else sum(b)) * 2
    )


def test_configured_missing_strategy_remains_visible(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[[book]]\nlabel = "Unbuilt strategy"\nkind = "historical"\ndir = "not-built"\n'
    )
    app = testing.AppTest.from_string(
        f"import attribution_dashboard.realized_page as page\npage.render_config({str(path)!r})"
    ).run()
    assert not app.exception
    assert app.selectbox(key="strategy").value == "Unbuilt strategy"
    assert app.error and "incomplete" in app.error[0].value
