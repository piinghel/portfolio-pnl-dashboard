"""Exercise factor drilldowns, period rebasing and missing forecast sessions."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl
import pytest
import streamlit.testing.v1 as testing


def _bundle(tmp_path: Path) -> Path:
    folder = tmp_path / "factors"
    folder.mkdir()
    dates = [dt.date(2024, 1, day) for day in (2, 3, 4)]
    daily = pl.DataFrame(
        [
            (day, factor, value, exposure)
            for day in dates
            for factor, value, exposure in [
                ("market", 0.003, 0.1),
                ("size", -0.002, -0.2),
                ("idio_pnl", 0.004, None),
                ("costs", -0.001, None),
                ("price_basis_gap", 0.0, None),
                ("unmodeled_pnl", 0.0, None),
            ]
        ],
        schema=["date", "factor", "pnl", "exposure"],
        orient="row",
    )
    daily.write_parquet(folder / "daily.parquet")
    daily.group_by("date").agg(
        pl.col("pnl").sum().alias("long_short_net")
    ).write_parquet(tmp_path / "daily.parquet")
    stocks = pl.DataFrame(
        [
            (day, asset, side, "Same label", "Sector", pnl + idio, pnl, idio, 0.0, 0.0)
            for day in dates
            for asset, side, pnl, idio in [
                ("A", "long", 0.002, 0.006),
                ("B", "short", -0.001, -0.002),
            ]
        ],
        schema=[
            "date",
            "asset_id",
            "side",
            "label",
            "sector",
            "asset_pnl",
            "factor_pnl",
            "idio_pnl",
            "price_basis_gap",
            "unmodeled_pnl",
        ],
        orient="row",
    )
    stocks.write_parquet(folder / "stocks.parquet")
    factors = pl.DataFrame(
        [
            (day, asset, side, factor, value)
            for day in dates
            for asset, side, factor, value in [
                ("A", "long", "market", 0.003),
                ("A", "long", "size", -0.001),
                ("B", "short", "market", 0.0),
                ("B", "short", "size", -0.001),
            ]
        ],
        schema=["date", "asset_id", "view", "factor", "pnl"],
        orient="row",
    )
    factors.write_parquet(folder / "asset_factors.parquet")
    pl.DataFrame(
        [
            (day, factor, value)
            for day in (dates[0], dates[2])
            for factor, value in [("market", -0.01), ("size", 0.02), ("idio", 0.03)]
        ],
        schema=["date", "factor", "risk_vol"],
        orient="row",
    ).write_parquet(folder / "risk.parquet")
    pl.DataFrame(
        {
            "date": dates,
            "gross_weight": [1.0, 1.0, 1.0],
            "pnl_coverage": [1.0, 1.0, 1.0],
            "risk_coverage": [1.0, 0.8, 1.0],
            "exposure_coverage": [1.0, 1.0, 1.0],
        }
    ).write_parquet(folder / "coverage.parquet")
    (folder / "manifest.json").write_text(json.dumps({"description": "Test model"}))
    return tmp_path


def _app(folder: Path) -> testing.AppTest:
    return testing.AppTest.from_string(
        "import datetime as dt\nfrom pathlib import Path\nimport streamlit as st\n"
        "import attribution_dashboard.factor_panels as panels\nimport polars as pl\n"
        "period = st.date_input('Dates', (dt.date(2024, 1, 2), dt.date(2024, 1, 4)), key='dates')\n"
        "scale = st.selectbox('Units', [100., 15.], key='units')\n"
        f"net = pl.read_parquet(Path({str(folder)!r}) / 'daily.parquet').filter(pl.col('date').is_between(*period))\n"
        f"panels.render(Path({str(folder)!r}), period[0], period[1], scale, 'test units', net_daily=net)\n",
        default_timeout=30,
    ).run()


def test_factor_controls_rebase_and_keep_stock_identity(tmp_path: Path) -> None:
    app = _app(_bundle(tmp_path))
    assert not app.exception
    traces = json.loads(app.get("plotly_chart")[0].proto.spec)["data"]
    assert next(t["y"][-1] for t in traces if t["name"] == "Net") == pytest.approx(1.2)
    app.segmented_control(key="factor_view").set_value("Stock drivers").run()
    assert not app.exception
    app.selectbox(key="factor_component").set_value("size").run()
    assert sum(
        json.loads(app.get("plotly_chart")[0].proto.spec)["data"][0]["x"]
    ) == pytest.approx(-0.6)
    # Two securities with identical names retain distinct IDs and the side sign.
    assert set(app.selectbox(key="factor_stock").options) == {
        "Same label · A",
        "Same label · B",
    }
    app.selectbox(key="factor_stock").set_value("B").run()
    app.selectbox(key="units").set_value(15.0).run()
    assert not app.exception
    assert sum(
        json.loads(app.get("plotly_chart")[0].proto.spec)["data"][0]["x"]
    ) == pytest.approx(-0.09)
    app.selectbox(key="factor_component").set_value("idio_pnl").run()
    assert sum(
        json.loads(app.get("plotly_chart")[0].proto.spec)["data"][0]["x"]
    ) == pytest.approx(0.18)
    app.date_input(key="dates").set_value(
        (dt.date(2024, 1, 4), dt.date(2024, 1, 4))
    ).run()
    assert not app.exception
    assert sum(
        json.loads(app.get("plotly_chart")[0].proto.spec)["data"][0]["x"]
    ) == pytest.approx(0.06)
    trace = json.loads(app.get("plotly_chart")[-1].proto.spec)["data"][0]
    assert trace["x"] == ["2024-01-03", "2024-01-04"]
    assert trace["y"] == pytest.approx([0, -0.03])
    app.date_input(key="dates").set_value(
        (dt.date(2025, 1, 1), dt.date(2025, 1, 2))
    ).run()
    assert not app.exception
    assert app.info and "No factor observations" in app.info[0].value


def test_forecast_risk_keeps_missing_sessions_and_units(tmp_path: Path) -> None:
    folder = _bundle(tmp_path)
    path = folder / "factors" / "daily.parquet"
    pl.read_parquet(path).filter(
        ~(
            (pl.col("date") == dt.date(2024, 1, 3))
            & pl.col("factor").is_in(["market", "size"])
        )
    ).with_columns(
        pl.when(
            (pl.col("date") == dt.date(2024, 1, 3))
            & (pl.col("factor") == "unmodeled_pnl")
        )
        .then(0.001)
        .otherwise(pl.col("pnl"))
        .alias("pnl")
    ).write_parquet(path)
    app = _app(folder)
    pnl_traces = json.loads(app.get("plotly_chart")[0].proto.spec)["data"]
    assert next(
        trace["y"] for trace in pnl_traces if trace["name"] == "Net"
    ) == pytest.approx([0.4, 0.8, 1.2])
    app.segmented_control(key="factor_view").set_value("Exposure and risk").run()
    assert not app.exception
    # Verify the emitted chart contract, not just that the view did not crash.
    charts = app.get("plotly_chart")
    exposure = json.loads(charts[0].proto.spec)
    assert all(trace["connectgaps"] is False for trace in exposure["data"])
    assert "Intercept" not in [trace["name"] for trace in exposure["data"]]
    assert sum(trace["y"].count(None) for trace in exposure["data"]) == 1
    dollars = json.loads(charts[1].proto.spec)
    assert dollars["layout"]["title"]["text"] == "Net dollars (% notional)"
    assert dollars["data"][0]["y"] == [10.0, None, 10.0]
    assert all(len(trace["x"]) == 3 for trace in exposure["data"])
    risk_spec = json.loads(charts[-1].proto.spec)
    assert all(trace["connectgaps"] is False for trace in risk_spec["data"])
    values = [v for trace in risk_spec["data"] for v in trace["y"]]
    assert values.count(None) == 3
    assert sorted(set(v for v in values if v is not None)) == [-1.0, 2.0, 3.0]
    app.selectbox(key="units").set_value(15.0).run()
    assert not app.exception
    assert json.loads(app.get("plotly_chart")[-1].proto.spec) == risk_spec
    app.date_input(key="dates").set_value(
        (dt.date(2024, 1, 3), dt.date(2024, 1, 3))
    ).run()
    assert not app.exception
    assert not any("Forecast risk" in item.value for item in app.subheader)


def test_missing_bundle_is_optional(tmp_path: Path) -> None:
    pl.DataFrame(
        {"date": [dt.date(2024, 1, 2)], "long_short_net": [0.0]}
    ).write_parquet(tmp_path / "daily.parquet")
    app = _app(tmp_path)
    assert not app.exception
    assert app.info and "not been built" in app.info[0].value


def test_signed_composition_preserves_cash_and_risk_units(tmp_path: Path) -> None:
    folder = _bundle(tmp_path)
    path = folder / "factors" / "daily.parquet"
    first = pl.read_parquet(path).filter(pl.col("date") == dt.date(2024, 1, 2))
    daily = pl.concat(
        [
            first.with_columns(
                pl.lit(dt.date(2024, 1, 2) + dt.timedelta(days=i)).alias("date"),
                (pl.col("pnl") * (1 + (i % 7) / 10)).alias("pnl"),
            )
            for i in range(80)
        ]
    )
    daily.write_parquet(path)
    daily.group_by("date").agg(
        pl.col("pnl").sum().alias("long_short_net")
    ).write_parquet(folder / "daily.parquet")
    app = _app(folder)
    app.date_input(key="dates").set_value(
        (dt.date(2024, 3, 1), dt.date(2024, 3, 21))
    ).run()
    assert not app.exception
    spec = json.loads(app.get("plotly_chart")[0].proto.spec)
    pnl = [t for t in spec["data"] if t["type"] == "bar"]
    risks = [t for t in spec["data"] if t.get("yaxis") == "y3"]
    net = next(t for t in spec["data"] if t["name"] == "Net")
    assert sum(sum(t["y"]) for t in pnl) == pytest.approx(net["y"][-1])
    assert net["y"][0] == 0
    assert net["x"][0] == "2024-02-29"
    assert all(
        all(v is None for v in values) or sum(values) == pytest.approx(100.0)
        for values in zip(*(t["y"] for t in risks), strict=True)
    )
    assert any(v is not None and v < 0 for t in risks for v in t["y"])
    assert spec["layout"]["xaxis"]["matches"] == "x3"
    app.selectbox(key="units").set_value(15.0).run()
    changed = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert [t["y"] for t in changed["data"] if t.get("yaxis") == "y3"] == [
        t["y"] for t in risks
    ]
    assert next(t for t in changed["data"] if t["name"] == "Net")["y"][
        -1
    ] == pytest.approx(net["y"][-1] * 0.15)
    app.multiselect(key="factor_focus_selection").set_value(["Size"]).run()
    focused = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert {t["name"] for t in focused["data"]} == {
        "Size",
        "Other components combined",
        "Net",
    }
    focus_risk = [t for t in focused["data"] if t.get("yaxis") == "y3"]
    assert all(
        all(v is None for v in values) or sum(values) == pytest.approx(100.0)
        for values in zip(*(t["y"] for t in focus_risk), strict=True)
    )
    for frequency in ["Weekly", "Monthly", "Quarterly", "Yearly"]:
        app.selectbox(key="factor_frequency").set_value(frequency).run()
        assert not app.exception
        traces = json.loads(app.get("plotly_chart")[0].proto.spec)["data"]
        assert (
            next(t for t in traces if t["name"] == "Net")["y"]
            == next(t for t in focused["data"] if t["name"] == "Net")["y"]
        )
        assert [t["y"] for t in traces if t.get("yaxis") == "y3"] == [
            t["y"] for t in focus_risk
        ]
        assert sum(sum(t["y"]) for t in traces if t["type"] == "bar") == pytest.approx(
            net["y"][-1] * 0.15
        )
    # Add before estimating risk: the selected sum remains a partition of net.
    app.multiselect(key="factor_focus_selection").set_value(["Size", "Intercept"]).run()
    assert not app.exception
    combined = json.loads(app.get("plotly_chart")[0].proto.spec)["data"]
    selected = next(
        t for t in combined if t["name"] == "Intercept + Size" and t.get("yaxis") == "y"
    )
    assert selected["y"][-1] == pytest.approx(net["y"][-1] * 0.15 / 4)
    shares = next(
        t
        for t in combined
        if t["name"] == "Intercept + Size" and t.get("yaxis") == "y3"
    )
    assert all(v is None or v == pytest.approx(25.0) for v in shares["y"])
    assert next(t for t in combined if t["name"] == "Net")["y"] == pytest.approx(
        next(t for t in focused["data"] if t["name"] == "Net")["y"]
    )
    app.multiselect(key="factor_focus_selection").set_value([]).run()
    assert not app.exception
    overview = json.loads(app.get("plotly_chart")[0].proto.spec)["data"]
    assert "Modeled factors" in {t["name"] for t in overview}


def test_stale_factor_bundle_is_rejected_before_display(tmp_path: Path) -> None:
    folder = _bundle(tmp_path)
    parent = pl.read_parquet(folder / "daily.parquet").with_columns(
        (pl.col("long_short_net") + 0.01).alias("long_short_net")
    )
    parent.write_parquet(folder / "daily.parquet")
    app = _app(folder).run()
    assert not app.exception
    assert "Rebuild the factor bundle" in app.error[0].value
    assert len(app.dataframe) == 0


def test_grouped_sectors_keep_net_and_exposure_detail(tmp_path: Path) -> None:
    folder = _bundle(tmp_path)
    path = folder / "factors" / "daily.parquet"
    original = pl.read_parquet(path)
    sectors = pl.concat(
        [
            original.filter(pl.col("factor") == "size").with_columns(
                pl.lit(name).alias("factor")
            )
            for name in ("sector:Energy", "sector:Industrials")
        ]
    )
    daily = pl.concat([original, sectors])
    daily.write_parquet(path)
    daily.group_by("date").agg(
        pl.col("pnl").sum().alias("long_short_net")
    ).write_parquet(folder / "daily.parquet")
    app = _app(folder)
    assert not app.exception
    app.multiselect(key="factor_focus_selection").set_value(["Sectors"]).run()
    grouped = {
        trace["name"]: trace["y"]
        for trace in json.loads(app.get("plotly_chart")[0].proto.spec)["data"]
        if trace.get("yaxis") == "y"
    }
    assert grouped["Sectors"][-1] == pytest.approx(-1.2)
    assert grouped["Net"][-1] == pytest.approx(0)
    assert "Energy" not in grouped
    app.segmented_control(key="factor_view").set_value("Exposure and risk").run()
    assert not app.exception
    exposures = json.loads(app.get("plotly_chart")[0].proto.spec)["data"]
    assert all(trace["name"] not in {"Energy", "Industrials"} for trace in exposures)
    sector_chart = next(
        chart
        for chart in app.get("plotly_chart")
        if "Sector exposure (% notional)" in chart.proto.spec
    )
    sector_traces = json.loads(sector_chart.proto.spec)["data"]
    assert {trace["name"] for trace in sector_traces} == {"Energy", "Industrials"}
    assert all(trace["y"] == [-20, -20, -20] for trace in sector_traces)


def test_covered_exposure_is_explicit_and_never_scaled_to_full_book(
    tmp_path: Path,
) -> None:
    folder = _bundle(tmp_path)
    path = folder / "factors" / "daily.parquet"
    pl.read_parquet(path).with_columns(
        pl.col("exposure").alias("covered_exposure"),
        pl.lit(None, dtype=pl.Float64).alias("exposure"),
    ).write_parquet(path)
    path = folder / "factors" / "coverage.parquet"
    pl.read_parquet(path).with_columns(
        pl.lit(0.8).alias("exposure_coverage")
    ).write_parquet(path)
    app = _app(folder)
    app.segmented_control(key="factor_view").set_value("Exposure and risk").run()
    assert not app.exception
    assert not any(
        widget.key == "factor_exposure_scope" for widget in app.segmented_control
    )
    traces = json.loads(app.get("plotly_chart")[0].proto.spec)["data"]
    assert next(trace["y"] for trace in traces if trace["name"] == "Size") == [
        -0.2,
        -0.2,
        -0.2,
    ]
    assert any(
        "average 80.0%; minimum 80.0%" in caption.value for caption in app.caption
    )
