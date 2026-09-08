"""Security price, holdings and contribution drilldowns."""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path

import polars as pl
import streamlit as st

import attribution_dashboard.accounting.realized as realized
import attribution_dashboard.accounting.realized_io as realized_io
import attribution_dashboard.accounting.realized_risk as risk
import attribution_dashboard.accounting.stock_history as stock_history
import attribution_dashboard.chart_settings as visual
import attribution_dashboard.factor_data as factor_data
import attribution_dashboard.ledger_charts as charts
import attribution_dashboard.stock_prices as stock_prices


def render(
    report: realized.RealizedPnlReport,
    scale: float,
    unit: str,
    *,
    directory: Path | None = None,
    opening_date: dt.date | None = None,
    settings: visual.ChartSettings = visual.DEFAULT_CHARTS,
) -> None:
    """Render one security on the complete portfolio calendar."""
    stocks = stock_history.summarize_stocks(report.assets, report.daily.select("date"))
    selected = st.session_state.get("stock")
    if (
        directory is not None
        and selected is not None
        and not st.session_state.get("reset_stock_detail", False)
        and selected not in stocks["asset_id"].to_list()
    ):
        identity = realized_io.read_stock_identity(directory, selected)
        if identity.is_empty():
            st.info("The selected stock is unavailable in this dataset.")
            return
        empty = identity.with_columns(
            pl.lit(0.0).alias("pnl"),
            *[
                pl.lit(
                    0.0 if "gross_weight" in report.assets.columns else None,
                    dtype=pl.Float64,
                ).alias(column)
                for column in (
                    "average_long",
                    "average_short",
                    "ending_long",
                    "ending_short",
                )
            ],
        ).select(stocks.columns)
        stocks = pl.concat([stocks, empty], how="vertical_relaxed")
    if stocks.is_empty():
        st.info("No stock positions in this period.")
        return
    names = dict(zip(stocks["asset_id"], stocks["label"], strict=True))
    labels = {
        r["asset_id"]: f"{r['label']} · {r['asset_id']}"
        for r in stocks.iter_rows(named=True)
    }
    options = stocks["asset_id"].to_list()
    if (
        st.session_state.pop("reset_stock_detail", False)
        or st.session_state.get("stock") not in options
    ):
        st.session_state["stock"] = options[0]
    chosen = st.selectbox(
        "Find a stock", options, format_func=lambda key: labels[key], key="stock"
    )
    selected_stock = stocks.filter(pl.col("asset_id") == chosen).row(0, named=True)
    if selected_stock["average_long"] is not None:
        st.caption(
            f"Stock P&L {selected_stock['pnl'] * scale:+.3f} {unit} · {selected_stock['sector']} · Average exposure: long {selected_stock['average_long']:z.2%}, short {selected_stock['average_short']:z.2%} · "
            f"End: long {selected_stock['ending_long']:z.2%}, short {selected_stock['ending_short']:z.2%}"
        )
    else:
        st.caption(
            f"Stock P&L {selected_stock['pnl'] * scale:+.3f} {unit} · {selected_stock['sector']}"
        )
    rows = report.assets.lazy().filter(pl.col("asset_id") == chosen)
    if not report.assets.filter(pl.col("asset_id") == chosen).height:
        st.caption(
            "No portfolio holdings in this period; price and model history remain visible."
        )
    has_exposure = "gross_weight" in report.assets.columns
    aggregations = [pl.col("asset_pnl").sum()]
    if has_exposure:
        aggregations.extend(
            [
                pl.col("gross_weight")
                .filter(pl.col("side") == "long")
                .sum()
                .alias("Long exposure"),
                (-pl.col("gross_weight").filter(pl.col("side") == "short").sum()).alias(
                    "Short exposure"
                ),
            ]
        )
    series = rows.group_by("date").agg(aggregations)
    frame = (
        report.daily.lazy()
        .select("date")
        .join(series, on="date", how="left")
        .fill_null(0)
        .sort("date")
        .collect()
    )
    rendered = directory is not None and stock_prices.render(
        directory,
        chosen,
        report.daily["date"][0],
        report.daily["date"][-1],
        contributions=frame,
        label=names[chosen],
        scale=scale,
        unit=unit,
        opening_date=opening_date,
        settings=settings,
    )
    if not rendered:
        charts.lines(
            frame.select(
                "date",
                (pl.col("asset_pnl").cum_sum() * scale).alias("value"),
                pl.lit(names[chosen]).alias("series"),
            ),
            title=f"Stock cumulative P&L ({unit})",
            opening_date=opening_date,
            opening_value=0.0,
            settings=settings,
        )
    if not has_exposure:
        st.caption(
            "Position weights were not supplied for this ledger; exposure is unavailable."
        )
    with st.expander("Stock risk and factor exposures"):
        _risk_detail(report, chosen, directory=directory, settings=settings)
    st.download_button(
        "Download stock daily history", frame.write_csv(), f"{chosen}.csv", "text/csv"
    )


def _risk_detail(
    report: realized.RealizedPnlReport,
    chosen: str,
    *,
    directory: Path | None,
    settings: visual.ChartSettings,
) -> None:
    window = st.selectbox(
        "Stock risk window (trading days)",
        [21, 63, 126, 252],
        index=1,
        key="stock_window",
    )
    has_holdings = report.assets.filter(pl.col("asset_id") == chosen).height > 0
    if not has_holdings:
        st.caption(
            "No portfolio risk contribution: this stock was not held in the selected period."
        )
    elif report.daily.height >= window:
        # Preserve every contribution while grouping all other stocks together.
        coarse = dataclasses.replace(
            report,
            assets=report.assets.lazy()
            .with_columns(
                pl.when(pl.col("asset_id") == chosen)
                .then(pl.lit("Selected stock"))
                .otherwise(pl.lit("Other stocks"))
                .alias("label")
            )
            .collect(),
        )
        trailing = risk.rolling_realized_risk(
            coarse, group_by=("label",), window=window
        )
        charts.lines(
            trailing.lazy()
            .filter(pl.col("label") == "Selected stock")
            .with_columns(
                (100 * pl.col("risk_vol")).alias("value"),
                pl.col("label").alias("series"),
            )
            .collect(),
            value="value",
            title=f"Stock {window}-day risk (vol pp)",
            trim_warmup=True,
            settings=settings,
        )
    else:
        st.info(
            f"At least {window} trading days are needed for the selected stock risk window."
        )
    if directory is not None:
        _factor_exposure(directory, chosen, report.daily["date"][-1])


@st.cache_data(max_entries=8, ttl=300, show_spinner=False)
def _read_exposure(
    path: Path, chosen: str, date: dt.date, stamp: tuple[int, int]
) -> pl.DataFrame:
    del stamp
    return (
        pl.scan_parquet(path)
        .filter((pl.col("asset_id") == chosen) & (pl.col("date") == date))
        .collect()
    )


def _factor_exposure(directory: Path, chosen: str, date: dt.date) -> None:
    path = directory / "factors" / "asset_factors.parquet"
    if not path.is_file():
        return
    rows = _read_exposure(path, chosen, date, factor_data.stamp(path))
    if rows.is_empty():
        st.info(
            f"No modeled stock exposures on {date}; the stock may be flat or lack model inputs."
        )
        return
    names = factor_data.names(rows["factor"].unique().to_list())
    display = (
        rows.lazy()
        .filter(
            ~pl.col("factor").str.starts_with("sector:")
            | (pl.col("factor_loading") != 0)
        )
        .group_by("factor")
        .agg(
            pl.col("factor_loading").first().alias("Descriptor loading"),
            pl.col("signed_factor_exposure").sum().alias("Signed portfolio exposure"),
        )
        .with_columns(pl.col("factor").replace(names).alias("Component"))
        .select(
            "Component",
            "Descriptor loading",
            "Signed portfolio exposure",
        )
        .collect()
    )
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            c: st.column_config.NumberColumn(c, format="%.3f")
            for c in display.columns
            if display.schema[c].is_float()
        },
    )
    st.caption(
        f"Model inputs for {date}: factor loading × signed beginning weight. Covered positions only; loading definitions follow the supplied model. Missing inputs are not zero."
    )


def factor_drivers(
    folder: Path,
    daily: pl.DataFrame,
    start: dt.date,
    end: dt.date,
    scale: float,
    unit: str,
    *,
    opening_date: dt.date | None = None,
    settings: visual.ChartSettings = visual.DEFAULT_CHARTS,
) -> None:
    names = factor_data.names(daily["factor"].unique().to_list())
    factors = sorted(
        daily.lazy()
        .filter(
            ~pl.col("factor").is_in(
                ["idio_pnl", "price_basis_gap", "unmodeled_pnl", "costs"]
            )
        )
        .select("factor")
        .unique()
        .collect()["factor"]
        .to_list()
    )
    options = factors + ["idio_pnl", "price_basis_gap", "unmodeled_pnl"]
    chosen = st.selectbox(
        "Explain this component",
        options,
        format_func=lambda value: names.get(value, value),
        key="factor_component",
        index=None if "factor_component" in st.session_state else 0,
    )
    if chosen is None:
        st.info("Choose a component to see its stock drivers.")
        return
    rows = factor_data.driver_rows(
        folder,
        start,
        end,
        chosen,
        factor_data.stamp(folder / "stocks.parquet"),
        factor_data.stamp(folder / "asset_factors.parquet"),
    )
    if rows.is_empty():
        st.info("No stock contributions for this component in the selected period.")
        return
    ranked = (
        rows.lazy()
        .group_by("asset_id", "label", "sector")
        .agg((pl.col("pnl").sum() * scale).alias("P&L"))
        .with_columns(pl.lit("total").alias("side"))
        .sort("P&L", "asset_id")
        .collect()
    )
    ranked = charts.stock_labels(ranked)
    bars = pl.concat([ranked.head(8), ranked.tail(8)]).unique().sort("P&L")
    charts.bars(
        bars,
        value="P&L",
        title=f"{names.get(chosen, chosen)} P&L ({unit})",
        settings=settings,
    )
    st.caption(
        "Both sides combined; up to eight stocks from each end. This is the selected component only, not total stock P&L."
    )
    labels = dict(zip(ranked["asset_id"], ranked["label"], strict=True))
    if st.session_state.pop("reset_factor_stock_detail", False):
        st.session_state["factor_stock"] = ranked["asset_id"][0]
    stock = st.selectbox(
        "Follow a stock",
        list(dict.fromkeys(ranked["asset_id"].to_list())),
        format_func=lambda value: f"{labels[value]} · {value}",
        key="factor_stock",
        index=None if "factor_stock" in st.session_state else 0,
    )
    if stock is None:
        st.info("Choose a stock to see its history.")
        return
    history = (
        daily.lazy()
        .select("date")
        .unique()
        .join(
            rows.lazy()
            .filter(pl.col("asset_id") == stock)
            .group_by("date")
            .agg(pl.col("pnl").sum()),
            on="date",
            how="left",
        )
        .with_columns(pl.col("pnl").fill_null(0))
        .sort("date")
        .with_columns(
            (pl.col("pnl").cum_sum() * scale).alias("value"),
            pl.lit(names.get(chosen, chosen)).alias("series"),
        )
        .collect()
    )
    charts.lines(
        history,
        value="value",
        title=f"{names.get(chosen, chosen)} P&L ({unit})",
        opening_date=opening_date,
        opening_value=0.0,
        settings=settings,
    )
    st.download_button(
        "Download stock contributions",
        ranked.write_csv(),
        "factor-stock-contributions.csv",
        "text/csv",
        key="factor_stock_download",
    )
