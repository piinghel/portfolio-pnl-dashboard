"""Original prices and recorded holding boundaries for the stock drilldown."""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import plotly.graph_objects as go
import plotly.subplots as subplots
import polars as pl
import streamlit as st

import attribution_dashboard.accounting.stock_history as stock_history
import attribution_dashboard.chart_settings as visual
import attribution_dashboard.factor_data as data
import attribution_dashboard.ledger_charts as charts
import attribution_dashboard.linear_history as linear_history
import attribution_dashboard.prediction_detail as predictions
import attribution_dashboard.prediction_history as prediction_history


@st.cache_data(max_entries=2, ttl=300, show_spinner=False)
def _calendar(path: Path, stamp: tuple[int, int]) -> pl.DataFrame:
    del stamp
    return pl.scan_parquet(path).select("date").unique().sort("date").collect()


@st.cache_data(max_entries=8, ttl=300, show_spinner=False)
def _read_stock(
    directory: Path, security: str, stamps: tuple[tuple[int, int], ...]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    del stamps
    quotes = (
        pl.scan_parquet(directory / "prices.parquet")
        .filter(pl.col("asset_id") == security)
        .sort("date")
        .collect()
    )
    holdings = (
        pl.scan_parquet(directory / "positions.parquet")
        .filter(pl.col("asset_id") == security)
        .collect()
    )
    calendar = _calendar(
        directory / "positions.parquet", data.stamp(directory / "positions.parquet")
    )
    return quotes, stock_history.position_events(holdings, calendar)


def render(
    directory: Path,
    security: str,
    start: dt.date,
    end: dt.date,
    *,
    contributions: pl.DataFrame,
    label: str,
    scale: float,
    unit: str,
    opening_date: dt.date | None,
    settings: visual.ChartSettings = visual.DEFAULT_CHARTS,
) -> bool:
    """Align adjusted prices, cumulative gross P&L and marked position sizes."""
    paths = [directory / f"{name}.parquet" for name in ("prices", "positions")]
    if not all(path.is_file() for path in paths):
        st.info(
            "Original prices and recorded holdings were not supplied for this ledger."
        )
        return False
    quotes, events = _read_stock(
        directory, security, tuple(data.stamp(path) for path in paths)
    )
    with st.container(horizontal=True):
        basis = st.segmented_control(
            "Stock price",
            ["Adjusted close", "Original close"],
            default="Adjusted close",
            required=True,
            key="stock_price_basis",
        )
        price_scale = st.segmented_control(
            "Price scale",
            ["Linear", "Log"],
            default="Linear",
            required=True,
            key="stock_price_scale",
            help="Log scale gives equal percentage moves equal vertical spacing.",
        )
    column = "px_last_unadjusted" if basis == "Original close" else "px_last"
    visible = (
        quotes.lazy()
        .filter(pl.col("date").is_between(opening_date or start, end))
        .collect()
    )
    if visible.is_empty():
        st.info("No saved stock prices in this selection.")
        return False
    log_price = price_scale == "Log"
    if log_price and (visible[column].drop_nulls() <= 0).any():
        st.info(
            "Log scale requires positive prices. Showing linear scale for this selection."
        )
        log_price = False
    currencies = visible["price_currency"].drop_nulls().unique().to_list()
    currency = "/".join(sorted(currencies)) or "source currency"
    selected = (
        events.lazy()
        .filter(pl.col("date").is_between(start, end))
        .join(
            quotes.lazy().select("date", pl.col(column).alias("price")),
            on="date",
            how="left",
            validate="m:1",
        )
        .collect()
    )
    try:
        prediction_bundle = predictions.load(directory, security)
    except (OSError, ValueError, pl.exceptions.PolarsError) as error:
        st.warning(f"Prediction explanation unavailable: {error}")
        prediction_bundle = None
    explanations = (
        {
            (
                r["date"].isoformat(),
                r["side"],
            ): f"Score {r['score']:+.3f} · rank {r['rank']}/{r['universe_size']}<br>Click to explain this decision"
            for r in prediction_bundle[0].iter_rows(named=True)
        }
        if prediction_bundle is not None
        else {}
    )
    figure = subplots.make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.45, 0.30, 0.25],
        vertical_spacing=0.08,
        subplot_titles=(
            f"{label} · {basis.lower()} ({currency})",
            f"Gross cumulative P&L ({unit})",
            "Position size (% notional)",
        ),
    )
    price = charts.line_figure(
        visible.select(
            "date", pl.col(column).alias("value"), pl.lit(basis).alias("series")
        ),
        title="",
        settings=settings,
    )
    price.data[0].line.color = "#182f42"
    price.data[0].showlegend = False
    figure.add_trace(price.data[0], row=1, col=1)
    pnl = charts.line_figure(
        contributions.select(
            "date",
            (pl.col("asset_pnl").cum_sum() * scale).alias("value"),
            pl.lit("Gross stock P&L").alias("series"),
        ),
        title="",
        opening_date=opening_date,
        opening_value=0.0,
        settings=settings,
    )
    pnl.data[0].line.color = "#20766b"
    pnl.data[0].showlegend = False
    figure.add_trace(pnl.data[0], row=2, col=1)
    for name, color in [("Long exposure", "#3275a8"), ("Short exposure", "#c65d36")]:
        if name not in contributions.columns:
            continue
        figure.add_trace(
            go.Scatter(
                x=contributions["date"].cast(pl.String).to_list(),
                y=(contributions[name] * 100).to_list(),
                name=name.replace(" exposure", " position"),
                mode="lines",
                line={"color": color, "shape": "hv", "width": 1.5},
                fill="tozeroy",
                fillcolor="rgba(50,117,168,.16)"
                if name.startswith("Long")
                else "rgba(198,93,54,.16)",
                hovertemplate="%{y:.3f}%<extra>%{fullData.name}</extra>",
            ),
            row=3,
            col=1,
        )
    _add_holding_markers(
        figure,
        selected,
        settings.stock_guide_limit,
        log_price=log_price,
        explanations=explanations,
    )
    figure.update_layout(
        template="plotly_white",
        height=settings.stock_history_height,
        font={"size": settings.font_size, "color": "#37424a"},
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 15, "r": 20, "t": 30, "b": 55},
        hovermode="closest" if prediction_bundle is not None else "x unified",
        hoversubplots="axis",
        clickmode="event+select",
        uirevision=f"stock_{security}_{start}_{end}",
        legend={
            "orientation": "h",
            "y": -0.10,
            "x": 0,
            "itemclick": "toggle",
            "itemdoubleclick": "toggleothers",
        },
    )
    axis = pnl.layout.xaxis.to_plotly_json()
    prediction_history.align_date_axis(figure, axis)
    figure.update_xaxes(matches="x3", showticklabels=False, row=1, col=1)
    figure.update_xaxes(matches="x3", showticklabels=False, row=2, col=1)
    figure.update_xaxes(
        showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1
    )
    figure.update_yaxes(rangemode="tozero", row=2, col=1)
    figure.update_yaxes(rangemode="tozero", row=3, col=1)
    figure.update_yaxes(
        type="log" if log_price else "linear",
        minorloglabels="complete",
        dtick="D2"
        if log_price
        and visible[column].min() is not None
        and visible[column].max() / visible[column].min() >= 3
        else None,
        autorange=True,
        uirevision=f"{security}_{basis}_{log_price}",
        row=1,
        col=1,
    )
    figure.update_yaxes(gridcolor="#e6e9ec", zerolinecolor="#a5adb3")
    revision = st.session_state.get("prediction_selection_revision", 0)
    chart_key = f"stock_price_{security}_{start}_{end}_{revision}"

    def open_decision() -> None:
        points = st.session_state[chart_key].get("selection", {}).get("points", [])
        choice = predictions.clicked_decision(points, set(explanations))
        if choice:
            st.session_state["open_prediction"] = (security, *choice)

    st.plotly_chart(
        figure,
        width="stretch",
        theme=None,
        key=chart_key,
        on_select=open_decision if prediction_bundle is not None else "ignore",
        selection_mode="points",
        config={"displaylogo": False},
    )
    st.caption(
        "▲ Entry · ▼ Exit · dotted guides align the panels."
        if selected.height <= settings.stock_guide_limit
        else "▲ Entry · ▼ Exit · select a shorter period to show event labels and guides."
    )
    if prediction_bundle is not None:
        predictions.controls(prediction_bundle, security, label, start, end, xaxis=axis)
    elif (directory / "linear_history.json").exists():
        try:
            rows = linear_history.load(directory, security, start, end)
            prediction_history.render(
                rows,
                security,
                "model",
                start,
                end,
                xaxis=axis,
                events=selected
                if selected.height <= settings.stock_guide_limit
                else None,
            )
        except (OSError, ValueError, pl.exceptions.PolarsError) as error:
            st.warning(f"Predictor history unavailable: {error}")
    elif not (directory / "predictions").exists():
        st.caption("Prediction breakdowns have not been supplied for this portfolio.")
    with st.expander("Holding dates and price definitions"):
        st.caption(
            "Markers show holding boundaries, not execution fills. Resizing is not an entry; exits mark the first flat session. Positions already open at the start are not shown as new entries."
        )
        st.caption(
            "Exposure is closing position value as a share of fixed notional. Averages include flat days; End is the final selected session. Adjusted close is the ledger's price basis; original close can jump at corporate actions. Price changes alone do not equal P&L."
        )
        missing = selected["price"].null_count()
        if missing:
            st.caption(
                f"{missing} boundaries lack a source price; the download retains them without an invented marker."
            )
        st.download_button(
            "Download holding boundaries",
            selected.write_csv(),
            f"{security}-holding-boundaries.csv",
            "text/csv",
        )
    return True


def _add_holding_markers(
    figure: go.Figure,
    selected: pl.DataFrame,
    guide_limit: int,
    *,
    log_price: bool = False,
    explanations: dict[tuple[str, str], str] | None = None,
) -> None:
    """Align holding boundaries across price, P&L and position panels."""
    for (side, event), rows in selected.partition_by(
        "side", "event", as_dict=True
    ).items():
        color = "#3275a8" if side == "long" else "#c65d36"
        figure.add_trace(
            go.Scatter(
                x=rows["date"].cast(pl.String).to_list(),
                y=rows["price"].to_list(),
                name=f"{side.title()} · {event.lower()}",
                mode="markers",
                customdata=[[date.isoformat(), side, event] for date in rows["date"]],
                text=[
                    (explanations or {}).get(
                        (date.isoformat(), side),
                        "No saved prediction for this boundary",
                    )
                    for date in rows["date"]
                ],
                marker={
                    "symbol": "triangle-up"
                    if event == "Entry"
                    else "triangle-down"
                    if event == "Exit"
                    else "circle-open",
                    "size": 12,
                    "color": color,
                    "line": {"width": 1, "color": "white"},
                },
                hovertemplate="%{x}<br>%{y:,.2f}<br>%{text}<extra>%{fullData.name}</extra>",
            ),
            row=1,
            col=1,
        )
        if selected.height <= guide_limit:
            for date, price in rows.select("date", "price").iter_rows():
                if price is None or (log_price and price <= 0):
                    continue
                figure.add_annotation(
                    x=date.isoformat(),
                    y=math.log10(price) if log_price else price,
                    xref="x",
                    yref="y",
                    text=f"{side.title()} {event.lower()}",
                    showarrow=True,
                    arrowhead=0,
                    arrowcolor=color,
                    ax=0,
                    ay=-32 if event == "Entry" else 32,
                    font={"size": 11, "color": color},
                    bgcolor="rgba(255,255,255,0.9)",
                    borderpad=2,
                    xanchor="center",
                )
        for date in rows["date"] if selected.height <= guide_limit else []:
            figure.add_shape(
                type="line",
                x0=date.isoformat(),
                x1=date.isoformat(),
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line={"color": color, "width": 1, "dash": "dot"},
                opacity=0.3,
            )
