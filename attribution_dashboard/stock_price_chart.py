"""Construct aligned stock price, P&L and position figures without I/O or UI state."""

from __future__ import annotations

import datetime as dt
import math
from typing import cast

import plotly.graph_objects as go
import plotly.subplots as subplots
import polars as pl

import attribution_dashboard.chart_settings as visual
import attribution_dashboard.ledger_charts as charts


def build(
    visible: pl.DataFrame,
    selected: pl.DataFrame,
    contributions: pl.DataFrame,
    *,
    security: str,
    label: str,
    basis: str,
    column: str,
    currency: str,
    start: dt.date,
    end: dt.date,
    scale: float,
    unit: str,
    opening_date: dt.date | None,
    log_price: bool,
    settings: visual.ChartSettings,
    explanations: dict[tuple[str, str], str] | None = None,
) -> tuple[go.Figure, dict]:
    """Return the three stock panels and the date axis for predictor alignment."""
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
        hovermode="closest" if explanations is not None else "x unified",
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
    figure.update_xaxes(**axis)
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
        and cast(float, visible[column].max()) / cast(float, visible[column].min()) >= 3
        else None,
        autorange=True,
        uirevision=f"{security}_{basis}_{log_price}",
        row=1,
        col=1,
    )
    figure.update_yaxes(gridcolor="#e6e9ec", zerolinecolor="#a5adb3")
    return figure, axis


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
