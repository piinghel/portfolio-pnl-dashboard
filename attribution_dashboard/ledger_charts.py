"""Readable Plotly displays; legend actions never change attribution calculations."""

from __future__ import annotations

import datetime as dt

import plotly.graph_objects as go
import plotly.subplots as subplots
import polars as pl
import streamlit as st

import attribution_dashboard.chart_settings as visual


def palette() -> dict[str, str]:
    """Return stable colors shared by P&L, exposure and risk views."""
    return {
        "Long": "#3275a8",
        "Long exposure": "#3275a8",
        "long": "#3275a8",
        "Short": "#c65d36",
        "Short exposure": "#c65d36",
        "short": "#c65d36",
        "Net": "#182f42",
        "Net dollars": "#68737d",
        "costs": "#42484e",
        "Costs": "#42484e",
        "Intercept": "#68737d",
        "Universe intercept": "#68737d",
        "Size": "#3275a8",
        "Momentum": "#20766b",
        "Beta": "#9c755f",
        "Reversal": "#b05a91",
        "Sectors": "#56a4a5",
        "Volatility": "#c65d36",
        "Residual": "#8167a9",
        "Drawdown": "#c65d36",
        "Unexplained": "#8167a9",
        "Reconciliation": "#9b9b9b",
        "Unmodeled": "#9a773b",
        "Communications": "#4c78a8",
        "Consumer Discretionary": "#f58518",
        "Consumer Staples": "#e45756",
        "Energy": "#54a24b",
        "Financials": "#b279a2",
        "Health Care": "#9d755d",
        "Industrials": "#d7ad35",
        "Materials": "#8a6da9",
        "Real Estate": "#79706e",
        "Technology": "#d67195",
        "Utilities": "#72a5c9",
    }


def line_figure(
    frame: pl.DataFrame,
    *,
    value: str = "value",
    title: str,
    label: str = "series",
    key: str | None = None,
    points: bool = False,
    opening_date: dt.date | None = None,
    opening_value: float | None = None,
    trim_warmup: bool = False,
    settings: visual.ChartSettings = visual.DEFAULT_CHARTS,
) -> go.Figure:
    """Build paths with native legends, explicit gaps and an optional display baseline.

    ``opening_date`` is the preceding ledger session when available. Otherwise,
    an opening marker uses the calendar day before the first plotted session.
    It is a display anchor, never a return observation or a change to ``frame``.
    ``opening_value`` applies only to cumulative paths (zero for P&L, one for an
    indexed dollar). Exposure and risk paths retain their observed starting values.
    """
    first_date = frame["date"].min()
    if opening_value is not None and first_date is not None:
        if not isinstance(first_date, dt.date):
            raise TypeError("Cumulative paths require calendar dates")
        opening_date = opening_date or first_date - dt.timedelta(days=1)
        if opening_date >= first_date:
            raise ValueError("Opening marker must precede the first plotted session")
    if trim_warmup:
        first = frame.filter(pl.col(value).is_not_null())["date"].min()
        if first is not None:
            frame = frame.filter(pl.col("date") >= first)
    figure = go.Figure()
    colors = palette()
    for (name,), rows in frame.sort("date").partition_by(label, as_dict=True).items():
        observed = rows[value].drop_nulls()
        dates = rows["date"].cast(pl.String).to_list()
        values = rows[value].to_list()
        descriptions = ["" for _ in values]
        if opening_value is not None and opening_date is not None:
            dates.insert(0, opening_date.isoformat())
            values.insert(0, opening_value)
            descriptions.insert(0, "Opening baseline<br>")
        figure.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                customdata=descriptions,
                name=name,
                legendrank=0 if name == "Net" else 1000,
                mode="lines+markers" if points or len(observed) == 1 else "lines",
                connectgaps=False,
                line={
                    "color": colors.get(name),
                    "width": 3 if name == "Net" else 2,
                    "dash": "dot" if name in {"Costs", "costs"} else "solid",
                },
                marker={"size": 5},
                hovertemplate=f"%{{customdata}}%{{y:,.{settings.pnl_decimals}f}}<extra>%{{fullData.name}}</extra>",
            )
        )
    _layout(figure, title, settings.line_height, settings)
    figure.update_layout(hovermode="x unified", uirevision=key or title)
    dates = frame["date"].unique().sort()
    if opening_value is not None and opening_date is not None:
        dates = pl.concat([pl.Series([opening_date], dtype=pl.Date), dates])
    date_values = dates.to_list()
    days = (date_values[-1] - date_values[0]).days if date_values else 0
    figure.update_xaxes(
        range=[date_values[0], date_values[-1]] if len(date_values) > 1 else None,
        type="date",
        nticks=7,
        showgrid=False,
        tickformat="%Y" if days > 730 else "%b %Y" if days > 180 else "%b %d",
        hoverformat="%b %d, %Y",
    )
    if len(dates) <= 8:
        figure.update_xaxes(
            tickmode="array",
            tickvals=dates.cast(pl.String).to_list(),
            ticklabeloverflow="allow",
        )
    return figure


def lines(frame: pl.DataFrame, **kwargs) -> None:
    """Render a line figure with native legend interactions."""
    _show(line_figure(frame, **kwargs), kwargs.get("key"))


def bars(
    frame: pl.DataFrame,
    *,
    value: str,
    title: str,
    settings: visual.ChartSettings = visual.DEFAULT_CHARTS,
) -> None:
    """Compare signed amounts, retaining full names on hover and a zero baseline."""
    rows = frame.sort(value)
    names = rows["name"].to_list()
    short = [_short_label(name, settings.label_length) for name in names]
    figure = go.Figure(
        go.Bar(
            x=rows[value].to_list(),
            y=list(range(rows.height)),
            orientation="h",
            marker_color=[
                "#20766b" if amount >= 0 else "#c65d36" for amount in rows[value]
            ],
            customdata=names,
            text=[f"{amount:+.2f}" for amount in rows[value]],
            textposition="auto",
            insidetextfont={"color": "white"},
            outsidetextfont={"color": "#37424a"},
            cliponaxis=False,
            hovertemplate=f"%{{customdata}}<br>%{{x:,.{settings.pnl_decimals}f}}<extra></extra>",
        )
    )
    _layout(
        figure, title, max(210, rows.height * settings.bar_row_height + 85), settings
    )
    figure.update_yaxes(
        tickmode="array",
        tickvals=list(range(rows.height)),
        ticktext=short,
        showgrid=False,
        zeroline=False,
    )
    figure.update_xaxes(rangemode="tozero", nticks=6, zeroline=True)
    figure.update_layout(showlegend=False, margin={"l": 0, "r": 55, "t": 45, "b": 30})
    _show(figure)


def _layout(
    figure: go.Figure, title: str, height: int, settings: visual.ChartSettings
) -> None:
    # Keep a usable plotting area when optional sector legends wrap on laptops.
    legend_rows = (len(figure.data) + 1) // 2
    legend_allowance = max(0, legend_rows - 2) * (settings.font_size + 12)
    figure.update_layout(
        height=height + legend_allowance,
        template="plotly_white",
        colorway=[
            "#4e79a7",
            "#f28e2b",
            "#e15759",
            "#76b7b2",
            "#59a14f",
            "#b07aa1",
            "#9c755f",
            "#edc948",
            "#af7aa1",
            "#79706e",
            "#d37295",
        ],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title={
            "text": title,
            "x": 0,
            "xanchor": "left",
            "font": {"size": settings.font_size + 2},
        },
        font={
            "size": settings.font_size,
            "family": "Arial, sans-serif",
            "color": "#37424a",
        },
        margin={"l": 40, "r": 40, "t": 45, "b": 45},
        legend={
            "orientation": "h",
            "font": {"size": settings.font_size},
            "y": -0.2,
            "yanchor": "top",
            "x": 0,
            "itemclick": "toggle",
            "itemdoubleclick": "toggleothers",
            "title": None,
        },
        hoverlabel={"font_size": settings.font_size},
    )
    figure.update_xaxes(title_text=None, automargin=True, zerolinecolor="#a5adb3")
    figure.update_yaxes(
        title_text=None,
        automargin=True,
        zerolinecolor="#a5adb3",
        nticks=6,
    )


def _show(figure: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(
        figure,
        width="stretch",
        theme=None,
        key=key,
        config={"displaylogo": False, "scrollZoom": False},
    )


def stock_labels(frame: pl.DataFrame) -> pl.DataFrame:
    """Keep chart labels short; append the ID only to ambiguous name/side pairs."""
    return frame.with_columns(
        pl.concat_str(
            "label",
            pl.lit(" · "),
            "side",
            pl.when(pl.col("asset_id").n_unique().over("label", "side") > 1)
            .then(pl.concat_str(pl.lit(" · "), "asset_id"))
            .otherwise(pl.lit("")),
        ).alias("name")
    )


def _short_label(name: str, limit: int) -> str:
    if len(name) <= limit:
        return name
    parts = name.split(" · ")
    suffix = " · " + parts[1] if len(parts) > 1 else ""
    return parts[0][: max(1, limit - len(suffix) - 1)] + "…" + suffix


def pnl_drawdown(
    frame: pl.DataFrame,
    drawdown: pl.DataFrame,
    *,
    unit: str,
    key: str,
    opening_date: dt.date | None = None,
    settings: visual.ChartSettings = visual.DEFAULT_CHARTS,
) -> go.Figure:
    """Align selected cumulative contributions with historical total-net drawdown."""
    top = line_figure(
        frame,
        title=f"Cumulative P&L ({unit})",
        key=key,
        opening_date=opening_date,
        opening_value=0.0,
        settings=settings,
    )
    figure = subplots.make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.10
    )
    for trace in top.data:
        figure.add_trace(trace, row=1, col=1)
    figure.add_trace(
        go.Scatter(
            x=drawdown["date"].cast(pl.String).to_list(),
            y=drawdown["value"].to_list(),
            name="Total net drawdown",
            mode="lines",
            showlegend=False,
            line={"color": "#c65d36", "width": 1.7},
            fill="tozeroy",
            fillcolor="rgba(198,93,54,0.12)",
            connectgaps=False,
            hovertemplate=f"%{{y:,.{settings.pnl_decimals}f}}<extra>Total net drawdown</extra>",
        ),
        row=2,
        col=1,
    )
    _layout(figure, f"Cumulative P&L ({unit})", settings.pnl_drawdown_height, settings)
    axis = top.layout.xaxis.to_plotly_json()
    figure.update_xaxes(**axis)
    figure.update_xaxes(matches="x2", showticklabels=False, row=1, col=1)
    figure.update_layout(hovermode="x unified", uirevision=key, legend={"y": -0.10})
    figure.add_annotation(
        text=f"<b>Total net P&L drawdown ({unit})</b>",
        x=0,
        y=0.32,
        xref="paper",
        yref="paper",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
    )
    figure.update_yaxes(rangemode="tozero", row=2, col=1)
    return figure


def paired_bars(
    frame: pl.DataFrame,
    *,
    unit: str,
    risk_value: str,
    risk_title: str,
    settings: visual.ChartSettings = visual.DEFAULT_CHARTS,
) -> None:
    """Compare P&L and covariance risk using identical rows and distinct units."""
    rows = frame.sort("P&L", "name")
    figure = subplots.make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.10,
        subplot_titles=(f"P&L ({unit})", risk_title),
    )
    for col, field in enumerate(("P&L", risk_value), start=1):
        values = rows[field].to_list()
        figure.add_trace(
            go.Bar(
                x=values,
                y=list(range(rows.height)),
                orientation="h",
                marker_color=[
                    "#9b9b9b"
                    if v is None
                    else "#4e79a7"
                    if field == risk_value
                    else "#20766b"
                    if v >= 0
                    else "#c65d36"
                    for v in values
                ],
                customdata=rows["name"].to_list(),
                hovertemplate=f"%{{customdata}}<br>%{{x:,.{settings.pnl_decimals}f}}<extra></extra>",
            ),
            row=1,
            col=col,
        )
    _layout(figure, "", max(240, rows.height * settings.bar_row_height + 85), settings)
    figure.update_layout(showlegend=False, margin={"l": 0, "r": 25, "t": 45, "b": 30})
    figure.update_yaxes(
        tickmode="array",
        tickvals=list(range(rows.height)),
        ticktext=[_short_label(n, settings.label_length) for n in rows["name"]],
        showgrid=False,
        zeroline=False,
    )
    figure.update_xaxes(rangemode="tozero", nticks=5, zeroline=True)
    _show(figure)
