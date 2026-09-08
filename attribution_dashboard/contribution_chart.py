"""Aligned cumulative rewards, period rewards and daily realized risk shares."""

from __future__ import annotations

import datetime as dt

import plotly.graph_objects as go
import plotly.subplots as subplots
import polars as pl

import attribution_dashboard.chart_settings as visual
import attribution_dashboard.ledger_charts as charts


def figure(
    periods: pl.DataFrame,
    cumulative: pl.DataFrame,
    trailing: pl.DataFrame,
    *,
    scale: float,
    unit: str,
    frequency: str,
    settings: visual.ChartSettings = visual.DEFAULT_CHARTS,
    opening_date: dt.date | None = None,
) -> go.Figure:
    """Link accumulated P&L to period gains/losses on the same real date axis.

    Inputs already contain the complete chosen component grouping. ``cumulative``
    contains cumulative P&L, ``periods`` contains additive bucket P&L, and
    ``trailing`` contains daily signed covariance shares. No regrouping,
    normalization or risk estimation occurs here.
    """
    first, last = cumulative["date"].min(), cumulative["date"].max()
    if not isinstance(first, dt.date) or not isinstance(last, dt.date):
        raise ValueError("The contribution timeline requires cumulative Date rows")
    if opening_date is not None and opening_date >= first:
        raise ValueError("The opening date must precede the first contribution")
    left = opening_date or first - dt.timedelta(days=1)
    colors = {
        **charts.palette(),
        "Modeled factors": "#3275a8",
        "Residual": "#8064a7",
        "Unmodeled": "#b49c7e",
        "Other components combined": "#b7c1c8",
    }
    names = sorted(
        cumulative["factor"].unique().to_list(),
        key=lambda name: (
            name
            in {
                "Other components combined",
                "Residual",
                "Unmodeled",
                "Costs",
            },
            name,
        ),
    )
    result = subplots.make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.4, 0.3, 0.3],
        vertical_spacing=0.08,
        subplot_titles=(
            f"Cumulative P&L ({unit})",
            f"{frequency} P&L ({unit})",
            f"{settings.contribution_risk_window}-day variance share (%)",
        ),
    )
    _add_daily_paths(
        result, cumulative, trailing, names, colors, scale, unit, opening_date
    )
    _add_period_bars(result, periods, names, colors, scale, unit, left)
    _style_timeline(result, left, last, settings)
    if trailing["variance_share"].drop_nulls().is_empty():
        result.add_annotation(
            text="Insufficient history or zero net variance",
            x=0.5,
            y=0.12,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
    return result


def _add_daily_paths(
    result: go.Figure,
    cumulative: pl.DataFrame,
    trailing: pl.DataFrame,
    names: list[str],
    colors: dict[str, str],
    scale: float,
    unit: str,
    opening_date: dt.date | None,
) -> None:
    """Keep component colors aligned and preserve the first return and risk gaps."""
    for name in names:
        color = colors.get(name, "#3275a8")
        for panel, frame, column, multiplier in [
            (1, cumulative, "pnl", scale),
            (3, trailing, "variance_share", 100.0),
        ]:
            rows = frame.filter(pl.col("factor") == name).sort("date")
            dates = rows["date"].cast(pl.String).to_list()
            values = (rows[column] * multiplier).to_list()
            if panel == 1 and opening_date is not None:
                dates.insert(0, opening_date.isoformat())
                values.insert(0, 0.0)
            result.add_trace(
                go.Scatter(
                    x=dates,
                    y=values,
                    name=name,
                    legendgroup=name,
                    showlegend=panel == 1,
                    connectgaps=False,
                    mode="lines+markers" if len(dates) == 1 else "lines",
                    line={
                        "color": color,
                        "width": 2,
                        "dash": "dot" if name == "Costs" else "solid",
                    },
                    marker={"size": 5},
                    hovertemplate=f"{name}: %{{y:.3f}} {unit if panel == 1 else '%'}<extra></extra>",
                ),
                row=panel,
                col=1,
            )
    net = cumulative.group_by("date").agg(pl.col("pnl").sum()).sort("date")
    dates, values = (
        net["date"].cast(pl.String).to_list(),
        (net["pnl"] * scale).to_list(),
    )
    if opening_date is not None:
        dates.insert(0, opening_date.isoformat())
        values.insert(0, 0.0)
    result.add_trace(
        go.Scatter(
            x=dates,
            y=values,
            name="Net",
            legendrank=0,
            mode="lines+markers" if len(dates) == 1 else "lines",
            line={"color": "#182f42", "width": 3},
            marker={"size": 6},
            hovertemplate=f"Net: %{{y:.3f}} {unit}<extra></extra>",
        ),
        row=1,
        col=1,
    )


def _add_period_bars(
    result: go.Figure,
    periods: pl.DataFrame,
    names: list[str],
    colors: dict[str, str],
    scale: float,
    unit: str,
    left: dt.date,
) -> None:
    """Fit bars inside observed periods, with no extension past the last session.

    A one-session bucket uses the preceding calendar day as its left display
    boundary, clipped to the chart start. Its hover retains the actual session.
    """
    for name in names:
        rows = periods.filter(pl.col("factor") == name).sort("date")
        centers, widths, bounds = [], [], []
        for row in rows.iter_rows(named=True):
            start, end = row["start"], row["date"]
            display_start = (
                max(left, end - dt.timedelta(days=1)) if start == end else start
            )
            first = dt.datetime.combine(display_start, dt.time())
            last = dt.datetime.combine(end, dt.time())
            centers.append(first + (last - first) / 2)
            widths.append((last - first).total_seconds() * 1000 * 0.85)
            bounds.append([str(start), str(end), row["sessions"]])
        result.add_trace(
            go.Bar(
                x=centers,
                y=(rows["pnl"] * scale).to_list(),
                width=widths,
                name=name,
                legendgroup=name,
                showlegend=False,
                marker_color=colors.get(name, "#3275a8"),
                customdata=bounds,
                hovertemplate=f"{name}: %{{y:.3f}} {unit}"
                "<br>%{customdata[0]} → %{customdata[1]}"
                "<br>%{customdata[2]} sessions<extra></extra>",
            ),
            row=2,
            col=1,
        )


def _style_timeline(
    result: go.Figure, first: dt.date, last: dt.date, settings: visual.ChartSettings
) -> None:
    """Share real dates while keeping return units separate from risk shares."""
    result.update_layout(
        height=settings.contribution_height,
        barmode="relative",
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 45, "r": 20, "t": 35, "b": 65},
        font={
            "family": "Arial, sans-serif",
            "size": settings.font_size,
            "color": "#37424a",
        },
        legend={
            "orientation": "h",
            "y": -0.10,
            "x": 0,
            "itemclick": False,
            "itemdoubleclick": False,
        },
        hovermode="x unified",
        hoversubplots="axis",
    )
    days = (last - first).days
    result.update_xaxes(
        type="date",
        range=[first.isoformat(), last.isoformat()],
        nticks=7,
        tickformat="%Y" if days > 730 else "%b %Y" if days > 180 else "%b %d",
        hoverformat="%b %d, %Y",
        showgrid=False,
        tickfont={"size": settings.font_size - 1},
    )
    result.update_xaxes(matches="x3", showticklabels=False, row=1, col=1)
    result.update_xaxes(matches="x3", showticklabels=False, row=2, col=1)
    result.update_yaxes(
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="#818b92",
        gridcolor="#edf0f2",
        nticks=4,
        rangemode="tozero",
    )
    result.add_hline(y=100, line_dash="dot", line_color="#bac2c8", row=3, col=1)
