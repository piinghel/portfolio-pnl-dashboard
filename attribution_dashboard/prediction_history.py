"""Compact history of one stock's saved model inputs and contributions."""

from __future__ import annotations

import datetime as dt
import re
import textwrap

import plotly.graph_objects as go
import polars as pl
import streamlit as st


def _label(name: str) -> str:
    label = name.removeprefix("X_feature_")
    match = re.fullmatch(
        r"price_sharpe_ratio_compound_r(\d+)_volatility(\d+)_rolling", label
    )
    if match:
        return f"Trailing Sharpe ({match[1]}d / {match[2]}d)"
    for prefix, replacement in [
        ("price_price_to_min", "Price / minimum "),
        ("price_price_to_max", "Price / maximum "),
        ("price_price_to_ma", "Price / moving average "),
        ("market_cap_log_diff_", "Log market cap change "),
        ("price_trend_streak", "Trend streak "),
    ]:
        if label.startswith(prefix):
            label = replacement + label.removeprefix(prefix)
            break
    return label.replace("_", " ")


def ranked_predictors(rows: pl.DataFrame) -> list[str]:
    """Order predictors by average absolute contribution in the supplied history."""
    return (
        rows.lazy()
        .group_by("predictor")
        .agg(pl.col("contribution").abs().mean().alias("importance"))
        .sort(["importance", "predictor"], descending=[True, False])
        .collect()["predictor"]
        .to_list()
    )


def heatmap(
    rows: pl.DataFrame,
    metric: str,
    *,
    calendar_axis: bool = False,
    predictors: list[str] | None = None,
) -> go.Figure:
    """Keep one row order and one zero-centred colour scale across all dates."""
    names = ranked_predictors(rows)[:5] if predictors is None else predictors
    height = max(310, 44 * len(names) + 90)
    dates = sorted(rows["date"].unique().to_list())
    if calendar_axis and dates:
        observed = set(dates)
        dates = [
            date
            for day in range(dates[0].toordinal(), dates[-1].toordinal() + 1)
            if (date := dt.date.fromordinal(day)).weekday() < 5 or date in observed
        ]
    shown = rows.lazy().filter(pl.col("predictor").is_in(names)).collect()
    lookup = {
        (row["predictor"], row["date"]): row for row in shown.iter_rows(named=True)
    }
    cells = [[lookup.get((name, date)) for date in dates] for name in names]
    values = [[cell[metric] if cell else None for cell in row] for row in cells]
    extent = (
        max(
            (abs(value) for row in values for value in row if value is not None),
            default=0,
        )
        or 1
    )
    unit = "Score contribution" if metric == "contribution" else "Model input"
    fig = go.Figure(
        go.Heatmap(
            x=[date.isoformat() for date in dates],
            y=names,
            z=values,
            customdata=[
                [
                    [
                        f"{cell[field]:+.4f}"
                        for field in ("input_value", "coefficient", "contribution")
                    ]
                    if cell
                    else [None, None, None]
                    for cell in row
                ]
                for row in cells
            ],
            zmin=-extent,
            zmax=extent,
            colorscale=[[0, "#c65d36"], [0.5, "#f5f6f7"], [1, "#20766b"]],
            colorbar={
                "title": {"text": unit},
                "orientation": "h",
                "len": 0.5,
                "thickness": 9,
                "y": -65 / (height - 90),
            },
            ygap=2,
            hoverongaps=False,
            hovertemplate=(
                "%{y} · %{x}<br>Model input: %{customdata[0]}"
                "<br>Coefficient: %{customdata[1]}"
                "<br>Score contribution: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 165, "r": 10, "t": 5, "b": 85},
        font={"size": 12, "color": "#37424a"},
    )
    fig.update_yaxes(
        autorange="reversed",
        automargin=True,
        tickmode="array",
        tickvals=names,
        ticktext=["<br>".join(textwrap.wrap(_label(name), 26)) for name in names],
    )
    ticks = dates[:: max(1, (len(dates) + 4) // 5)]
    fig.update_xaxes(
        type="category",
        tickmode="array",
        tickvals=[date.isoformat() for date in ticks],
        ticktext=[
            date.strftime("%d %b %Y" if (dates[-1] - dates[0]).days < 180 else "%b %Y")
            for date in ticks
        ],
        tickangle=0,
    )
    if calendar_axis:
        fig.update_xaxes(
            type="date",
            tickmode="auto",
            nticks=5,
            tickformat="%d %b" if (dates[-1] - dates[0]).days < 180 else "%b %Y",
        )
    return fig


def render(
    contributions: pl.DataFrame,
    security: str,
    side: str,
    start: dt.date,
    end: dt.date,
    *,
    events: pl.DataFrame | None = None,
) -> None:
    rows = (
        contributions.lazy()
        .filter(
            (pl.col("asset_id") == security)
            & (pl.col("side") == side)
            & pl.col("date").is_between(start, end)
        )
        .collect()
    )
    if rows["date"].n_unique() < 2:
        st.caption(
            "Predictor history needs at least two saved decision dates in this period."
        )
        return
    with st.expander("Predictors over time", expanded=True):
        metric = st.segmented_control(
            "Heatmap values",
            ["Score contribution", "Model input"],
            default="Score contribution",
            required=True,
            key="prediction_history_metric",
            label_visibility="collapsed",
        )
        ranked = ranked_predictors(rows)
        preset = st.selectbox(
            "Predictors",
            ["Top 5", "Top 10", "Top 20", "Choose predictors"],
            key="prediction_history_selection",
        )
        if preset == "Choose predictors":
            names = st.multiselect(
                "Search and choose predictors",
                ranked,
                default=ranked[:5],
                format_func=_label,
                placeholder="Type a predictor name…",
                key=f"history_predictors_{security}_{side}",
                select_all=False,
            )
        else:
            names = ranked[: int(preset.split()[-1])]
        if not names:
            st.info("Choose at least one predictor to display its history.")
            return
        figure = heatmap(
            rows,
            "contribution" if metric == "Score contribution" else "input_value",
            calendar_axis=side == "model",
            predictors=names,
        )
        if side == "model":
            figure.update_xaxes(range=[start.isoformat(), end.isoformat()])
            if events is not None and events.height <= 30:
                for event in events.iter_rows(named=True):
                    figure.add_vline(
                        x=event["date"].isoformat(),
                        line_width=1,
                        line_dash="dot",
                        line_color="#3275a8",
                        opacity=0.45,
                    )
        st.plotly_chart(
            figure,
            theme=None,
            config={"displayModeBar": False},
            key=f"prediction_history_{security}_{side}",
        )
        st.caption(
            "Saved Ridge model · "
            "Daily normalized input × the coefficient from the model in use that day; intercept excluded. "
            "These explain the model score, not the optimizer's trades."
            if side == "model"
            else f"{side.title()} book · "
            "Each column is one saved decision, including dates outside holdings; gaps stay blank. "
            "Orange is negative, green positive. Inputs keep their saved model scaling."
        )
        if preset != "Choose predictors":
            st.caption(
                f"Showing {len(names)} predictors ranked by average absolute contribution in this period."
            )
