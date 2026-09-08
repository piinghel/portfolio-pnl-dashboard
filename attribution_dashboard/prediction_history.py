"""Compact history of one stock's saved model inputs and contributions."""

from __future__ import annotations

import datetime as dt

import plotly.graph_objects as go
import polars as pl
import streamlit as st


def heatmap(rows: pl.DataFrame, metric: str) -> go.Figure:
    """Keep one row order and one zero-centred colour scale across all dates."""
    names = (
        rows.lazy()
        .group_by("predictor")
        .agg(pl.col("contribution").abs().mean().alias("importance"))
        .sort(["importance", "predictor"], descending=[True, False])
        .head(5)
        .collect()["predictor"]
        .to_list()
    )
    dates = sorted(rows["date"].unique().to_list())
    lookup = {
        (row["predictor"], row["date"]): row for row in rows.iter_rows(named=True)
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
                "y": -0.35,
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
        height=310,
        margin={"l": 10, "r": 10, "t": 5, "b": 85},
        font={"size": 12, "color": "#37424a"},
    )
    fig.update_yaxes(autorange="reversed", automargin=True)
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
    return fig


def render(
    contributions: pl.DataFrame, security: str, side: str, start: dt.date, end: dt.date
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
        st.plotly_chart(
            heatmap(
                rows,
                "contribution" if metric == "Score contribution" else "input_value",
            ),
            theme=None,
            config={"displayModeBar": False},
            key=f"prediction_history_{security}_{side}",
        )
        st.caption(
            f"{side.title()} book · Top five by average absolute contribution in the selected period. "
            "Each column is one saved decision, including dates outside holdings; gaps stay blank. "
            "Orange is negative, green positive. Inputs keep their saved model scaling."
        )
