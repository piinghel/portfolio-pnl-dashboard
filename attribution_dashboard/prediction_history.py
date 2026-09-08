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
        margin={"l": 165, "r": 20, "t": 5, "b": 85, "autoexpand": False},
        font={"size": 12, "color": "#37424a"},
    )
    fig.update_yaxes(
        autorange="reversed",
        automargin=False,
        tickmode="array",
        tickvals=names,
        ticktext=["<br>".join(textwrap.wrap(_label(name), 23)) for name in names],
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


def align_date_axis(figure: go.Figure, xaxis: dict) -> None:
    """Use the same calendar and fixed plotting gutter for stock history panels."""
    figure.update_layout(margin_l=165, margin_r=20, margin_autoexpand=False)
    figure.update_xaxes(**xaxis)
    figure.update_yaxes(automargin=False)


def render(
    contributions: pl.DataFrame,
    security: str,
    side: str,
    start: dt.date,
    end: dt.date,
    *,
    events: pl.DataFrame | None = None,
    xaxis: dict | None = None,
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
    st.markdown("**Predictors over time**")
    with st.container():
        metric = st.segmented_control(
            "Heatmap values",
            ["Both", "Score contribution", "Model input"],
            default="Both",
            required=True,
            key="prediction_history_metric",
            label_visibility="collapsed",
        )
        ranked = ranked_predictors(rows)
        preset = st.selectbox(
            "Predictors",
            ["Top 5", "Top 10", "Top 20", "Choose predictors"],
            key="prediction_history_selection",
            help="Top predictors are ranked by average absolute score contribution in the selected period.",
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
        metrics = (
            [("contribution", "Score contribution"), ("input_value", "Model input")]
            if metric == "Both"
            else [("contribution", "Score contribution")]
            if metric == "Score contribution"
            else [("input_value", "Model input")]
        )
        for field, title in metrics:
            figure = heatmap(
                rows,
                field,
                calendar_axis=side == "model" or xaxis is not None,
                predictors=names,
            )
            if xaxis is not None:
                align_date_axis(figure, xaxis)
            elif side == "model":
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
            figure.update_layout(
                title={"text": title, "x": 0, "xref": "paper", "font": {"size": 14}},
                margin_t=35,
            )
            st.plotly_chart(
                figure,
                theme=None,
                config={"displayModeBar": False},
                key=f"prediction_history_{security}_{side}_{field}_{start}_{end}",
            )
        with st.expander("About these charts"):
            st.caption(
                "Saved Ridge model · Daily normalized input × the coefficient from "
                "the model in use that day; intercept excluded. These explain the "
                "model score, not the optimizer's trades."
                if side == "model"
                else f"{side.title()} book · Each column is one saved decision, "
                "including dates outside holdings; gaps stay blank. "
                "Inputs keep their saved model scaling."
            )
            st.caption(
                "Orange is negative; green is positive. Inputs and contributions "
                "have separate colour scales. Both panels keep the same predictor order. "
                "Use the date filter to compare the same period across all charts; "
                "chart zoom is independent."
            )
            if preset != "Choose predictors":
                st.caption(
                    f"Showing {len(names)} predictors ranked by average absolute contribution in this period."
                )
