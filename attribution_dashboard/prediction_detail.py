"""Point-in-time linear prediction explanations, separate from realized P&L."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st

import attribution_dashboard.factor_data as data
import attribution_dashboard.prediction_history as history


def validate(decisions: pl.DataFrame, contributions: pl.DataFrame) -> None:
    """Reject incomplete or inconsistent linear decompositions."""
    keys = ["date", "asset_id", "side"]
    if (
        decisions.select(pl.any_horizontal(pl.all().is_null()).any()).item()
        or contributions.select(pl.any_horizontal(pl.all().is_null()).any()).item()
    ):
        raise ValueError("Prediction snapshots must be complete.")
    invalid_selection = decisions.filter(
        ~pl.col("side").is_in(["long", "short"])
        | ~pl.col("rank").is_between(1, pl.col("universe_size"))
        | ~pl.col("selection_count").is_between(1, pl.col("universe_size"))
        | (pl.col("selected") != (pl.col("rank") <= pl.col("selection_count")))
    )
    if invalid_selection.height:
        raise ValueError("Invalid rank or selection decision.")
    signed_margin = (
        pl.when(pl.col("side") == "long")
        .then(pl.col("score") - pl.col("cutoff"))
        .otherwise(pl.col("cutoff") - pl.col("score"))
    )
    if decisions.filter(
        (pl.col("selected") & (signed_margin < -1e-10))
        | (~pl.col("selected") & (signed_margin > 1e-10))
    ).height:
        raise ValueError("Selection decision contradicts the saved cutoff.")
    if (
        decisions.select(keys).is_duplicated().any()
        or contributions.select(*keys, "predictor").is_duplicated().any()
    ):
        raise ValueError("Duplicate prediction keys.")
    for frame, columns in [
        (decisions, ["score", "intercept", "cutoff"]),
        (contributions, ["input_value", "coefficient", "contribution"]),
    ]:
        if frame.select(
            pl.any_horizontal(
                [pl.col(c).is_null() | ~pl.col(c).is_finite() for c in columns]
            ).any()
        ).item():
            raise ValueError("Prediction values must be finite and complete.")
    if contributions.filter(
        (pl.col("input_value") * pl.col("coefficient") - pl.col("contribution")).abs()
        > 1e-10
    ).height:
        raise ValueError("Predictor contributions do not equal input × coefficient.")
    totals = (
        contributions.lazy()
        .group_by(keys)
        .agg(pl.col("contribution").sum().alias("total"))
    )
    checked = (
        decisions.lazy()
        .join(totals, on=keys, how="full", coalesce=True, validate="1:1")
        .collect()
    )
    if checked.filter(
        pl.col("total").is_null()
        | pl.col("score").is_null()
        | ((pl.col("intercept") + pl.col("total") - pl.col("score")).abs() > 1e-10)
    ).height:
        raise ValueError("Prediction components do not reconcile to the saved score.")


@st.cache_data(max_entries=8, ttl=300, show_spinner=False)
def _read(
    directory: Path, security: str, stamps: tuple[tuple[int, int], ...]
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    del stamps
    frames = [
        pl.scan_parquet(directory / f"{name}.parquet")
        .filter(pl.col("asset_id") == security)
        .collect()
        for name in ("decisions", "contributions")
    ]
    validate(*frames)
    metadata = json.loads((directory / "manifest.json").read_text())
    if not isinstance(metadata, dict) or any(
        not isinstance(metadata.get(key), str) or not metadata[key].strip()
        for key in ("model", "description", "selection_rule", "timing")
    ):
        raise ValueError(
            "Prediction metadata must describe the model, selection rule and timing."
        )
    return *frames, metadata


def load(
    directory: Path, security: str
) -> tuple[pl.DataFrame, pl.DataFrame, dict] | None:
    folder = directory / "predictions"
    if not folder.exists():
        return None
    paths = [
        folder / name
        for name in ("decisions.parquet", "contributions.parquet", "manifest.json")
    ]
    if not all(path.is_file() for path in paths):
        raise ValueError("The saved prediction bundle is incomplete.")
    return _read(folder, security, tuple(data.stamp(path) for path in paths))


def clicked_decision(
    points: list[dict], available: set[tuple[str, str]]
) -> tuple[str, str] | None:
    """Only marker payloads with an exact saved snapshot may open a decision."""
    for point in reversed(points):
        payload = point.get("customdata")
        if isinstance(payload, (list, tuple)) and len(payload) == 3:
            date, side, event = payload
            if (
                isinstance(date, str)
                and isinstance(side, str)
                and event in ("Entry", "Exit", "Already held")
                and (date, side) in available
            ):
                return date, side
    return None


def waterfall(decision: dict, rows: pl.DataFrame, limit: int) -> go.Figure:
    ranked = rows.sort(pl.col("contribution").abs(), descending=True)
    shown = ranked.head(limit)
    names = ["Intercept", *shown["predictor"].to_list()]
    values = [decision["intercept"], *shown["contribution"].to_list()]
    if ranked.height > limit:
        names.append(f"Other {ranked.height - limit} predictors")
        values.append(ranked.slice(limit)["contribution"].sum())
    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            y=[*names, "Prediction score"],
            x=[*values, 0],
            measure=["absolute", *(["relative"] * (len(names) - 1)), "total"],
            text=[f"{value:+.3f}" for value in [*values, decision["score"]]],
            textposition="outside",
            cliponaxis=False,
            increasing={"marker": {"color": "#20766b"}},
            decreasing={"marker": {"color": "#c65d36"}},
            totals={"marker": {"color": "#182f42"}},
            connector={"line": {"color": "#c9cfd3", "width": 1}},
            hovertemplate="%{y}: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=max(320, (len(names) + 1) * 34),
        margin={"l": 10, "r": 65, "t": 10, "b": 35},
        font={"size": 13, "color": "#37424a"},
        showlegend=False,
    )
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.update_xaxes(
        title="Prediction score units", zeroline=True, zerolinecolor="#9aa5ad"
    )
    return fig


def _dismiss() -> None:
    # A fresh selection widget lets the same point fire again after dismissal.
    st.session_state["prediction_selection_revision"] = (
        st.session_state.get("prediction_selection_revision", 0) + 1
    )


@st.dialog("Why this position?", width="large", on_dismiss=_dismiss)
def show(decision: dict, rows: pl.DataFrame, metadata: dict, label: str) -> None:
    st.markdown(f"**{label} · {decision['side'].title()} · {decision['date']}**")
    st.caption(metadata["model"])
    with st.container(horizontal=True):
        st.metric("Prediction score", f"{decision['score']:+.3f}")
        st.metric("Selection rank", f"{decision['rank']} / {decision['universe_size']}")
        st.metric("Cutoff score", f"{decision['cutoff']:+.3f}")
    direction = "highest" if decision["side"] == "long" else "lowest"
    outcome = "Selected" if decision["selected"] else "Not selected"
    st.write(
        f"{outcome}: this book takes the {decision['selection_count']} {direction} scores in its eligible universe."
    )
    limit = st.segmented_control(
        "Largest contributions", [5, 10], default=5, required=True, key="prediction_top"
    )
    st.plotly_chart(
        waterfall(decision, rows, limit),
        theme=None,
        config={"displaylogo": False},
        key="prediction_waterfall",
    )
    st.caption(
        "Green raises the score; orange lowers it. Lower scores favour shorts. Correlated predictors can share the same information; these are model contributions, not causal explanations or realized P&L."
    )
    with st.expander("Inputs and model details"):
        st.write(metadata["model"])
        st.write(metadata["description"])
        st.write(metadata["selection_rule"])
        st.write(metadata["timing"])
        st.caption(
            "Score = intercept + Σ (model input × coefficient). Inputs use the exact scaling applied by the model at that decision."
        )
        st.dataframe(
            rows.select("predictor", "input_value", "coefficient", "contribution").sort(
                pl.col("contribution").abs(), descending=True
            ),
            hide_index=True,
            column_config={
                "predictor": "Predictor",
                **{
                    key: st.column_config.NumberColumn(name, format="%.4f")
                    for key, name in [
                        ("input_value", "Model input"),
                        ("coefficient", "Coefficient"),
                        ("contribution", "Contribution"),
                    ]
                },
            },
        )
        st.download_button(
            "Download prediction inputs",
            rows.write_csv(),
            f"{decision['asset_id']}-{decision['date']}-prediction.csv",
            "text/csv",
        )


def controls(
    bundle: tuple[pl.DataFrame, pl.DataFrame, dict],
    security: str,
    label: str,
    start: dt.date,
    end: dt.date,
    *,
    xaxis: dict | None = None,
) -> None:
    decisions, contributions, metadata = bundle
    visible = decisions.filter(pl.col("date").is_between(start, end)).sort(
        "date", descending=True
    )
    requested = st.session_state.pop("open_prediction", None)
    if visible.is_empty():
        st.caption("No saved prediction decisions in this period.")
        return
    options = [
        (r["date"].isoformat(), r["side"]) for r in visible.iter_rows(named=True)
    ]
    st.caption(
        "Click an entry or exit marker to see its saved prediction, or choose a rebalance below."
    )
    with st.container(horizontal=True, vertical_alignment="bottom"):
        choice = st.selectbox(
            "Decision date",
            options,
            format_func=lambda item: f"{item[0]} · {item[1].title()}",
            key=f"decision_{security}_{start}_{end}",
        )
        if st.button("Explain decision"):
            requested = (security, *choice)
    history.render(contributions, security, choice[1], start, end, xaxis=xaxis)
    if requested and requested[0] == security and tuple(requested[1:]) in options:
        date, side = dt.date.fromisoformat(requested[1]), requested[2]
        decision = visible.filter(
            (pl.col("date") == date) & (pl.col("side") == side)
        ).row(0, named=True)
        rows = contributions.filter((pl.col("date") == date) & (pl.col("side") == side))
        show(decision, rows, metadata, label)
