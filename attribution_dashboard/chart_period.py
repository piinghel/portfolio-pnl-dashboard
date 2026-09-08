"""Date selections change the analysis period, not just a chart's viewport."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence

import plotly.graph_objects as go
import streamlit as st


def selected_dates(
    selection: dict, calendar: Sequence[dt.date]
) -> tuple[dt.date, dt.date] | None:
    """Resolve a horizontal box against saved sessions, excluding display anchors."""
    boxes = selection.get("box", [])
    if not boxes:
        return None
    bounds = boxes[-1].get("x", [])
    if len(bounds) != 2:
        return None
    try:
        start, end = sorted(dt.date.fromisoformat(str(x)[:10]) for x in bounds)
    except (TypeError, ValueError):
        return None
    dates = [date for date in calendar if start <= date <= end]
    return (dates[0], dates[-1]) if dates else None


def queue(start: dt.date, end: dt.date) -> None:
    """Request one period for every view, preserving the original reset window."""
    context = st.session_state.get("analysis_context")
    if context is None or (start, end) == (context["start"], context["end"]):
        return
    origin = st.session_state.get("analysis_origin")
    if not origin or origin["directory"] != context["directory"]:
        st.session_state["analysis_origin"] = {
            "directory": context["directory"],
            "start": context["start"],
            "end": context["end"],
        }
    st.session_state["analysis_pending"] = {
        "directory": context["directory"],
        "start": start,
        "end": end,
    }


def reset() -> None:
    """Restore the period before the first chart selection."""
    origin = st.session_state.pop("analysis_origin", None)
    if origin is not None:
        st.session_state["analysis_pending"] = origin


def clear_origin() -> None:
    """A manually chosen sidebar period becomes the new starting point."""
    st.session_state.pop("analysis_origin", None)


def apply_pending(directory: str) -> None:
    """Apply queued dates before Streamlit creates the sidebar date widgets."""
    pending = st.session_state.pop("analysis_pending", None)
    if pending and pending["directory"] == directory:
        st.session_state["preset"] = "Custom"
        st.session_state[f"dates_{directory}_Custom"] = (
            pending["start"],
            pending["end"],
        )


def plot(
    figure: go.Figure,
    *,
    key: str | None = None,
    on_points: Callable[[list[dict]], None] | None = None,
    **options,
) -> None:
    """Link horizontal date selection; retain point actions such as stock decisions."""
    context = st.session_state.get("analysis_context")
    if context is None or figure.layout.xaxis.type != "date":
        st.plotly_chart(figure, key=key, **options)
        return
    mode = "points" if on_points is not None else "box"
    revision = f"{context['directory']}_{context['start']}_{context['end']}_{mode}"
    chart_key = f"{key or figure.layout.title.text or 'date_chart'}_{revision}"
    figure.update_layout(
        # Streamlit disables point-selection callbacks in box-drag mode.
        dragmode="pan" if on_points is not None else "select",
        selectdirection="h",
        selectionrevision=revision,
        uirevision=revision,
    )
    figure.update_xaxes(fixedrange=True)
    # Plotly only reports selections for selectable traces. Tiny markers retain
    # the continuous line appearance while allowing a drag on any line panel.
    for trace in figure.data:
        if trace.type in {"scatter", "scattergl"} and trace.mode == "lines":
            trace.update(mode="lines+markers", marker={"size": 2, "opacity": 0.15})
        elif trace.type == "heatmap" and len(trace.y):
            # Heatmaps cannot emit Plotly selections themselves. A transparent
            # hit target on their existing dates makes the same horizontal
            # gesture work here too, without changing cells or their hover data.
            figure.add_trace(
                go.Scatter(
                    x=list(trace.x),
                    y=[trace.y[0]] * len(trace.x),
                    xaxis=trace.xaxis,
                    yaxis=trace.yaxis,
                    mode="markers",
                    marker={"size": 1, "opacity": 0},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    def select() -> None:
        selection = st.session_state[chart_key].get("selection", {})
        dates = selected_dates(selection, context["calendar"])
        if dates is not None:
            queue(*dates)
        elif not selection.get("box") and on_points is not None:
            on_points(selection.get("points", []))

    config = dict(options.pop("config", {}))
    config.update(
        displaylogo=False,
        scrollZoom=False,
        doubleClick=False,
        modeBarButtonsToRemove=[
            "zoom2d",
            "pan2d",
            "zoomIn2d",
            "zoomOut2d",
            "autoScale2d",
            "resetScale2d",
            "lasso2d",
        ],
    )
    if on_points is not None:
        config["modeBarButtonsToRemove"].append("select2d")
    st.plotly_chart(
        figure,
        key=chart_key,
        on_select=select,
        selection_mode="points" if on_points is not None else "box",
        config=config,
        **options,
    )
