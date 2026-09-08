"""Reconciled period contributions and a common, readable waterfall display."""

from __future__ import annotations

import html
import textwrap

import plotly.graph_objects as go
import polars as pl

import attribution_dashboard.accounting.factor_risk as factor_risk
import attribution_dashboard.chart_settings as visual
import attribution_dashboard.factor_data as factor_data


def group_totals(assets: pl.DataFrame, group: str, side: str) -> pl.DataFrame:
    """Sum signed security P&L by classification and actual daily holding side.

    Classification changes stay on their recorded dates. Missing labels retain
    their P&L in Unclassified. Long and short are never renormalized.
    """
    column = {"Stocks": "asset_id", "Sectors": "sector", "Industries": "industry"}[
        group
    ]
    rows = assets.lazy()
    if side != "Combined":
        rows = rows.filter(pl.col("side") == side.lower())
    if column not in assets.columns:
        rows = rows.with_columns(pl.lit(None, pl.String).alias(column))
    return (
        rows.with_columns(
            pl.col(column).fill_null("Unclassified").replace("", "Unclassified")
        )
        .group_by(column)
        .agg(
            pl.col("label").first()
            if group == "Stocks"
            else pl.col(column).first().alias("label"),
            pl.col("asset_pnl").sum().alias("pnl"),
            pl.col("asset_pnl").filter(pl.col("side") == "long").sum().alias("long"),
            pl.col("asset_pnl").filter(pl.col("side") == "short").sum().alias("short"),
        )
        .rename({column: "id"})
        .sort("pnl", "id")
        .collect()
    )


def factor_totals(
    daily: pl.DataFrame,
    parent: pl.DataFrame,
    *,
    conventions: factor_data.FactorConventions = factor_data.DEFAULT_CONVENTIONS,
) -> pl.DataFrame:
    """Validate a complete factor partition before summarizing its gross drivers.

    ``parent.long_short_net`` is the comparison total: whole-book net P&L,
    or one side's gross P&L. Costs are displayed separately by the caller.
    Sector model terms form one group; residual and reconciliation stay explicit.
    """
    factor_risk.validate_factor_partition(daily, parent)
    labels = factor_data.names(
        daily["factor"].unique().to_list(), conventions=conventions
    )
    return (
        daily.lazy()
        .filter(pl.col("factor") != "costs")
        .with_columns(
            pl.when(conventions.sector_mask())
            .then(pl.lit("Sector effects"))
            .otherwise(pl.col("factor").replace(labels))
            .alias("label")
        )
        .group_by("label")
        .agg(pl.col("pnl").sum())
        .with_columns(pl.col("label").alias("id"))
        .sort("pnl", "id")
        .collect()
    )


def figure(
    contributions: pl.DataFrame,
    costs: float | None,
    scale: float,
    unit: str,
    *,
    group: str,
    side: str,
    limit: int | None = 10,
    settings: visual.ChartSettings = visual.DEFAULT_CHARTS,
) -> go.Figure:
    """Display signed drivers with wrapped names, complete totals and readable values.

    The largest absolute contributions are shown; an Other row retains every
    omitted contribution. Side totals are gross, with no invented cost allocation.
    """
    ranked = contributions.lazy().sort(pl.col("pnl").abs(), descending=True)
    shown = (
        (ranked.head(limit) if limit is not None else ranked)
        .sort("pnl", "id")
        .collect()
    )
    names = shown["label"].to_list()
    values = [float(v) * scale for v in shown["pnl"]]
    identifiers = shown["id"].to_list() if group == "Stocks" else [""] * shown.height
    notes = []
    for row in shown.iter_rows(named=True):
        notes.append(
            f"Long {row['long'] * scale:+,.{settings.pnl_decimals}f} · "
            f"Short {row['short'] * scale:+,.{settings.pnl_decimals}f}"
            + ("<br>Click to inspect stock" if group == "Stocks" else "")
            if "long" in row
            else "Model-based attribution"
        )
    if shown.height < contributions.height:
        names.append(f"Other {group.lower()}")
        values.append(
            (float(contributions["pnl"].sum()) - float(shown["pnl"].sum())) * scale
        )
        identifiers.append("")
        notes.append(f"Remaining {contributions.height - shown.height} {group.lower()}")
    if costs is not None:
        names.append("Trading costs")
        values.append(costs * scale)
        identifiers.append("")
        notes.append("Recorded whole-book costs")
    total = sum(values)
    names.append("Net P&L" if costs is not None else f"{side} gross P&L")
    identifiers.append("")
    notes.append("Sum of every contribution above")
    labels = [
        "<br>".join(html.escape(part) for part in textwrap.wrap(name, 28))
        for name in names
    ]
    formatted = [f"{value:+,.{settings.pnl_decimals}f}" for value in [*values, total]]
    result = go.Figure(
        go.Waterfall(
            orientation="h",
            y=list(range(len(names))),
            x=[*values, 0.0],
            measure=["relative"] * len(values) + ["total"],
            customdata=[
                [identifier, html.escape(name), value, note]
                for identifier, name, value, note in zip(
                    identifiers, names, formatted, notes, strict=True
                )
            ],
            decreasing={"marker": {"color": "#c65d36"}},
            increasing={"marker": {"color": "#20766b"}},
            totals={"marker": {"color": "#182f42"}},
            connector={"line": {"color": "#c7cdd1", "width": 1}},
            hovertemplate="%{customdata[1]}<br>%{customdata[2]} "
            + unit
            + "<br>%{customdata[3]}<extra></extra>",
            text=formatted,
            textposition="outside",
            cliponaxis=False,
            textfont={"size": settings.font_size},
        )
    )
    # Outside labels need space beyond every intermediate step, not just the net.
    levels = [0.0]
    for value in values:
        levels.append(levels[-1] + value)
    low, high = min(levels), max(levels)
    span = high - low or 1.0
    row_height = max(
        34,
        settings.bar_row_height,
        max(label.count("<br>") + 1 for label in labels) * 19 + 10,
    )
    result.update_layout(
        height=row_height * len(names) + 65,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        margin={"l": 10, "r": 20, "t": 8, "b": 45},
        clickmode="event+select",
        dragmode=False,
        font={
            "family": "Arial, sans-serif",
            "size": settings.font_size,
            "color": "#37424a",
        },
        hoverlabel={"font_size": settings.font_size},
    )
    result.update_yaxes(
        autorange="reversed",
        tickmode="array",
        tickvals=list(range(len(names))),
        ticktext=labels,
        automargin=True,
        showgrid=False,
        zeroline=False,
        fixedrange=True,
    )
    result.update_xaxes(
        title_text=f"Contribution ({unit})",
        range=[low - span * 0.22, high + span * 0.22],
        nticks=5,
        gridcolor="#edf0f2",
        zeroline=True,
        zerolinecolor="#a5adb3",
        automargin=True,
        fixedrange=True,
    )
    return result
