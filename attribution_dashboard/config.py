"""Read the public dashboard's YAML portfolio and chart configuration."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

import attribution_dashboard.chart_settings as visual


@dataclass(frozen=True)
class BookSource:
    """A named ledger directory and its optional display defaults."""

    label: str
    directory: Path
    group: str = ""
    default_start: dt.date | None = None
    default_end: dt.date | None = None
    benchmark_label: str | None = None

    @property
    def display_label(self) -> str:
        """Include the group when several portfolios need distinguishing."""
        return f"{self.group} · {self.label}" if self.group else self.label


@dataclass(frozen=True)
class DashboardConfig:
    """Available portfolios, in display order, and chart settings."""

    books: tuple[BookSource, ...]
    charts: visual.ChartSettings = visual.DEFAULT_CHARTS


def load_config(path: Path | str) -> DashboardConfig:
    """Load YAML, rejecting misspelled keys and ambiguous portfolio defaults.

    Relative data paths resolve beside the configuration. Optional saved dates
    must be supplied together as unquoted ISO dates.
    """
    config_path = Path(path).expanduser().resolve()
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Dashboard configuration must use .yaml or .yml")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {config_path}: {error}") from error
    if not isinstance(data, dict) or any(not isinstance(key, str) for key in data):
        raise ValueError("Dashboard configuration must be a mapping with string keys")
    unknown = data.keys() - {"books", "charts"}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    entries = data.get("books", [])
    if not isinstance(entries, list):
        raise ValueError("Books must be a list of mappings")
    books = tuple(
        _parse_book(entry, index, config_path.parent)
        for index, entry in enumerate(entries)
    )
    labels = [book.display_label for book in books]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Duplicate book labels: {sorted(labels)}")
    chart_values = data.get("charts", {})
    if not isinstance(chart_values, dict) or any(
        not isinstance(key, str) for key in chart_values
    ):
        raise ValueError("charts must be a mapping with string keys")
    unknown_charts = (
        chart_values.keys() - visual.ChartSettings.__dataclass_fields__.keys()
    )
    if unknown_charts:
        raise ValueError(f"Unknown chart settings: {sorted(unknown_charts)}")
    return DashboardConfig(books=books, charts=visual.ChartSettings(**chart_values))


def _parse_book(entry: dict, index: int, base: Path) -> BookSource:
    if not isinstance(entry, dict) or any(not isinstance(key, str) for key in entry):
        raise ValueError(f"Book #{index} must be a mapping with string keys")
    unknown = entry.keys() - {
        "label",
        "dir",
        "group",
        "default_start",
        "default_end",
        "benchmark_label",
    }
    if unknown:
        raise ValueError(f"Book #{index} has unknown keys: {sorted(unknown)}")
    for key in ("label", "dir"):
        if not isinstance(entry.get(key), str) or not entry[key].strip():
            raise ValueError(f"Book #{index} {key} must be a nonempty string")
    if not isinstance(entry.get("group", ""), str):
        raise ValueError(f"Book #{index} group must be a string")
    benchmark = entry.get("benchmark_label")
    if benchmark is not None and (
        not isinstance(benchmark, str) or not benchmark.strip()
    ):
        raise ValueError("benchmark_label must be a nonempty string")
    start, end = entry.get("default_start"), entry.get("default_end")
    if start is not None or end is not None:
        if type(start) is not dt.date or type(end) is not dt.date:
            raise ValueError(
                "default_start and default_end must both be unquoted ISO dates"
            )
        if start > end:
            raise ValueError("default_start must be on or before default_end")
    directory = Path(entry["dir"]).expanduser()
    return BookSource(
        label=entry["label"],
        directory=directory
        if directory.is_absolute()
        else (base / directory).resolve(),
        group=entry.get("group", ""),
        default_start=cast(dt.date | None, start),
        default_end=cast(dt.date | None, end),
        benchmark_label=benchmark,
    )
