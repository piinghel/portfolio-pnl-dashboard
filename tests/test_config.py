"""Public YAML configuration preserves paths, dates and validation boundaries."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

import attribution_dashboard.config as config


def test_portfolio_paths_and_saved_dates(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "books:\n"
        "  - label: First\n    dir: first\n"
        "  - label: Second\n    dir: other\n    group: Research\n"
        "    default_start: 2023-01-03\n    default_end: 2023-01-31\n"
    )
    result = config.load_config(path)
    assert [book.display_label for book in result.books] == [
        "First",
        "Research · Second",
    ]
    assert result.books[1].directory == tmp_path / "other"
    assert result.books[1].default_start == dt.date(2023, 1, 3)
    assert result.books[1].default_end == dt.date(2023, 1, 31)


@pytest.mark.parametrize(
    "configuration, message",
    [
        ("book: []", "Unknown configuration keys"),
        ("books: [", "Invalid YAML"),
        ("books: {}", "Books must be a list"),
        ("books: [{label: A, dir: data, defaut_start: 2023-01-03}]", "unknown keys"),
        (
            "books: [{label: A, dir: data}, {label: A, dir: other}]",
            "Duplicate book labels",
        ),
        ("books: [{label: A, dir: data, default_start: 2023-01-03}]", "default_start"),
        (
            "books: [{label: A, dir: data, default_start: '2023-01-03', default_end: 2023-01-31}]",
            "default_start",
        ),
        (
            "books: [{label: A, dir: data, default_start: 2023-02-01, default_end: 2023-01-31}]",
            "default_start",
        ),
    ],
)
def test_invalid_configuration(
    tmp_path: Path, configuration: str, message: str
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(configuration)
    with pytest.raises(ValueError, match=message):
        config.load_config(path)


def test_chart_settings(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("books: []\ncharts: {line_height: 420, font_size: 16}\n")
    result = config.load_config(path)
    assert result.charts.line_height == 420
    assert result.charts.font_size == 16
    for invalid in (
        "{font_size: true}",
        "{line_height: 40}",
        "{font_szie: 14}",
        "{pnl_decimals: 1}",
    ):
        path.write_text(f"books: []\ncharts: {invalid}\n")
        with pytest.raises(ValueError):
            config.load_config(path)
