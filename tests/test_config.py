from __future__ import annotations

import datetime as dt
import pathlib

import pytest

from attribution_dashboard import config as config_mod

_CONFIG = """
state_dir = "state"

[[book]]
label = "Live book"
kind  = "live"
dir   = "live"

[[book]]
label = "LS 75/75"
kind  = "historical"
dir   = "hist/ls"
group = "US ML"
default_start = 2023-01-03
default_end = 2023-01-31
"""


def test_load_and_select(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(_CONFIG, encoding="utf-8")
    config = config_mod.load_config(path)

    assert config.state_dir == tmp_path / "state"
    assert [book.label for book in config.books] == ["Live book", "LS 75/75"]
    assert [book.display_label for book in config.books] == [
        "Live book",
        "US ML · LS 75/75",
    ]

    assert config.books[1].default_start == dt.date(2023, 1, 3)

    # Both bundles exist -> live first, then historical.
    both = {tmp_path / "live", tmp_path / "hist/ls"}
    assert [
        b.label
        for b in config_mod.selectable_books(config, is_bundle=both.__contains__)
    ] == [
        "Live book",
        "LS 75/75",
    ]
    # Only the historical bundle exists -> the live book drops out.
    only_hist = {tmp_path / "hist/ls"}
    assert [
        b.label
        for b in config_mod.selectable_books(config, is_bundle=only_hist.__contains__)
    ] == ["LS 75/75"]


def test_bad_kind_raises(tmp_path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(
        '[[book]]\nlabel = "x"\nkind = "weird"\ndir = "d"\n', encoding="utf-8"
    )
    with pytest.raises(ValueError):
        config_mod.load_config(path)


@pytest.mark.parametrize(
    "dates",
    [
        "default_start = 2023-01-03",
        'default_start = "2023-01-03"\ndefault_end = 2023-01-31',
        "default_start = 2023-02-01\ndefault_end = 2023-01-31",
    ],
)
def test_invalid_saved_period_rejected(tmp_path: pathlib.Path, dates: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[[book]]\nlabel = "x"\nkind = "historical"\ndir = "data"\n' + dates
    )
    with pytest.raises(ValueError, match="default_"):
        config_mod.load_config(path)


def test_yaml_two_strategies_match_existing_config_contract(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "dashboard:\n  port: 8505\n"
        "state_dir: state\n"
        "books:\n"
        "  - label: Live book\n    kind: live\n    dir: live\n"
        "  - label: LS 75/75\n    kind: historical\n    dir: hist/ls\n"
        "    group: US ML\n    default_start: 2023-01-03\n    default_end: 2023-01-31\n"
    )
    yaml_config = config_mod.load_config(path)
    legacy = tmp_path / "config.toml"
    legacy.write_text(_CONFIG)
    assert yaml_config == config_mod.load_config(legacy)
    assert yaml_config.books[1].directory == tmp_path / "hist/ls"


@pytest.mark.parametrize(
    "configuration, message",
    [
        ("book: []", "Unknown configuration keys"),
        ("books: [", "Invalid YAML"),
        ("books: {}", "Books must be a list"),
        (
            "books: [{label: A, kind: live, dir: data, defaut_start: 2023-01-03}]",
            "unknown keys",
        ),
        (
            "books: [{label: A, kind: live, dir: data}, {label: A, kind: live, dir: other}]",
            "Duplicate book labels",
        ),
        (
            "books: [{label: A, kind: live, dir: data, default_start: 2023-01-03}]",
            "default_start",
        ),
    ],
)
def test_yaml_rejects_ambiguous_configuration(
    tmp_path: pathlib.Path, configuration: str, message: str
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(configuration)
    with pytest.raises(ValueError, match=message):
        config_mod.load_config(path)


def test_explicit_chart_settings(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("books: []\ncharts: {line_height: 420, font_size: 16}\n")
    result = config_mod.load_config(path)
    assert result.charts.line_height == 420
    assert result.charts.font_size == 16
    for decimals in (2, 3):
        path.write_text(f"books: []\ncharts: {{pnl_decimals: {decimals}}}\n")
        assert config_mod.load_config(path).charts.pnl_decimals == decimals
    for charts in (
        "{font_size: true}",
        "{line_height: 40}",
        "{font_szie: 14}",
        "{pnl_decimals: 1}",
        "{pnl_decimals: 4}",
    ):
        path.write_text(f"books: []\ncharts: {charts}\n")
        with pytest.raises(ValueError):
            config_mod.load_config(path)
