"""Dashboard configuration: which books the app can open.

Streamlit-free. A runnable instance outside the package points the app at a
YAML configuration via ``$ATTRIBUTION_DASHBOARD_CONFIG``; the package itself holds
no paths. A book is one attribution bundle directory, tagged ``live`` or
``historical`` so the picker can group them (live first).
"""

from __future__ import annotations

import datetime as dt
import pathlib
import tomllib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast, get_args

import yaml

import attribution_dashboard.chart_settings as visual

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "BookKind",
    "BookSource",
    "DashboardConfig",
    "load_config",
    "selectable_books",
]

# The two book kinds; the Literal is the single owner, _KINDS its runtime view.
BookKind = Literal["live", "historical"]
_KINDS: tuple[BookKind, ...] = get_args(BookKind)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BookSource:
    """One openable book: a labelled attribution bundle directory.

    ``group`` is an optional project tag ("US ML", "FactSet", …) shown as a
    label prefix in the picker; the bundle file contract is identical for
    every book regardless of group.
    """

    label: str
    kind: BookKind
    directory: pathlib.Path
    group: str = ""
    default_start: dt.date | None = None
    default_end: dt.date | None = None
    benchmark_label: str | None = None

    @property
    def display_label(self) -> str:
        """Picker text: ``group · label`` when a group is set."""

        return f"{self.group} · {self.label}" if self.group else self.label


@dataclass(frozen=True)
class DashboardConfig:
    """The books the dashboard can open, plus the shared risk-state directory."""

    books: tuple[BookSource, ...]
    state_dir: pathlib.Path | None
    charts: visual.ChartSettings = visual.DEFAULT_CHARTS


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_config(path: pathlib.Path | str) -> DashboardConfig:
    """Parse an external YAML instance into a :class:`DashboardConfig`.

    Schema: an optional ``state_dir`` string and a ``books`` list of mappings,
    each with ``label``, ``kind`` (``live``|``historical``), ``dir``, and an
    optional ``group`` project tag. Relative directories resolve beside the config.
    An optional saved period uses paired ISO dates ``default_start`` / ``default_end``.
    A ``dashboard`` mapping is reserved for the external launcher. Legacy TOML
    ``[[book]]`` input remains supported for the existing Windows bundle explorer.

    Args:
        path: Path to the config file.

    Returns:
        The parsed config.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a book entry is missing a field or has a bad kind.
    """

    config_path = pathlib.Path(path).expanduser().resolve()
    if config_path.suffix.lower() == ".toml":
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
        entries = data.get("book", [])
    elif config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid YAML in {config_path}: {error}") from error
        if not isinstance(data, dict):
            raise ValueError("Dashboard configuration must be a mapping")
        if any(not isinstance(key, str) for key in data):
            raise ValueError("Configuration keys must be strings")
        unknown = data.keys() - {"books", "state_dir", "dashboard", "charts"}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        entries = data.get("books", [])
    else:
        raise ValueError("Dashboard configuration must use .yaml, .yml or .toml")
    if not isinstance(entries, list):
        raise ValueError("Books must be a list of mappings")
    books = tuple(
        _parse_book(entry, index, config_path.parent)
        for index, entry in enumerate(entries)
    )
    labels = [book.display_label for book in books]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Duplicate book labels in {config_path}: {sorted(labels)}")
    state_dir = data.get("state_dir")
    chart_values = data.get("charts", {})
    if not isinstance(chart_values, dict):
        raise ValueError("charts must be a mapping")
    unknown_charts = (
        chart_values.keys() - visual.ChartSettings.__dataclass_fields__.keys()
    )
    if unknown_charts:
        raise ValueError(f"Unknown chart settings: {sorted(unknown_charts)}")
    return DashboardConfig(
        charts=visual.ChartSettings(**chart_values),
        books=books,
        state_dir=(
            _resolve_path(state_dir, config_path.parent)
            if state_dir is not None
            else None
        ),
    )


def _parse_book(
    entry: dict, index: int, base: pathlib.Path, *, kinds: tuple[str, ...] = _KINDS
) -> BookSource:
    """Validate one book mapping and build a :class:`BookSource`."""

    if not isinstance(entry, dict):
        raise ValueError(f"Book #{index} must be a mapping")
    unknown = entry.keys() - {
        "label",
        "kind",
        "dir",
        "group",
        "default_start",
        "default_end",
        "benchmark_label",
    }
    if unknown:
        raise ValueError(f"Book #{index} has unknown keys: {sorted(unknown)}")
    missing = [key for key in ("label", "kind", "dir") if key not in entry]
    if missing:
        raise ValueError(f"Book #{index} is missing {missing}.")
    for key in ("label", "dir"):
        if not isinstance(entry[key], str) or not entry[key].strip():
            raise ValueError(f"Book #{index} {key} must be a nonempty string")
    if not isinstance(entry.get("group", ""), str):
        raise ValueError(f"Book #{index} group must be a string")
    if entry["kind"] not in kinds:
        raise ValueError(
            f"Book {entry['label']!r} has kind {entry['kind']!r}; "
            f"expected one of {kinds}."
        )
    if entry.get("benchmark_label") is not None and (
        not isinstance(entry["benchmark_label"], str)
        or not entry["benchmark_label"].strip()
    ):
        raise ValueError("benchmark_label must be a nonempty string")
    dates = [entry.get("default_start"), entry.get("default_end")]
    if any(value is not None for value in dates):
        if not all(type(value) is dt.date for value in dates):
            raise ValueError(
                "default_start and default_end must both be ISO dates (YYYY-MM-DD, unquoted)"
            )
        if cast(dt.date, dates[0]) > cast(dt.date, dates[1]):
            raise ValueError("default_start must be on or before default_end")
    return BookSource(
        label=entry["label"],
        kind=entry["kind"],
        directory=_resolve_path(entry["dir"], base),
        group=entry.get("group", ""),
        default_start=dates[0],
        default_end=dates[1],
        benchmark_label=entry.get("benchmark_label"),
    )


def _resolve_path(value: str, base: pathlib.Path) -> pathlib.Path:
    """Resolve native relative paths beside their configuration file."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Configured paths must be nonempty strings")
    path = pathlib.Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def selectable_books(
    config: DashboardConfig, *, is_bundle: Callable[[pathlib.Path], bool]
) -> tuple[BookSource, ...]:
    """Books whose bundle exists on disk, live first then historical.

    ``is_bundle`` is injected (e.g. ``bundle.is_bundle``) so this stays free of
    the bundle file contract and is trivially testable. Within a kind the
    config order is preserved.

    Args:
        config: The loaded dashboard config.
        is_bundle: Predicate telling whether a directory holds a full bundle.

    Returns:
        The openable books, live first.
    """

    ordered = sorted(config.books, key=lambda book: book.kind != "live")
    return tuple(book for book in ordered if is_bundle(book.directory))
