"""Normalize configurable source columns before accounting or dashboard logic."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import polars as pl


@dataclasses.dataclass(frozen=True)
class SourceColumns:
    """Source names for the shared date, identity and classification fields.

    Fields not present in a table remain optional. Required fields are checked
    by the consuming reader. Values must be distinct nonempty column names.
    """

    date: str = "date"
    asset_id: str = "asset_id"
    label: str = "label"
    sector: str = "sector"
    industry: str = "industry"

    def __post_init__(self) -> None:
        names = dataclasses.astuple(self)
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("Source column names must be nonempty strings")
        if len(set(names)) != len(names):
            raise ValueError("Source column names must be distinct")


DEFAULT_COLUMNS = SourceColumns()


def parse_columns(values: object) -> SourceColumns:
    """Validate an optional YAML mapping from logical fields to source names."""
    if not isinstance(values, dict) or any(not isinstance(key, str) for key in values):
        raise ValueError("columns must be a mapping with string keys")
    unknown = values.keys() - SourceColumns.__dataclass_fields__.keys()
    if unknown:
        raise ValueError(f"Unknown source column settings: {sorted(unknown)}")
    return SourceColumns(**values)


def scan_parquet(
    path: Path | str,
    *,
    columns: SourceColumns = DEFAULT_COLUMNS,
    canonical: SourceColumns = DEFAULT_COLUMNS,
) -> pl.LazyFrame:
    """Scan a table and normalize its configured names without copying the data.

    A conflicting canonical column is rejected rather than silently selected.
    Dates retain their original dtype; this mapping never guesses date formats,
    changes calendars, interprets identifier values or reclassifies securities.
    """
    frame = pl.scan_parquet(path)
    present = frame.collect_schema().names()
    rename = {}
    for source, target in zip(
        dataclasses.astuple(columns), dataclasses.astuple(canonical), strict=True
    ):
        if source == target:
            continue
        if source in present:
            rename[source] = target
        elif target in present:
            raise ValueError(
                f"{Path(path).name}: configured column {source!r} for {target!r} is missing"
            )
    normalized = [rename.get(name, name) for name in present]
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            f"{Path(path).name}: source mapping produces duplicate columns"
        )
    return frame.rename(rename)
