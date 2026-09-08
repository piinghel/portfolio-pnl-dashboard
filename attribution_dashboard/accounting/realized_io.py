"""Read a strategy-independent realized P&L bundle without a dashboard dependency.

The bundle contains ``assets.parquet``, ``daily.parquet`` and ``manifest.json``.
The manifest's ``portfolio`` object owns display metadata: required ``name`` and
positive fixed ``notional``; optional ``currency``, ``description``,
``classification`` and ``pnl_method`` are plain explanatory strings. Other
manifest fields belong to the producer and are retained for provenance.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path

import polars as pl

import attribution_dashboard.accounting.realized as realized
import attribution_dashboard.accounting.source_schema as source_schema


@dataclass
class PortfolioMetadata:
    """Strategy metadata independent of the source backtest configuration."""

    name: str
    notional: float
    currency: str = "backtest currency"
    description: str = ""
    classification: str = ""
    pnl_method: str = ""


def read_metadata(directory: Path | str) -> PortfolioMetadata:
    """Read and validate the portfolio metadata in a ledger manifest.

    Parameters
    ----------
    directory
        Directory containing the ledger manifest.

    Returns
    -------
    PortfolioMetadata
        Fixed-notional interpretation and producer-supplied descriptions.
    """
    path = Path(directory) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    data = manifest.get("portfolio") if isinstance(manifest, dict) else None
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a portfolio metadata object")
    name, notional = data.get("name"), data.get("notional")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("portfolio.name must be a nonempty string")
    if (
        isinstance(notional, bool)
        or not isinstance(notional, (int, float))
        or not math.isfinite(notional)
        or notional <= 0
    ):
        raise ValueError("portfolio.notional must be finite and positive")
    optional = {}
    for key in ("currency", "description", "classification", "pnl_method"):
        if key in data:
            if not isinstance(data[key], str):
                raise ValueError(f"portfolio.{key} must be a string")
            optional[key] = data[key]
    return PortfolioMetadata(name=name, notional=float(notional), **optional)


def load_period(
    directory: Path | str,
    start: dt.date,
    end: dt.date,
    *,
    tolerance: float = 1e-10,
    columns: source_schema.SourceColumns = source_schema.DEFAULT_COLUMNS,
) -> realized.RealizedPnlReport:
    """Load an inclusive date window and recompute its accounting and linking.

    Parameters
    ----------
    directory
        Ledger directory. Assets contain signed ``asset_pnl`` and the canonical
        date/security/side keys. Label, sector, industry and gross_weight are optional.
        Daily rows use the six saved return columns accepted by
        ``build_realized_pnl_report``. Previously linked columns are ignored.
    start, end
        Inclusive calendar bounds. A window without trading days fails clearly.
    tolerance
        Absolute daily accounting reconciliation tolerance in return units.

    Returns
    -------
    RealizedPnlReport
        Reconciled report rebased to this window, callable without Streamlit.
    """
    if start > end:
        raise ValueError("start must be on or before end")
    root = Path(directory)
    assets = source_schema.scan_parquet(root / "assets.parquet", columns=columns)
    selected_columns = ["date", "asset_id", "side", "asset_pnl"]
    selected_columns.extend(
        name
        for name in ("label", "sector", "industry", "gross_weight")
        if name in assets.collect_schema()
    )
    assets_frame = (
        assets.filter(pl.col("date").is_between(start, end))
        .select(selected_columns)
        .collect()
    )
    returns = (
        source_schema.scan_parquet(root / "daily.parquet", columns=columns)
        .filter(pl.col("date").is_between(start, end))
        .select(
            "date",
            "long_gross",
            "short_gross",
            "long_short_gross",
            "long_net",
            "short_net",
            "long_short_net",
        )
        .collect()
    )
    return realized.build_realized_pnl_report(
        assets_frame, returns, tolerance=tolerance
    )


def read_stock_identity(
    directory: Path | str,
    security: str,
    *,
    columns: source_schema.SourceColumns = source_schema.DEFAULT_COLUMNS,
) -> pl.DataFrame:
    """Read a stock's display identity even when its selected period is flat.

    Parameters
    ----------
    directory
        Ledger directory containing assets.parquet.
    security
        Security identifier to retain across period changes.

    Returns
    -------
    pl.DataFrame
        At most one row with identifier, label and sector. Missing labels use
        the identifier; missing or empty sectors use ``Unclassified``, matching
        the realized report's optional metadata policy.
    """
    assets = source_schema.scan_parquet(
        Path(directory) / "assets.parquet", columns=columns
    )
    schema = assets.collect_schema()
    label = pl.col("label") if "label" in schema else pl.lit(None, pl.String)
    sector = pl.col("sector") if "sector" in schema else pl.lit(None, pl.String)
    return (
        assets.filter(pl.col("asset_id") == security)
        .select(
            "asset_id",
            label.fill_null(pl.col("asset_id")).alias("label"),
            sector.fill_null("Unclassified")
            .replace("", "Unclassified")
            .alias("sector"),
        )
        .head(1)
        .collect()
    )
