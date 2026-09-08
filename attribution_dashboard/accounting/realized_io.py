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
) -> realized.RealizedPnlReport:
    """Load an inclusive date window and recompute its accounting and linking.

    Parameters
    ----------
    directory
        Ledger directory. Assets contain signed ``asset_pnl`` and the canonical
        date/security/side keys. Label, sector and gross_weight are optional.
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
    assets = pl.scan_parquet(root / "assets.parquet")
    columns = ["date", "asset_id", "side", "asset_pnl"]
    columns.extend(
        name
        for name in ("label", "sector", "gross_weight")
        if name in assets.collect_schema()
    )
    assets_frame = (
        assets.filter(pl.col("date").is_between(start, end)).select(columns).collect()
    )
    returns = (
        pl.scan_parquet(root / "daily.parquet")
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
