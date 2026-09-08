"""Generate a reproducible fictional portfolio; never reads private data."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
from pathlib import Path

import polars as pl


def generate(destination: Path, *, seed: int = 20260908) -> None:
    """Write synthetic prices, holdings and an exact factor P&L partition.

    Portfolio decisions use only current prices at the close. P&L uses the
    previous close's shares; positions are marked after close-time rebalancing.
    Factor covariance is known from the simulation, not estimated from history.
    """
    rng = random.Random(seed)
    dates = pl.date_range(dt.date(2022, 1, 3), dt.date(2025, 12, 31), eager=True)
    dates = dates.filter(dates.dt.weekday() <= 5).to_list()
    n, notional, residual_sigma = 36, 1_000_000.0, 0.012
    factors = ["market", "beta", "size", "momentum"]
    sigmas = [0.002, 0.009, 0.004, 0.004]
    sectors = [
        "Industrials",
        "Technology",
        "Healthcare",
        "Consumer",
        "Energy",
        "Financials",
    ]
    loadings = [
        [1.0, rng.uniform(0.5, 1.4), rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)]
        for _ in range(n)
    ]
    prices = [rng.uniform(25, 150) for _ in range(n)]
    shares = [0.0] * n
    asset_rows, daily_rows, quote_rows, holding_rows = [], [], [], []
    factor_rows, stock_rows, asset_factor_rows, risk_rows, coverage_rows = (
        [],
        [],
        [],
        [],
        [],
    )
    for t, date in enumerate(dates):
        # Independent innovations; no source data or fitted strategy is used.
        returns = [rng.gauss(0, sigma) for sigma in sigmas]
        long_pnl = short_pnl = long_cost = short_cost = 0.0
        exposures, factor_pnl = [0.0] * len(factors), [0.0] * len(factors)
        residual_pnl = residual_variance = gross_start = 0.0
        selected = set()
        if t % 21 == 0:
            selected = set(rng.sample(range(18), 12) + rng.sample(range(18, 36), 12))
        for i in range(n):
            asset, label, sector = (
                f"DEMO{i + 1:03d}",
                f"Company {i + 1:02d}",
                sectors[i % len(sectors)],
            )
            side, sign = ("long", 1) if i < 18 else ("short", -1)
            previous_price, previous_shares = prices[i], shares[i]
            weight = previous_shares * previous_price / notional
            components = [
                loading * value
                for loading, value in zip(loadings[i], returns, strict=True)
            ]
            residual = rng.gauss(0, residual_sigma)
            stock_return = sum(components) + residual
            prices[i] *= 1 + stock_return
            pnl, explained, unexplained = (
                weight * stock_return,
                weight * sum(components),
                weight * residual,
            )
            residual_pnl += unexplained
            residual_variance += (weight * residual_sigma) ** 2
            gross_start += abs(weight)
            for k, factor in enumerate(factors):
                exposure = weight * loadings[i][k]
                exposures[k] += exposure
                factor_pnl[k] += weight * components[k]
                asset_factor_rows.append(
                    (
                        date,
                        asset,
                        side,
                        factor,
                        weight * components[k],
                        loadings[i][k],
                        exposure,
                    )
                )
            if t % 21 == 0:
                shares[i] = (
                    sign * rng.uniform(0.035, 0.065) * notional / prices[i]
                    if i in selected
                    else 0.0
                )
            cost = abs(shares[i] - previous_shares) * prices[i] / notional * 0.0005
            if side == "long":
                long_pnl += pnl
                long_cost += cost
            else:
                short_pnl += pnl
                short_cost += cost
            asset_rows.append(
                (
                    date,
                    asset,
                    side,
                    pnl,
                    abs(shares[i] * prices[i] / notional),
                    label,
                    sector,
                )
            )
            quote_rows.append((date, asset, prices[i], prices[i], "USD"))
            holding_rows.append((date, asset, side, abs(shares[i])))
            stock_rows.append(
                (
                    date,
                    asset,
                    side,
                    label,
                    sector,
                    pnl,
                    explained,
                    unexplained,
                    0.0,
                    0.0,
                )
            )
        costs = -long_cost - short_cost
        net = long_pnl + short_pnl + costs
        daily_rows.append(
            (
                date,
                long_pnl,
                -short_pnl,
                long_pnl + short_pnl,
                long_pnl - long_cost,
                -short_pnl + short_cost,
                net,
                returns[1],
            )
        )
        for k, factor in enumerate(factors):
            factor_rows.append((date, factor, factor_pnl[k], exposures[k]))
        for factor, value in [
            ("idio_pnl", residual_pnl),
            ("costs", costs),
            ("price_basis_gap", 0.0),
            ("unmodeled_pnl", 0.0),
        ]:
            factor_rows.append((date, factor, value, None))
        variance = (
            sum(
                (exposure * sigma) ** 2
                for exposure, sigma in zip(exposures, sigmas, strict=True)
            )
            + residual_variance
        )
        if variance > 0:
            for factor, exposure, sigma in zip(factors, exposures, sigmas, strict=True):
                risk_rows.append(
                    (
                        date,
                        factor,
                        (exposure * sigma) ** 2 / math.sqrt(variance) * math.sqrt(252),
                    )
                )
            risk_rows.append(
                (date, "idio", residual_variance / math.sqrt(variance) * math.sqrt(252))
            )
        coverage_rows.append((date, gross_start, 1.0, 1.0, 1.0))
    tables = {
        "assets": (
            asset_rows,
            [
                "date",
                "asset_id",
                "side",
                "asset_pnl",
                "gross_weight",
                "label",
                "sector",
            ],
        ),
        "daily": (
            daily_rows,
            [
                "date",
                "long_gross",
                "short_gross",
                "long_short_gross",
                "long_net",
                "short_net",
                "long_short_net",
                "benchmark",
            ],
        ),
        "prices": (
            quote_rows,
            ["date", "asset_id", "px_last", "px_last_unadjusted", "price_currency"],
        ),
        "positions": (holding_rows, ["date", "asset_id", "side", "holding_qty"]),
        "factors/daily": (factor_rows, ["date", "factor", "pnl", "exposure"]),
        "factors/stocks": (
            stock_rows,
            [
                "date",
                "asset_id",
                "side",
                "label",
                "sector",
                "asset_pnl",
                "factor_pnl",
                "idio_pnl",
                "price_basis_gap",
                "unmodeled_pnl",
            ],
        ),
        "factors/asset_factors": (
            asset_factor_rows,
            [
                "date",
                "asset_id",
                "view",
                "factor",
                "pnl",
                "factor_loading",
                "signed_factor_exposure",
            ],
        ),
        "factors/risk": (risk_rows, ["date", "factor", "risk_vol"]),
        "factors/coverage": (
            coverage_rows,
            [
                "date",
                "gross_weight",
                "pnl_coverage",
                "risk_coverage",
                "exposure_coverage",
            ],
        ),
    }
    for name, (rows, schema) in tables.items():
        path = destination / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(rows, schema=schema, orient="row").write_parquet(path)
    manifest = {
        "synthetic": True,
        "seed": seed,
        "generator": "scripts/generate_demo.py",
        "portfolio": {
            "name": "Synthetic long/short portfolio",
            "notional": notional,
            "currency": "USD",
            "description": "Synthetic demo: fictional companies, prices and holdings. These results do not represent a real investment strategy.",
            "classification": "fictional, fixed sector assignments",
            "pnl_method": "P&L uses previous-close shares × price change, divided by fixed notional. Holdings are marked after rebalancing every 21 weekdays. Costs are 5 basis points of traded notional. No dividends or corporate actions are simulated, so adjusted and original prices coincide.",
        },
        "conventions": {"omitted_costs": "Borrow, financing and market impact"},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    model = {
        "synthetic": True,
        "factor_names": factors,
        "description": "Known simulation components, not a fitted attribution model. Independent Gaussian factor and stock innovations; random holdings have no intended predictive edge.",
        "model": {
            "scope_note": "The factor partition is exact by construction. Forecast risk uses the simulation's known covariance and previous-close weights, not estimated real-market risk. Residuals are independently simulated innovations.",
            "exposure_units": "signed starting weight × simulated loading",
        },
    }
    (destination / "factors/manifest.json").write_text(
        json.dumps(model, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/demo"
    )
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    generate(args.output, seed=args.seed)
