"""Generate a reproducible fictional portfolio; never reads private data."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
from pathlib import Path

import polars as pl

PREDICTORS = [
    ("Book yield", 0.24),
    ("Earnings yield", 0.18),
    ("Profitability", 0.20),
    ("Operating margin", 0.12),
    ("Medium-term momentum", 0.22),
    ("Short-term momentum", 0.08),
    ("Low volatility", 0.16),
    ("Earnings stability", 0.10),
    ("Liquidity", 0.07),
    ("Company size", 0.05),
]


def prediction_snapshot(
    rng: random.Random, previous: list[list[float]]
) -> tuple[list[list[float]], list[float]]:
    """Simulate correlated, persistent inputs; no future prices enter the score."""
    raw = []
    for old in previous:
        groups = [rng.gauss(0, 1) for _ in range(5)]
        raw.append(
            [
                0.65 * old[k]
                + math.sqrt(1 - 0.65**2)
                * (math.sqrt(0.7) * groups[k // 2] + math.sqrt(0.3) * rng.gauss(0, 1))
                for k in range(len(PREDICTORS))
            ]
        )
    means = [sum(row[k] for row in raw) / len(raw) for k in range(len(PREDICTORS))]
    scales = [
        math.sqrt(sum((row[k] - means[k]) ** 2 for row in raw) / len(raw))
        for k in range(len(PREDICTORS))
    ]
    features = [
        [(value - means[k]) / scales[k] for k, value in enumerate(row)] for row in raw
    ]
    scores = [
        0.05 + sum(x * beta for x, (_, beta) in zip(row, PREDICTORS, strict=True))
        for row in features
    ]
    return features, scores


def _regime(date: dt.date) -> tuple[float, float, float]:
    """Hand-designed market, selection drift and volatility regimes.

    These are illustrative scenarios, not fitted or exported real returns.
    """
    market, selection = {
        2021: (0.18, 0.14),
        2022: (-0.14, 0.08),
        2023: (0.16, 0.08),
        2024: (0.10, 0.06),
        2025: (0.03, 0.27),
    }[date.year]
    market, selection, volatility = market / 252, selection / 252, 1.0
    if date.year == 2022:
        volatility = 1.35
    if dt.date(2021, 7, 5) <= date <= dt.date(2021, 8, 6):
        selection, volatility = -0.0012, 1.3
    if dt.date(2023, 1, 2) <= date <= dt.date(2023, 2, 3):
        market, selection, volatility = 0.0010, -0.0025, 1.5
    if dt.date(2025, 3, 17) <= date <= dt.date(2025, 4, 11):
        market, selection, volatility = -0.003, -0.0012, 1.8
    if dt.date(2025, 4, 14) <= date <= dt.date(2025, 6, 13):
        market, selection, volatility = 0.0018, 0.0010, 1.2
    return market, selection, volatility


def generate(destination: Path, *, seed: int = 20260908) -> None:
    """Write synthetic prices, holdings and an exact factor P&L partition.

    Portfolio decisions use only current prices at the close. P&L uses the
    previous close's shares; positions are marked after close-time rebalancing.
    Factor covariance is known from the simulation, not estimated from history.
    """
    rng = random.Random(seed)
    signal_rng = random.Random(seed + 1)
    dates = pl.date_range(dt.date(2021, 1, 4), dt.date(2025, 12, 31), eager=True)
    dates = dates.filter(dates.dt.weekday() <= 5).to_list()
    n, notional, residual_sigma = 36, 1_000_000.0, 0.010
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
    selection_loadings = [rng.uniform(0.6, 1.4) for _ in range(n)]
    prices = [rng.uniform(25, 150) for _ in range(n)]
    shares = [0.0] * n
    features = [[0.0] * len(PREDICTORS) for _ in range(n)]
    decision_rows, predictor_rows = [], []
    asset_rows, daily_rows, quote_rows, holding_rows = [], [], [], []
    factor_rows, stock_rows, asset_factor_rows, risk_rows, coverage_rows = (
        [],
        [],
        [],
        [],
        [],
    )
    for t, date in enumerate(dates):
        market_drift, selection_drift, volatility = _regime(date)
        daily_sigmas = [sigma * volatility for sigma in sigmas]
        stock_sigma, selection_sigma = residual_sigma * volatility, 0.0025 * volatility
        returns = [rng.gauss(0, sigma) for sigma in daily_sigmas]
        returns[1] += market_drift
        selection_move = selection_drift + rng.gauss(0, selection_sigma)
        long_pnl = short_pnl = long_cost = short_cost = 0.0
        exposures, factor_pnl = [0.0] * len(factors), [0.0] * len(factors)
        residual_pnl = residual_variance = gross_start = 0.0
        selection_exposure = 0.0
        selected = set()
        if t % 21 == 0:
            features, scores = prediction_snapshot(signal_rng, features)
            for side, candidates in [("long", range(18)), ("short", range(18, 36))]:
                ranked = sorted(
                    candidates,
                    key=lambda i: (scores[i] if side == "short" else -scores[i], i),
                )
                selected.update(ranked[:12])
                for rank, i in enumerate(ranked, start=1):
                    asset = f"DEMO{i + 1:03d}"
                    decision_rows.append(
                        (
                            date,
                            asset,
                            side,
                            scores[i],
                            0.05,
                            rank,
                            18,
                            12,
                            scores[ranked[11]],
                            rank <= 12,
                        )
                    )
                    for x, (name, beta) in zip(features[i], PREDICTORS, strict=True):
                        predictor_rows.append(
                            (date, asset, side, name, x, beta, x * beta)
                        )
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
            residual = sign * selection_loadings[i] * selection_move + rng.gauss(
                0, stock_sigma
            )
            stock_return = sum(components) + residual
            prices[i] *= 1 + stock_return
            pnl, explained, unexplained = (
                weight * stock_return,
                weight * sum(components),
                weight * residual,
            )
            residual_pnl += unexplained
            residual_variance += (weight * stock_sigma) ** 2
            selection_exposure += weight * sign * selection_loadings[i]
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
                    sign * rng.uniform(0.065, 0.095) * notional / prices[i]
                    if i in selected and side == "long"
                    else sign * rng.uniform(0.045, 0.075) * notional / prices[i]
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
        # Residuals include a shared selection shock; keep its covariance.
        residual_variance += (selection_exposure * selection_sigma) ** 2
        variance = (
            sum(
                (exposure * sigma) ** 2
                for exposure, sigma in zip(exposures, daily_sigmas, strict=True)
            )
            + residual_variance
        )
        if variance > 0:
            for factor, exposure, sigma in zip(
                factors, exposures, daily_sigmas, strict=True
            ):
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
        "predictions/decisions": (
            decision_rows,
            [
                "date",
                "asset_id",
                "side",
                "score",
                "intercept",
                "rank",
                "universe_size",
                "selection_count",
                "cutoff",
                "selected",
            ],
        ),
        "predictions/contributions": (
            predictor_rows,
            [
                "date",
                "asset_id",
                "side",
                "predictor",
                "input_value",
                "coefficient",
                "contribution",
            ],
        ),
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
        "description": "Illustrative market and selection regimes create trends, drawdowns and recoveries. A synthetic linear score ranks fictional long/short populations with designed drift differences. This is a scenario demonstration, not evidence of a predictive strategy.",
        "model": {
            "scope_note": "The partition is exact by construction. The four displayed factors omit a shared synthetic selection component, which remains in the correlated residual. Forecast risk uses the known regime covariance, including this residual correlation, and previous-close weights. This is an illustrative model-risk calculation, not a real forecast.",
            "exposure_units": "signed starting weight × simulated loading",
        },
    }
    (destination / "factors/manifest.json").write_text(
        json.dumps(model, indent=2) + "\n"
    )
    (destination / "predictions/manifest.json").write_text(
        json.dumps(
            {
                "model": "Synthetic linear score v1",
                "description": "Illustrative score, not a fitted forecast or predicted Sharpe ratio. Ten simulated inputs are standardized across 36 companies at each decision; paired inputs are correlated. Fixed coefficients are hand-chosen. Inputs are not measured company fundamentals, and the separate price simulation does not imply predictive power for these inputs.",
                "selection_rule": "At each rebalance, buy the 12 highest scores in the 18 long-eligible names and short the 12 lowest in the 18 short-eligible names. Rank 1 is best for that side. Scores choose membership; position sizes are random within the documented ranges.",
                "timing": "Inputs are generated before the rebalance session's return; holdings change at that session's close. New holdings earn P&L from the following session. No future returns enter the score.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/demo"
    )
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    generate(args.output, seed=args.seed)
