# Data bundle

Paths in `config.yaml` resolve relative to that file. The reader expects a
directory containing `assets.parquet`, `daily.parquet` and `manifest.json`.
Dates use Polars `Date`, numeric values use finite floating-point numbers, and
identifiers use strings. P&L is in decimal units of a single fixed notional.

## Configuring source column names

Each book may map its source names to the dashboard's shared fields in YAML:

```yaml
books:
  - label: Another strategy
    dir: data/another-strategy
    columns:
      date: trading_date
      asset_id: security_id
      label: security_name
      sector: sector_name
      industry: industry_name
    default_start: 2024-01-02
    default_end: 2024-12-31
```

Omit `columns` to use the schema below. Omitted mapping entries retain their
canonical names. The same mapping applies wherever these fields appear in
the ledger, classification sidecar, prices, holdings, factors and saved
prediction bundle. Normalize inconsistent names across files before supplying
one bundle. The separate `linear_history.json` raw-model sources keep their own
source contract, including their existing `asset_column` setting.

Readers rename columns lazily at the input boundary. Calculations and charts
continue to use one internal schema; no source data is rewritten. Mappings are
validated and participate in the read-cache keys. A mapping change clears stale
stock/date navigation for that directory. It never changes identifier values,
classifications, date formats or P&L conventions.

Date bounds normally come from the supplied ledger calendar. The optional saved
period above belongs to that book's YAML; no strategy dates live in the UI code.
Column names such as `industry` describe fields, not a fixed list of industries:
the actual categories and security IDs come from the data.

## Canonical accounting schema

| File | Required columns / fields |
| --- | --- |
| `assets.parquet` | `date`, `asset_id`, `side`, `asset_pnl` |
| `daily.parquet` | `date`, `long_gross`, `short_gross`, `long_short_gross`, `long_net`, `short_net`, `long_short_net` |
| `manifest.json` | `portfolio.name`, positive `portfolio.notional`; optional `currency`, `description`, `classification`, `pnl_method` inside `portfolio` |

Asset keys `(date, asset_id, side)` and daily dates must be unique. `side` is
`long` or `short`. Security `asset_pnl` is signed, including for shorts. Optional
asset columns are `label`, `sector`, `industry` and nonnegative `gross_weight`, where weight
is the same-date marked absolute position divided by fixed notional.

The saved daily short-leg columns use the **opposite sign** from short security
P&L, preserving the original engine convention:

```text
sum(long security P&L)  = long_gross
sum(short security P&L) = -short_gross
long_short_gross        = long_gross - short_gross
long_short_net          = long_net - short_net
cost P&L               = long_short_net - long_short_gross <= 0
```

Pass the complete daily calendar, including flat days. Missing security days
imply zero contribution. Missing prices and factor estimates remain gaps.
An optional daily `benchmark` column contains the benchmark's own decimal price
returns; set `benchmark_label` in configuration to expose its display toggle.

## Optional stock context

- `prices.parquet`: `date`, `asset_id`, `px_last`, `px_last_unadjusted`,
  `price_currency`. Log scale requires positive prices.
- `positions.parquet`: `date`, `asset_id`, `side`, `holding_qty` (absolute shares).
  These are end-of-session holdings, not execution records.

Industry labels can instead be supplied in `classifications.parquet`, with one
unique `asset_id` per row and an `industry` string. A dated industry column in
`assets.parquet` takes precedence. Missing classifications retain their P&L in
**Unclassified**. Describe the source and snapshot date in `portfolio.classification`;
a retrospective snapshot is not point-in-time classification. The demo file
contains fictional classifications only.

## Optional prediction context

Create a `predictions/` directory. This is a saved linear model decomposition,
separate from realized factor attribution. No explanation is inferred from P&L.

| File | Required fields |
| --- | --- |
| `decisions.parquet` | `date`, `asset_id`, `side`, `score`, `intercept`, `rank`, `universe_size`, `selection_count`, `cutoff`, `selected` |
| `contributions.parquet` | `date`, `asset_id`, `side`, `predictor`, `input_value`, `coefficient`, `contribution` |
| `manifest.json` | `model`, `description`, `selection_rule`, `timing` |

Decision keys `(date, asset_id, side)` are unique; predictor keys add `predictor`.
Each contribution must equal the **actual model input** times its saved
coefficient, and their sum plus intercept must equal the saved score. Export
the exact point-in-time transformed/scaled inputs, never today's values or betas.
No inverse target transformation is applied by this viewer.

`rank` is one-based and ordered for the side: highest score first for longs,
lowest first for shorts. This viewer assumes top-N membership within each
side's eligible universe; `selected` means rank ≤ `selection_count`, and `cutoff`
is the last included candidate's score. Export all rebalances, including retained
and rejected candidates. If an optimizer can override membership, this simple
rule is insufficient: export and display the optimizer decision separately.

`date` is the holdings decision date. Document input availability and execution
timing in `timing`. Marker clicks require an exact saved date and side match;
missing snapshots are never filled with a later prediction. The decision selector
also covers resizing and retained positions that have no entry/exit marker.

## Optional factor context

Create a `factors/` subdirectory with these files:

| File | Columns |
| --- | --- |
| `daily.parquet` | `date`, `factor`, `pnl`, `exposure` |
| `stocks.parquet` | `date`, `asset_id`, `side`, `label`, `sector`, `asset_pnl`, `factor_pnl`, `idio_pnl`, `price_basis_gap`, `unmodeled_pnl` |
| `asset_factors.parquet` | `date`, `asset_id`, `view`, `factor`, `pnl`, `factor_loading`, `signed_factor_exposure` |
| `risk.parquet` | `date`, `factor`, `risk_vol` |
| `coverage.parquet` | `date`, `gross_weight`, `pnl_coverage`, `risk_coverage`, `exposure_coverage` |
| `manifest.json` | `description`, optional `factor_names` and `model` metadata |

`view` is `long` or `short`. The daily partition includes factor contributions,
`idio_pnl`, `price_basis_gap`, `unmodeled_pnl` and `costs`; its sum must match the
parent net ledger for every date. Coverage columns are fractions of gross starting
exposure. Supply nullable `covered_exposure` in the factor daily file when only
part of the portfolio has known inputs; known weights must not be scaled up.

`risk_vol` is an annualized volatility contribution in decimal units. The
generator supplies known-model values from starting weights. Missing forecast
dates remain gaps; they are not forward-filled. Cost risk is not included in the
demo's model forecast, while realized factor risk includes the full net partition.

Document the model's factor definitions, exposure units, timestamp convention,
coverage and limitations in `model.scope_note` and `model.exposure_units`.
Manifests are downloadable: do not put private paths or credentials in them.

### Factor display conventions

Set model-specific roles in `factors/manifest.json`; the dashboard does not fit
or select the risk model. Existing bundles default to `intercept_factor: "market"`
(unit loading, so portfolio exposure is net invested weight) and
`sector_prefix: "sector:"` (categorical indicator loadings, so exposure is signed
sector weight). These roles determine percentage scaling and sector grouping.
They must match the supplied model, not merely its factor names.

For a model whose `market` factor is market beta rather than an intercept:

```json
{
  "model": {
    "intercept_factor": null,
    "sector_prefix": null,
    "factor_labels": {"market": "Market beta"},
    "exposure_units": "signed weight × beta"
  }
}
```

Use another intercept ID or sector prefix when those roles exist under different
names. `null` disables that role; it does not remove the factor from P&L, risk or
exposure charts. `factor_labels` changes display names only. Labels must distinguish
components and cannot reuse the reserved accounting/group labels. Residual,
reconciliation, uncovered P&L and costs retain their accounting identities.
