# Portfolio Optimization 2026

DBA5106 group project examining overfitting in portfolio models through the
linear-regression representation of the minimum-variance portfolio and its
LASSO and Ridge regularized extensions.

## Project structure

| Directory | Responsibility |
|---|---|
| `A_Data/` | Data parsing, cleaning, validation, and rolling-window construction |
| `B_EW_MinVar/` | Equal-weighted and unrestricted minimum-variance benchmarks |
| `C_LASSO/` | LASSO portfolio implementation |
| `D_Ridge/` | Ridge portfolio implementation |
| `Analysis/` | Cross-model tables, diagnostics, and figures |
| `data/` | A single shared copy of the project data |
| `results/` | Model outputs separated by workstream |
| `figures/` | Report figures separated by purpose |
| `report/` | Final submitted report |

The repository currently contains the organized A and B workstreams. C and D
are reserved for integration after their baselines and extensions are frozen.

## Environment

```bash
python -m pip install -r requirements.txt
```

The current unified dependency file was verified with Python 3.14.6. All
returns remain in percentage-point units during estimation: `1.00` means a
daily return of `1.00%`.

## Reproduce the data

Download the official `100 Portfolios Formed on Size and Book-to-Market
(10 x 10) [Daily]` CSV ZIP from Kenneth French's Data Library and place it in
`data/raw/`. Then run from the repository root:

```bash
python -m A_Data.data_preprocessing \
  --input data/raw/100_Portfolios_10x10_Daily_CSV.zip
```

The preprocessing pipeline keeps the second, equal-weighted daily table,
converts missing-value sentinels, removes `ME9 BM10` and `BIG HiBM`, and
produces 98 asset-return series plus the rolling schedules.

## Reproduce EW and MinVar

```bash
python -m B_EW_MinVar.baseline_models
```

This runs the explicit 126-day rolling backtest over Validation 2
(`2026-01-02` through `2026-06-30`), checks the regression/OLS equivalence of
the minimum-variance solution, and writes outputs to `results/baselines/`.

## Data availability

The processed July 2026 release contains no observations for the reserved
August 2026 final-test period. No final-test result is fabricated. The frozen
pipeline must be rerun when the official final-test data become available.

## Submission policy

The course ZIP and this GitHub repository use the same source files. Generated
caches, operating-system metadata, personal learning notes, draft Word files,
and duplicate copies of the dataset are excluded.
