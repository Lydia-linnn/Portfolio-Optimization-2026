"""Data preparation and rolling-window utilities for DBA5106 Project 1.

This module owns only the data and experimental-design layer. It deliberately
does not implement the team's final LASSO/Ridge portfolio models.

The Ken French source file stores returns in percentage points: a value of
1.25 means 1.25%, not 125%. The processed output preserves that scale because
the assignment's regularization grid is defined for it. Use
to_decimal_returns only for wealth compounding or APIs that require decimals.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterator, Sequence
from zipfile import ZipFile

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EQUAL_WEIGHTED_MARKER = "Average Equal Weighted Returns -- Daily"
MISSING_SENTINELS = (-99.99, -999.0)
ASSETS_TO_DROP = ("ME9 BM10", "BIG HiBM")
EXPECTED_RAW_ASSETS = 100
EXPECTED_CLEAN_ASSETS = 98
WINDOW_LENGTH = 126

VALIDATION_2_START = "2026-01-02"
VALIDATION_2_END = "2026-06-30"
FINAL_TEST_START = "2026-08-01"
FINAL_TEST_END = "2026-08-30"

DEBUG_DATE = "2026-01-02"
DEBUG_ONE_BASED_ROW = 26152
DEBUG_TRAIN_START = "2025-07-03"
DEBUG_TRAIN_END = "2025-12-31"
DEBUG_FIRST_SIX_WEIGHTS = np.array(
    [-0.111, -0.199, 0.348, 0.460, 0.006, 0.496], dtype=float
)
DEBUG_FIRST_SIX_RETURN = 1.393


@dataclass(frozen=True)
class RollingWindow:
    """One no-look-ahead training/test pair."""

    target_date: pd.Timestamp
    train: pd.DataFrame
    test: pd.Series


def _normalise_asset_name(name: str) -> str:
    """Normalise whitespace and case for safe column matching."""

    return re.sub(r"\s+", "", str(name)).upper()


def read_equal_weighted_daily_zip(zip_path: str | Path) -> pd.DataFrame:
    """Read the second (equal-weighted daily) table from the official ZIP."""

    zip_path = Path(zip_path).expanduser().resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    with ZipFile(zip_path) as archive:
        csv_names = [
            name for name in archive.namelist() if name.lower().endswith(".csv")
        ]
        if len(csv_names) != 1:
            raise ValueError(
                "Expected exactly one CSV inside the ZIP; "
                f"found {len(csv_names)}: {csv_names}"
            )
        raw_text = archive.read(csv_names[0]).decode("utf-8-sig", errors="replace")

    if EQUAL_WEIGHTED_MARKER not in raw_text:
        raise ValueError(
            f"Could not find section marker: {EQUAL_WEIGHTED_MARKER!r}"
        )

    section = raw_text.split(EQUAL_WEIGHTED_MARKER, 1)[1].splitlines()
    try:
        header_index = next(
            index
            for index, line in enumerate(section)
            if "SMALL LoBM" in line and "," in line
        )
    except StopIteration as exc:
        raise ValueError("Could not locate the equal-weighted table header") from exc

    table_lines = [section[header_index]]
    for line in section[header_index + 1 :]:
        if re.match(r"^\s*\d{8},", line):
            table_lines.append(line)
        elif len(table_lines) > 1:
            break

    if len(table_lines) == 1:
        raise ValueError("The equal-weighted section contains no daily observations")

    frame = pd.read_csv(StringIO("\n".join(table_lines)))
    frame.columns = [str(column).strip() for column in frame.columns]
    date_column = frame.columns[0]
    frame = frame.rename(columns={date_column: "date"})

    date_text = (
        frame["date"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    )
    frame["date"] = pd.to_datetime(date_text, format="%Y%m%d", errors="raise")
    frame = frame.set_index("date").sort_index()
    frame = frame.apply(pd.to_numeric, errors="raise")

    if frame.index.has_duplicates:
        duplicate_dates = frame.index[frame.index.duplicated()].unique().tolist()
        raise ValueError(f"Duplicate dates detected: {duplicate_dates[:5]}")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Dates are not in increasing order")
    if frame.shape[1] != EXPECTED_RAW_ASSETS:
        raise ValueError(
            f"Expected {EXPECTED_RAW_ASSETS} raw assets, found {frame.shape[1]}"
        )

    return frame


def clean_returns(
    raw_returns: pd.DataFrame,
    assets_to_drop: Sequence[str] = ASSETS_TO_DROP,
) -> pd.DataFrame:
    """Replace official missing sentinels and remove specified assets.

    Rows are never dropped and missing values are never silently imputed.
    """

    clean = raw_returns.copy().replace(list(MISSING_SENTINELS), np.nan)

    normalised_to_original: dict[str, str] = {}
    for column in clean.columns:
        normalised = _normalise_asset_name(column)
        if normalised in normalised_to_original:
            raise ValueError(f"Duplicate normalised asset name: {normalised}")
        normalised_to_original[normalised] = str(column)

    wanted = [_normalise_asset_name(name) for name in assets_to_drop]
    missing_columns = [name for name in wanted if name not in normalised_to_original]
    if missing_columns:
        raise ValueError(
            "Required deletion columns were not found: " + ", ".join(missing_columns)
        )

    drop_columns = [normalised_to_original[name] for name in wanted]
    clean = clean.drop(columns=drop_columns)

    if clean.shape[1] != EXPECTED_CLEAN_ASSETS:
        raise ValueError(
            f"Expected {EXPECTED_CLEAN_ASSETS} cleaned assets, found {clean.shape[1]}"
        )
    return clean


def to_decimal_returns(percentage_point_returns: pd.DataFrame | pd.Series):
    """Convert percentage points to decimal returns (1.25 -> 0.0125)."""

    return percentage_point_returns / 100.0


def iter_rolling_windows(
    returns: pd.DataFrame,
    evaluation_start: str,
    evaluation_end: str,
    window_length: int = WINDOW_LENGTH,
) -> Iterator[RollingWindow]:
    """Yield fixed-length windows without using the target day's return."""

    if window_length <= 0:
        raise ValueError("window_length must be positive")
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError("Return index must be unique and increasing")

    evaluation = returns.loc[evaluation_start:evaluation_end]
    if evaluation.empty:
        return

    for target_date in evaluation.index:
        position = returns.index.get_loc(target_date)
        if not isinstance(position, (int, np.integer)):
            raise ValueError(f"Date lookup was not unique: {target_date.date()}")
        if position < window_length:
            raise ValueError(
                f"Insufficient history for {target_date.date()}: "
                f"need {window_length}, have {position}"
            )

        train = returns.iloc[position - window_length : position].copy()
        test = returns.iloc[position].copy()
        if train.shape != (window_length, returns.shape[1]):
            raise AssertionError("Unexpected rolling-window dimensions")

        train_missing = int(train.isna().sum().sum())
        test_missing = int(test.isna().sum())
        if train_missing or test_missing:
            raise ValueError(
                f"Missing data for target {target_date.date()}: "
                f"train={train_missing}, test={test_missing}"
            )

        yield RollingWindow(target_date=target_date, train=train, test=test)


def rolling_schedule(
    returns: pd.DataFrame,
    evaluation_start: str,
    evaluation_end: str,
    window_length: int = WINDOW_LENGTH,
) -> pd.DataFrame:
    """Return auditable dates and dimensions for every rolling iteration."""

    columns = [
        "target_date",
        "train_start",
        "train_end",
        "n_train_days",
        "n_assets",
    ]
    rows = []
    for window in iter_rolling_windows(
        returns, evaluation_start, evaluation_end, window_length
    ):
        rows.append(
            {
                "target_date": window.target_date,
                "train_start": window.train.index[0],
                "train_end": window.train.index[-1],
                "n_train_days": len(window.train),
                "n_assets": window.train.shape[1],
            }
        )
    return pd.DataFrame(rows, columns=columns)


def first_six_minvar_debug(returns: pd.DataFrame) -> dict[str, object]:
    """Reproduce the assignment's six-asset 2026-01-02 debug calculation."""

    position = returns.index.get_loc(pd.Timestamp(DEBUG_DATE))
    if not isinstance(position, (int, np.integer)):
        raise ValueError(f"Date lookup was not unique: {DEBUG_DATE}")

    train = returns.iloc[position - WINDOW_LENGTH : position, :6]
    test = returns.iloc[position, :6]
    if train.isna().any().any() or test.isna().any():
        raise ValueError("Missing values prevent the six-asset debug check")

    covariance = train.cov().to_numpy()
    ones = np.ones(6)
    direction = np.linalg.solve(covariance, ones)
    weights = direction / (ones @ direction)
    realised_return = float(test.to_numpy() @ weights)

    return {
        "asset_names": list(train.columns),
        "weights": weights.tolist(),
        "realised_return_percentage_points": realised_return,
        "weights_match_assignment": bool(
            np.allclose(weights, DEBUG_FIRST_SIX_WEIGHTS, atol=0.002)
        ),
        "return_matches_assignment": bool(
            np.isclose(realised_return, DEBUG_FIRST_SIX_RETURN, atol=0.002)
        ),
    }


def validate_dataset(returns: pd.DataFrame) -> dict[str, object]:
    """Validate dimensions, key dates, missingness, and debug values."""

    if returns.shape[1] != EXPECTED_CLEAN_ASSETS:
        raise AssertionError(
            f"Expected {EXPECTED_CLEAN_ASSETS} assets, found {returns.shape[1]}"
        )

    debug_timestamp = pd.Timestamp(DEBUG_DATE)
    if debug_timestamp not in returns.index:
        raise AssertionError(f"Missing debug date: {DEBUG_DATE}")
    debug_position = returns.index.get_loc(debug_timestamp)
    if int(debug_position) + 1 != DEBUG_ONE_BASED_ROW:
        raise AssertionError(
            f"{DEBUG_DATE} should be row {DEBUG_ONE_BASED_ROW}, "
            f"found {int(debug_position) + 1}"
        )

    first_window = next(
        iter_rolling_windows(returns, DEBUG_DATE, DEBUG_DATE, WINDOW_LENGTH)
    )
    if first_window.train.index[0] != pd.Timestamp(DEBUG_TRAIN_START):
        raise AssertionError("Unexpected first training-window start")
    if first_window.train.index[-1] != pd.Timestamp(DEBUG_TRAIN_END):
        raise AssertionError("Unexpected first training-window end")

    validation2 = returns.loc[VALIDATION_2_START:VALIDATION_2_END]
    if validation2.empty:
        raise AssertionError("Validation 2 contains no observations")
    validation2_missing = int(validation2.isna().sum().sum())
    if validation2_missing:
        raise AssertionError(
            f"Validation 2 contains {validation2_missing} missing values"
        )

    debug = first_six_minvar_debug(returns)
    if not debug["weights_match_assignment"] or not debug["return_matches_assignment"]:
        raise AssertionError(
            "Six-asset MinVar debug values do not match the assignment; "
            "check the data table and release version"
        )

    final_test = returns.loc[FINAL_TEST_START:FINAL_TEST_END]
    return {
        "data_start": returns.index.min().date().isoformat(),
        "data_end": returns.index.max().date().isoformat(),
        "n_daily_observations": int(len(returns)),
        "n_assets": int(returns.shape[1]),
        "return_unit": "percentage points (1.00 means 1.00%)",
        "dropped_assets": list(ASSETS_TO_DROP),
        "debug_date_one_based_row": int(debug_position) + 1,
        "first_training_start": first_window.train.index[0].date().isoformat(),
        "first_training_end": first_window.train.index[-1].date().isoformat(),
        "window_length": WINDOW_LENGTH,
        "validation_2_start": VALIDATION_2_START,
        "validation_2_end": VALIDATION_2_END,
        "validation_2_observations": int(len(validation2)),
        "final_test_start": FINAL_TEST_START,
        "final_test_end": FINAL_TEST_END,
        "final_test_observations_currently_available": int(len(final_test)),
        "six_asset_debug": debug,
    }


def preprocess(zip_path: str | Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run the complete read-clean-validate pipeline."""

    raw = read_equal_weighted_daily_zip(zip_path)
    clean = clean_returns(raw)
    summary = validate_dataset(clean)
    return clean, summary


def save_outputs(
    returns: pd.DataFrame,
    summary: dict[str, object],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save processed returns, metadata, and window schedules."""

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    returns_path = output_dir / "returns_equal_weighted_98.csv"
    summary_path = output_dir / "preprocessing_summary.json"
    validation_schedule_path = output_dir / "validation2_rolling_schedule.csv"
    final_schedule_path = output_dir / "final_test_rolling_schedule.csv"

    returns.to_csv(returns_path, index_label="date")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rolling_schedule(
        returns, VALIDATION_2_START, VALIDATION_2_END
    ).to_csv(validation_schedule_path, index=False)
    rolling_schedule(
        returns, FINAL_TEST_START, FINAL_TEST_END
    ).to_csv(final_schedule_path, index=False)

    return {
        "returns": returns_path,
        "summary": summary_path,
        "validation_schedule": validation_schedule_path,
        "final_schedule": final_schedule_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the Ken French 100 Portfolios 10x10 Daily data."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to 100_Portfolios_10x10_Daily_CSV.zip",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Directory for cleaned data and schedules (default: data/processed)",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    returns, summary = preprocess(args.input)
    outputs = save_outputs(returns, summary, args.output_dir)

    print("Data preprocessing completed successfully.")
    print(f"Observations: {summary['n_daily_observations']}")
    print(f"Assets: {summary['n_assets']}")
    print(f"Date range: {summary['data_start']} to {summary['data_end']}")
    print(
        "Six-asset debug passed:",
        summary["six_asset_debug"]["weights_match_assignment"]
        and summary["six_asset_debug"]["return_matches_assignment"],
    )
    print("Outputs:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
