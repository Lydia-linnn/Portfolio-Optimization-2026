"""Explainable EW and minimum-variance portfolio baselines.

This module consumes the 98-asset return file produced by
``data_preprocessing.py``.  Returns remain in percentage-point units during
estimation: a stored value of 1.00 means a daily return of 1.00%.

The two benchmark portfolios deliberately stay simple:

* Equal weighted (EW) assigns 1/N to every asset and estimates nothing.
* Minimum variance (MinVar) uses the sample covariance matrix from the prior
  126 trading days and permits short positions, as required by the assignment.

The target day's return is never included in its training window.  This is the
key timing rule that makes the recorded returns genuinely out of sample.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from A_Data.data_preprocessing import (
    DEBUG_DATE,
    DEBUG_FIRST_SIX_RETURN,
    DEBUG_FIRST_SIX_WEIGHTS,
    EXPECTED_CLEAN_ASSETS,
    VALIDATION_2_END,
    VALIDATION_2_START,
    WINDOW_LENGTH,
    iter_rolling_windows,
)


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
VERSION = "1.1.0"
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "returns_equal_weighted_98.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "baselines"
)
DEFAULT_VALIDATION_SCHEDULE = (
    DEFAULT_INPUT_PATH.parent / "validation2_rolling_schedule.csv"
)
TRADING_DAYS_PER_YEAR = 252
MODEL_ORDER = ("EW", "MinVar")


@dataclass(frozen=True)
class BaselineBacktestResult:
    """All auditable outputs from one EW/MinVar rolling backtest."""

    daily_returns: pd.DataFrame
    daily_weights: pd.DataFrame
    daily_diagnostics: pd.DataFrame
    performance_summary: pd.DataFrame


def load_returns(
    path: str | Path,
    expected_assets: int = EXPECTED_CLEAN_ASSETS,
) -> pd.DataFrame:
    """Load the teammate-prepared return CSV and validate its structure.

    Missing observations outside the requested backtest period are retained.
    ``iter_rolling_windows`` performs the stricter no-missing-data check for
    every training and target row that is actually used.
    """

    input_path = Path(path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Processed return file not found: {input_path}")

    frame = pd.read_csv(input_path)
    if "date" not in frame.columns:
        raise ValueError("Processed return file must contain a 'date' column")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate columns found: {duplicates}")

    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.set_index("date")
    frame = frame.apply(pd.to_numeric, errors="raise")

    if frame.index.has_duplicates:
        duplicates = frame.index[frame.index.duplicated()].unique().tolist()
        raise ValueError(f"Duplicate dates found: {duplicates[:5]}")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Return dates must be in increasing order")
    if frame.shape[1] != expected_assets:
        raise ValueError(
            f"Expected {expected_assets} assets, found {frame.shape[1]}"
        )
    if frame.columns.isna().any() or any(
        str(name).strip() == "" for name in frame.columns
    ):
        raise ValueError("Asset names must be non-empty")

    remaining_sentinels = int(frame.isin([-99.99, -999.0]).sum().sum())
    if remaining_sentinels:
        raise ValueError(
            f"Processed returns still contain {remaining_sentinels} missing sentinels"
        )

    return frame


def equal_weight(number_of_assets: int) -> np.ndarray:
    """Return the 1/N benchmark portfolio.

    Economically, this portfolio avoids covariance and expected-return
    estimation altogether.  It is therefore a useful benchmark for judging
    whether optimization adds value after estimation error.
    """

    if number_of_assets <= 0:
        raise ValueError("number_of_assets must be positive")
    return np.full(number_of_assets, 1.0 / number_of_assets, dtype=float)


def sample_covariance(train_returns: pd.DataFrame) -> np.ndarray:
    """Estimate the conventional sample covariance matrix (ddof=1)."""

    if train_returns.shape[0] < 2:
        raise ValueError("At least two observations are required for covariance")
    if train_returns.shape[1] < 1:
        raise ValueError("At least one asset is required")
    if train_returns.isna().any().any():
        raise ValueError("Training returns contain missing values")

    covariance = train_returns.cov().to_numpy(dtype=float)
    if not np.isfinite(covariance).all():
        raise ValueError("Covariance matrix contains non-finite values")
    if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0.0):
        raise ValueError("Covariance matrix is not symmetric")
    return covariance


def _minimum_variance_weights_from_covariance(
    covariance: np.ndarray,
) -> np.ndarray:
    """Solve the fully invested, unrestricted global MinVar portfolio."""

    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix")

    ones = np.ones(covariance.shape[0], dtype=float)
    try:
        # Solving Sigma x = 1 is more stable than explicitly forming Sigma^-1.
        minimum_variance_direction = np.linalg.solve(covariance, ones)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "Sample covariance is singular; this inverse-based MinVar solver "
            "cannot be used. No pseudoinverse or regularization was applied"
        ) from exc

    normalizing_constant = float(ones @ minimum_variance_direction)
    if np.isclose(normalizing_constant, 0.0, atol=1e-14):
        raise np.linalg.LinAlgError(
            "MinVar normalization is numerically zero"
        )

    weights = minimum_variance_direction / normalizing_constant
    if not np.isfinite(weights).all():
        raise ValueError("MinVar weights contain non-finite values")
    if not np.isclose(weights.sum(), 1.0, atol=1e-10):
        raise AssertionError("MinVar weights do not satisfy the budget constraint")
    return weights


def minimum_variance_weights(train_returns: pd.DataFrame) -> np.ndarray:
    """Estimate traditional MinVar weights from a return training window."""

    covariance = sample_covariance(train_returns)
    return _minimum_variance_weights_from_covariance(covariance)


def minimum_variance_weights_via_ols(
    train_returns: pd.DataFrame,
) -> np.ndarray:
    """Recover the same MinVar portfolio through the assignment's OLS form.

    For p assets, N contains an identity matrix in its first p-1 rows and a
    final row of -1 values.  With y = R w_EW and X = R N, the OLS residual is
    R(w_EW - N beta) - alpha * 1, where alpha = mean(R w).  Demeaning X and y
    profiles out this intercept, so the residual is centered portfolio return.
    Its squared norm equals (n-1) * w.T @ sample_covariance @ w.  Without the
    intercept, OLS minimizes the second moment, not the variance.

    In this assignment R is (126, 98), N is (98, 97), X is (126, 97), and
    beta has 97 entries.  Since 1.T @ N = 0, w = w_EW - N beta always sums
    to one.  A full-rank fit has 126 - 97 - 1 = 28 residual degrees of freedom.

    Beta is only a mathematical coordinate; it is not an asset weight and has
    no standalone economic interpretation.
    """

    if train_returns.isna().any().any():
        raise ValueError("Training returns contain missing values")

    return_matrix = train_returns.to_numpy(dtype=float)
    number_of_assets = return_matrix.shape[1]
    if number_of_assets < 2:
        raise ValueError("OLS reformulation requires at least two assets")

    equal_weights = equal_weight(number_of_assets)
    transformation = np.vstack(
        [
            np.eye(number_of_assets - 1, dtype=float),
            -np.ones((1, number_of_assets - 1), dtype=float),
        ]
    )
    response = return_matrix @ equal_weights
    design = return_matrix @ transformation

    centered_response = response - response.mean()
    centered_design = design - design.mean(axis=0)
    beta, _, _, _ = np.linalg.lstsq(
        centered_design, centered_response, rcond=None
    )
    weights = equal_weights - transformation @ beta

    if not np.isclose(weights.sum(), 1.0, atol=1e-10):
        raise AssertionError("OLS-transformed weights do not sum to one")
    return weights


def portfolio_return(
    weights: Sequence[float],
    asset_returns: pd.Series | Sequence[float],
) -> float:
    """Calculate one day's portfolio return in percentage points."""

    weight_array = np.asarray(weights, dtype=float)
    return_array = np.asarray(asset_returns, dtype=float)
    if weight_array.ndim != 1 or return_array.ndim != 1:
        raise ValueError("weights and asset_returns must be one-dimensional")
    if weight_array.shape != return_array.shape:
        raise ValueError(
            "weights and asset_returns must have the same number of assets"
        )
    if not np.isfinite(weight_array).all() or not np.isfinite(return_array).all():
        raise ValueError("weights and asset_returns must be finite")
    return float(weight_array @ return_array)


def performance_metrics(
    daily_returns_percentage_points: pd.Series | Sequence[float],
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float | int]:
    """Calculate the assignment metrics from daily percentage-point returns."""

    returns_pp = pd.Series(daily_returns_percentage_points, dtype=float)
    if not np.isfinite(returns_pp.to_numpy()).all():
        raise ValueError(
            "Daily portfolio returns contain missing or non-finite values; "
            "do not drop dates silently when comparing models"
        )
    if len(returns_pp) < 2:
        raise ValueError("At least two daily returns are required")
    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive")

    # Percentage points are divided by 100 only for wealth compounding.
    decimal_returns = returns_pp / 100.0
    growth_factors = 1.0 + decimal_returns
    if (growth_factors <= 0).any():
        raise ValueError("A daily portfolio return is at or below -100%")

    daily_mean_pp = float(returns_pp.mean())
    daily_volatility_pp = float(returns_pp.std(ddof=1))
    if np.isclose(daily_volatility_pp, 0.0):
        daily_sharpe = np.nan
    else:
        # The assignment defines Sharpe as mean/std with a zero risk-free rate.
        daily_sharpe = daily_mean_pp / daily_volatility_pp

    return {
        "n_observations": int(len(returns_pp)),
        "cumulative_return_pct": float((growth_factors.prod() - 1.0) * 100.0),
        "daily_mean_return_pp": daily_mean_pp,
        "daily_volatility_pp": daily_volatility_pp,
        "daily_sharpe": float(daily_sharpe),
        "annualized_volatility_pct": float(
            daily_volatility_pp * np.sqrt(annualization_factor)
        ),
        "annualized_sharpe": float(
            daily_sharpe * np.sqrt(annualization_factor)
        ),
    }


def drifted_weights(
    previous_weights: Sequence[float],
    previous_asset_returns_pp: Sequence[float],
) -> np.ndarray:
    """Holdings immediately before rebalancing, after the previous day's return.

    漂移后的持仓 = 昨日目标权重 * (1 + 昨日资产收益) / (1 + 昨日组合收益)。
    Even EW needs trades to restore equal weights after unequal asset returns.
    No costs, financing charges, or external cash flows are modeled here.
    """

    weights = np.asarray(previous_weights, dtype=float)
    asset_returns = np.asarray(previous_asset_returns_pp, dtype=float)
    portfolio_growth = 1.0 + portfolio_return(weights, asset_returns) / 100.0
    if portfolio_growth <= 0:
        raise ValueError("Cannot rebalance a portfolio with non-positive wealth")
    return weights * (1.0 + asset_returns / 100.0) / portfolio_growth


def weight_diagnostics(
    weights: Sequence[float],
    covariance: np.ndarray,
    previous_weights: Sequence[float] | None = None,
    previous_asset_returns_pp: Sequence[float] | None = None,
) -> dict[str, float | int]:
    """Describe concentration, gross exposure, target changes, and actual trades.

    One-way turnover = half the absolute change from drifted holdings to the
    new target.  Target-weight change is a separate stability diagnostic.
    Both are undefined on the first evaluation day (initial funding excluded).
    """

    weight_array = np.asarray(weights, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (len(weight_array), len(weight_array)):
        raise ValueError("covariance dimensions do not match the weights")

    positive_weights = weight_array[weight_array > 0]
    negative_weights = weight_array[weight_array < 0]
    portfolio_variance = float(weight_array @ covariance @ weight_array)
    estimated_volatility_pp = float(np.sqrt(max(portfolio_variance, 0.0)))

    if (previous_weights is None) != (previous_asset_returns_pp is None):
        raise ValueError("Provide both previous weights and previous asset returns")
    if previous_weights is None:
        one_way_turnover = np.nan
        target_weight_change = np.nan
    else:
        previous_array = np.asarray(previous_weights, dtype=float)
        if previous_array.shape != weight_array.shape:
            raise ValueError("previous_weights dimensions do not match")
        target_weight_change = float(
            0.5 * np.abs(weight_array - previous_array).sum()
        )
        pretrade_weights = drifted_weights(previous_array, previous_asset_returns_pp)
        one_way_turnover = float(0.5 * np.abs(weight_array - pretrade_weights).sum())

    return {
        "weight_sum": float(weight_array.sum()),
        "minimum_weight": float(weight_array.min()),
        "maximum_weight": float(weight_array.max()),
        "maximum_absolute_weight": float(np.abs(weight_array).max()),
        "gross_exposure": float(np.abs(weight_array).sum()),
        "long_exposure": float(positive_weights.sum()),
        "short_exposure": float(-negative_weights.sum()),
        "number_of_short_positions": int((weight_array < 0).sum()),
        "one_way_turnover": one_way_turnover,
        "half_l1_target_weight_change": target_weight_change,
        "estimated_daily_volatility_pp": estimated_volatility_pp,
        "covariance_condition_number": float(np.linalg.cond(covariance)),
    }


def assignment_debug_check(returns: pd.DataFrame) -> dict[str, object]:
    """Reproduce the PDF's six-asset MinVar debugging example."""

    target_date = pd.Timestamp(DEBUG_DATE)
    if target_date not in returns.index:
        raise ValueError(f"Debug date is not present: {DEBUG_DATE}")

    target_position = returns.index.get_loc(target_date)
    if not isinstance(target_position, (int, np.integer)):
        raise ValueError(f"Debug date is not unique: {DEBUG_DATE}")

    train_returns = returns.iloc[
        target_position - WINDOW_LENGTH : target_position, :6
    ]
    target_returns = returns.iloc[target_position, :6]
    weights = minimum_variance_weights(train_returns)
    realized_return_pp = portfolio_return(weights, target_returns)

    weights_match = bool(
        np.allclose(weights, DEBUG_FIRST_SIX_WEIGHTS, atol=0.002, rtol=0.0)
    )
    return_matches = bool(
        np.isclose(
            realized_return_pp,
            DEBUG_FIRST_SIX_RETURN,
            atol=0.002,
            rtol=0.0,
        )
    )
    if not weights_match or not return_matches:
        raise AssertionError(
            "Six-asset debug values do not match the assignment PDF"
        )

    return {
        "asset_names": list(train_returns.columns),
        "weights": weights.tolist(),
        "realized_return_pp": realized_return_pp,
        "weights_match": weights_match,
        "return_matches": return_matches,
    }


def _validate_evaluation_period(
    returns: pd.DataFrame,
    start_date: str,
    end_date: str,
    reference_schedule: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Reject truncated periods; optionally verify teammate A's trading schedule.

    Boundary checks only roll weekends, not exchange holidays.  If the file
    ends on a holiday boundary, require a later data release instead of assuming
    coverage.  Internal missing trading dates require a reference schedule;
    observed dates alone cannot distinguish missing rows from market holidays.
    """

    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("Evaluation start/end must be valid and start <= end")
    if returns.empty:
        raise ValueError("Return dataset is empty")
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError("Return dates must be unique and increasing")
    evaluation_dates = returns.loc[start:end].index
    if evaluation_dates.empty:
        raise ValueError(
            f"No observations are available between {start_date} and {end_date}; "
            f"the processed data end on {returns.index.max().date()}"
        )
    first_required = pd.offsets.BDay().rollforward(start)
    last_required = pd.offsets.BDay().rollback(end)
    if returns.index.min() > first_required or returns.index.max() < last_required:
        raise ValueError(
            f"Incomplete evaluation period {start_date} to {end_date}; "
            f"dataset covers {returns.index.min().date()} to "
            f"{returns.index.max().date()}. Do not report partial results as complete."
        )
    if reference_schedule is None:
        return None
    required = {"target_date", "train_start", "train_end", "n_train_days", "n_assets"}
    if not required.issubset(reference_schedule.columns):
        raise ValueError(f"Rolling schedule must contain {sorted(required)}")
    schedule = reference_schedule.copy()
    for column in ("target_date", "train_start", "train_end"):
        schedule[column] = pd.to_datetime(schedule[column], errors="raise")
        if schedule[column].isna().any():
            raise ValueError(f"Rolling schedule contains missing {column}")
    if schedule.target_date.duplicated().any():
        raise ValueError("Rolling schedule has duplicate target dates")
    schedule = schedule.set_index("target_date").sort_index().loc[start:end]
    if not schedule.index.equals(evaluation_dates):
        raise ValueError("Evaluation dates do not match the reference rolling schedule")
    return schedule


def run_baseline_backtest(
    returns: pd.DataFrame,
    start_date: str = VALIDATION_2_START,
    end_date: str = VALIDATION_2_END,
    window_length: int = WINDOW_LENGTH,
    verify_ols_equivalence: bool = True,
    ols_tolerance: float = 1e-10,
    reference_schedule: pd.DataFrame | None = None,
) -> BaselineBacktestResult:
    """Run the daily, one-step-ahead EW and MinVar backtest."""

    schedule = _validate_evaluation_period(
        returns, start_date, end_date, reference_schedule
    )
    asset_names = list(returns.columns)
    number_of_assets = len(asset_names)
    equal_weights = equal_weight(number_of_assets)

    return_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []

    previous_minvar_weights: np.ndarray | None = None
    previous_asset_returns_pp: np.ndarray | None = None
    ew_wealth_index = 1.0
    minvar_wealth_index = 1.0

    for window in iter_rolling_windows(
        returns,
        evaluation_start=start_date,
        evaluation_end=end_date,
        window_length=window_length,
    ):
        train_returns = window.train
        next_day_returns = window.test
        if schedule is not None:
            expected = schedule.loc[window.target_date]
            if (
                train_returns.index[0] != expected.train_start
                or train_returns.index[-1] != expected.train_end
                or len(train_returns) != expected.n_train_days
                or train_returns.shape[1] != expected.n_assets
            ):
                raise ValueError(
                    f"Training window differs from reference schedule on "
                    f"{window.target_date.date()}"
                )
        covariance = sample_covariance(train_returns)

        try:
            minvar_weights = _minimum_variance_weights_from_covariance(covariance)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                f"MinVar failed on {window.target_date.date()}: {exc}"
            ) from exc
        if verify_ols_equivalence:
            ols_weights = minimum_variance_weights_via_ols(train_returns)
            ols_max_abs_difference = float(
                np.max(np.abs(minvar_weights - ols_weights))
            )
            if ols_max_abs_difference > ols_tolerance:
                raise AssertionError(
                    f"Direct and OLS MinVar differ on "
                    f"{window.target_date.date()}: "
                    f"{ols_max_abs_difference:.3e}"
                )
        else:
            ols_max_abs_difference = np.nan

        ew_return_pp = portfolio_return(equal_weights, next_day_returns)
        minvar_return_pp = portfolio_return(minvar_weights, next_day_returns)

        ew_wealth_index *= 1.0 + ew_return_pp / 100.0
        minvar_wealth_index *= 1.0 + minvar_return_pp / 100.0

        train_start = train_returns.index[0]
        train_end = train_returns.index[-1]
        for model, daily_return_pp, wealth_index in (
            ("EW", ew_return_pp, ew_wealth_index),
            ("MinVar", minvar_return_pp, minvar_wealth_index),
        ):
            return_rows.append(
                {
                    "date": window.target_date,
                    "model": model,
                    "train_start": train_start,
                    "train_end": train_end,
                    "portfolio_return_pp": daily_return_pp,
                    "wealth_index": wealth_index,
                }
            )

        for model, model_weights in (
            ("EW", equal_weights),
            ("MinVar", minvar_weights),
        ):
            for asset_name, weight in zip(
                asset_names, model_weights, strict=True
            ):
                weight_rows.append(
                    {
                        "date": window.target_date,
                        "model": model,
                        "asset": asset_name,
                        "weight": float(weight),
                    }
                )

        ew_diagnostics = weight_diagnostics(
            equal_weights,
            covariance,
            previous_weights=(
                equal_weights if previous_asset_returns_pp is not None else None
            ),
            previous_asset_returns_pp=previous_asset_returns_pp,
        )
        minvar_diagnostics = weight_diagnostics(
            minvar_weights,
            covariance,
            previous_weights=previous_minvar_weights,
            previous_asset_returns_pp=previous_asset_returns_pp,
        )
        for model, diagnostics, ols_difference in (
            ("EW", ew_diagnostics, np.nan),
            ("MinVar", minvar_diagnostics, ols_max_abs_difference),
        ):
            diagnostic_rows.append(
                {
                    "date": window.target_date,
                    "model": model,
                    **diagnostics,
                    "ols_max_abs_weight_difference": ols_difference,
                }
            )

        previous_minvar_weights = minvar_weights.copy()
        previous_asset_returns_pp = next_day_returns.to_numpy(dtype=float).copy()

    if not return_rows:
        data_end = returns.index.max().date().isoformat()
        raise ValueError(
            f"No observations are available between {start_date} and "
            f"{end_date}; the processed data end on {data_end}"
        )

    daily_returns = pd.DataFrame(return_rows)
    daily_weights = pd.DataFrame(weight_rows)
    daily_diagnostics = pd.DataFrame(diagnostic_rows)

    summary_rows = []
    for model in MODEL_ORDER:
        model_returns = daily_returns.loc[
            daily_returns["model"] == model, "portfolio_return_pp"
        ]
        summary_rows.append(
            {
                "model": model,
                "evaluation_start": daily_returns["date"].min().date().isoformat(),
                "evaluation_end": daily_returns["date"].max().date().isoformat(),
                "n_assets": number_of_assets,
                "window_length": window_length,
                "reference_schedule_checked": schedule is not None,
                **performance_metrics(model_returns),
            }
        )
    performance_summary = pd.DataFrame(summary_rows)

    return BaselineBacktestResult(
        daily_returns=daily_returns,
        daily_weights=daily_weights,
        daily_diagnostics=daily_diagnostics,
        performance_summary=performance_summary,
    )


def save_backtest_outputs(
    result: BaselineBacktestResult,
    output_dir: str | Path,
    output_prefix: str = "validation2",
) -> dict[str, Path]:
    """Save tidy CSV outputs that other model scripts can reuse."""

    if not output_prefix or any(character.isspace() for character in output_prefix):
        raise ValueError("output_prefix must be non-empty and contain no spaces")

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    paths = {
        "returns": destination / f"{output_prefix}_returns.csv",
        "weights": destination / f"{output_prefix}_weights.csv",
        "metrics": destination / f"{output_prefix}_metrics.csv",
        "diagnostics": destination / f"{output_prefix}_weight_diagnostics.csv",
    }
    result.daily_returns.to_csv(
        paths["returns"], index=False, date_format="%Y-%m-%d"
    )
    result.daily_weights.to_csv(
        paths["weights"], index=False, date_format="%Y-%m-%d"
    )
    result.performance_summary.to_csv(paths["metrics"], index=False)
    result.daily_diagnostics.to_csv(
        paths["diagnostics"], index=False, date_format="%Y-%m-%d"
    )
    return paths


def summarize_minvar_diagnostics(
    result: BaselineBacktestResult,
) -> pd.Series:
    """Return the weight statistics used in the benchmark interpretation."""

    minvar = result.daily_diagnostics.loc[
        result.daily_diagnostics["model"] == "MinVar"
    ]
    minvar_weights = result.daily_weights.loc[
        result.daily_weights["model"] == "MinVar", "weight"
    ]
    minvar_returns = result.daily_returns.loc[
        result.daily_returns["model"] == "MinVar", "portfolio_return_pp"
    ]

    return pd.Series(
        {
            "median_maximum_absolute_weight": minvar[
                "maximum_absolute_weight"
            ].median(),
            "maximum_positive_weight": minvar_weights.max(),
            "minimum_negative_weight": minvar_weights.min(),
            "median_gross_exposure": minvar["gross_exposure"].median(),
            "median_short_exposure": minvar["short_exposure"].median(),
            "median_number_of_short_positions": minvar[
                "number_of_short_positions"
            ].median(),
            "median_one_way_turnover": minvar["one_way_turnover"].median(),
            "median_half_l1_target_weight_change": minvar[
                "half_l1_target_weight_change"
            ].median(),
            "median_covariance_condition_number": minvar[
                "covariance_condition_number"
            ].median(),
            "average_estimated_daily_volatility_pp": minvar[
                "estimated_daily_volatility_pp"
            ].mean(),
            "root_mean_estimated_daily_variance_pp": np.sqrt(
                (minvar["estimated_daily_volatility_pp"] ** 2).mean()
            ),
            "realized_oos_daily_volatility_pp": minvar_returns.std(ddof=1),
            "maximum_ols_weight_difference": minvar[
                "ols_max_abs_weight_difference"
            ].max(),
        },
        dtype=float,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run explainable equal-weighted and traditional minimum-variance "
            "rolling portfolio benchmarks."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Processed 98-asset return CSV (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--start", default=VALIDATION_2_START)
    parser.add_argument("--end", default=VALIDATION_2_END)
    parser.add_argument("--window-length", type=int, default=WINDOW_LENGTH)
    parser.add_argument("--output-prefix", default="validation2")
    parser.add_argument(
        "--schedule", type=Path,
        help="Reference rolling schedule; Validation 2 defaults to teammate A's file",
    )
    parser.add_argument(
        "--skip-ols-check",
        action="store_true",
        help="Skip the direct-MinVar versus OLS equivalence check",
    )
    parser.add_argument(
        "--skip-assignment-debug",
        action="store_true",
        help="Skip the PDF's six-asset debugging check",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    returns = load_returns(args.input)

    if not args.skip_assignment_debug:
        debug = assignment_debug_check(returns)
        print("Assignment six-asset debug: passed")
        print(
            "  realised return (percentage points): "
            f"{debug['realized_return_pp']:.6f}"
        )

    schedule_path = args.schedule
    if (
        schedule_path is None
        and pd.Timestamp(VALIDATION_2_START) <= pd.Timestamp(args.start)
        and pd.Timestamp(args.end) <= pd.Timestamp(VALIDATION_2_END)
    ):
        schedule_path = DEFAULT_VALIDATION_SCHEDULE
    reference_schedule = pd.read_csv(schedule_path) if schedule_path else None

    result = run_baseline_backtest(
        returns,
        start_date=args.start,
        end_date=args.end,
        window_length=args.window_length,
        verify_ols_equivalence=not args.skip_ols_check,
        reference_schedule=reference_schedule,
    )
    output_paths = save_backtest_outputs(
        result,
        args.output_dir,
        output_prefix=args.output_prefix,
    )

    print(f"Baseline version: {VERSION}")
    print(f"Input: {Path(args.input).expanduser().resolve()}")
    print(f"Reference schedule: {schedule_path or 'not supplied (boundary checks only)'}")
    print(f"Assets: {returns.shape[1]}")
    print(f"Evaluation period: {args.start} to {args.end}")
    print("\nPerformance summary:")
    print(result.performance_summary.to_string(index=False))
    print("\nMinVar weight diagnostics:")
    print(summarize_minvar_diagnostics(result).to_string())
    print("\nSaved outputs:")
    for label, path in output_paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
