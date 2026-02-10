"""
Advanced validation utilities for strategy evaluation.

This module provides:
- walk-forward split generation
- purged cross-validation split generation with embargo
- regime-sliced return evaluation
"""

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationFold:
    """A single validation fold definition."""

    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray

    @property
    def train_size(self) -> int:
        return int(len(self.train_indices))

    @property
    def test_size(self) -> int:
        return int(len(self.test_indices))


@dataclass(frozen=True)
class FoldEvaluation:
    """Metrics emitted for one validation fold."""

    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    metrics: Dict[str, float]


def _validate_positive_int(name: str, value: int) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be > 0")


def _to_datetime_index(index: Sequence) -> pd.DatetimeIndex:
    if isinstance(index, pd.DatetimeIndex):
        return index
    return pd.DatetimeIndex(pd.to_datetime(list(index)))


def generate_walk_forward_splits(
    index: Sequence,
    *,
    train_size: int,
    test_size: int,
    step_size: int = None,
    gap: int = 0,
    expanding: bool = True,
) -> List[ValidationFold]:
    """
    Generate walk-forward folds.

    Args:
        index: Sequence of timestamps/labels.
        train_size: Initial train window size.
        test_size: Test window size for each fold.
        step_size: Forward movement after each fold (defaults to test_size).
        gap: Purge gap between train and test sections.
        expanding: If True, train starts at index 0 and expands.

    Returns:
        List of ValidationFold.
    """
    _validate_positive_int("train_size", train_size)
    _validate_positive_int("test_size", test_size)
    if step_size is None:
        step_size = test_size
    _validate_positive_int("step_size", step_size)
    if gap < 0:
        raise ValueError("gap must be >= 0")

    dt_index = _to_datetime_index(index)
    n = len(dt_index)
    if n < train_size + gap + test_size:
        return []

    folds: List[ValidationFold] = []
    fold_id = 0
    train_end = train_size

    while True:
        train_start = 0 if expanding else max(0, train_end - train_size)
        test_start = train_end + gap
        test_end = test_start + test_size
        if test_end > n:
            break

        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) > 0 and len(test_idx) > 0:
            folds.append(
                ValidationFold(
                    fold_id=fold_id,
                    train_indices=train_idx,
                    test_indices=test_idx,
                )
            )
            fold_id += 1

        train_end += step_size

    return folds


def generate_purged_kfold_splits(
    index: Sequence,
    *,
    n_splits: int = 5,
    purge_window: int = 0,
    embargo: int = 0,
    min_train_size: int = 1,
) -> List[ValidationFold]:
    """
    Generate purged K-fold splits with embargo to avoid leakage.

    Train section excludes:
    - `purge_window` samples immediately before test fold
    - `embargo` samples immediately after test fold
    """
    _validate_positive_int("n_splits", n_splits)
    _validate_positive_int("min_train_size", min_train_size)
    if purge_window < 0:
        raise ValueError("purge_window must be >= 0")
    if embargo < 0:
        raise ValueError("embargo must be >= 0")

    dt_index = _to_datetime_index(index)
    n = len(dt_index)
    if n < n_splits:
        raise ValueError("n_splits cannot exceed sample count")

    all_indices = np.arange(n)
    test_folds = np.array_split(all_indices, n_splits)

    folds: List[ValidationFold] = []
    for fold_id, test_idx in enumerate(test_folds):
        if len(test_idx) == 0:
            continue
        test_start = int(test_idx[0])
        test_end = int(test_idx[-1]) + 1

        left_train_end = max(0, test_start - purge_window)
        right_train_start = min(n, test_end + embargo)

        left_train = all_indices[:left_train_end]
        right_train = all_indices[right_train_start:]
        train_idx = np.concatenate([left_train, right_train])

        if len(train_idx) < min_train_size:
            continue

        folds.append(
            ValidationFold(
                fold_id=fold_id,
                train_indices=train_idx,
                test_indices=test_idx,
            )
        )

    return folds


def regime_sliced_evaluation(
    returns: Sequence[float],
    regimes: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate return quality by market regime.
    """
    if len(returns) != len(regimes):
        raise ValueError("returns and regimes must have the same length")

    grouped: Dict[str, List[float]] = {}
    for ret, regime in zip(returns, regimes):
        grouped.setdefault(str(regime), []).append(float(ret))

    summary: Dict[str, Dict[str, float]] = {}
    for regime, values in grouped.items():
        if not values:
            continue
        wins = sum(1 for v in values if v > 0)
        avg = mean(values)
        vol = pstdev(values) if len(values) > 1 else 0.0
        sharpe_like = (avg / vol) if vol > 0 else 0.0
        summary[regime] = {
            "sample_count": float(len(values)),
            "mean_return": float(avg),
            "volatility": float(vol),
            "win_rate": float(wins / len(values)),
            "sharpe_like": float(sharpe_like),
        }
    return summary


class AdvancedValidationFramework:
    """High-level helpers to execute split-based strategy evaluation."""

    def __init__(self, timestamp_col: str = "timestamp"):
        self.timestamp_col = timestamp_col

    def _fold_to_evaluation(
        self,
        *,
        fold: ValidationFold,
        frame: pd.DataFrame,
        metrics: Mapping[str, float],
    ) -> FoldEvaluation:
        ts = pd.to_datetime(frame[self.timestamp_col])
        train_start = ts.iloc[int(fold.train_indices[0])]
        train_end = ts.iloc[int(fold.train_indices[-1])]
        test_start = ts.iloc[int(fold.test_indices[0])]
        test_end = ts.iloc[int(fold.test_indices[-1])]
        return FoldEvaluation(
            fold_id=fold.fold_id,
            train_start=train_start.isoformat(),
            train_end=train_end.isoformat(),
            test_start=test_start.isoformat(),
            test_end=test_end.isoformat(),
            metrics={k: float(v) for k, v in metrics.items()},
        )

    def walk_forward_validate(
        self,
        *,
        frame: pd.DataFrame,
        evaluator: Callable[[pd.DataFrame, pd.DataFrame], Mapping[str, float]],
        train_size: int,
        test_size: int,
        step_size: int = None,
        gap: int = 0,
        expanding: bool = True,
    ) -> List[FoldEvaluation]:
        if self.timestamp_col not in frame.columns:
            raise ValueError(f"DataFrame must contain '{self.timestamp_col}'")

        folds = generate_walk_forward_splits(
            frame[self.timestamp_col],
            train_size=train_size,
            test_size=test_size,
            step_size=step_size,
            gap=gap,
            expanding=expanding,
        )

        evaluations: List[FoldEvaluation] = []
        for fold in folds:
            train_df = frame.iloc[fold.train_indices].copy()
            test_df = frame.iloc[fold.test_indices].copy()
            metrics = evaluator(train_df, test_df)
            evaluations.append(
                self._fold_to_evaluation(fold=fold, frame=frame, metrics=metrics)
            )
        return evaluations

    def purged_cv_validate(
        self,
        *,
        frame: pd.DataFrame,
        evaluator: Callable[[pd.DataFrame, pd.DataFrame], Mapping[str, float]],
        n_splits: int = 5,
        purge_window: int = 0,
        embargo: int = 0,
        min_train_size: int = 1,
    ) -> List[FoldEvaluation]:
        if self.timestamp_col not in frame.columns:
            raise ValueError(f"DataFrame must contain '{self.timestamp_col}'")

        folds = generate_purged_kfold_splits(
            frame[self.timestamp_col],
            n_splits=n_splits,
            purge_window=purge_window,
            embargo=embargo,
            min_train_size=min_train_size,
        )

        evaluations: List[FoldEvaluation] = []
        for fold in folds:
            train_df = frame.iloc[fold.train_indices].copy()
            test_df = frame.iloc[fold.test_indices].copy()
            metrics = evaluator(train_df, test_df)
            evaluations.append(
                self._fold_to_evaluation(fold=fold, frame=frame, metrics=metrics)
            )
        return evaluations

    @staticmethod
    def summarize_metrics(folds: Iterable[FoldEvaluation]) -> Dict[str, Dict[str, float]]:
        """Aggregate fold metrics with mean/std per metric key."""
        fold_list = list(folds)
        if not fold_list:
            return {}

        metric_keys = set()
        for fold in fold_list:
            metric_keys.update(fold.metrics.keys())

        summary: Dict[str, Dict[str, float]] = {}
        for key in sorted(metric_keys):
            values = [float(fold.metrics[key]) for fold in fold_list if key in fold.metrics]
            if not values:
                continue
            metric_mean = mean(values)
            metric_std = pstdev(values) if len(values) > 1 else 0.0
            summary[key] = {
                "mean": float(metric_mean),
                "std": float(metric_std),
                "min": float(min(values)),
                "max": float(max(values)),
                "fold_count": float(len(values)),
            }
        return summary
