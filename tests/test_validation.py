import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.backtesting.validation import (
    AdvancedValidationFramework,
    generate_purged_kfold_splits,
    generate_walk_forward_splits,
    regime_sliced_evaluation,
)


def _frame(length: int = 40) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=length, freq="H")
    close = [100.0 + (i * 0.5) for i in range(length)]
    returns = [0.0] + [((close[i] / close[i - 1]) - 1.0) for i in range(1, length)]
    return pd.DataFrame({"timestamp": ts, "close": close, "returns": returns})


def test_walk_forward_splits_expand_without_overlap():
    frame = _frame(30)
    folds = generate_walk_forward_splits(
        frame["timestamp"],
        train_size=10,
        test_size=5,
        step_size=5,
        gap=1,
        expanding=True,
    )

    assert len(folds) == 3
    assert [fold.train_size for fold in folds] == [10, 15, 20]
    for fold in folds:
        assert int(fold.train_indices[-1]) < int(fold.test_indices[0])


def test_walk_forward_splits_rolling_window_train_size_constant():
    frame = _frame(24)
    folds = generate_walk_forward_splits(
        frame["timestamp"],
        train_size=8,
        test_size=4,
        step_size=4,
        gap=0,
        expanding=False,
    )

    assert len(folds) == 4
    assert all(fold.train_size == 8 for fold in folds)
    assert all(fold.test_size == 4 for fold in folds)


def test_purged_kfold_excludes_neighboring_samples():
    frame = _frame(20)
    folds = generate_purged_kfold_splits(
        frame["timestamp"],
        n_splits=4,
        purge_window=2,
        embargo=1,
        min_train_size=1,
    )

    assert len(folds) == 4
    for fold in folds:
        test_start = int(fold.test_indices[0])
        test_end = int(fold.test_indices[-1]) + 1
        purged_from = max(0, test_start - 2)
        purged_to = min(len(frame), test_end + 1)
        forbidden = set(range(purged_from, purged_to))
        assert forbidden.isdisjoint(set(int(i) for i in fold.train_indices))


def test_regime_sliced_evaluation_returns_per_regime_metrics():
    returns = [0.01, 0.02, -0.01, -0.03, 0.01, -0.02]
    regimes = ["BULL", "BULL", "SIDEWAYS", "BEAR", "SIDEWAYS", "BEAR"]
    result = regime_sliced_evaluation(returns, regimes)

    assert set(result.keys()) == {"BULL", "SIDEWAYS", "BEAR"}
    assert result["BULL"]["sample_count"] == 2.0
    assert result["BULL"]["mean_return"] > 0
    assert result["BEAR"]["mean_return"] < 0


def test_framework_walk_forward_and_summary():
    frame = _frame(36)
    framework = AdvancedValidationFramework(timestamp_col="timestamp")

    def evaluator(train_df: pd.DataFrame, test_df: pd.DataFrame):
        # Simple stability score: train/test mean return consistency.
        train_mean = float(train_df["returns"].mean())
        test_mean = float(test_df["returns"].mean())
        gap = abs(train_mean - test_mean)
        return {
            "train_mean_return": train_mean,
            "test_mean_return": test_mean,
            "consistency": max(0.0, 1.0 - gap * 100.0),
        }

    folds = framework.walk_forward_validate(
        frame=frame,
        evaluator=evaluator,
        train_size=12,
        test_size=6,
        step_size=6,
        gap=0,
        expanding=True,
    )
    assert len(folds) == 4
    assert folds[0].fold_id == 0
    assert "test_mean_return" in folds[0].metrics

    summary = framework.summarize_metrics(folds)
    assert "test_mean_return" in summary
    assert summary["test_mean_return"]["fold_count"] == 4.0


def test_framework_purged_cv_validate():
    frame = _frame(30)
    framework = AdvancedValidationFramework(timestamp_col="timestamp")

    def evaluator(_train_df: pd.DataFrame, test_df: pd.DataFrame):
        return {"test_total_return": float(test_df["returns"].sum())}

    folds = framework.purged_cv_validate(
        frame=frame,
        evaluator=evaluator,
        n_splits=5,
        purge_window=1,
        embargo=1,
        min_train_size=5,
    )

    assert len(folds) == 5
    assert all("test_total_return" in fold.metrics for fold in folds)
