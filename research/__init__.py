"""Research velocity utilities for experiments and reproducibility."""

from research.benchmarks import (
    PromotionBenchmarkDecision,
    PromotionBenchmarkThresholds,
    build_promotion_benchmark_spec,
    evaluate_promotion_candidate,
    export_promotion_benchmark_spec,
    promotion_thresholds_from_config,
)
from research.registry import (
    DatasetRecord,
    ExperimentRecord,
    ExperimentRegistry,
    fingerprint_dataframe,
)

__all__ = [
    "DatasetRecord",
    "ExperimentRecord",
    "ExperimentRegistry",
    "fingerprint_dataframe",
    "PromotionBenchmarkDecision",
    "PromotionBenchmarkThresholds",
    "build_promotion_benchmark_spec",
    "evaluate_promotion_candidate",
    "export_promotion_benchmark_spec",
    "promotion_thresholds_from_config",
]
