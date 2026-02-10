from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint_dataframe(
    dataframe: pd.DataFrame,
    *,
    sort_by: Optional[List[str]] = None,
    include_columns: Optional[List[str]] = None,
) -> str:
    """
    Build deterministic fingerprint from a dataframe snapshot.
    """
    if dataframe is None:
        raise ValueError("dataframe is required")

    frame = dataframe.copy()
    if include_columns:
        missing = [col for col in include_columns if col not in frame.columns]
        if missing:
            raise ValueError(f"Missing columns for fingerprint: {missing}")
        frame = frame[include_columns]

    sort_fields = [field for field in (sort_by or ["timestamp"]) if field in frame.columns]
    if sort_fields:
        frame = frame.sort_values(sort_fields)
    frame = frame.reset_index(drop=True)

    payload = frame.to_json(orient="split", date_format="iso", double_precision=12)
    return _sha256(payload)


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    name: str
    fingerprint: str
    row_count: int
    created_at: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    experiment_name: str
    strategy_name: str
    dataset_fingerprint: str
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    validation_summary: Dict[str, Any]
    execution_config: Dict[str, Any]
    created_at: str


class ExperimentRegistry:
    """
    Persistent research registry for reproducible experiments and leaderboard ranking.
    """

    def __init__(self, db_path: str = "research/experiments.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    fingerprint TEXT NOT NULL UNIQUE,
                    row_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    experiment_key TEXT NOT NULL UNIQUE,
                    experiment_name TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    dataset_fingerprint TEXT NOT NULL,
                    params_json TEXT,
                    metrics_json TEXT,
                    validation_json TEXT,
                    execution_config_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def register_dataset(
        self,
        *,
        name: str,
        dataframe: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
        sort_by: Optional[List[str]] = None,
        include_columns: Optional[List[str]] = None,
    ) -> DatasetRecord:
        fingerprint = fingerprint_dataframe(
            dataframe,
            sort_by=sort_by,
            include_columns=include_columns,
        )
        created_at = _now_iso()
        row_count = int(len(dataframe))

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT dataset_id, name, fingerprint, row_count, created_at, metadata_json
                FROM datasets WHERE fingerprint = ?
                """,
                (fingerprint,),
            )
            existing = cursor.fetchone()
            if existing:
                return DatasetRecord(
                    dataset_id=existing[0],
                    name=existing[1],
                    fingerprint=existing[2],
                    row_count=int(existing[3]),
                    created_at=existing[4],
                    metadata=json.loads(existing[5]) if existing[5] else {},
                )

            dataset_id = uuid.uuid4().hex
            cursor.execute(
                """
                INSERT INTO datasets (dataset_id, name, fingerprint, row_count, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    str(name),
                    fingerprint,
                    row_count,
                    created_at,
                    _stable_json(metadata or {}),
                ),
            )
            conn.commit()

        return DatasetRecord(
            dataset_id=dataset_id,
            name=str(name),
            fingerprint=fingerprint,
            row_count=row_count,
            created_at=created_at,
            metadata=dict(metadata or {}),
        )

    def register_experiment(
        self,
        *,
        experiment_name: str,
        strategy_name: str,
        dataset_fingerprint: str,
        params: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        validation_summary: Optional[Dict[str, Any]] = None,
        execution_config: Optional[Dict[str, Any]] = None,
    ) -> ExperimentRecord:
        params = dict(params or {})
        metrics = dict(metrics or {})
        validation_summary = dict(validation_summary or {})
        execution_config = dict(execution_config or {})

        experiment_key = _sha256(
            _stable_json(
                {
                    "experiment_name": experiment_name,
                    "strategy_name": strategy_name,
                    "dataset_fingerprint": dataset_fingerprint,
                    "params": params,
                }
            )
        )

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    experiment_id, experiment_name, strategy_name, dataset_fingerprint,
                    params_json, metrics_json, validation_json, execution_config_json, created_at
                FROM experiments
                WHERE experiment_key = ?
                """,
                (experiment_key,),
            )
            existing = cursor.fetchone()
            if existing:
                return ExperimentRecord(
                    experiment_id=existing[0],
                    experiment_name=existing[1],
                    strategy_name=existing[2],
                    dataset_fingerprint=existing[3],
                    params=json.loads(existing[4]) if existing[4] else {},
                    metrics=json.loads(existing[5]) if existing[5] else {},
                    validation_summary=json.loads(existing[6]) if existing[6] else {},
                    execution_config=json.loads(existing[7]) if existing[7] else {},
                    created_at=existing[8],
                )

            experiment_id = uuid.uuid4().hex
            created_at = _now_iso()
            cursor.execute(
                """
                INSERT INTO experiments (
                    experiment_id, experiment_key, experiment_name, strategy_name, dataset_fingerprint,
                    params_json, metrics_json, validation_json, execution_config_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    experiment_key,
                    str(experiment_name),
                    str(strategy_name),
                    str(dataset_fingerprint),
                    _stable_json(params),
                    _stable_json(metrics),
                    _stable_json(validation_summary),
                    _stable_json(execution_config),
                    created_at,
                ),
            )
            conn.commit()

        return ExperimentRecord(
            experiment_id=experiment_id,
            experiment_name=str(experiment_name),
            strategy_name=str(strategy_name),
            dataset_fingerprint=str(dataset_fingerprint),
            params=params,
            metrics=metrics,
            validation_summary=validation_summary,
            execution_config=execution_config,
            created_at=created_at,
        )

    def list_experiments(
        self,
        *,
        strategy_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[ExperimentRecord]:
        limit = max(1, int(limit))
        query = """
            SELECT
                experiment_id, experiment_name, strategy_name, dataset_fingerprint,
                params_json, metrics_json, validation_json, execution_config_json, created_at
            FROM experiments
        """
        params: List[Any] = []
        if strategy_name:
            query += " WHERE strategy_name = ?"
            params.append(str(strategy_name))
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        return [
            ExperimentRecord(
                experiment_id=row[0],
                experiment_name=row[1],
                strategy_name=row[2],
                dataset_fingerprint=row[3],
                params=json.loads(row[4]) if row[4] else {},
                metrics=json.loads(row[5]) if row[5] else {},
                validation_summary=json.loads(row[6]) if row[6] else {},
                execution_config=json.loads(row[7]) if row[7] else {},
                created_at=row[8],
            )
            for row in rows
        ]

    def get_leaderboard(
        self,
        *,
        metric: str = "sharpe_ratio",
        strategy_name: Optional[str] = None,
        descending: bool = True,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        records = self.list_experiments(strategy_name=strategy_name, limit=max(1, int(limit)) * 10)
        sorted_records = sorted(
            records,
            key=lambda rec: float(rec.metrics.get(metric, float("-inf"))),
            reverse=bool(descending),
        )[: max(1, int(limit))]

        leaderboard: List[Dict[str, Any]] = []
        for idx, record in enumerate(sorted_records, start=1):
            leaderboard.append(
                {
                    "rank": idx,
                    "experiment_id": record.experiment_id,
                    "experiment_name": record.experiment_name,
                    "strategy_name": record.strategy_name,
                    "dataset_fingerprint": record.dataset_fingerprint,
                    "metric": metric,
                    "metric_value": float(record.metrics.get(metric, 0.0)),
                    "metrics": record.metrics,
                    "created_at": record.created_at,
                }
            )
        return leaderboard

    def check_reproducibility(
        self,
        *,
        experiment_id: str,
        metrics: Dict[str, Any],
        tolerance: float = 1e-9,
    ) -> Dict[str, Any]:
        records = [record for record in self.list_experiments(limit=10000) if record.experiment_id == experiment_id]
        if not records:
            raise ValueError(f"Unknown experiment_id: {experiment_id}")
        baseline = records[0].metrics

        deltas: Dict[str, float] = {}
        reproducible = True
        for key, baseline_value in baseline.items():
            if key not in metrics:
                reproducible = False
                deltas[key] = float("inf")
                continue
            try:
                delta = abs(float(metrics[key]) - float(baseline_value))
            except Exception:
                reproducible = False
                deltas[key] = float("inf")
                continue
            deltas[key] = float(delta)
            if delta > float(tolerance):
                reproducible = False

        return {
            "experiment_id": experiment_id,
            "reproducible": bool(reproducible),
            "tolerance": float(tolerance),
            "deltas": deltas,
            "baseline_metrics": baseline,
            "candidate_metrics": dict(metrics),
        }
