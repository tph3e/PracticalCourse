from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .BranchingUtils import extract_current_activity


DEFAULT_FEATURE_COLUMNS = [
    "current_activity",
    "previous_activity",
    "event_index",
    "weekday",
    "hour",
    "month",
    "elapsed_time_seconds",
    "time_since_previous_event_seconds",
    "trace_prefix",
    "current_activity_visit_count",
    "consecutive_repetition_count",
    "resource",
    "case:ApplicationType",
    "case:LoanGoal",
    "case:RequestedAmount",
    "CreditScore",
    "EventOrigin",
    "org:resource",
    "candidate_set_signature",
    "decision_point_id",
]


@dataclass
class FeatureDiagnostics:
    feature_rows_built: int = 0
    missing_feature_count: int = 0
    default_count_by_feature: Counter[str] = field(default_factory=Counter)
    unknown_category_count: int = 0
    timestamp_fallback_count: int = 0
    history_fallback_count: int = 0
    runtime_feature_schema_mismatch: int = 0
    train_runtime_parity_failures: int = 0

    @property
    def missing_feature_rate(self) -> float:
        denominator = max(1, self.feature_rows_built * len(DEFAULT_FEATURE_COLUMNS))
        return self.missing_feature_count / denominator

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_rows_built": self.feature_rows_built,
            "missing_feature_count": self.missing_feature_count,
            "missing_feature_rate": self.missing_feature_rate,
            "default_count_by_feature": dict(self.default_count_by_feature),
            "unknown_category_count": self.unknown_category_count,
            "timestamp_fallback_count": self.timestamp_fallback_count,
            "history_fallback_count": self.history_fallback_count,
            "runtime_feature_schema_mismatch": self.runtime_feature_schema_mismatch,
            "train_runtime_parity_failures": self.train_runtime_parity_failures,
        }


class BranchingFeatureBuilder:
    def __init__(
        self,
        case_col: str = "case:concept:name",
        activity_col: str = "concept:name",
        timestamp_col: str = "time:timestamp",
        resource_col: str = "org:resource",
        feature_columns: list[str] | None = None,
    ):
        self.case_col = case_col
        self.activity_col = activity_col
        self.timestamp_col = timestamp_col
        self.resource_col = resource_col
        self.feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
        self.diagnostics = FeatureDiagnostics()

    def build_from_log_occurrence(
        self,
        records: list[dict[str, Any]],
        index: int,
        case_start_time: Any | None = None,
        decision_point_id: str | None = None,
        candidate_set_signature: str | None = None,
    ) -> dict[str, Any]:
        timestamps = [
            pd.to_datetime(record.get(self.timestamp_col), utc=True, errors="coerce")
            for record in records
        ]
        activities = [str(record.get(self.activity_col, "UNKNOWN")) for record in records]
        start_time = (
            pd.to_datetime(case_start_time, utc=True, errors="coerce")
            if case_start_time is not None
            else timestamps[0]
        )
        row = self._build_row(
            attributes=records[index],
            activities=activities,
            timestamps=timestamps,
            index=index,
            case_start_time=start_time,
            decision_point_id=decision_point_id,
            candidate_set_signature=candidate_set_signature,
        )
        return self._finalize(row)

    def build_from_runtime_event(
        self,
        event: Any,
        history: list[str] | None = None,
        timestamp: Any | None = None,
        case_start_time: Any | None = None,
        previous_timestamp: Any | None = None,
        event_index: int | None = None,
        decision_point_id: str | None = None,
        candidate_set_signature: str | None = None,
    ) -> dict[str, Any]:
        attributes = self._extract_attributes(event)
        current_activity = extract_current_activity(event, self.activity_col) or attributes.get(self.activity_col)
        timestamp_value = timestamp if timestamp is not None else self._extract_timestamp(event, attributes)
        current_timestamp = pd.to_datetime(timestamp_value, utc=True, errors="coerce")
        if pd.isna(current_timestamp):
            self.diagnostics.timestamp_fallback_count += 1
            current_timestamp = pd.Timestamp(0, tz="UTC")

        history_values = history if history is not None else self._extract_history(event)
        if history_values is None:
            history_values = []
        history_values = [str(item) for item in history_values if item is not None]
        if current_activity and (not history_values or history_values[-1] != str(current_activity)):
            history_with_current = [*history_values, str(current_activity)]
        else:
            history_with_current = history_values
        if not history_with_current:
            self.diagnostics.history_fallback_count += 1
            history_with_current = [str(current_activity or "UNKNOWN")]

        inferred_index = event_index
        if inferred_index is None:
            inferred_index = self._coerce_int(attributes.get("event_index"), len(history_with_current) - 1)

        start_time = pd.to_datetime(
            case_start_time if case_start_time is not None else attributes.get("case_start_time"),
            utc=True,
            errors="coerce",
        )
        if pd.isna(start_time):
            start_time = current_timestamp

        previous_time = pd.to_datetime(
            previous_timestamp if previous_timestamp is not None else attributes.get("previous_timestamp"),
            utc=True,
            errors="coerce",
        )
        timestamps = [current_timestamp] * len(history_with_current)
        if len(timestamps) >= 2 and not pd.isna(previous_time):
            timestamps[-2] = previous_time

        row = self._build_row(
            attributes=attributes,
            activities=history_with_current,
            timestamps=timestamps,
            index=len(history_with_current) - 1,
            case_start_time=start_time,
            event_index_override=inferred_index,
            decision_point_id=decision_point_id,
            candidate_set_signature=candidate_set_signature,
        )
        return self._finalize(row)

    def _build_row(
        self,
        attributes: dict[str, Any],
        activities: list[str],
        timestamps: list[Any],
        index: int,
        case_start_time: Any,
        event_index_override: int | None = None,
        decision_point_id: str | None = None,
        candidate_set_signature: str | None = None,
    ) -> dict[str, Any]:
        current_activity = activities[index]
        prior = activities[:index]
        previous_activity = prior[-1] if prior else "START"
        prefix = "|".join(prior[-3:]) or "START"
        visit_count = sum(1 for activity in prior if activity == current_activity)
        consecutive = 0
        for activity in reversed(prior):
            if activity != current_activity:
                break
            consecutive += 1

        timestamp = pd.to_datetime(timestamps[index], utc=True, errors="coerce")
        if pd.isna(timestamp):
            self.diagnostics.timestamp_fallback_count += 1
            timestamp = pd.Timestamp(0, tz="UTC")
        start_time = pd.to_datetime(case_start_time, utc=True, errors="coerce")
        if pd.isna(start_time):
            start_time = timestamp
        previous_time = pd.to_datetime(timestamps[index - 1], utc=True, errors="coerce") if index > 0 else timestamp

        row = {
            "current_activity": current_activity,
            "previous_activity": previous_activity,
            "event_index": int(event_index_override if event_index_override is not None else index),
            "weekday": int(timestamp.weekday()),
            "hour": int(timestamp.hour),
            "month": int(timestamp.month),
            "elapsed_time_seconds": float((timestamp - start_time).total_seconds()),
            "time_since_previous_event_seconds": float((timestamp - previous_time).total_seconds()) if index > 0 else 0.0,
            "trace_prefix": prefix,
            "current_activity_visit_count": int(visit_count),
            "consecutive_repetition_count": int(consecutive),
            "resource": self._clean(attributes.get(self.resource_col, attributes.get("resource", "UNKNOWN"))),
            "candidate_set_signature": candidate_set_signature or "UNKNOWN",
            "decision_point_id": decision_point_id or "UNKNOWN",
        }
        for column in self.feature_columns:
            if column in row:
                continue
            row[column] = self._clean(attributes.get(column, "UNKNOWN"))
        return row

    def _finalize(self, row: dict[str, Any]) -> dict[str, Any]:
        self.diagnostics.feature_rows_built += 1
        finalized = {}
        for column in self.feature_columns:
            value = self._clean(row.get(column, "UNKNOWN"))
            if value == "UNKNOWN":
                self.diagnostics.missing_feature_count += 1
                self.diagnostics.default_count_by_feature[column] += 1
                self.diagnostics.unknown_category_count += 1
            finalized[column] = value
        return finalized

    def _extract_attributes(self, event: Any) -> dict[str, Any]:
        attrs: dict[str, Any] = {}
        if event is None:
            return attrs
        if hasattr(event, "getAttribs"):
            value = event.getAttribs()
            if isinstance(value, dict):
                attrs.update(value)
        for attr in ("attributes", "data"):
            value = getattr(event, attr, None)
            if isinstance(value, dict):
                attrs.update(value)
        if isinstance(event, dict):
            attrs.update(event)
        if hasattr(event, "activity"):
            attrs.setdefault(self.activity_col, event.activity)
            attrs.setdefault("activity", event.activity)
        return attrs

    def _extract_timestamp(self, event: Any, attributes: dict[str, Any]) -> Any:
        for key in (self.timestamp_col, "timestamp", "time"):
            if key in attributes:
                return attributes[key]
        return getattr(event, "time", None)

    def _extract_history(self, event: Any) -> list[str] | None:
        if event is None:
            return []
        event_case = getattr(event, "eventCase", None)
        activities = getattr(event_case, "activities", None)
        if isinstance(activities, list):
            return activities
        history = getattr(event, "history", None)
        if isinstance(history, list):
            return history
        if isinstance(event, dict) and isinstance(event.get("activity_history"), list):
            return event["activity_history"]
        return []

    def _clean(self, value: Any) -> Any:
        if value is None:
            return "UNKNOWN"
        try:
            if pd.isna(value):
                return "UNKNOWN"
        except (TypeError, ValueError):
            pass
        return value

    def _coerce_int(self, value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
