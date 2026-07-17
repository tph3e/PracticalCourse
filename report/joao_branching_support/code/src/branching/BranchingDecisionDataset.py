from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .BranchingFeatureBuilder import BranchingFeatureBuilder


@dataclass(frozen=True)
class BranchingDatasetResult:
    observations: pd.DataFrame
    coverage: dict[str, Any]
    metadata: dict[str, Any]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decision_point_id(
    marking_signature: str,
    candidate_transition_ids: list[str],
    candidate_activity_labels: list[str],
) -> str:
    payload = json.dumps(
        {
            "marking": marking_signature,
            "transition_ids": sorted(candidate_transition_ids),
            "activity_labels": sorted(candidate_activity_labels),
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def candidate_set_signature(
    candidate_transition_ids: list[str],
    candidate_activity_labels: list[str],
) -> str:
    payload = json.dumps(
        {
            "transition_ids": sorted(candidate_transition_ids),
            "activity_labels": sorted(candidate_activity_labels),
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class BranchingDecisionDatasetBuilder:
    def __init__(
        self,
        bpmn_model_path: str = "models/v4_replay.bpmn",
        case_col: str = "case:concept:name",
        activity_col: str = "concept:name",
        timestamp_col: str = "time:timestamp",
        lifecycle_col: str = "lifecycle:transition",
        lifecycle_filter: str | None = "complete",
        feature_builder: BranchingFeatureBuilder | None = None,
    ):
        self.bpmn_model_path = bpmn_model_path
        self.case_col = case_col
        self.activity_col = activity_col
        self.timestamp_col = timestamp_col
        self.lifecycle_col = lifecycle_col
        self.lifecycle_filter = lifecycle_filter
        self.feature_builder = feature_builder or BranchingFeatureBuilder(
            case_col=case_col,
            activity_col=activity_col,
            timestamp_col=timestamp_col,
        )

    def build(
        self,
        event_log: pd.DataFrame,
        split_case_ids: dict[str, set[str] | list[str]] | None = None,
        log_path: str | Path | None = None,
    ) -> BranchingDatasetResult:
        prepared = self._prepare_log(event_log)
        split_lookup = self._build_split_lookup(split_case_ids)
        bpmn_hash = file_sha256(self.bpmn_model_path)
        log_hash = file_sha256(log_path) if log_path else ""
        rows: list[dict[str, Any]] = []
        coverage = {
            "total_events": int(len(prepared)),
            "cases_seen": 0,
            "synchronized_events": 0,
            "nonconformant_events": 0,
            "deterministic_single_candidate_states": 0,
            "multi_candidate_decision_observations": 0,
            "skipped_true_label_not_enabled": 0,
            "duplicate_label_ambiguities": 0,
            "cases_fully_synchronized": 0,
            "cases_partially_synchronized": 0,
        }

        engine = self._new_bpmn_engine()
        for case_id, case_events in prepared.groupby(self.case_col, sort=False):
            coverage["cases_seen"] += 1
            case_events = case_events.sort_values(self.timestamp_col).reset_index(drop=True)
            records = case_events.to_dict("records")
            activities = [str(value) for value in case_events[self.activity_col].tolist()]
            normalized_case_id = str(case_id)
            engine.initialize_case(normalized_case_id)
            case_synchronized = True

            for index, activity in enumerate(activities):
                candidates = engine.getPossibleNextTransitionCandidates(normalized_case_id)
                matching = [candidate for candidate in candidates if str(candidate.activity_label) == activity]
                if len(matching) != 1:
                    if len(matching) > 1:
                        coverage["duplicate_label_ambiguities"] += 1
                    else:
                        coverage["nonconformant_events"] += 1
                    case_synchronized = False
                    break

                if not engine.fire_transition_candidate(normalized_case_id, matching[0]):
                    coverage["nonconformant_events"] += 1
                    case_synchronized = False
                    break
                coverage["synchronized_events"] += 1

                if index >= len(activities) - 1:
                    continue

                next_candidates = engine.getPossibleNextTransitionCandidates(normalized_case_id)
                transition_ids = [str(candidate.transition_id) for candidate in next_candidates]
                labels = [str(candidate.activity_label) for candidate in next_candidates]
                unique_labels = sorted(set(labels))
                duplicate_count = len(labels) - len(unique_labels)
                if duplicate_count:
                    coverage["duplicate_label_ambiguities"] += duplicate_count

                if len(unique_labels) <= 1:
                    coverage["deterministic_single_candidate_states"] += 1
                    continue

                true_next = activities[index + 1]
                true_present = true_next in unique_labels
                if not true_present:
                    coverage["skipped_true_label_not_enabled"] += 1
                    continue

                marking_signature = engine.current_marking_signature(normalized_case_id)
                candidate_sig = candidate_set_signature(transition_ids, labels)
                dp_id = decision_point_id(marking_signature, transition_ids, labels)
                features = self.feature_builder.build_from_log_occurrence(
                    records=records,
                    index=index,
                    case_start_time=records[0][self.timestamp_col],
                    decision_point_id=dp_id,
                    candidate_set_signature=candidate_sig,
                )
                row = {
                    **features,
                    "case_id": normalized_case_id,
                    "split": split_lookup.get(normalized_case_id, "unspecified"),
                    "source_event_index": index,
                    "timestamp": records[index][self.timestamp_col],
                    "post_fire_marking_signature": marking_signature,
                    "decision_point_id": dp_id,
                    "candidate_transition_ids": "|".join(sorted(transition_ids)),
                    "candidate_activity_labels": "|".join(unique_labels),
                    "candidate_set_signature": candidate_sig,
                    "candidate_count": len(unique_labels),
                    "duplicate_label_count": duplicate_count,
                    "true_next_activity": true_next,
                    "next_activity": true_next,
                    "true_next_label_present_in_candidates": true_present,
                    "synchronized": True,
                    "skip_reason": "",
                    "lifecycle_filter": self.lifecycle_filter or "none",
                    "bpmn_hash": bpmn_hash,
                    "log_hash": log_hash,
                }
                rows.append(row)
                coverage["multi_candidate_decision_observations"] += 1

            if case_synchronized:
                coverage["cases_fully_synchronized"] += 1
            else:
                coverage["cases_partially_synchronized"] += 1

        observations = pd.DataFrame(rows)
        metadata = {
            "bpmn_model": self.bpmn_model_path,
            "bpmn_sha256": bpmn_hash,
            "log_sha256": log_hash,
            "lifecycle_filter": self.lifecycle_filter,
            "feature_schema": list(self.feature_builder.feature_columns),
            "feature_diagnostics": self.feature_builder.diagnostics.as_dict(),
        }
        return BranchingDatasetResult(observations=observations, coverage=coverage, metadata=metadata)

    def _prepare_log(self, event_log: pd.DataFrame) -> pd.DataFrame:
        prepared = event_log.copy()
        prepared[self.timestamp_col] = pd.to_datetime(prepared[self.timestamp_col], utc=True, errors="coerce")
        prepared = prepared.dropna(subset=[self.case_col, self.activity_col, self.timestamp_col])
        if self.lifecycle_filter and self.lifecycle_col in prepared.columns:
            prepared = prepared[prepared[self.lifecycle_col] == self.lifecycle_filter]
        return prepared.sort_values([self.case_col, self.timestamp_col], kind="mergesort").reset_index(drop=True)

    def _build_split_lookup(
        self,
        split_case_ids: dict[str, set[str] | list[str]] | None,
    ) -> dict[str, str]:
        lookup: dict[str, str] = {}
        if not split_case_ids:
            return lookup
        for split, ids in split_case_ids.items():
            for case_id in ids:
                normalized = str(case_id)
                if normalized in lookup:
                    raise ValueError(f"Case {normalized} occurs in multiple splits.")
                lookup[normalized] = split
        return lookup

    def _new_bpmn_engine(self) -> Any:
        try:
            from BPMN_engine import BPMNEngine
        except ModuleNotFoundError:
            repo_root = Path(__file__).resolve().parents[3]
            module_path = repo_root / "BPMN_engine.py"
            spec = importlib.util.spec_from_file_location("BPMN_engine", module_path)
            if spec is None or spec.loader is None:
                raise
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            BPMNEngine = module.BPMNEngine
        return BPMNEngine(model_filename=self.bpmn_model_path)
