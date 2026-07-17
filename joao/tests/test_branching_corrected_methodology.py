from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.branching.AttributeBasedBranchingEngine import AttributeBasedBranchingEngine
from src.branching.BranchingFeatureBuilder import BranchingFeatureBuilder
from src.branching.BranchingUtils import temporal_train_validation_test_split_by_case
from src.branching.CompositeBranchingArtifact import (
    export_composite_branching_artifact,
    load_artifact_payload,
)
from src.branching.CompositeBranchingEngine import CompositeBranchingEngine
from src.branching.ProbabilityBranchingEngine import ProbabilityBranchingEngine


def small_log() -> pd.DataFrame:
    rows = []
    for case_index in range(10):
        case_id = f"C{case_index:02d}"
        start = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=case_index)
        rows.append(
            {
                "case:concept:name": case_id,
                "concept:name": "A",
                "time:timestamp": start,
                "case:RequestedAmount": str(1000 + case_index),
            }
        )
        rows.append(
            {
                "case:concept:name": case_id,
                "concept:name": "B" if case_index % 2 else "C",
                "time:timestamp": start + pd.Timedelta(minutes=5),
                "case:RequestedAmount": str(1000 + case_index),
            }
        )
    return pd.DataFrame(rows)


def test_temporal_split_has_zero_overlap_and_is_deterministic() -> None:
    log = small_log()
    split_a = temporal_train_validation_test_split_by_case(log, outer_train_ratio=0.7, inner_train_ratio=0.85)
    split_b = temporal_train_validation_test_split_by_case(log, outer_train_ratio=0.7, inner_train_ratio=0.85)

    assert split_a == split_b
    assert set(split_a.outer_train_cases).isdisjoint(split_a.outer_test_cases)
    assert set(split_a.inner_train_cases).isdisjoint(split_a.inner_validation_cases)
    assert set(split_a.inner_train_cases).issubset(split_a.outer_train_cases)
    assert set(split_a.inner_validation_cases).issubset(split_a.outer_train_cases)
    assert split_a.outer_train_cases
    assert split_a.outer_test_cases


def test_feature_builder_training_runtime_parity_for_log_occurrence() -> None:
    records = [
        {
            "case:concept:name": "C1",
            "concept:name": "A",
            "time:timestamp": pd.Timestamp("2026-01-05 09:00:00", tz="UTC"),
            "case:RequestedAmount": 1000,
        },
        {
            "case:concept:name": "C1",
            "concept:name": "B",
            "time:timestamp": pd.Timestamp("2026-01-05 09:10:00", tz="UTC"),
            "case:RequestedAmount": 1000,
        },
    ]
    builder = BranchingFeatureBuilder()
    train_features = builder.build_from_log_occurrence(
        records,
        index=1,
        case_start_time=records[0]["time:timestamp"],
        decision_point_id="dp",
        candidate_set_signature="cand",
    )

    runtime_event = {
        "concept:name": "B",
        "time:timestamp": records[1]["time:timestamp"],
        "previous_timestamp": records[0]["time:timestamp"],
        "case_start_time": records[0]["time:timestamp"],
        "event_index": 1,
        "case:RequestedAmount": 1000,
        "activity_history": ["A", "B"],
    }
    runtime_features = builder.build_from_runtime_event(
        runtime_event,
        decision_point_id="dp",
        candidate_set_signature="cand",
    )

    assert runtime_features["weekday"] == train_features["weekday"] == 0
    assert runtime_features["hour"] == train_features["hour"] == 9
    assert runtime_features["month"] == train_features["month"] == 1
    assert runtime_features["previous_activity"] == train_features["previous_activity"] == "A"
    assert runtime_features["trace_prefix"] == train_features["trace_prefix"] == "A"
    assert runtime_features["event_index"] == train_features["event_index"] == 1


def test_probability_branching_uses_state_then_activity_backoff() -> None:
    log = small_log()
    engine = ProbabilityBranchingEngine(seed=1, alpha=1.0, min_state_support=100)
    engine.train(log)

    event = {"concept:name": "A", "activity_history": ["A"]}
    decision = engine.decide(event, ["B", "C"])

    assert decision is not None
    assert decision.activities[0] in {"B", "C"}
    assert decision.probability_source == "activity_level_probability"
    assert decision.is_valid_for(["B", "C"])


def test_attribute_based_rule_match_is_separate_from_fallback() -> None:
    engine = AttributeBasedBranchingEngine(
        rules=[
            {
                "decision_point": "A",
                "attribute": "case:RequestedAmount",
                "operator": ">",
                "value": 500,
                "preferred_activities": ["B"],
            }
        ],
        seed=1,
    )
    decision = engine.decide(
        {"concept:name": "A", "case:RequestedAmount": "1000"},
        ["B", "C"],
    )

    assert decision is not None
    assert decision.activities == ["B"]
    assert decision.decision_source == "attribute_rule"
    assert engine.get_diagnostics()["fallback_count"] == 0


def test_composite_records_abstention_and_source() -> None:
    class AbstainingEngine:
        def decide(self, event, possible, context=None):
            return None

    class ChoosingEngine:
        def decide(self, event, possible, context=None):
            from src.branching.BranchDecision import BranchDecision

            return BranchDecision(activities=[possible[-1]], decision_source="choosing_engine")

    composite = CompositeBranchingEngine(
        engines=[AbstainingEngine(), ChoosingEngine()],
        use_default_hierarchy=False,
    )

    decision = composite.decide(object(), ["A", "B"])

    assert decision.activities == ["B"]
    assert decision.decision_source == "choosing_engine"
    stats = composite.get_statistics()
    assert stats["engine_abstention_counts"]["AbstainingEngine"] == 1
    assert stats["engine_success_counts"]["ChoosingEngine"] == 1


def test_artifact_scope_validation(tmp_path) -> None:
    composite = CompositeBranchingEngine(engines=[], use_default_hierarchy=False)
    path = tmp_path / "artifact.pkl"
    export_composite_branching_artifact(
        composite,
        path,
        metadata={"training_log": "small", "training_case_ids_sha256": "abc"},
        artifact_scope="evaluation",
    )

    payload = load_artifact_payload(path, expected_scope="evaluation")
    assert payload["artifact_scope"] == "evaluation"
    with pytest.raises(ValueError):
        load_artifact_payload(path, expected_scope="deployment")
