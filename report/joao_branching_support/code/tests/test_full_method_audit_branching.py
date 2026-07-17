import pickle
from types import SimpleNamespace

import pandas as pd

from src.branching.AttributeBasedBranchingEngine import AttributeBasedBranchingEngine
from src.branching.AttributeSamplingBranchingEngine import AttributeSamplingBranchingEngine
from src.branching.CompositeBranchingArtifact import (
    export_composite_branching_artifact,
    load_composite_branching_artifact,
)
from src.branching.CompositeBranchingEngine import CompositeBranchingEngine
from src.branching.PredictiveBranchingEngine import PredictiveBranchingEngine
from src.branching.ProbabilityBranchingEngine import ProbabilityBranchingEngine


def event(activity="A", **attrs):
    return {"concept:name": activity, **attrs}


class EmptyResultEngine:
    def getNextActivities(self, event, possibleActivities):
        return []


def replay_candidate(transition_id, label):
    return SimpleNamespace(transition_id=transition_id, activity_label=label)


class ReplayStubEngine:
    def __init__(self):
        self.positions = {}

    def initialize_case(self, case_id):
        self.positions[str(case_id)] = 0

    def getPossibleNextTransitionCandidates(self, case_id):
        position = self.positions[str(case_id)]
        if position == 0:
            return [replay_candidate("t_a", "A")]
        if position == 1:
            return [
                replay_candidate("t_b", "B"),
                replay_candidate("t_c", "C"),
            ]
        return []

    def fire_transition_candidate(self, case_id, candidate):
        self.positions[str(case_id)] += 1
        return True


def training_log():
    rows = []
    for case_id, next_activity in [("C1", "B"), ("C2", "B"), ("C3", "C"), ("C4", "C")]:
        rows.append(
            {
                "case:concept:name": case_id,
                "concept:name": "A",
                "time:timestamp": pd.Timestamp("2026-01-01 09:00:00"),
                "org:resource": "R1",
            }
        )
        rows.append(
            {
                "case:concept:name": case_id,
                "concept:name": next_activity,
                "time:timestamp": pd.Timestamp("2026-01-01 09:01:00"),
                "org:resource": "R1",
            }
        )
    return pd.DataFrame(rows)


def test_probability_branching_filters_to_bpmn_valid_and_is_seeded():
    first = ProbabilityBranchingEngine(seed=9)
    second = ProbabilityBranchingEngine(seed=9)
    first.branch_probabilities = {"A": {"B": 0.9, "INVALID": 0.1}}
    second.branch_probabilities = {"A": {"B": 0.9, "INVALID": 0.1}}

    assert first.getNextActivities(event("A"), ["B", "C"]) == ["B"]
    assert first.getNextActivities(event("UNKNOWN"), ["B", "C"]) == second.getNextActivities(
        event("UNKNOWN"),
        ["B", "C"],
    )


def test_probability_branching_uses_stateful_history_before_activity_only_fallback():
    rows = []
    for case_id, middle, target in [
        ("C1", "A", "B"),
        ("C2", "A", "B"),
        ("C3", "X", "C"),
        ("C4", "X", "C"),
    ]:
        rows.extend(
            [
                {
                    "case:concept:name": case_id,
                    "concept:name": middle,
                    "time:timestamp": pd.Timestamp("2026-01-01 09:00:00"),
                },
                {
                    "case:concept:name": case_id,
                    "concept:name": "O_Sent",
                    "time:timestamp": pd.Timestamp("2026-01-01 09:01:00"),
                },
                {
                    "case:concept:name": case_id,
                    "concept:name": target,
                    "time:timestamp": pd.Timestamp("2026-01-01 09:02:00"),
                },
            ]
        )
    engine = ProbabilityBranchingEngine(seed=7)
    engine.train(pd.DataFrame(rows))

    runtime_event = {
        "concept:name": "O_Sent",
        "previous_activity": "X",
        "activity_history": ["X", "O_Sent"],
    }

    assert engine.getNextActivities(runtime_event, ["B", "C"]) == ["C"]


def test_predictive_training_dataset_contains_state_features():
    engine = PredictiveBranchingEngine(seed=5, n_estimators=5, n_jobs=1)

    dataset = engine.build_training_dataset(training_log())

    assert "trace_prefix" in dataset.columns
    assert "current_activity_visit_count" in dataset.columns
    assert "consecutive_repetition_count" in dataset.columns
    assert "time_since_previous_event_seconds" in dataset.columns


def test_predictive_bpmn_replay_dataset_uses_model_synchronized_decisions():
    engine = PredictiveBranchingEngine(
        seed=5,
        n_estimators=5,
        n_jobs=1,
        use_bpmn_replay=True,
        bpmn_engine_factory=ReplayStubEngine,
    )

    dataset = engine.build_training_dataset(training_log())

    assert engine.dataset_mode == "bpmn_replay"
    assert engine.decision_points == {"A"}
    assert set(dataset["current_activity"]) == {"A"}
    assert set(dataset["next_activity"]) == {"B", "C"}
    assert engine.bpmn_replay_diagnostics["decision_observations"] == 4


def test_attribute_based_falls_back_when_attribute_missing_and_validates_rule():
    fallback = ProbabilityBranchingEngine(seed=3)
    fallback.branch_probabilities = {"A": {"C": 1.0}}
    engine = AttributeBasedBranchingEngine(
        rules=[
            {
                "decision_point": "A",
                "attribute": "risk",
                "operator": "==",
                "value": "high",
                "preferred_activities": ["B"],
            }
        ],
        fallback_engine=fallback,
        seed=3,
    )

    assert engine.getNextActivities(event("A", risk="high"), ["B", "C"]) == ["B"]
    assert engine.getNextActivities(event("A"), ["B", "C"]) == ["C"]


def test_attribute_sampling_supplies_missing_attribute_to_base_rule():
    base = AttributeBasedBranchingEngine(
        rules=[
            {
                "decision_point": "A",
                "attribute": "customer_type",
                "operator": "==",
                "value": "premium",
                "preferred_activities": ["B"],
            }
        ],
        seed=1,
    )
    engine = AttributeSamplingBranchingEngine(
        base_engine=base,
        sampling_config={"customer_type": {"premium": 1.0}},
        seed=1,
    )

    assert engine.getNextActivities(event("A"), ["B", "C"]) == ["B"]
    assert engine.sampled_attribute_count == 1


def test_predictive_branching_trains_once_and_runtime_does_not_retrain():
    engine = PredictiveBranchingEngine(seed=5, n_estimators=5, n_jobs=1)
    engine.train(training_log())
    model_id = id(engine.model)

    result = engine.getNextActivities(event("A"), ["B", "C"])

    assert result[0] in {"B", "C"}
    assert id(engine.model) == model_id
    assert engine.total_predictions == 1


def test_composite_artifact_load_resets_runtime_counters(tmp_path):
    composite = CompositeBranchingEngine(
        engines=[ProbabilityBranchingEngine(seed=2)],
        seed=2,
        use_default_hierarchy=False,
    )
    composite.engines[0].branch_probabilities = {"A": {"B": 1.0}}
    composite.getNextActivities(event("A"), ["B", "C"])
    path = tmp_path / "branching.pkl"

    export_composite_branching_artifact(composite, path, metadata={"purpose": "test"})
    loaded = load_composite_branching_artifact(path)

    assert loaded.get_statistics()["total_decisions"] == 0
    assert loaded.getNextActivities(event("A"), ["B", "C"]) == ["B"]


def test_legacy_pickle_roundtrip_keeps_prediction_execution_consistency(tmp_path):
    engine = CompositeBranchingEngine(
        engines=[ProbabilityBranchingEngine(seed=4)],
        seed=4,
        use_default_hierarchy=False,
    )
    engine.engines[0].branch_probabilities = {"A": {"B": 1.0}}
    path = tmp_path / "legacy.pkl"
    with path.open("wb") as file:
        pickle.dump(engine, file)
    with path.open("rb") as file:
        loaded = pickle.load(file)

    assert loaded.getNextActivities(event("A"), ["B", "C"]) == ["B"]


def test_predictive_evaluate_keeps_training_decision_points():
    engine = PredictiveBranchingEngine(seed=5, n_estimators=5, n_jobs=1)
    engine.train(training_log())
    training_decision_points = set(engine.decision_points)
    test_log = pd.DataFrame(
        [
            {
                "case:concept:name": "T1",
                "concept:name": "X",
                "time:timestamp": pd.Timestamp("2026-01-02 09:00:00"),
                "org:resource": "R1",
            },
            {
                "case:concept:name": "T1",
                "concept:name": "Y",
                "time:timestamp": pd.Timestamp("2026-01-02 09:01:00"),
                "org:resource": "R1",
            },
            {
                "case:concept:name": "T2",
                "concept:name": "X",
                "time:timestamp": pd.Timestamp("2026-01-02 09:00:00"),
                "org:resource": "R1",
            },
            {
                "case:concept:name": "T2",
                "concept:name": "Z",
                "time:timestamp": pd.Timestamp("2026-01-02 09:01:00"),
                "org:resource": "R1",
            },
        ]
    )

    metrics = engine.evaluate(test_log)

    assert metrics["n_samples"] == 0
    assert engine.decision_points == training_decision_points


def test_predictive_bpmn_replay_evaluate_keeps_training_decision_points():
    engine = PredictiveBranchingEngine(
        seed=5,
        n_estimators=5,
        n_jobs=1,
        use_bpmn_replay=True,
        bpmn_engine_factory=ReplayStubEngine,
    )
    engine.train(training_log())
    training_decision_points = set(engine.decision_points)

    metrics = engine.evaluate(training_log())

    assert metrics["n_samples"] == 4
    assert engine.decision_points == training_decision_points


def test_individual_engines_reject_empty_fallback_when_candidates_exist():
    predictive = PredictiveBranchingEngine(fallback_engine=EmptyResultEngine(), seed=1)
    attribute = AttributeBasedBranchingEngine(fallback_engine=EmptyResultEngine(), seed=1)
    sampling = AttributeSamplingBranchingEngine(
        base_engine=EmptyResultEngine(),
        fallback_engine=EmptyResultEngine(),
        seed=1,
    )

    assert predictive.fallback(event("A"), ["B", "C"]) in [["B"], ["C"]]
    assert attribute.fallback(event("A"), ["B", "C"]) in [["B"], ["C"]]
    assert sampling.getNextActivities(event("A"), ["B", "C"]) in [["B"], ["C"]]
