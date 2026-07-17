from datetime import datetime

import pandas as pd

from src.branching.CompositeBranchingArtifact import (
    export_composite_branching_artifact,
    load_artifact_payload,
    load_composite_branching_artifact,
)
from src.branching.CompositeBranchingEngine import CompositeBranchingEngine


class Event:
    activity = "A"

    class eventCase:
        caseId = "C_runtime"

    def getAttribs(self):
        return {
            "concept:name": "A",
            "case:concept:name": "C_runtime",
            "time:timestamp": datetime(2026, 1, 5, 9, 0),
            "case:ApplicationType": "New credit",
            "case:LoanGoal": "Car",
            "case:RequestedAmount": 1000,
            "EventOrigin": "Application",
            "org:resource": "R1",
        }


def build_small_log():
    rows = []
    for case_id, next_activity in [
        ("C1", "B"),
        ("C2", "B"),
        ("C3", "C"),
        ("C4", "B"),
        ("C5", "C"),
        ("C6", "B"),
    ]:
        rows.append(
            {
                "case:concept:name": case_id,
                "concept:name": "A",
                "time:timestamp": pd.Timestamp("2026-01-01 09:00:00"),
                "case:ApplicationType": "New credit",
                "case:LoanGoal": "Car",
                "case:RequestedAmount": 1000,
                "EventOrigin": "Application",
                "org:resource": "R1",
            }
        )
        rows.append(
            {
                "case:concept:name": case_id,
                "concept:name": next_activity,
                "time:timestamp": pd.Timestamp("2026-01-01 09:01:00"),
                "case:ApplicationType": "New credit",
                "case:LoanGoal": "Car",
                "case:RequestedAmount": 1000,
                "EventOrigin": "Application",
                "org:resource": "R1",
            }
        )
    return pd.DataFrame(rows)


def test_composite_branching_artifact_round_trip(tmp_path):
    composite = CompositeBranchingEngine(
        log=build_small_log(),
        seed=1,
        use_default_hierarchy=True,
        train_on_init=True,
    )
    before = composite.getNextActivities(Event(), ["B", "C"])
    assert composite.total_decisions == 1

    path = tmp_path / "composite.pkl"
    export_composite_branching_artifact(
        composite,
        path,
        metadata={
            "training_log": "small",
            "selected_validation_train_ratio": 0.7,
            "deployment_training_mode": "full_log_after_model_selection",
        },
    )

    payload = load_artifact_payload(path)
    assert payload["metadata"]["runtime_state_persisted"] is False
    assert payload["metadata"]["deployment_training_mode"] == "full_log_after_model_selection"

    loaded = load_composite_branching_artifact(path)
    loaded_again = load_composite_branching_artifact(path)

    assert loaded.total_decisions == 0
    assert loaded_again.total_decisions == 0
    after = loaded.getNextActivities(Event(), ["B", "C"])
    assert after == before
    assert loaded.total_decisions == 1
    assert loaded_again.total_decisions == 0
    assert loaded.engines is not loaded_again.engines
