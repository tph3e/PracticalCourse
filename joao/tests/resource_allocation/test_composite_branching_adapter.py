from datetime import datetime

from src.resource_allocation.integration.CompositeBranchingAdapter import (
    CompositeBranchingAdapter,
)


class FakeProbabilityEngine:
    branch_probabilities = {"A": {"B": 0.75, "C": 0.25}}


class FakeComposite:
    def __init__(self):
        self.engines = [FakeProbabilityEngine()]
        self.calls = 0

    def get_statistics(self):
        return {
            "engine_success_counts": (
                {"ProbabilityBranchingEngine": self.calls}
                if self.calls
                else {}
            ),
            "random_fallback_count": 0,
        }

    def getNextActivities(self, event, possible):
        self.calls += 1
        return ["B"]


class Event:
    activity = "A"

    class eventCase:
        caseId = "C1"


def test_composite_branching_adapter_records_source_and_probabilities():
    adapter = CompositeBranchingAdapter(FakeComposite())

    prediction = adapter.predict(
        event=Event(),
        possible_activities=["B", "C"],
        prediction_time=datetime(2026, 1, 1, 9, 0),
    )

    assert prediction.case_id == "C1"
    assert prediction.current_activity == "A"
    assert prediction.selected_activity == "B"
    assert prediction.prediction_source == "ProbabilityBranchingEngine"
    assert prediction.probabilities == {"B": 0.75, "C": 0.25}
    assert prediction.confidence == 0.75
