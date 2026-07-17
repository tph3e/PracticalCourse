from datetime import datetime
from types import SimpleNamespace

from src.resource_allocation.integration.CompositeBranchingAdapter import (
    CompositeBranchingAdapter,
)
from src.resource_allocation.integration.TransitionAwareBranching import (
    TransitionDisambiguationModel,
)


class LabelEngine:
    def __init__(self, label):
        self.label = label

    def getNextActivities(self, event, possible_activities):
        return [self.label]

    def get_statistics(self):
        return {}


def candidate(transition_id, label, marking="m0"):
    return SimpleNamespace(
        transition_id=transition_id,
        activity_label=label,
        source_marking=marking,
        pre_visible_marking=marking,
        marking_before=marking,
        resulting_marking=f"{marking}->{transition_id}",
        silent_transition_path=(),
        duplicate_label_count=1,
    )


def test_transition_adapter_records_duplicate_label_ambiguity():
    adapter = CompositeBranchingAdapter(LabelEngine("A"))

    prediction = adapter.predict_transition(
        event=SimpleNamespace(activity="X"),
        transition_candidates=[
            candidate("t2", "A", "m2"),
            candidate("t1", "A", "m1"),
        ],
        prediction_time=datetime(2026, 1, 1),
    )

    assert prediction.selected_activity == "A"
    assert prediction.selected_transition_id == "t1"
    assert prediction.transition_ambiguity is True
    assert adapter.diagnostics()["transition_ambiguities"] == 1


def test_transition_adapter_rejects_invalid_prediction_and_uses_fallback():
    model = TransitionDisambiguationModel()
    model.observe(
        transition_id="t2",
        activity_label="B",
        marking_signature="m0",
        previous_activity="START",
        current_activity="X",
        visit_count_bucket=0,
        repetition_bucket=0,
    )
    adapter = CompositeBranchingAdapter(LabelEngine("INVALID"), seed=1, transition_model=model)

    prediction = adapter.predict_transition(
        event=SimpleNamespace(activity="X"),
        transition_candidates=[candidate("t1", "A"), candidate("t2", "B")],
        prediction_time=datetime(2026, 1, 1),
    )

    assert prediction.rejected_activity == "INVALID"
    assert prediction.selected_activity == "B"
    assert prediction.selected_transition_id == "t2"
    assert prediction.fallback_source
    assert adapter.diagnostics()["invalid_branch_predictions"] == 1


def test_transition_adapter_uses_train_derived_duplicate_label_resolution():
    model = TransitionDisambiguationModel()
    model.observe(
        transition_id="t2",
        activity_label="A",
        marking_signature="m0",
        previous_activity="START",
        current_activity="X",
        visit_count_bucket=0,
        repetition_bucket=0,
    )
    adapter = CompositeBranchingAdapter(LabelEngine("A"), transition_model=model)

    prediction = adapter.predict_transition(
        event=SimpleNamespace(activity="X"),
        transition_candidates=[
            candidate("t1", "A", "m0"),
            candidate("t2", "A", "m0"),
        ],
        prediction_time=datetime(2026, 1, 1),
    )

    assert prediction.transition_ambiguity is True
    assert prediction.selected_transition_id == "t2"
    assert prediction.fallback_source == "marking_label_probability"
