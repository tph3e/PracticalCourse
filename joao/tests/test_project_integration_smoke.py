from datetime import datetime

import pandas as pd

from BPMN_engine import BPMNEngine
from resources import ResourceEngine
from joao.src.branching.CompositeBranchingEngine import CompositeBranchingEngine
from joao.src.resource_allocation.AllocationStrategy import Resource
from joao.src.resource_allocation.MLPredictionAdapter import MLPredictionAdapter
from joao.src.resource_allocation.ParkSongAllocation import ParkSongAllocation
from joao.src.resource_allocation.ParkSongMLIntegration import ParkSongMLIntegration


class _Event:
    def __init__(self, activity, time=None):
        self.activity = activity
        self.time = time or datetime(2026, 1, 5, 9, 0)
        self.resource = ""

    def getAttribs(self):
        return {
            "concept:name": self.activity,
            "time:timestamp": self.time,
            "case:concept:name": "SMOKE_CASE",
        }


class _FakePredictiveModel:
    def predict_proba(self, X):
        return [[0.9, 0.1]]


class _FakePredictiveEngine:
    is_trained = True
    model = _FakePredictiveModel()
    classes_ = ["A_Create Application", "A_Submitted"]
    feature_names = ["current_activity", "event_index"]

    def extract_features_from_event(self, event):
        return {
            "current_activity": getattr(event, "activity", "UNKNOWN"),
            "event_index": 0,
        }

    def prepare_features_for_prediction(self, X):
        return X[self.feature_names]


def test_bpmn_composite_branching_resource_and_parksong_smoke():
    bpmn = BPMNEngine()
    assert bpmn.model_filename in {"models/v4_replay.bpmn", "models/v5_simulation.bpmn"}

    start_activity = bpmn.getStartActivity()
    assert start_activity

    case_id = "SMOKE_CASE"
    bpmn.initialize_case(case_id)
    possible_activities = bpmn.getPossibleNextActivities("", case_id=case_id)
    assert start_activity in possible_activities

    branching = CompositeBranchingEngine(
        engines=[],
        seed=1,
        use_default_hierarchy=False,
    )
    selected_activities = branching.getNextActivities(
        _Event(start_activity),
        possible_activities,
    )
    assert selected_activities
    assert set(selected_activities).issubset(set(possible_activities))

    log = pd.DataFrame(
        [
            {
                "concept:name": start_activity,
                "org:resource": "R_SMOKE",
                "time:timestamp": datetime(2026, 1, 5, 9, 0),
            }
        ]
    )
    resource_engine = ResourceEngine(log, seed=1)

    # Keep the facade real while forcing a tiny deterministic basic-mode fixture.
    resource_engine.availability.calendars = None
    resource_engine.availability._all_resources = {"R_SMOKE"}
    resource_engine.permissions._activity_to_resources = {
        start_activity: {"R_SMOKE"}
    }

    event = _Event(start_activity)
    assert resource_engine.allocateResource(event) is True
    assert event.resource == "R_SMOKE"
    assert "R_SMOKE" in resource_engine.busy

    adapter = MLPredictionAdapter(
        predictive_engine=_FakePredictiveEngine(),
        default_expected_delay=1.0,
    )
    parksong_ml = ParkSongMLIntegration(
        prediction_adapter=adapter,
        allocator=ParkSongAllocation(prediction_probability_threshold=0.5),
    )
    decisions = parksong_ml.allocate_with_ml_predictions(
        event=_Event(start_activity),
        possible_activities=[start_activity, "A_Submitted"],
        resources=[Resource("R_SMOKE", available=True, skills=[start_activity])],
        waiting_tasks=[],
        current_time=0.0,
    )

    assert decisions
    assert decisions[0].decision_type == "reservation"
    assert decisions[0].activity == start_activity
