from src.resource_allocation.AllocationStrategy import Resource
from src.resource_allocation.MLPredictionAdapter import MLPredictionAdapter
from src.resource_allocation.ParkSongAllocation import ParkSongAllocation
from src.resource_allocation.ParkSongMLIntegration import ParkSongMLIntegration


class FakeModel:
    def predict_proba(self, X):
        return [[0.9, 0.1]]


class FakePredictiveEngine:
    def __init__(self):
        self.is_trained = True
        self.model = FakeModel()
        self.feature_names = ["current_activity", "event_index"]
        self.classes_ = ["A_APPROVED", "A_REJECTED"]

    def extract_features_from_event(self, event):
        return {
            "current_activity": event.get("concept:name", "UNKNOWN"),
            "event_index": event.get("event_index", 0),
        }

    def _default_value_for_feature(self, feature_name):
        if feature_name == "event_index":
            return 0

        return "UNKNOWN"


def test_parksong_ml_integration_creates_reservation_from_ml_prediction():
    predictive_engine = FakePredictiveEngine()

    adapter = MLPredictionAdapter(
        predictive_engine=predictive_engine,
        default_expected_delay=1.0,
    )

    allocator = ParkSongAllocation(
        prediction_probability_threshold=0.5,
        allow_strategic_idling=True,
    )

    integration = ParkSongMLIntegration(
        prediction_adapter=adapter,
        allocator=allocator,
    )

    event = {
        "case:concept:name": "C1",
        "concept:name": "A_START",
        "event_index": 0,
    }

    possible_activities = ["A_APPROVED", "A_REJECTED"]

    resources = [
        Resource(
            resource_id="R1",
            available=True,
            skills=["A_APPROVED", "A_REJECTED"],
        )
    ]

    waiting_tasks = []

    decisions = integration.allocate_with_ml_predictions(
        event=event,
        possible_activities=possible_activities,
        resources=resources,
        waiting_tasks=waiting_tasks,
        current_time=0.0,
    )

    assert len(decisions) == 1
    assert decisions[0].resource_id == "R1"
    assert decisions[0].decision_type == "reservation"
    assert decisions[0].activity == "A_APPROVED"
    assert decisions[0].case_id == "C1"


def test_parksong_ml_integration_assigns_current_task_when_current_is_better():
    from src.resource_allocation.AllocationStrategy import Task

    predictive_engine = FakePredictiveEngine()

    adapter = MLPredictionAdapter(
        predictive_engine=predictive_engine,
        default_expected_delay=10.0,
    )

    allocator = ParkSongAllocation(
        prediction_probability_threshold=0.5,
        uncertainty_weight=5.0,
        idling_weight=1.0,
        allow_strategic_idling=True,
    )

    integration = ParkSongMLIntegration(
        prediction_adapter=adapter,
        allocator=allocator,
    )

    event = {
        "case:concept:name": "C2",
        "concept:name": "A_START",
        "event_index": 0,
    }

    possible_activities = ["A_APPROVED", "A_REJECTED"]

    resources = [
        Resource(
            resource_id="R1",
            available=True,
            skills=["A_APPROVED", "A_REJECTED", "A_CURRENT"],
        )
    ]

    waiting_tasks = [
        Task(
            task_id="T1",
            case_id="C_CURRENT",
            activity="A_CURRENT",
            enabled_time=0.0,
        )
    ]

    decisions = integration.allocate_with_ml_predictions(
        event=event,
        possible_activities=possible_activities,
        resources=resources,
        waiting_tasks=waiting_tasks,
        current_time=0.0,
    )

    assert len(decisions) == 1
    assert decisions[0].decision_type == "assignment"
    assert decisions[0].task_id == "T1"
    assert decisions[0].activity == "A_CURRENT"
