from src.resource_allocation.MLPredictionAdapter import MLPredictionAdapter


class FakeModel:
    def predict_proba(self, X):
        return [[0.8, 0.2]]


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


def test_ml_prediction_adapter_converts_ml_output_to_predictions():
    engine = FakePredictiveEngine()

    adapter = MLPredictionAdapter(
        predictive_engine=engine,
        default_expected_delay=1.5,
    )

    event = {
        "case:concept:name": "C1",
        "concept:name": "A_START",
        "event_index": 3,
    }

    predictions = adapter.predict_for_event(
        event=event,
        possible_activities=["A_APPROVED", "A_REJECTED"],
    )

    assert len(predictions) == 2

    assert predictions[0].case_id == "C1"
    assert predictions[0].activity == "A_APPROVED"
    assert predictions[0].probability == 0.8
    assert predictions[0].expected_delay == 1.5
    assert predictions[0].source == "PredictiveBranchingEngine"
    assert predictions[0].confidence == 0.8

    assert predictions[1].case_id == "C1"
    assert predictions[1].activity == "A_REJECTED"
    assert predictions[1].probability == 0.2
    assert predictions[1].expected_delay == 1.5


def test_ml_prediction_adapter_filters_impossible_activities():
    engine = FakePredictiveEngine()

    adapter = MLPredictionAdapter(
        predictive_engine=engine,
        default_expected_delay=1.0,
    )

    event = {
        "case:concept:name": "C1",
        "concept:name": "A_START",
    }

    predictions = adapter.predict_for_event(
        event=event,
        possible_activities=["A_REJECTED"],
    )

    assert len(predictions) == 1
    assert predictions[0].activity == "A_REJECTED"
    assert predictions[0].probability == 0.2


def test_ml_prediction_adapter_returns_empty_when_engine_is_not_trained():
    engine = FakePredictiveEngine()
    engine.is_trained = False

    adapter = MLPredictionAdapter(
        predictive_engine=engine,
        default_expected_delay=1.0,
    )

    predictions = adapter.predict_for_event(
        event={
            "case:concept:name": "C1",
            "concept:name": "A_START",
        },
        possible_activities=["A_APPROVED", "A_REJECTED"],
    )

    assert predictions == []


def test_ml_prediction_adapter_returns_empty_without_possible_activities():
    engine = FakePredictiveEngine()

    adapter = MLPredictionAdapter(
        predictive_engine=engine,
        default_expected_delay=1.0,
    )

    predictions = adapter.predict_for_event(
        event={
            "case:concept:name": "C1",
            "concept:name": "A_START",
        },
        possible_activities=[],
    )

    assert predictions == []
