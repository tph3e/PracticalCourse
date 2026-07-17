from src.resource_allocation.MLPredictionAdapter import MLPredictionAdapter
import pandas as pd
import pytest


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
    assert predictions[0].confidence == 0.2


class ConfigurableFakeModel:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, X):
        return [self.probabilities]


@pytest.mark.parametrize("raw_probability", [0.2, 0.49, 0.51])
def test_ml_prediction_adapter_preserves_raw_probability_after_bpmn_filtering(
    raw_probability,
):
    engine = FakePredictiveEngine()
    engine.model = ConfigurableFakeModel([1.0 - raw_probability, raw_probability])

    adapter = MLPredictionAdapter(
        predictive_engine=engine,
        default_expected_delay=1.0,
    )

    predictions = adapter.predict_for_event(
        event={
            "case:concept:name": "C1",
            "concept:name": "A_START",
        },
        possible_activities=["A_REJECTED"],
    )

    assert len(predictions) == 1
    assert predictions[0].activity == "A_REJECTED"
    assert predictions[0].probability == pytest.approx(raw_probability)
    assert predictions[0].confidence == pytest.approx(raw_probability)


def test_ml_prediction_adapter_preserves_raw_probabilities_with_multiple_valid_activities():
    engine = FakePredictiveEngine()
    engine.model = ConfigurableFakeModel([0.51, 0.49])

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

    assert [prediction.activity for prediction in predictions] == [
        "A_APPROVED",
        "A_REJECTED",
    ]
    assert [prediction.probability for prediction in predictions] == [
        pytest.approx(0.51),
        pytest.approx(0.49),
    ]


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


def test_ml_prediction_adapter_returns_empty_without_model():
    engine = FakePredictiveEngine()
    engine.model = None

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
    assert adapter.diagnostics()["missing_model"] == 1


def test_ml_prediction_adapter_returns_empty_when_no_model_class_is_bpmn_valid():
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
        possible_activities=["A_UNKNOWN"],
    )

    assert predictions == []
    assert adapter.diagnostics()["zero_valid_probability_mass"] == 1


class FailingModel:
    def predict_proba(self, X):
        raise ValueError("bad schema")


def test_ml_prediction_adapter_records_predict_proba_failures():
    engine = FakePredictiveEngine()
    engine.model = FailingModel()

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

    diagnostics = adapter.diagnostics()
    assert predictions == []
    assert diagnostics["predict_proba_failures"] == 1
    assert diagnostics["last_error"] == "bad schema"


class SchemaAwareEngine(FakePredictiveEngine):
    def __init__(self):
        super().__init__()
        self.feature_names = ["current_activity", "event_index", "amount"]
        self.numeric_feature_names = ["event_index", "amount"]
        self.categorical_feature_names = ["current_activity"]
        self.seen_columns = None

    def extract_features_from_event(self, event):
        return {
            "current_activity": event.get("concept:name", "UNKNOWN"),
            "amount": event.get("amount", "not numeric"),
        }

    def prepare_features_for_prediction(self, X):
        for feature_name in self.feature_names:
            if feature_name not in X.columns:
                X[feature_name] = self._default_value_for_feature(feature_name)
        X = X[self.feature_names]
        X["amount"] = pd.to_numeric(X["amount"], errors="coerce").fillna(0.0)
        self.seen_columns = list(X.columns)
        return X


def test_ml_prediction_adapter_uses_engine_schema_alignment():
    engine = SchemaAwareEngine()

    adapter = MLPredictionAdapter(
        predictive_engine=engine,
        default_expected_delay=1.0,
    )

    predictions = adapter.predict_for_event(
        event={
            "case:concept:name": "C1",
            "concept:name": "A_START",
            "amount": "invalid",
        },
        possible_activities=["A_APPROVED", "A_REJECTED"],
    )

    assert [prediction.activity for prediction in predictions] == [
        "A_APPROVED",
        "A_REJECTED",
    ]
    assert engine.seen_columns == ["current_activity", "event_index", "amount"]
    assert adapter.diagnostics()["predictions_returned"] == 2


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
