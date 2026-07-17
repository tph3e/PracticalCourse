from __future__ import annotations

from collections import Counter
from typing import Any, List

import pandas as pd

from .AllocationStrategy import Prediction


class MLPredictionAdapter:
    """
    Adapter between PredictiveBranchingEngine and ParkSongAllocation

    It converts ML next-activity probabilities into Prediction objects
    that ParkSongAllocation can use as predicted future task candidates
    """

    def __init__(
            self,
            predictive_engine: Any,
            default_expected_delay: float = 1.0,
            source_name: str = "PredictiveBranchingEngine",
    ):
        self.predictive_engine = predictive_engine
        self.default_expected_delay = default_expected_delay
        self.source_name = source_name
        self.counters: Counter[str] = Counter()
        self.last_error: str | None = None

    def predict_for_event(
            self,
            event: Any,
            possible_activities: List[str],
    ) -> List[Prediction]:
        """
        Generate ParkSongAllocation-compatible predictions for one simulation event
        """
        self.counters["prediction_calls"] += 1
        self.last_error = None

        if not possible_activities:
            self.counters["empty_possible_activities"] += 1
            return []
        
        if not getattr(self.predictive_engine, "is_trained", False):
            self.counters["untrained_engine"] += 1
            return []
        
        if self.predictive_engine.model is None:
            self.counters["missing_model"] += 1
            return []
        
        try:
            features = self.predictive_engine.extract_features_from_event(event)
            X = self._prepare_prediction_features(features)
        except Exception as exc:
            self.counters["feature_preparation_failures"] += 1
            self.last_error = str(exc)
            return []

        try:
            probabilities = self.predictive_engine.model.predict_proba(X)[0]
        except Exception as exc:
            self.counters["predict_proba_failures"] += 1
            self.last_error = str(exc)
            return []

        classes = self._model_classes()

        case_id = self._extract_case_id(event)
        valid_activity_set = set(possible_activities)
        class_probabilities = [
            (activity, float(probability))
            for activity, probability in zip(classes, probabilities)
            if activity in valid_activity_set
        ]
        if not class_probabilities:
            self.counters["zero_valid_probability_mass"] += 1
            return []

        predictions: List[Prediction] = []

        for activity, raw_probability in class_probabilities:
            probability = max(0.0, raw_probability)

            predictions.append(
                Prediction(
                    case_id=case_id,
                    activity=activity,
                    probability=probability,
                    expected_delay=self.default_expected_delay,
                    source=self.source_name,
                    confidence=raw_probability,
                )
            )

        predictions.sort(
            key=lambda prediction: prediction.probability,
            reverse=True,
        )
        self.counters["predictions_returned"] += len(predictions)

        return predictions

    def diagnostics(self) -> dict[str, Any]:
        """
        Return adapter runtime counters for audit/debugging.
        """

        return {
            **dict(self.counters),
            "last_error": self.last_error,
        }

    def _prepare_prediction_features(self, features: dict[str, Any]) -> pd.DataFrame:
        """
        Prepare features for real PredictiveBranchingEngine instances and
        older/fake engines used by tests.
        """

        X = pd.DataFrame([features])

        if hasattr(self.predictive_engine, "prepare_features_for_prediction"):
            return self.predictive_engine.prepare_features_for_prediction(X)

        for feature_name in self.predictive_engine.feature_names:
            if feature_name not in X.columns:
                X[feature_name] = self._default_value_for_feature(feature_name)

        return X[self.predictive_engine.feature_names]

    def _model_classes(self) -> list[str]:
        if getattr(self.predictive_engine, "classes_", None):
            return list(self.predictive_engine.classes_)

        model = getattr(self.predictive_engine, "model", None)
        if model is not None and hasattr(model, "named_steps"):
            classifier = model.named_steps.get("classifier")
            if classifier is not None and hasattr(classifier, "classes_"):
                return list(classifier.classes_)

        return []

    def _default_value_for_feature(self, feature_name: str) -> Any:
        if hasattr(self.predictive_engine, "_default_value_for_feature"):
            return self.predictive_engine._default_value_for_feature(feature_name)

        numeric_defaults = {
            "event_index": 0,
            "weekday": 0,
            "hour": 0,
            "month": 0,
            "elapsed_time_seconds": 0.0,
        }

        return numeric_defaults.get(feature_name, "UNKNOWN")

    def _extract_case_id(self, event: Any) -> str:
        """
        Extract case id from dict-like or object-like events
        """

        if event is None:
            return "UNKNOWN_CASE"

        if isinstance(event, dict):
            return (
                event.get("case:concept:name")
                or event.get("case_id")
                or event.get("case")
                or "UNKNOWN_CASE"
            )

        if hasattr(event, "getAttribs"):
            attributes = event.getAttribs()
            if isinstance(attributes, dict):
                return (
                    attributes.get("case:concept:name")
                    or attributes.get("case_id")
                    or attributes.get("case")
                    or "UNKNOWN_CASE"
                )

        if hasattr(event, "case_id"):
            return event.case_id

        return "UNKNOWN_CASE"
