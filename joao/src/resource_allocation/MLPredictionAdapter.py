from __future__ import annotations

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

    
    def predict_for_event(
            self,
            event: Any,
            possible_activities: List[str],
    ) -> List[Prediction]:
        """
        Generate ParkSongAllocation-compatible predictions for one simulation event
        """

        if not possible_activities:
            return []
        
        if not getattr(self.predictive_engine, "is_trained", False):
            return []
        
        if self.predictive_engine.model is None:
            return []
        
        features = self.predictive_engine.extract_features_from_event(event)
        X = pd.DataFrame([features])

        for feature_name in self.predictive_engine.feature_names:
            if feature_name not in X.columns:
                X[feature_name] = self.predictive_engine._default_value_for_feature(feature_name)

        X = X[self.predictive_engine.feature_names]

        probabilities = self.predictive_engine.model.predict_proba(X)[0]
        classes = self.predictive_engine.classes_

        case_id = self._extract_case_id(event)

        predictions: List[Prediction] = []

        for activity, probability in zip(classes, probabilities):
            if activity not in possible_activities:
                continue

            predictions.append(
                Prediction(
                    case_id=case_id,
                    activity=activity,
                    probability=float(probability),
                    expected_delay=self.default_expected_delay,
                    source=self.source_name,
                    confidence=float(probability),
                )
            )

        predictions.sort(
            key=lambda prediction: prediction.probability,
            reverse=True,
        )

        return predictions


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
    
    