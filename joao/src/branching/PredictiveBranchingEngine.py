from __future__ import annotations
from pandas.api.types import is_numeric_dtype

from typing import Any
import random

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .BranchingLogHandler import BranchingLogHandler
from .BranchingUtils import choose_random_valid_activity, extract_current_activity


class PredictiveBranchingEngine:
    """
    Predictive branching engine for Task 1.5 Advanced 2.

    This engine formulates branching decisions as a supervised next-activity
    prediction task.

    Training:
        event-log features -> next_activity

    Runtime:
        event + possibleActivities -> BPMN-valid predicted next activity

    The engine uses RandomForestClassifier as the main ML model.
    """

    def __init__(
        self,
        fallback_engine: Any | None = None,
        log_handler: BranchingLogHandler | None = None,
        seed: int = 1,
        case_col: str = "case:concept:name",
        activity_col: str = "concept:name",
        timestamp_col: str = "time:timestamp",
        resource_col: str = "org:resource",
        feature_columns: list[str] | None = None,
        n_estimators: int = 100,
        max_depth: int | None = 8,
        min_samples_leaf: int = 1,
        class_weight: str | dict | None = "balanced",
    ):
        self.fallback_engine = fallback_engine
        self.seed = seed
        self.random = random.Random(seed)

        self.case_col = case_col
        self.activity_col = activity_col
        self.timestamp_col = timestamp_col
        self.resource_col = resource_col
        self.feature_columns = feature_columns or []

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight

        self.log_handler = log_handler or BranchingLogHandler(
            case_col=case_col,
            activity_col=activity_col,
            timestamp_col=timestamp_col,
        )

        self.model: Pipeline | None = None
        self.feature_names: list[str] = []
        self.decision_points: set[str] = set()
        self.classes_: list[str] = []

        self.is_trained = False

        self.total_predictions = 0
        self.valid_ml_predictions = 0
        self.fallback_count = 0

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    def train(self, event_log: pd.DataFrame) -> None:
        """
        Trains the Random Forest model on decision-point occurrences.
        """

        dataset = self.build_training_dataset(event_log)

        if dataset.empty:
            raise ValueError(
                "Cannot train PredictiveBranchingEngine: training dataset is empty."
            )

        if "next_activity" not in dataset.columns:
            raise ValueError("Training dataset must contain 'next_activity'.")

        X = dataset.drop(columns=["next_activity"])
        y = dataset["next_activity"]

        self.feature_names = list(X.columns)

        categorical_features = [
            col
            for col in X.columns
            if not is_numeric_dtype(X[col])
        ]

        numeric_features = [
            col
            for col in X.columns
            if is_numeric_dtype(X[col])
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_features,
                ),
                (
                    "numeric",
                    "passthrough",
                    numeric_features,
                ),
            ]
        )

        classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.seed,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
        )

        self.model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        self.model.fit(X, y)

        self.classes_ = list(self.model.named_steps["classifier"].classes_)
        self.is_trained = True

    # --------------------------------------------------
    # Dataset construction
    # --------------------------------------------------

    def build_training_dataset(self, event_log: pd.DataFrame) -> pd.DataFrame:
        """
        Builds a supervised dataset from the event log.

        Each row represents an occurrence of a decision point.
        The target is the observed next activity.
        """

        prepared_log = self.log_handler.prepare_log(event_log)

        transition_counts = self.log_handler.extract_transition_counts(prepared_log)
        self.decision_points = self.log_handler.discover_decision_points(
            transition_counts
        )

        rows: list[dict[str, Any]] = []

        for _, case_events in prepared_log.groupby(self.case_col):
            case_events = case_events.sort_values(
                self.timestamp_col
            ).reset_index(drop=True)

            if len(case_events) < 2:
                continue

            case_start_time = case_events.iloc[0][self.timestamp_col]

            for index in range(len(case_events) - 1):
                current_row = case_events.iloc[index]
                next_row = case_events.iloc[index + 1]

                current_activity = current_row[self.activity_col]

                if current_activity not in self.decision_points:
                    continue

                previous_activity = (
                    "START"
                    if index == 0
                    else case_events.iloc[index - 1][self.activity_col]
                )

                timestamp = current_row[self.timestamp_col]

                row = {
                    "current_activity": current_activity,
                    "previous_activity": previous_activity,
                    "event_index": index,
                    "weekday": timestamp.weekday(),
                    "hour": timestamp.hour,
                    "month": timestamp.month,
                    "elapsed_time_seconds": (
                        timestamp - case_start_time
                    ).total_seconds(),
                    "next_activity": next_row[self.activity_col],
                }

                if self.resource_col in prepared_log.columns:
                    resource_value = current_row.get(self.resource_col, "UNKNOWN")
                    row["resource"] = (
                        "UNKNOWN" if pd.isna(resource_value) else resource_value
                    )

                for column in self.feature_columns:
                    if column in prepared_log.columns:
                        value = current_row.get(column)
                        row[column] = "UNKNOWN" if pd.isna(value) else value

                rows.append(row)

        return pd.DataFrame(rows)

    # --------------------------------------------------
    # Simulation interface
    # --------------------------------------------------

    def getNextActivities(
        self,
        event: Any,
        possibleActivities: list[str],
    ) -> list[str]:
        """
        Predicts the next activity and filters prediction by BPMN possibleActivities.
        """

        if not possibleActivities:
            return []

        if len(possibleActivities) == 1:
            return possibleActivities

        self.total_predictions += 1

        if not self.is_trained or self.model is None:
            return self.fallback(event, possibleActivities)

        features = self.extract_features_from_event(event)
        X = pd.DataFrame([features])

        for feature_name in self.feature_names:
            if feature_name not in X.columns:
                X[feature_name] = self._default_value_for_feature(feature_name)

        X = X[self.feature_names]

        selected_activity = self._predict_best_bpmn_valid_activity(
            X=X,
            possibleActivities=possibleActivities,
        )

        if selected_activity is not None:
            self.valid_ml_predictions += 1
            return [selected_activity]

        return self.fallback(event, possibleActivities)

    def extract_features_from_event(self, event: Any) -> dict[str, Any]:
        """
        Extracts runtime features from a simulation event.
        """

        attributes = self._extract_attributes(event)

        current_activity = extract_current_activity(
            event,
            activity_col=self.activity_col,
        )

        previous_activity = self._extract_previous_activity(event)

        features = {
            "current_activity": current_activity or "UNKNOWN",
            "previous_activity": previous_activity or "UNKNOWN",
            "event_index": attributes.get("event_index", 0),
            "weekday": attributes.get("weekday", 0),
            "hour": attributes.get("hour", 0),
            "month": attributes.get("month", 0),
            "elapsed_time_seconds": attributes.get("elapsed_time_seconds", 0.0),
        }

        if "resource" in self.feature_names:
            features["resource"] = attributes.get(
                self.resource_col,
                attributes.get("resource", "UNKNOWN"),
            )

        for column in self.feature_columns:
            if column in self.feature_names:
                features[column] = attributes.get(column, "UNKNOWN")

        return features

    def _predict_best_bpmn_valid_activity(
        self,
        X: pd.DataFrame,
        possibleActivities: list[str],
    ) -> str | None:
        """
        Uses predicted probabilities and selects the highest-probability
        BPMN-valid activity.
        """

        if self.model is None:
            return None

        probabilities = self.model.predict_proba(X)[0]

        class_probability_pairs = list(zip(self.classes_, probabilities))
        class_probability_pairs.sort(key=lambda pair: pair[1], reverse=True)

        for activity, _ in class_probability_pairs:
            if activity in possibleActivities:
                return activity

        return None

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    def evaluate(self, test_log: pd.DataFrame) -> dict[str, Any]:
        """
        Evaluates the trained model on a test event log.
        """

        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before evaluation.")

        dataset = self.build_training_dataset(test_log)

        if dataset.empty:
            return {
                "accuracy": None,
                "macro_f1": None,
                "weighted_f1": None,
                "confusion_matrix": None,
                "n_samples": 0,
            }

        X = dataset.drop(columns=["next_activity"])
        y_true = dataset["next_activity"]

        for feature_name in self.feature_names:
            if feature_name not in X.columns:
                X[feature_name] = self._default_value_for_feature(feature_name)

        X = X[self.feature_names]

        y_pred = self.model.predict(X)

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            "weighted_f1": f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "n_samples": len(dataset),
        }

    # --------------------------------------------------
    # Fallback and helpers
    # --------------------------------------------------

    def fallback(
        self,
        event: Any,
        possibleActivities: list[str],
    ) -> list[str]:
        """
        Uses fallback engine if available. Otherwise uses random BPMN-valid choice.
        """

        self.fallback_count += 1

        if self.fallback_engine is not None:
            result = self.fallback_engine.getNextActivities(
                event,
                possibleActivities,
            )

            if self._is_valid_result(result, possibleActivities):
                return result

        return choose_random_valid_activity(
            possible_activities=possibleActivities,
            random_generator=self.random,
        )

    def _extract_attributes(self, event: Any) -> dict[str, Any]:
        """
        Extracts attributes from different possible event structures.
        """

        attributes: dict[str, Any] = {}

        if event is None:
            return attributes

        if hasattr(event, "getAttribs"):
            event_attributes = event.getAttribs()
            if isinstance(event_attributes, dict):
                attributes.update(event_attributes)

        if hasattr(event, "attributes"):
            event_attributes = event.attributes
            if isinstance(event_attributes, dict):
                attributes.update(event_attributes)

        if hasattr(event, "data"):
            event_data = event.data
            if isinstance(event_data, dict):
                attributes.update(event_data)

        if isinstance(event, dict):
            attributes.update(event)

        return attributes

    def _extract_previous_activity(self, event: Any) -> str | None:
        """
        Extracts previous activity from event history if available.
        """

        if event is None:
            return None

        if hasattr(event, "history"):
            history = event.history
            if isinstance(history, list) and history:
                return history[-1]

        if hasattr(event, "getAttribOfLastEvents"):
            try:
                history_attributes = event.getAttribOfLastEvents(-1)

                if isinstance(history_attributes, dict):
                    history = history_attributes.get("history")

                    if isinstance(history, list) and history:
                        return history[-1]
            except TypeError:
                pass

        return None

    def _default_value_for_feature(self, feature_name: str) -> Any:
        """
        Returns default values for missing runtime features.
        """

        numeric_defaults = {
            "event_index": 0,
            "weekday": 0,
            "hour": 0,
            "month": 0,
            "elapsed_time_seconds": 0.0,
        }

        return numeric_defaults.get(feature_name, "UNKNOWN")

    def _is_valid_result(
        self,
        result: list[str],
        possibleActivities: list[str],
    ) -> bool:
        """
        Checks whether a result is BPMN-valid.
        """

        if not isinstance(result, list):
            return False

        if not possibleActivities:
            return result == []

        return all(activity in possibleActivities for activity in result)

    def get_prediction_statistics(self) -> dict[str, int | float]:
        """
        Returns prediction and fallback statistics.
        """

        valid_rate = (
            self.valid_ml_predictions / self.total_predictions
            if self.total_predictions > 0
            else 0.0
        )

        fallback_rate = (
            self.fallback_count / self.total_predictions
            if self.total_predictions > 0
            else 0.0
        )

        return {
            "total_predictions": self.total_predictions,
            "valid_ml_predictions": self.valid_ml_predictions,
            "fallback_count": self.fallback_count,
            "valid_ml_prediction_rate": valid_rate,
            "fallback_rate": fallback_rate,
        }