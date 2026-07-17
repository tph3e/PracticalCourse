from __future__ import annotations
from pandas.api.types import is_numeric_dtype

from typing import Any
import importlib.util
import random
import sys
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .BranchingLogHandler import BranchingLogHandler
from .BranchDecision import BranchDecision
from .BranchingFeatureBuilder import BranchingFeatureBuilder
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
        n_jobs: int | None = -1,
        use_bpmn_replay: bool = False,
        bpmn_model_path: str = "models/v4_replay.bpmn",
        bpmn_engine_factory: Any | None = None,
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
        self.n_jobs = n_jobs
        self.use_bpmn_replay = use_bpmn_replay
        self.bpmn_model_path = bpmn_model_path
        self.bpmn_engine_factory = bpmn_engine_factory

        self.log_handler = log_handler or BranchingLogHandler(
            case_col=case_col,
            activity_col=activity_col,
            timestamp_col=timestamp_col,
        )

        self.model: Pipeline | None = None
        self.feature_names: list[str] = []
        self.categorical_feature_names: list[str] = []
        self.numeric_feature_names: list[str] = []
        self.decision_points: set[str] = set()
        self.classes_: list[str] = []

        self.is_trained = False

        self.total_predictions = 0
        self.valid_ml_predictions = 0
        self.fallback_count = 0
        self.dataset_mode = "directly_follows"
        self.bpmn_replay_diagnostics: dict[str, int] = {}
        self.training_bpmn_replay_diagnostics: dict[str, int] = {}
        self.evaluation_bpmn_replay_diagnostics: dict[str, int] = {}
        self.feature_builder = BranchingFeatureBuilder(
            case_col=case_col,
            activity_col=activity_col,
            timestamp_col=timestamp_col,
            resource_col=resource_col,
        )
        self.raw_top_class_rejected_by_bpmn = 0
        self.no_valid_rf_class_count = 0

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
        self.numeric_feature_names = self._infer_numeric_features(X)
        self.categorical_feature_names = [
            col for col in self.feature_names
            if col not in self.numeric_feature_names
        ]
        X = self.prepare_features_for_prediction(X)

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    self.categorical_feature_names,
                ),
                (
                    "numeric",
                    "passthrough",
                    self.numeric_feature_names,
                ),
            ]
        )

        classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.seed,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
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

        if self.use_bpmn_replay:
            return self.build_bpmn_replay_dataset(
                prepared_log,
                update_decision_points=True,
            )

        transition_counts = self.log_handler.extract_transition_counts(prepared_log)
        self.decision_points = self.log_handler.discover_decision_points(
            transition_counts
        )
        self.dataset_mode = "directly_follows"
        return self._build_dataset_from_prepared_log(
            prepared_log=prepared_log,
            decision_points=self.decision_points,
        )

    def build_bpmn_replay_dataset(
        self,
        event_log: pd.DataFrame,
        update_decision_points: bool = False,
    ) -> pd.DataFrame:
        """
        Builds decision observations by replaying each trace on the BPMN model.

        A row is emitted only after the current event has been synchronized and
        fired in the model, the resulting marking exposes multiple BPMN-valid
        next labels, and the observed next activity is one of those labels.
        """

        prepared_log = self.log_handler.prepare_log(event_log)
        engine = self._new_bpmn_engine()
        rows: list[dict[str, Any]] = []
        diagnostics = {
            "cases_seen": 0,
            "events_seen": 0,
            "events_synchronized": 0,
            "nonconformant_events": 0,
            "ambiguous_events": 0,
            "decision_observations": 0,
            "skipped_single_candidate": 0,
            "skipped_unobserved_next_label": 0,
        }
        decision_points: set[str] = set()

        for case_id, case_events in prepared_log.groupby(self.case_col):
            diagnostics["cases_seen"] += 1
            case_events = case_events.sort_values(
                self.timestamp_col
            ).reset_index(drop=True)

            if len(case_events) < 2:
                continue

            normalized_case_id = str(case_id)
            engine.initialize_case(normalized_case_id)
            records = case_events.to_dict("records")
            case_start_time = records[0][self.timestamp_col]
            activities = [str(activity) for activity in case_events[self.activity_col]]
            timestamps = case_events[self.timestamp_col].tolist()

            for index, activity in enumerate(activities):
                diagnostics["events_seen"] += 1
                candidates = engine.getPossibleNextTransitionCandidates(
                    normalized_case_id
                )
                matches = [
                    candidate
                    for candidate in candidates
                    if str(candidate.activity_label) == activity
                ]

                if len(matches) != 1:
                    if len(matches) > 1:
                        diagnostics["ambiguous_events"] += 1
                    else:
                        diagnostics["nonconformant_events"] += 1
                    break

                fired = engine.fire_transition_candidate(
                    normalized_case_id,
                    matches[0],
                )
                if not fired:
                    diagnostics["nonconformant_events"] += 1
                    break

                diagnostics["events_synchronized"] += 1

                if index >= len(activities) - 1:
                    continue

                next_candidates = engine.getPossibleNextTransitionCandidates(
                    normalized_case_id
                )
                possible_next_labels = sorted(
                    {str(candidate.activity_label) for candidate in next_candidates}
                )

                if len(possible_next_labels) <= 1:
                    diagnostics["skipped_single_candidate"] += 1
                    continue

                next_activity = activities[index + 1]
                if next_activity not in possible_next_labels:
                    diagnostics["skipped_unobserved_next_label"] += 1
                    continue

                row = self._build_feature_row(
                    records=records,
                    activities=activities,
                    timestamps=timestamps,
                    index=index,
                    case_start_time=case_start_time,
                )
                row["next_activity"] = next_activity
                rows.append(row)
                decision_points.add(activity)
                diagnostics["decision_observations"] += 1

        if update_decision_points:
            self.decision_points = decision_points
            self.dataset_mode = "bpmn_replay"
            self.training_bpmn_replay_diagnostics = diagnostics
        else:
            self.evaluation_bpmn_replay_diagnostics = diagnostics

        self.bpmn_replay_diagnostics = diagnostics
        return pd.DataFrame(rows)

    def _build_dataset_from_prepared_log(
        self,
        prepared_log: pd.DataFrame,
        decision_points: set[str],
    ) -> pd.DataFrame:
        """
        Builds feature rows for the provided decision points without changing
        the trained decision-point set.
        """

        rows: list[dict[str, Any]] = []

        for _, case_events in prepared_log.groupby(self.case_col):
            case_events = case_events.sort_values(
                self.timestamp_col
            ).reset_index(drop=True)

            if len(case_events) < 2:
                continue

            records = case_events.to_dict("records")
            case_start_time = records[0][self.timestamp_col]
            activities = case_events[self.activity_col].tolist()
            timestamps = case_events[self.timestamp_col].tolist()

            for index in range(len(case_events) - 1):
                current_row = records[index]

                current_activity = activities[index]

                if current_activity not in decision_points:
                    continue

                row = self._build_feature_row(
                    records=records,
                    activities=activities,
                    timestamps=timestamps,
                    index=index,
                    case_start_time=case_start_time,
                )
                row["next_activity"] = activities[index + 1]

                rows.append(row)

        return pd.DataFrame(rows)

    def _build_feature_row(
        self,
        records: list[dict[str, Any]],
        activities: list[str],
        timestamps: list[Any],
        index: int,
        case_start_time: Any,
    ) -> dict[str, Any]:
        current_row = records[index]
        current_activity = activities[index]
        previous_activity = "START" if index == 0 else activities[index - 1]
        prefix_start = max(0, index - 3)
        trace_prefix = "|".join(
            str(activity)
            for activity in activities[prefix_start:index]
        )
        prior_activities = activities[:index]
        current_visit_count = sum(
            1 for activity in prior_activities if activity == current_activity
        )
        consecutive_repetition_count = 0
        for activity in reversed(prior_activities):
            if activity != current_activity:
                break
            consecutive_repetition_count += 1

        timestamp = timestamps[index]

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
            "time_since_previous_event_seconds": (
                timestamp - timestamps[index - 1]
            ).total_seconds()
            if index > 0
            else 0.0,
            "trace_prefix": trace_prefix or "START",
            "current_activity_visit_count": current_visit_count,
            "consecutive_repetition_count": consecutive_repetition_count,
        }

        if self.resource_col in current_row:
            resource_value = current_row.get(self.resource_col, "UNKNOWN")
            row["resource"] = (
                "UNKNOWN" if pd.isna(resource_value) else resource_value
            )

        for column in self.feature_columns:
            if column in current_row:
                value = current_row.get(column)
                row[column] = "UNKNOWN" if pd.isna(value) else value

        return row

    def _new_bpmn_engine(self) -> Any:
        if self.bpmn_engine_factory is not None:
            return self.bpmn_engine_factory()

        try:
            from BPMN_engine import BPMNEngine
        except ModuleNotFoundError:
            repo_root = Path(__file__).resolve().parents[3]
            module_path = repo_root / "BPMN_engine.py"
            spec = importlib.util.spec_from_file_location(
                "BPMN_engine",
                module_path,
            )
            if spec is None or spec.loader is None:
                raise
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            BPMNEngine = module.BPMNEngine

        return BPMNEngine(model_filename=self.bpmn_model_path)

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

        decision = self.decide(event, possibleActivities)
        if decision is not None:
            return decision.activities
        return self.fallback(event, possibleActivities)

    def decide(
        self,
        event: Any,
        possibleActivities: list[str],
        context: dict[str, Any] | None = None,
    ) -> BranchDecision | None:
        if not possibleActivities:
            return BranchDecision(
                activities=[],
                decision_source="predictive_empty_candidates",
                used_fallback=True,
                candidate_activities=[],
            )

        if len(possibleActivities) == 1:
            return BranchDecision(
                activities=possibleActivities,
                decision_source="single_bpmn_candidate",
                probability_source="single_candidate",
                probabilities={possibleActivities[0]: 1.0},
                confidence=1.0,
                candidate_activities=list(possibleActivities),
            )

        self.total_predictions += 1
        if not self.is_trained or self.model is None:
            return None

        features = self.extract_features_from_event(event)
        X = self.prepare_features_for_prediction(pd.DataFrame([features]))
        probabilities = self.model.predict_proba(X)[0]
        class_probability_pairs = list(zip(self.classes_, probabilities))
        class_probability_pairs.sort(key=lambda pair: pair[1], reverse=True)
        raw_top_class = class_probability_pairs[0][0] if class_probability_pairs else None
        if raw_top_class not in possibleActivities:
            self.raw_top_class_rejected_by_bpmn += 1
        for activity, probability in class_probability_pairs:
            if activity in possibleActivities:
                self.valid_ml_predictions += 1
                return BranchDecision(
                    activities=[activity],
                    decision_source="predictive_rf_bpmn_constrained",
                    probability_source="rf_predict_proba",
                    probabilities={str(label): float(prob) for label, prob in class_probability_pairs},
                    confidence=float(probability),
                    support=None,
                    used_fallback=False,
                    candidate_activities=list(possibleActivities),
                    metadata={
                        "rf_raw_top_class": raw_top_class,
                        "rf_constrained_selected_class": activity,
                        "raw_class_invalid_under_bpmn": raw_top_class not in possibleActivities,
                    },
                )
        self.no_valid_rf_class_count += 1
        return None

    def extract_features_from_event(self, event: Any) -> dict[str, Any]:
        """
        Extracts runtime features from a simulation event.
        """

        features = self.feature_builder.build_from_runtime_event(event)

        for column in self.feature_columns:
            if column in self.feature_names:
                features[column] = features.get(column, "UNKNOWN")

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

        if self.use_bpmn_replay:
            dataset = self.build_bpmn_replay_dataset(
                test_log,
                update_decision_points=False,
            )
            if not dataset.empty:
                dataset = dataset[
                    dataset["current_activity"].isin(self.decision_points)
                ].reset_index(drop=True)
        else:
            prepared_log = self.log_handler.prepare_log(test_log)
            dataset = self._build_dataset_from_prepared_log(
                prepared_log=prepared_log,
                decision_points=set(self.decision_points),
            )

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

        X = self.prepare_features_for_prediction(X)

        y_pred_raw = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        labels = list(self.classes_)
        y_pred_constrained = []
        fallback_count = 0
        raw_rejected = 0
        for row_index, row in dataset.reset_index(drop=True).iterrows():
            candidate_labels = str(row.get("candidate_activity_labels", "")).split("|")
            candidate_labels = [label for label in candidate_labels if label]
            pairs = list(zip(labels, probabilities[row_index]))
            pairs.sort(key=lambda pair: pair[1], reverse=True)
            if pairs and pairs[0][0] not in candidate_labels:
                raw_rejected += 1
            selected = None
            for activity, _ in pairs:
                if activity in candidate_labels:
                    selected = activity
                    break
            if selected is None:
                fallback_count += 1
                selected = y_pred_raw[row_index]
            y_pred_constrained.append(selected)

        raw_metrics = self._classification_metrics(y_true, y_pred_raw)
        constrained_metrics = self._classification_metrics(y_true, y_pred_constrained)
        try:
            raw_log_loss = log_loss(y_true, probabilities, labels=labels)
        except ValueError:
            raw_log_loss = None

        return {
            "accuracy": raw_metrics["accuracy"],
            "macro_f1": raw_metrics["macro_f1"],
            "weighted_f1": raw_metrics["weighted_f1"],
            "raw_accuracy": raw_metrics["accuracy"],
            "raw_macro_f1": raw_metrics["macro_f1"],
            "raw_weighted_f1": raw_metrics["weighted_f1"],
            "raw_log_loss": raw_log_loss,
            "constrained_accuracy": constrained_metrics["accuracy"],
            "constrained_macro_f1": constrained_metrics["macro_f1"],
            "constrained_weighted_f1": constrained_metrics["weighted_f1"],
            "candidate_coverage": 1 - (fallback_count / len(dataset)),
            "raw_top_class_rejected_by_bpmn": raw_rejected,
            "fallback_rate": fallback_count / len(dataset),
            "confusion_matrix": confusion_matrix(y_true, y_pred_raw).tolist(),
            "n_samples": len(dataset),
        }

    def _classification_metrics(self, y_true: Any, y_pred: Any) -> dict[str, float]:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
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

        event_case = getattr(event, "eventCase", None)
        activities = getattr(event_case, "activities", None)
        if isinstance(activities, list) and len(activities) >= 2:
            current_activity = getattr(event, "activity", None)
            try:
                current_index = len(activities) - 1 - activities[::-1].index(
                    current_activity
                )
            except ValueError:
                current_index = len(activities) - 1
            if current_index > 0:
                return activities[current_index - 1]

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

    def _extract_activity_history(self, event: Any) -> list[str]:
        event_case = getattr(event, "eventCase", None)
        activities = getattr(event_case, "activities", None)
        if isinstance(activities, list):
            return [str(activity) for activity in activities if activity]

        if hasattr(event, "getAttribOfLastEvents"):
            try:
                history_attributes = event.getAttribOfLastEvents(-1)
            except TypeError:
                history_attributes = []
            if isinstance(history_attributes, list):
                return [
                    str(row.get(self.activity_col) or row.get("concept:name"))
                    for row in history_attributes
                    if isinstance(row, dict)
                    and (row.get(self.activity_col) or row.get("concept:name"))
                ]

        current_activity = getattr(event, "activity", None)
        return [str(current_activity)] if current_activity else []

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
            "time_since_previous_event_seconds": 0.0,
            "current_activity_visit_count": 0,
            "consecutive_repetition_count": 0,
        }

        return numeric_defaults.get(feature_name, "UNKNOWN")

    def prepare_features_for_prediction(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aligns feature columns and dtypes with the training-time schema.

        sklearn's OneHotEncoder expects categorical columns to keep a stable
        dtype. BPIC logs can contain mixed Python types and NaN values in the
        same categorical column, especially after XES conversion, so all
        categorical values are converted to strings with a stable missing-value
        placeholder before the ColumnTransformer sees them.
        """

        X = X.copy()
        self._ensure_feature_schema(X)

        for feature_name in self.feature_names:
            if feature_name not in X.columns:
                X[feature_name] = self._default_value_for_feature(feature_name)

        X = X[self.feature_names]

        for feature_name in self.numeric_feature_names:
            if feature_name in X.columns:
                X[feature_name] = pd.to_numeric(
                    X[feature_name],
                    errors="coerce",
                ).fillna(0.0)

        for feature_name in self.categorical_feature_names:
            if feature_name in X.columns:
                values = X[feature_name].astype("object")
                X[feature_name] = values.where(
                    values.notna(),
                    "UNKNOWN",
                ).astype(str)

        return X

    def _infer_numeric_features(self, X: pd.DataFrame) -> list[str]:
        """
        Infers numeric feature columns once during training.
        """

        known_categorical_features = {
            "current_activity",
            "previous_activity",
            "resource",
            self.resource_col,
            "EventOrigin",
            "case:ApplicationType",
            "case:LoanGoal",
        }
        known_numeric_features = {
            "event_index",
            "weekday",
            "hour",
            "month",
            "elapsed_time_seconds",
            "case:RequestedAmount",
            "CreditScore",
        }

        numeric_features = []

        for feature_name in X.columns:
            if feature_name in known_categorical_features:
                continue

            if feature_name in known_numeric_features:
                numeric_features.append(feature_name)
                continue

            if is_numeric_dtype(X[feature_name]):
                numeric_features.append(feature_name)
                continue

            non_missing = X[feature_name].dropna()
            if not non_missing.empty:
                converted = pd.to_numeric(non_missing, errors="coerce")
                if converted.notna().all():
                    numeric_features.append(feature_name)

        return numeric_features

    def _ensure_feature_schema(self, X: pd.DataFrame) -> None:
        """
        Ensures schema fields exist, including for older pickled engines.
        """

        if not hasattr(self, "feature_names") or not self.feature_names:
            self.feature_names = list(X.columns)

        has_categorical = (
            hasattr(self, "categorical_feature_names")
            and self.categorical_feature_names
        )
        has_numeric = (
            hasattr(self, "numeric_feature_names")
            and self.numeric_feature_names
        )

        if has_categorical or has_numeric:
            return

        recovered = self._recover_feature_schema_from_model()

        if recovered is not None:
            categorical_features, numeric_features = recovered
            self.categorical_feature_names = categorical_features
            self.numeric_feature_names = numeric_features
            return

        self.numeric_feature_names = self._infer_numeric_features(X)
        self.categorical_feature_names = [
            col for col in self.feature_names
            if col not in self.numeric_feature_names
        ]

    def _recover_feature_schema_from_model(self) -> tuple[list[str], list[str]] | None:
        """
        Recovers ColumnTransformer feature lists from pre-existing pickles.
        """

        if self.model is None or "preprocessor" not in self.model.named_steps:
            return None

        preprocessor = self.model.named_steps["preprocessor"]
        transformers = getattr(
            preprocessor,
            "transformers_",
            getattr(preprocessor, "transformers", []),
        )

        categorical_features: list[str] = []
        numeric_features: list[str] = []

        for name, _, columns in transformers:
            if columns is None or (
                isinstance(columns, str) and columns == "drop"
            ):
                continue

            if isinstance(columns, str):
                feature_list = [columns]
            else:
                feature_list = list(columns)

            if name == "categorical":
                categorical_features.extend(feature_list)
            elif name == "numeric":
                numeric_features.extend(feature_list)

        if not categorical_features and not numeric_features:
            return None

        return categorical_features, numeric_features

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

        if len(result) == 0:
            return False

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
