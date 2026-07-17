from __future__ import annotations

from datetime import datetime
from typing import Any
import random

from .BranchPredictionContext import BranchPrediction
from .TransitionAwareBranching import TransitionDisambiguationModel


class CompositeBranchingAdapter:
    """
    Records structured branch-prediction context around CompositeBranchingEngine.
    """

    def __init__(
        self,
        branching_engine,
        seed: int = 1,
        transition_model: TransitionDisambiguationModel | None = None,
    ):
        self.branching_engine = branching_engine
        self.transition_model = transition_model or TransitionDisambiguationModel()
        self.predictions: list[BranchPrediction] = []
        self._prediction_counter = 0
        self.random = random.Random(seed)
        self.transition_ambiguities = 0
        self.invalid_predictions = 0
        self.transition_fallbacks = 0
        self.unique_label_matches = 0
        self.duplicate_label_resolved = 0
        self.transition_candidates_seen = 0
        self.marking_signature_available = 0

    def predict(
        self,
        event: Any,
        possible_activities: list[str],
        prediction_time: datetime,
        expected_delay: float = 0.0,
        target_task_id: str | None = None,
    ) -> BranchPrediction:
        before = self._statistics_snapshot()
        result = self.branching_engine.getNextActivities(event, possible_activities)
        after = self._statistics_snapshot()

        selected_activity = self._selected_activity(result)
        source = self._infer_source(before, after, possible_activities)
        probabilities = self._probabilities_for(event, possible_activities)
        if selected_activity is not None and not probabilities:
            probabilities = {selected_activity: 1.0}

        prediction = BranchPrediction(
            prediction_id=f"BP{self._prediction_counter}",
            case_id=self._case_id(event),
            current_activity=getattr(event, "activity", ""),
            decision_point=getattr(event, "activity", ""),
            candidate_activities=tuple(possible_activities),
            selected_activity=selected_activity,
            probabilities=probabilities,
            prediction_source=source,
            prediction_time=prediction_time,
            target_task_id=target_task_id,
            expected_delay=expected_delay,
        )
        self._prediction_counter += 1
        self.predictions.append(prediction)
        return prediction

    def predict_transition(
        self,
        event: Any,
        transition_candidates: list[Any],
        prediction_time: datetime,
        expected_delay: float = 0.0,
        target_task_id: str | None = None,
    ) -> BranchPrediction:
        possible_activities = [
            str(candidate.activity_label) for candidate in transition_candidates
        ]
        self.transition_candidates_seen += len(transition_candidates)
        unique_activities = sorted(set(possible_activities))
        before = self._statistics_snapshot()
        result = self.branching_engine.getNextActivities(event, unique_activities)
        after = self._statistics_snapshot()

        selected_activity = self._selected_activity(result)
        source = self._infer_source(before, after, unique_activities)
        probabilities = self._probabilities_for(event, unique_activities)
        if selected_activity is not None and not probabilities:
            probabilities = {selected_activity: 1.0}

        selected_transition = None
        transition_ambiguity = False
        fallback_source = None
        rejected_activity = None

        context = self._transition_context(event, transition_candidates)
        if context.get("marking_signature"):
            self.marking_signature_available += 1

        selected_transition, selection_source = self.transition_model.choose_for_label(
            transition_candidates,
            str(selected_activity),
            context,
        ) if selected_activity is not None else (None, "no_label_prediction")
        matches = [
            candidate
            for candidate in transition_candidates
            if str(candidate.activity_label) == str(selected_activity)
        ] if selected_activity is not None else []

        if selection_source == "unique_label_match":
            self.unique_label_matches += 1
            selected_transition = matches[0]
        elif len(matches) > 1:
            transition_ambiguity = True
            self.transition_ambiguities += 1
            self.duplicate_label_resolved += 1
            fallback_source = selection_source
        elif selected_activity is not None and selection_source == "no_label_match":
            self.invalid_predictions += 1
            rejected_activity = selected_activity
            selected_transition, fallback_source = self.transition_model.choose(
                transition_candidates,
                context,
            )
            self.transition_fallbacks += 1
        elif transition_candidates:
            selected_transition, fallback_source = self.transition_model.choose(
                transition_candidates,
                context,
            )
            self.transition_fallbacks += 1

        if selected_transition is not None:
            selected_activity = str(selected_transition.activity_label)
            probabilities.setdefault(selected_activity, 1.0)

        prediction = BranchPrediction(
            prediction_id=f"BP{self._prediction_counter}",
            case_id=self._case_id(event),
            current_activity=getattr(event, "activity", ""),
            decision_point=getattr(event, "activity", ""),
            candidate_activities=tuple(unique_activities),
            selected_activity=selected_activity,
            probabilities=probabilities,
            prediction_source=source,
            prediction_time=prediction_time,
            target_task_id=target_task_id,
            expected_delay=expected_delay,
            candidate_transition_ids=tuple(
                str(candidate.transition_id) for candidate in transition_candidates
            ),
            selected_transition_id=(
                str(selected_transition.transition_id)
                if selected_transition is not None
                else None
            ),
            marking_signature=(
                str(selected_transition.marking_before)
                if selected_transition is not None
                else None
            ),
            transition_ambiguity=transition_ambiguity,
            fallback_source=fallback_source,
            rejected_activity=rejected_activity,
        )
        self._prediction_counter += 1
        self.predictions.append(prediction)
        return prediction

    def diagnostics(self) -> dict[str, int]:
        source_counts: dict[str, int] = {}
        for prediction in self.predictions:
            source_counts[prediction.prediction_source] = (
                source_counts.get(prediction.prediction_source, 0) + 1
            )
        return {
            "branch_predictions": len(self.predictions),
            "transition_candidates_seen": self.transition_candidates_seen,
            "unique_label_matches": self.unique_label_matches,
            "duplicate_label_ambiguities": self.transition_ambiguities,
            "duplicate_label_resolved": self.duplicate_label_resolved,
            "invalid_label_predictions": self.invalid_predictions,
            "transition_ambiguities": self.transition_ambiguities,
            "invalid_branch_predictions": self.invalid_predictions,
            "transition_fallbacks": self.transition_fallbacks,
            "marking_signature_available": self.marking_signature_available,
            **{
                f"branch_prediction_source_{source}": count
                for source, count in source_counts.items()
            },
        }

    def _statistics_snapshot(self) -> dict[str, Any]:
        if not hasattr(self.branching_engine, "get_statistics"):
            return {}
        return self.branching_engine.get_statistics()

    def _infer_source(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        possible_activities: list[str],
    ) -> str:
        if len(possible_activities) <= 1:
            return "BPMNDeterministic"

        before_success = before.get("engine_success_counts", {})
        after_success = after.get("engine_success_counts", {})
        for source, count in after_success.items():
            if count > before_success.get(source, 0):
                return source

        if after.get("random_fallback_count", 0) > before.get("random_fallback_count", 0):
            return "CompositeRandomFallback"

        return "CompositeBranchingEngine"

    def _probabilities_for(
        self,
        event: Any,
        possible_activities: list[str],
    ) -> dict[str, float]:
        current_activity = getattr(event, "activity", None)
        if current_activity is None:
            return {}

        for engine in getattr(self.branching_engine, "engines", []):
            probabilities = getattr(engine, "branch_probabilities", None)
            if not probabilities:
                continue
            current_probabilities = probabilities.get(current_activity)
            if not current_probabilities:
                continue
            filtered = {
                activity: probability
                for activity, probability in current_probabilities.items()
                if activity in possible_activities
            }
            total = sum(filtered.values())
            if total > 0:
                return {
                    activity: probability / total
                    for activity, probability in filtered.items()
                }
        return {}

    def _selected_activity(self, result) -> str | None:
        if isinstance(result, str):
            return result
        if isinstance(result, list) and result:
            return result[0]
        return None

    def _fallback_transition(self, event: Any, candidates: list[Any]) -> Any | None:
        if not candidates:
            return None

        probabilities = self._probabilities_for(
            event,
            sorted({str(candidate.activity_label) for candidate in candidates}),
        )
        weighted_candidates = [
            (candidate, probabilities.get(str(candidate.activity_label), 0.0))
            for candidate in candidates
        ]
        total = sum(weight for _, weight in weighted_candidates)
        if total > 0:
            draw = self.random.random() * total
            cumulative = 0.0
            for candidate, weight in sorted(
                weighted_candidates,
                key=lambda item: str(item[0].transition_id),
            ):
                cumulative += weight
                if draw <= cumulative:
                    return candidate

        return sorted(candidates, key=lambda candidate: str(candidate.transition_id))[0]

    def _transition_context(
        self,
        event: Any,
        transition_candidates: list[Any],
    ) -> dict[str, Any]:
        history = self._activity_history(event)
        current_activity = str(getattr(event, "activity", "") or "")
        previous_activity = history[-2] if len(history) >= 2 else "START"
        prior = history[:-1] if history and history[-1] == current_activity else history
        visit_count = sum(1 for activity in prior if activity == current_activity)
        consecutive = 0
        for activity in reversed(prior):
            if activity != current_activity:
                break
            consecutive += 1
        marking_signature = ""
        if transition_candidates:
            marking_signature = str(
                getattr(transition_candidates[0], "source_marking", "")
                or getattr(transition_candidates[0], "marking_before", "")
                or ""
            )
        return {
            "marking_signature": marking_signature,
            "current_activity": current_activity,
            "previous_activity": previous_activity,
            "visit_count_bucket": self._bucket_count(visit_count),
            "repetition_bucket": self._bucket_count(consecutive),
            "trace_prefix": "|".join(history[-4:]) or "START",
            "event_index": len(history),
        }

    def _activity_history(self, event: Any) -> list[str]:
        event_case = getattr(event, "eventCase", None)
        activities = getattr(event_case, "activities", None)
        if isinstance(activities, list):
            return [str(activity) for activity in activities if activity]
        if hasattr(event, "getAttribOfLastEvents"):
            try:
                records = event.getAttribOfLastEvents(-1)
            except TypeError:
                records = []
            if isinstance(records, list):
                return [
                    str(record.get("concept:name"))
                    for record in records
                    if isinstance(record, dict) and record.get("concept:name")
                ]
        current_activity = getattr(event, "activity", None)
        return [str(current_activity)] if current_activity else []

    def _bucket_count(self, value: int) -> int:
        if value <= 0:
            return 0
        if value == 1:
            return 1
        if value <= 3:
            return 3
        if value <= 10:
            return 10
        return 99

    def _case_id(self, event: Any) -> str:
        event_case = getattr(event, "eventCase", None)
        case_id = getattr(event_case, "caseId", None)
        return str(case_id if case_id is not None else "UNKNOWN_CASE")
