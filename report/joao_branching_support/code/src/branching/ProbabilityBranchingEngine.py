from __future__ import annotations

from collections import Counter
from typing import Any
import random

import pandas as pd

from .BranchingLogHandler import BranchingLogHandler
from .BranchDecision import BranchDecision
from .BranchingUtils import (
    choose_random_valid_activity,
    extract_current_activity,
    filter_probabilities_by_possible_activities,
    normalize_probabilities,
)


class ProbabilityBranchingEngine:
    """
    Static probabilistic branching engine

    This engine learns empirical transition probabilities from the event log
    
    During simulation, it samples the next activity among the BPMN-allowed possible activities

    Main rule:
    The engine must never return an activity outside possibleActivities
    """

    def __init__(
            self,
            log: pd.DataFrame | None = None,
            log_handler: BranchingLogHandler | None = None,
            seed: int = 1,
            case_col: str = "case:concept:name",
            activity_col: str = "concept:name",
            timestamp_col: str = "time:timestamp",
            alpha: float = 1.0,
            min_state_support: int = 5,
    ):
        self.log = log
        self.seed = seed
        self.random = random.Random(seed)

        self.case_col = case_col
        self.activity_col = activity_col
        self.timestamp_col = timestamp_col
        self.alpha = alpha
        self.min_state_support = min_state_support

        self.log_handler = log_handler or BranchingLogHandler(
            case_col=case_col,
            activity_col=activity_col,
            timestamp_col=timestamp_col,
        )

        self.transition_counts: dict[str, Counter[str]] = {}
        self.branch_probabilities: dict[str, dict[str, float]] = {}
        self.state_transition_counts: dict[tuple[str, str, int, int], Counter[str]] = {}
        self.state_branch_probabilities: dict[tuple[str, str, int, int], dict[str, float]] = {}
        self.decision_points: set[str] = set()
        self.diagnostics: Counter[str] = Counter()
        self.support_values: list[int] = []

        self.is_trained = False

    
    def train(self, log:pd.DataFrame | None = None) -> None:
        """
        Learns transition counts, decision points, and branch probabilities from the event log
        """

        if log is not None:
            self.log = log

        if self.log is None:
            raise ValueError("No event log provided for training")
        
        prepared_log = self.log_handler.prepare_log(self.log)

        self.transition_counts = self.log_handler.extract_transition_counts(prepared_log)
        self.decision_points = self.log_handler.discover_decision_points(
            self.transition_counts
        )
        self.branch_probabilities = self.log_handler.learn_branch_probabilities(
            self.transition_counts
        )
        self.state_transition_counts = self._extract_state_transition_counts(
            prepared_log,
        )
        self.state_branch_probabilities = {
            state: {
                activity: count / sum(counter.values())
                for activity, count in counter.items()
            }
            for state, counter in self.state_transition_counts.items()
            if sum(counter.values()) > 0
        }

        self.is_trained = True

    
    def getNextActivities(
            self,
            event: Any,
            possibleActivities: list[str],
    ) -> list[str]:
        """
        Selects the next activity during simulation

        Parameters
        -----------
        event:
            Current event or process state

        possibleActivities:
            List of BPMN-allowed next activities
        
        Returns
        ----------
        list[str]
            List containing the selected next activity
        """

        decision = self.decide(event, possibleActivities)
        if decision is not None:
            return decision.activities
        return choose_random_valid_activity(possibleActivities, self.random)

    def decide(
        self,
        event: Any,
        possibleActivities: list[str],
        context: dict[str, Any] | None = None,
    ) -> BranchDecision | None:
        if not possibleActivities:
            return BranchDecision(
                activities=[],
                decision_source="probability_empty_candidates",
                candidate_activities=[],
                used_fallback=True,
            )

        if len(possibleActivities) == 1:
            return BranchDecision(
                activities=possibleActivities,
                decision_source="single_bpmn_candidate",
                probability_source=None,
                probabilities={possibleActivities[0]: 1.0},
                confidence=1.0,
                support=None,
                used_fallback=False,
                candidate_activities=list(possibleActivities),
            )

        current_activity = extract_current_activity(
            event,
            activity_col=self.activity_col,
        )

        if current_activity is None:
            self.diagnostics["unseen_activity_count"] += 1
            selected = choose_random_valid_activity(possibleActivities, self.random)
            return BranchDecision(
                activities=selected,
                decision_source="probability_random_fallback",
                probability_source="random_fallback",
                probabilities=self._uniform_probabilities(possibleActivities),
                confidence=1 / len(possibleActivities),
                support=0,
                used_fallback=True,
                candidate_activities=list(possibleActivities),
                metadata={"reason": "missing_current_activity"},
            )

        state_key = self._state_key_from_event(event, current_activity)
        state_counter = self.state_transition_counts.get(state_key, Counter())
        state_support = sum(state_counter.values())
        state_nonzero_candidates = [
            activity for activity in possibleActivities
            if state_counter.get(activity, 0) > 0
        ]
        if state_support >= self.min_state_support or len(state_nonzero_candidates) == 1:
            probabilities = self._smoothed_probabilities(state_counter, possibleActivities)
            if probabilities:
                selected_activity = self._sample_from_probabilities(probabilities)
                self.diagnostics["state_model_used"] += 1
                self.support_values.append(state_support)
                return BranchDecision(
                    activities=[selected_activity],
                    decision_source="probability_branching",
                    probability_source="state_conditioned_probability",
                    probabilities=probabilities,
                    confidence=probabilities[selected_activity],
                    support=state_support,
                    used_fallback=False,
                    candidate_activities=list(possibleActivities),
                    metadata={"state_key": state_key, "state_support": state_support},
                )

        if state_counter:
            self.diagnostics["unseen_state_count"] += int(state_support < self.min_state_support)

        activity_counter = self.transition_counts.get(current_activity, Counter())
        activity_support = sum(activity_counter.values())
        probabilities = self._activity_probabilities(
            current_activity=current_activity,
            counter=activity_counter,
            possible_activities=possibleActivities,
        )
        if probabilities:
            selected_activity = self._sample_from_probabilities(probabilities)
            self.diagnostics["activity_model_used"] += 1
            self.support_values.append(activity_support)
            return BranchDecision(
                activities=[selected_activity],
                decision_source="probability_branching",
                probability_source="activity_level_probability",
                probabilities=probabilities,
                confidence=probabilities[selected_activity],
                support=activity_support,
                used_fallback=state_support < self.min_state_support,
                candidate_activities=list(possibleActivities),
                metadata={"activity_support": activity_support, "state_support": state_support},
            )

        self.diagnostics["random_fallback_used"] += 1
        selected = choose_random_valid_activity(possibleActivities, self.random)
        return BranchDecision(
            activities=selected,
            decision_source="probability_random_fallback",
            probability_source="candidate_uniform_fallback",
            probabilities=self._uniform_probabilities(possibleActivities),
            confidence=1 / len(possibleActivities),
            support=0,
            used_fallback=True,
            candidate_activities=list(possibleActivities),
            metadata={"activity_support": activity_support, "state_support": state_support},
        )

    def _extract_state_transition_counts(
        self,
        prepared_log: pd.DataFrame,
    ) -> dict[tuple[str, str, int, int], Counter[str]]:
        counts: dict[tuple[str, str, int, int], Counter[str]] = {}
        for _, case_events in prepared_log.groupby(self.case_col):
            case_events = case_events.sort_values(self.timestamp_col).reset_index(drop=True)
            activities = case_events[self.activity_col].astype(str).tolist()
            prior_counts: Counter[str] = Counter()
            previous_activity = "START"
            for index in range(len(activities) - 1):
                current_activity = activities[index]
                next_activity = activities[index + 1]
                consecutive = 0
                cursor = index - 1
                while cursor >= 0:
                    if activities[cursor] != current_activity:
                        break
                    consecutive += 1
                    cursor -= 1
                state = (
                    current_activity,
                    previous_activity,
                    self._bucket_count(prior_counts[current_activity]),
                    self._bucket_count(consecutive),
                )
                counts.setdefault(state, Counter())[next_activity] += 1
                prior_counts[current_activity] += 1
                previous_activity = current_activity
        return counts

    def _state_key_from_event(
        self,
        event: Any,
        current_activity: str,
    ) -> tuple[str, str, int, int]:
        history = self._extract_activity_history(event)
        previous_activity = history[-2] if len(history) >= 2 else "START"
        prior = history[:-1] if history and history[-1] == current_activity else history
        visit_count = sum(1 for activity in prior if activity == current_activity)
        consecutive = 0
        for activity in reversed(prior):
            if activity != current_activity:
                break
            consecutive += 1
        return (
            current_activity,
            previous_activity,
            self._bucket_count(visit_count),
            self._bucket_count(consecutive),
        )

    def _extract_activity_history(self, event: Any) -> list[str]:
        event_case = getattr(event, "eventCase", None)
        activities = getattr(event_case, "activities", None)
        if isinstance(activities, list):
            return [str(activity) for activity in activities if activity]
        if isinstance(event, dict):
            activities = event.get("activity_history")
            if isinstance(activities, list):
                return [str(activity) for activity in activities if activity]
            current_activity = event.get(self.activity_col) or event.get("concept:name") or event.get("activity")
            return [str(current_activity)] if current_activity else []
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
    

    def _sample_from_probabilities(
            self,
            probabilities: dict[str, float],
    ) -> str:
        """
        Samples one activity according to a probability distribution
        """

        activities = list(probabilities.keys())
        weights = list(probabilities.values())

        return self.random.choices(
            population=activities,
            weights=weights,
            k=1
        )[0]

    def _smoothed_probabilities(
        self,
        counter: Counter[str],
        possible_activities: list[str],
    ) -> dict[str, float]:
        if not possible_activities:
            self.diagnostics["candidate_filter_empty"] += 1
            return {}
        values = {
            activity: float(counter.get(activity, 0)) + self.alpha
            for activity in possible_activities
        }
        return normalize_probabilities(values)

    def _activity_probabilities(
        self,
        current_activity: str,
        counter: Counter[str],
        possible_activities: list[str],
    ) -> dict[str, float]:
        if sum(counter.values()) > 0:
            return self._smoothed_probabilities(counter, possible_activities)

        legacy_probabilities = self.branch_probabilities.get(current_activity, {})
        filtered = {
            activity: probability
            for activity, probability in legacy_probabilities.items()
            if activity in possible_activities
        }
        if filtered:
            return normalize_probabilities(filtered)

        return {}

    def _uniform_probabilities(self, possible_activities: list[str]) -> dict[str, float]:
        if not possible_activities:
            return {}
        return {activity: 1 / len(possible_activities) for activity in possible_activities}

    def get_diagnostics(self) -> dict[str, Any]:
        mean_support = (
            sum(self.support_values) / len(self.support_values)
            if self.support_values
            else 0.0
        )
        return {
            **dict(self.diagnostics),
            "min_state_support": self.min_state_support,
            "alpha": self.alpha,
            "mean_support": mean_support,
        }
    

    def get_transition_probabilities(self) -> dict[str, dict[str, float]]:
        """
        Returns learned branch probabilities
        """

        return self.branch_probabilities
    

    def get_transition_counts(self) -> dict[str, Counter[str]]:
        """
        Returns learned transition counts
        """

        return self.transition_counts
    

    def get_decision_points(self) -> set[str]:
        """
        Returns discovered decision points
        """

        return self.decision_points
    

    def export_transition_table(self) -> pd.DataFrame:
        """
        Exports transition counts and probabilities as a DataFrame.
        """

        return self.log_handler.export_transition_table(
            transition_counts=self.transition_counts,
            probabilities=self.branch_probabilities,
            decision_points=self.decision_points,
        )


    def export_decision_points_table(self) -> pd.DataFrame:
        """
        Exports decision points as a DataFrame.
        """

        return self.log_handler.export_decision_points_table(
            transition_counts=self.transition_counts,
            decision_points=self.decision_points,
        )
        
