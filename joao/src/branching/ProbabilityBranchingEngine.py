from __future__ import annotations

from collections import Counter
from typing import Any
import random

import pandas as pd

from .BranchingLogHandler import BranchingLogHandler
from .BranchingUtils import (
    choose_random_valid_activity,
    extract_current_activity,
    filter_probabilities_by_possible_activities,
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
    ):
        self.log = log
        self.seed = seed
        self.random = random.Random(seed)

        self.case_col = case_col
        self.activity_col = activity_col
        self.timestamp_col = timestamp_col

        self.log_handler = log_handler or BranchingLogHandler(
            case_col=case_col,
            activity_col=activity_col,
            timestamp_col=timestamp_col,
        )

        self.transition_counts: dict[str, Counter[str]] = {}
        self.branch_probabilities: dict[str, dict[str, float]] = {}
        self.decision_points: set[str] = set()

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

        if not possibleActivities:
            return []
        
        if len(possibleActivities) == 1:
            return possibleActivities
        
        current_activity = extract_current_activity(
            event,
            activity_col=self.activity_col,
        )

        if current_activity is None:
            return choose_random_valid_activity(possibleActivities, self.random)
        
        probabilities = self.branch_probabilities.get(current_activity)

        if not probabilities:
            return choose_random_valid_activity(possibleActivities, self.random)
        
        valid_probabilities = filter_probabilities_by_possible_activities(
            probabilities=probabilities,
            possible_activities=possibleActivities,
        )

        if not valid_probabilities:
            return choose_random_valid_activity(possibleActivities, self.random)
        
        selected_activity = self._sample_from_probabilities(valid_probabilities)

        return [selected_activity]
    

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
        
