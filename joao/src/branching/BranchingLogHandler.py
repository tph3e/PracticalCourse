from __future__ import annotations

from collections import Counter, defaultdict
from typing import DefaultDict

import pandas as pd


class BranchingLogHandler:
    """
    Handles event-log preprocessing and transition extraction for branching decisions

    This class is shared by several Task 1.5 components
    For Basic 1, it is mainly used to:
    - prepare the event log
    - extract transitions
    - identify decision points
    - compute branch probabilities
    """

    def __init__(
            self,
            case_col: str = "case:concept:name",
            activity_col: str = "concept:name",
            timestamp_col: str = "time:timestamp",
    ):
        self.case_col = case_col
        self.activity_col = activity_col
        self.timestamp_col = timestamp_col

    
    def prepare_log(self, log: pd.DataFrame) -> pd.DataFrame:
        """
        Validates, cleans, converts timestamps, and sorts the event log
        """

        required_columns = [
            self.case_col,
            self.activity_col,
            self.timestamp_col,
        ]

        missing_columns = [
            column for column in required_columns if column not in log.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required event-log columns: {missing_columns}"
            )
        
        prepared_log = log.copy()

        prepared_log[self.timestamp_col] = pd.to_datetime(
            prepared_log[self.timestamp_col],
            errors="coerce",
        )

        prepared_log = prepared_log.dropna(
            subset=[
                self.case_col,
                self.activity_col,
                self.timestamp_col,
            ]
        )

        prepared_log = prepared_log.sort_values(
            by=[self.case_col, self.timestamp_col]
        ).reset_index(drop=True)

        return prepared_log
    

    def extract_traces(self, log: pd.DataFrame) -> dict[str, list[str]]:
        """
        Extracts activity sequences per case
        """

        prepared_log = self.prepare_log(log)

        traces = (
            prepared_log
            .groupby(self.case_col)[self.activity_col]
            .apply(list)
            .to_dict()
        )

        return traces 
    

    def extract_transition_counts(
            self,
            log: pd.DataFrame,
    ) -> dict[str, Counter[str]]:
        """
        Extracts direct transition counts from the event log

        Returns:
            current_activit -> Counter(next_activity -> count)
        """

        traces = self.extract_traces(log)

        transition_counts: DefaultDict[str, Counter[str]] = defaultdict(Counter)

        for activities in traces.values():
            for current_activity, next_activity in zip(activities[:-1], activities[1:]):
                transition_counts[current_activity][next_activity] += 1

        return dict(transition_counts)
    

    def discover_decision_points(
            self,
            transition_counts: dict[str, Counter[str]],
    ) -> set[str]:
        """
        Identifies activities with more than one observed successor
        """

        return {
            activity
            for activity, successors in transition_counts.items()
            if len(successors) > 1
        }
    

    def learn_branch_probabilities(
            self,
            transition_counts: dict[str, Counter[str]],
    ) -> dict[str, dict[str, float]]:
        """
        Converts transition counts into empirical branch probabilities
        """

        probabilities: dict[str, dict[str, float]] = {}

        for current_activity, successor_counts in transition_counts.items():
            total = sum(successor_counts.values())

            if total <= 0:
                continue

            probabilities[current_activity] = {
                next_activity: count / total
                for next_activity, count in successor_counts.items()
            }

        return probabilities
        
    
    def export_transition_table(
            self,
            transition_counts: dict[str, Counter[str]],
            probabilities: dict[str, dict[str, float]],
            decision_points: set[str] | None = None,
    ) -> pd.DataFrame:
        """
        Exports transition counts and probabilities as a DataFrame
        """

        decision_points = decision_points or set()
        rows = []

        for current_activity, successors in transition_counts.items():
            for next_activity, count in successors.items():
                probability = probabilities.get(current_activity, {}).get(
                    next_activity, 
                    0.0,
                )

                rows.append(
                    {
                        "current_activity" : current_activity,
                        "next_activity" : next_activity,
                        "count" : count,
                        "probability" : probability,
                        "is_decision_point" : current_activity in decision_points,
                    }
                )

        return pd.DataFrame(rows)
    

    def export_decision_points_table(
            self,
            transition_counts: dict[str, Counter[str]],
            decision_points: set[str],
    ) -> pd.DataFrame:
        """
        Exports discovered decision points as a DataFrame
        """

        rows = []

        for decision_point in sorted(decision_points):
            successors = list(transition_counts[decision_point].keys())

            rows.append(
                {
                    "decision_point" : decision_point,
                    "number_of_successors" : len(successors),
                    "successor_activities" : ", ".join(successors),
                }
            )
        
        return pd.DataFrame(rows)
        
    

