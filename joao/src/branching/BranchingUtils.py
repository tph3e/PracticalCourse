from __future__ import annotations

from typing import Any
import random

import pandas as pd


def extract_current_activity(event: Any, activity_col: str = "concept:name") -> str | None:
    """
    Extracts the current activity from different possible event structures.

    Supported structures:
    - event.activity
    - event.getAttribs()
    - dictionary-like event
    """

    if event is None:
        return None

    if hasattr(event, "activity"):
        return event.activity

    if hasattr(event, "getAttribs"):
        attributes = event.getAttribs()
        if isinstance(attributes, dict):
            return (
                attributes.get(activity_col)
                or attributes.get("concept:name")
                or attributes.get("activity")
            )

    if isinstance(event, dict):
        return event.get(activity_col) or event.get("concept:name") or event.get("activity")

    return None


def normalize_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    """
    Normalizes a probability dictionary so that values sum to 1.
    """

    total = sum(probabilities.values())

    if total <= 0:
        return {}

    return {
        activity: probability / total
        for activity, probability in probabilities.items()
    }


def filter_probabilities_by_possible_activities(
    probabilities: dict[str, float],
    possible_activities: list[str],
) -> dict[str, float]:
    """
    Filters a probability dictionary by BPMN-allowed possible activities.

    After filtering, probabilities are renormalized.
    """

    filtered = {
        activity: probability
        for activity, probability in probabilities.items()
        if activity in possible_activities
    }

    return normalize_probabilities(filtered)


def choose_random_valid_activity(
    possible_activities: list[str],
    random_generator: random.Random | None = None,
) -> list[str]:
    """
    Chooses one random BPMN-valid activity.

    Returns a list because the simulation interface expects a list.
    """

    if not possible_activities:
        return []

    rng = random_generator or random.Random()

    return [rng.choice(possible_activities)]


def is_bpmn_valid(
    activity: str,
    possible_activities: list[str],
) -> bool:
    """
    Checks whether an activity is allowed by the BPMN engine.
    """

    return activity in possible_activities


def ensure_bpmn_valid_result(
    result: list[str],
    possible_activities: list[str],
) -> bool:
    """
    Checks whether all returned activities are BPMN-valid.
    """

    if not isinstance(result, list):
        return False

    if not possible_activities:
        return result == []

    return all(activity in possible_activities for activity in result)


def temporal_train_test_split_by_case(
    log: pd.DataFrame,
    case_col: str = "case:concept:name",
    timestamp_col: str = "time:timestamp",
    train_ratio: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits an event log into train and test sets by case using case start time.

    This prevents data leakage because all events of a case stay either in
    the train set or in the test set.

    Train set:
        older cases

    Test set:
        newer cases
    """

    if case_col not in log.columns:
        raise ValueError(f"Missing case column: {case_col}")

    if timestamp_col not in log.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    prepared_log = log.copy()

    prepared_log[timestamp_col] = pd.to_datetime(
        prepared_log[timestamp_col],
        errors="coerce",
    )

    prepared_log = prepared_log.dropna(
        subset=[
            case_col,
            timestamp_col,
        ]
    )

    case_start_times = (
        prepared_log
        .groupby(case_col)[timestamp_col]
        .min()
        .sort_values()
    )

    case_ids = case_start_times.index.tolist()

    split_index = int(len(case_ids) * train_ratio)

    if split_index <= 0 or split_index >= len(case_ids):
        raise ValueError(
            "Train/test split would be empty. "
            "Use more cases or adjust train_ratio."
        )

    train_cases = set(case_ids[:split_index])
    test_cases = set(case_ids[split_index:])

    train_log = prepared_log[prepared_log[case_col].isin(train_cases)].copy()
    test_log = prepared_log[prepared_log[case_col].isin(test_cases)].copy()

    train_log = train_log.sort_values(
        by=[case_col, timestamp_col]
    ).reset_index(drop=True)

    test_log = test_log.sort_values(
        by=[case_col, timestamp_col]
    ).reset_index(drop=True)

    return train_log, test_log