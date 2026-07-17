from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import hashlib
import random

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    outer_train_cases: list[str]
    outer_test_cases: list[str]
    inner_train_cases: list[str]
    inner_validation_cases: list[str]
    case_start_times: dict[str, str]
    case_hashes: dict[str, str]
    time_ranges: dict[str, tuple[str, str]]


def hash_case_ids(case_ids: list[str] | set[str]) -> str:
    payload = "\n".join(sorted(str(case_id) for case_id in case_ids)).encode()
    return hashlib.sha256(payload).hexdigest()


def temporal_train_validation_test_split_by_case(
    log: pd.DataFrame,
    case_col: str = "case:concept:name",
    timestamp_col: str = "time:timestamp",
    outer_train_ratio: float = 0.7,
    inner_train_ratio: float = 0.85,
) -> TemporalSplit:
    if case_col not in log.columns:
        raise ValueError(f"Missing case column: {case_col}")
    if timestamp_col not in log.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")
    if not 0 < outer_train_ratio < 1:
        raise ValueError("outer_train_ratio must be between 0 and 1.")
    if not 0 < inner_train_ratio < 1:
        raise ValueError("inner_train_ratio must be between 0 and 1.")

    prepared = log[[case_col, timestamp_col]].copy()
    prepared[timestamp_col] = pd.to_datetime(prepared[timestamp_col], utc=True, errors="coerce")
    prepared = prepared.dropna(subset=[case_col, timestamp_col])
    starts = (
        prepared.groupby(case_col)[timestamp_col]
        .min()
        .reset_index()
        .assign(_case_id=lambda frame: frame[case_col].astype(str))
        .sort_values([timestamp_col, "_case_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    case_ids = starts["_case_id"].tolist()
    if len(case_ids) < 3:
        raise ValueError("At least three cases are required for train/validation/test split.")

    outer_cut = int(len(case_ids) * outer_train_ratio)
    if outer_cut <= 0 or outer_cut >= len(case_ids):
        raise ValueError("Outer split would be empty.")

    outer_train = case_ids[:outer_cut]
    outer_test = case_ids[outer_cut:]
    inner_cut = int(len(outer_train) * inner_train_ratio)
    if inner_cut <= 0 or inner_cut >= len(outer_train):
        raise ValueError("Inner split would be empty.")

    inner_train = outer_train[:inner_cut]
    inner_validation = outer_train[inner_cut:]

    groups = {
        "outer_train": outer_train,
        "outer_test": outer_test,
        "inner_train": inner_train,
        "inner_validation": inner_validation,
    }
    if set(outer_train) & set(outer_test):
        raise AssertionError("Outer train/test split has overlapping cases.")
    if set(inner_train) & set(inner_validation):
        raise AssertionError("Inner train/validation split has overlapping cases.")
    if not set(inner_train).issubset(set(outer_train)):
        raise AssertionError("Inner train cases must be inside outer train.")
    if not set(inner_validation).issubset(set(outer_train)):
        raise AssertionError("Inner validation cases must be inside outer train.")

    start_map = dict(zip(starts["_case_id"], starts[timestamp_col].astype(str)))
    ranges: dict[str, tuple[str, str]] = {}
    for name, ids in groups.items():
        values = [pd.Timestamp(start_map[case_id]) for case_id in ids]
        ranges[name] = (str(min(values)), str(max(values)))

    return TemporalSplit(
        outer_train_cases=outer_train,
        outer_test_cases=outer_test,
        inner_train_cases=inner_train,
        inner_validation_cases=inner_validation,
        case_start_times=start_map,
        case_hashes={name: hash_case_ids(ids) for name, ids in groups.items()},
        time_ranges=ranges,
    )


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

    if len(result) == 0:
        return False

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
