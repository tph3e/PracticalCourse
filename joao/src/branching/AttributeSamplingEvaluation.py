from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import random

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class EmpiricalAttributeSampler:
    attributes: list[str]
    seed: int = 1

    def __post_init__(self) -> None:
        self.random = random.Random(self.seed)
        self.marginal: dict[str, list[Any]] = {}
        self.by_activity: dict[tuple[str, str], list[Any]] = {}
        self.modes: dict[str, Any] = {}

    def fit(self, rows: pd.DataFrame) -> "EmpiricalAttributeSampler":
        for attribute in self.attributes:
            if attribute not in rows.columns:
                continue
            values = [value for value in rows[attribute].tolist() if not pd.isna(value)]
            self.marginal[attribute] = values
            self.modes[attribute] = pd.Series(values).mode().iloc[0] if values else None
            for activity, group in rows.groupby("current_activity", dropna=False):
                activity_values = [value for value in group[attribute].tolist() if not pd.isna(value)]
                if activity_values:
                    self.by_activity[(str(activity), attribute)] = activity_values
        return self

    def sample_value(self, attribute: str, current_activity: str | None = None) -> Any:
        conditional = self.by_activity.get((str(current_activity), attribute), [])
        population = conditional or self.marginal.get(attribute, [])
        if not population:
            return None
        return self.random.choice(population)

    def fill_missing(self, rows: pd.DataFrame, *, mode: str = "sample") -> tuple[pd.DataFrame, dict[str, int]]:
        filled = rows.copy()
        counts = {"sampled_values": 0, "mode_values": 0}
        for index, row in filled.iterrows():
            current_activity = row.get("current_activity")
            for attribute in self.attributes:
                if attribute not in filled.columns:
                    continue
                if not pd.isna(row.get(attribute)):
                    continue
                if mode == "mode":
                    value = self.modes.get(attribute)
                    counts["mode_values"] += int(value is not None)
                else:
                    value = self.sample_value(attribute, current_activity=current_activity)
                    counts["sampled_values"] += int(value is not None)
                if value is not None:
                    filled.at[index, attribute] = value
        return filled, counts


def inject_missingness(
    rows: pd.DataFrame,
    *,
    attributes: list[str],
    rate: float,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    masked = rows.copy()
    for attribute in attributes:
        if attribute not in masked.columns:
            continue
        for index in masked.index:
            if rng.random() < rate:
                masked.at[index, attribute] = pd.NA
    return masked


def evaluate_missingness_sampling(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    *,
    attributes: list[str],
    predict_fn,
    rates: list[float] | None = None,
    seed: int = 1,
) -> pd.DataFrame:
    rates = rates or [0.1, 0.3, 0.5]
    sampler = EmpiricalAttributeSampler(attributes=attributes, seed=seed).fit(train_rows)
    y_true = test_rows["true_next_activity"].astype(str).tolist()
    records: list[dict[str, Any]] = []
    for rate in rates:
        masked = inject_missingness(test_rows, attributes=attributes, rate=rate, seed=seed)
        no_sampling_pred = predict_fn(masked)
        sampled_rows, sampled_counts = sampler.fill_missing(masked, mode="sample")
        sampled_pred = predict_fn(sampled_rows)
        mode_rows, mode_counts = sampler.fill_missing(masked, mode="mode")
        mode_pred = predict_fn(mode_rows)
        records.append(
            {
                "missing_rate": rate,
                "n_samples": len(test_rows),
                "sampling_rate": sampled_counts["sampled_values"] / max(1, len(test_rows) * len(attributes)),
                "mode_fill_rate": mode_counts["mode_values"] / max(1, len(test_rows) * len(attributes)),
                "no_sampling_accuracy": accuracy_score(y_true, no_sampling_pred),
                "sampled_accuracy": accuracy_score(y_true, sampled_pred),
                "mode_accuracy": accuracy_score(y_true, mode_pred),
                "delta_accuracy_sampled_vs_none": accuracy_score(y_true, sampled_pred) - accuracy_score(y_true, no_sampling_pred),
                "delta_accuracy_sampled_vs_mode": accuracy_score(y_true, sampled_pred) - accuracy_score(y_true, mode_pred),
                "no_sampling_macro_f1": f1_score(y_true, no_sampling_pred, average="macro", zero_division=0),
                "sampled_macro_f1": f1_score(y_true, sampled_pred, average="macro", zero_division=0),
                "mode_macro_f1": f1_score(y_true, mode_pred, average="macro", zero_division=0),
                "delta_macro_f1_sampled_vs_none": f1_score(y_true, sampled_pred, average="macro", zero_division=0)
                - f1_score(y_true, no_sampling_pred, average="macro", zero_division=0),
            }
        )
    return pd.DataFrame(records)
