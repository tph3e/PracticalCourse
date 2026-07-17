from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class MinedAttributeRule:
    rule_id: str
    decision_point_id: str
    attribute: str
    operator: str
    value: Any
    preferred_activity: str
    support: int
    confidence: float
    lift: float
    value_kind: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_candidate_labels(value: Any) -> list[str]:
    return [item for item in str(value).split("|") if item]


def mine_attribute_rules(
    rows: pd.DataFrame,
    *,
    categorical_attributes: list[str],
    numeric_attributes: list[str],
    min_support: int = 30,
    min_confidence: float = 0.70,
    min_lift: float = 1.05,
) -> list[MinedAttributeRule]:
    rules: list[MinedAttributeRule] = []
    for decision_point_id, group in rows.groupby("decision_point_id", dropna=False):
        if group.empty:
            continue
        outcome_counts = Counter(group["true_next_activity"].astype(str))
        total = sum(outcome_counts.values())
        priors = {activity: count / total for activity, count in outcome_counts.items()}

        for attribute in categorical_attributes:
            if attribute not in group.columns:
                continue
            values = group[attribute].where(~group[attribute].map(pd.isna), "MISSING").astype(str)
            for value, subset_index in values.groupby(values).groups.items():
                subset = group.loc[subset_index]
                rules.extend(
                    _rules_from_subset(
                        subset=subset,
                        decision_point_id=str(decision_point_id),
                        attribute=attribute,
                        operator="==",
                        value=value,
                        value_kind="categorical",
                        priors=priors,
                        min_support=min_support,
                        min_confidence=min_confidence,
                        min_lift=min_lift,
                    )
                )

        for attribute in numeric_attributes:
            if attribute not in group.columns:
                continue
            numeric = pd.to_numeric(group[attribute], errors="coerce")
            clean = group.loc[numeric.notna()].copy()
            if len(clean) < min_support * 2:
                continue
            clean["_numeric_value"] = numeric.loc[clean.index].astype(float)
            quantiles = sorted(set(clean["_numeric_value"].quantile([0.25, 0.5, 0.75]).dropna().tolist()))
            for threshold in quantiles:
                for operator, subset in [
                    ("<=", clean[clean["_numeric_value"] <= threshold]),
                    (">", clean[clean["_numeric_value"] > threshold]),
                ]:
                    rules.extend(
                        _rules_from_subset(
                            subset=subset,
                            decision_point_id=str(decision_point_id),
                            attribute=attribute,
                            operator=operator,
                            value=float(threshold),
                            value_kind="numeric_quantile",
                            priors=priors,
                            min_support=min_support,
                            min_confidence=min_confidence,
                            min_lift=min_lift,
                        )
                    )

    dedup: dict[tuple[str, str, str, str, str], MinedAttributeRule] = {}
    for rule in rules:
        key = (
            rule.decision_point_id,
            rule.attribute,
            rule.operator,
            str(rule.value),
            rule.preferred_activity,
        )
        existing = dedup.get(key)
        if existing is None or (rule.confidence, rule.support) > (existing.confidence, existing.support):
            dedup[key] = rule
    return sorted(dedup.values(), key=lambda item: (-item.lift, -item.confidence, -item.support, item.rule_id))


def select_rules_on_validation(
    candidate_rules: list[MinedAttributeRule],
    validation_rows: pd.DataFrame,
    *,
    max_rules_per_decision_point: int = 3,
    min_validation_support: int = 10,
    min_validation_accuracy: float = 0.60,
) -> list[MinedAttributeRule]:
    accepted: list[MinedAttributeRule] = []
    by_dp: dict[str, list[MinedAttributeRule]] = defaultdict(list)
    for rule in candidate_rules:
        by_dp[rule.decision_point_id].append(rule)

    for decision_point_id, rules in by_dp.items():
        selected_for_dp = 0
        dp_rows = validation_rows[validation_rows["decision_point_id"].astype(str) == decision_point_id]
        if dp_rows.empty:
            continue
        for rule in rules:
            covered = _covered_rows(dp_rows, rule)
            if len(covered) < min_validation_support:
                continue
            valid = covered[
                covered["candidate_activity_labels"].map(parse_candidate_labels).map(
                    lambda candidates: rule.preferred_activity in candidates
                )
            ]
            if len(valid) < min_validation_support:
                continue
            accuracy = float((valid["true_next_activity"].astype(str) == rule.preferred_activity).mean())
            if accuracy < min_validation_accuracy:
                continue
            accepted.append(rule)
            selected_for_dp += 1
            if selected_for_dp >= max_rules_per_decision_point:
                break
    return accepted


def evaluate_rules(
    rules: list[MinedAttributeRule],
    rows: pd.DataFrame,
    fallback_predictions: list[str] | None = None,
) -> dict[str, Any]:
    y_true = rows["true_next_activity"].astype(str).tolist()
    rule_predictions: list[str | None] = []
    applied_rule_ids: list[str | None] = []
    for _, row in rows.iterrows():
        prediction, rule_id = apply_rules_to_row(rules, row)
        rule_predictions.append(prediction)
        applied_rule_ids.append(rule_id)

    covered = [pred is not None for pred in rule_predictions]
    covered_true = [truth for truth, ok in zip(y_true, covered) if ok]
    covered_pred = [pred for pred in rule_predictions if pred is not None]
    coverage = sum(covered) / len(covered) if covered else 0.0
    rule_only_accuracy = (
        sum(1 for truth, pred in zip(covered_true, covered_pred) if truth == pred) / len(covered_true)
        if covered_true
        else float("nan")
    )
    if fallback_predictions is None:
        total_accuracy = rule_only_accuracy
        fallback_accuracy = float("nan")
    else:
        combined = [
            pred if pred is not None else fallback
            for pred, fallback in zip(rule_predictions, fallback_predictions)
        ]
        total_accuracy = sum(1 for truth, pred in zip(y_true, combined) if truth == pred) / len(y_true)
        fallback_true = [truth for truth, ok in zip(y_true, covered) if not ok]
        fallback_pred = [pred for pred, ok in zip(fallback_predictions, covered) if not ok]
        fallback_accuracy = (
            sum(1 for truth, pred in zip(fallback_true, fallback_pred) if truth == pred) / len(fallback_true)
            if fallback_true
            else float("nan")
        )

    return {
        "rule_count": len(rules),
        "n_samples": len(rows),
        "covered_samples": int(sum(covered)),
        "rule_coverage": coverage,
        "rule_only_accuracy": rule_only_accuracy,
        "fallback_accuracy": fallback_accuracy,
        "total_accuracy": total_accuracy,
        "applied_rule_counts": dict(Counter(rule_id for rule_id in applied_rule_ids if rule_id)),
    }


def apply_rules_to_row(
    rules: list[MinedAttributeRule],
    row: pd.Series,
) -> tuple[str | None, str | None]:
    decision_point_id = str(row.get("decision_point_id"))
    candidates = parse_candidate_labels(row.get("candidate_activity_labels"))
    for rule in rules:
        if rule.decision_point_id != decision_point_id:
            continue
        if rule.preferred_activity not in candidates:
            continue
        if _row_matches_rule(row, rule):
            return rule.preferred_activity, rule.rule_id
    return None, None


def _rules_from_subset(
    *,
    subset: pd.DataFrame,
    decision_point_id: str,
    attribute: str,
    operator: str,
    value: Any,
    value_kind: str,
    priors: dict[str, float],
    min_support: int,
    min_confidence: float,
    min_lift: float,
) -> list[MinedAttributeRule]:
    if len(subset) < min_support:
        return []
    counts = Counter(subset["true_next_activity"].astype(str))
    rules: list[MinedAttributeRule] = []
    for activity, support in counts.items():
        if support < min_support:
            continue
        confidence = support / len(subset)
        prior = priors.get(activity, 0.0)
        lift = confidence / prior if prior > 0 else float("inf")
        if confidence < min_confidence or lift < min_lift:
            continue
        stable_value = str(value).replace(" ", "_").replace("/", "_")[:32]
        rule_id = f"{decision_point_id[:10]}_{attribute}_{operator}_{stable_value}_{activity[:16]}"
        rules.append(
            MinedAttributeRule(
                rule_id=rule_id,
                decision_point_id=decision_point_id,
                attribute=attribute,
                operator=operator,
                value=value,
                preferred_activity=activity,
                support=int(len(subset)),
                confidence=float(confidence),
                lift=float(lift),
                value_kind=value_kind,
            )
        )
    return rules


def _covered_rows(rows: pd.DataFrame, rule: MinedAttributeRule) -> pd.DataFrame:
    if rule.attribute not in rows.columns:
        return rows.iloc[0:0]
    return rows[rows.apply(lambda row: _row_matches_rule(row, rule), axis=1)]


def _row_matches_rule(row: pd.Series, rule: MinedAttributeRule) -> bool:
    value = row.get(rule.attribute)
    if pd.isna(value):
        value = "MISSING"
    if rule.operator == "==":
        return str(value) == str(rule.value)
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return False
    threshold = float(rule.value)
    if rule.operator == "<=":
        return float(numeric) <= threshold
    if rule.operator == ">":
        return float(numeric) > threshold
    raise ValueError(f"Unsupported mined rule operator: {rule.operator}")
