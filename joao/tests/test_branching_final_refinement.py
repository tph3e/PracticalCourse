from __future__ import annotations

from collections import Counter

import pandas as pd

from joao.src.branching.AttributeRuleMining import (
    evaluate_rules,
    mine_attribute_rules,
    select_rules_on_validation,
)
from joao.src.branching.AttributeSamplingEvaluation import (
    EmpiricalAttributeSampler,
    evaluate_missingness_sampling,
    inject_missingness,
)
from joao.src.branching.TransitionContinuationEvaluation import evaluate_transition_continuation
from joao.src.resource_allocation.integration.TransitionAwareBranching import TransitionDisambiguationModel


def _rule_rows() -> pd.DataFrame:
    rows = []
    for idx in range(80):
        rows.append(
            {
                "case_id": f"T{idx}",
                "decision_point_id": "dp1",
                "segment": "premium",
                "true_next_activity": "Approve",
                "candidate_activity_labels": "Approve|Reject",
            }
        )
    for idx in range(30):
        rows.append(
            {
                "case_id": f"N{idx}",
                "decision_point_id": "dp1",
                "segment": "standard",
                "true_next_activity": "Reject",
                "candidate_activity_labels": "Approve|Reject",
            }
        )
    return pd.DataFrame(rows)


def test_attribute_rule_mining_uses_train_rows_and_validates_on_validation():
    train = _rule_rows()
    validation = _rule_rows().iloc[:40].copy()
    rules = mine_attribute_rules(
        train,
        categorical_attributes=["segment"],
        numeric_attributes=[],
        min_support=20,
        min_confidence=0.7,
        min_lift=1.01,
    )
    selected = select_rules_on_validation(
        rules,
        validation,
        min_validation_support=10,
        min_validation_accuracy=0.7,
    )
    metrics = evaluate_rules(selected, validation)

    assert selected
    assert metrics["rule_coverage"] > 0
    assert metrics["rule_only_accuracy"] == 1.0
    assert all(rule.attribute == "segment" for rule in selected)


def test_attribute_sampler_handles_nan_and_is_deterministic():
    train = pd.DataFrame(
        {
            "current_activity": ["A", "A", "B"],
            "score": [10.0, 20.0, 30.0],
            "true_next_activity": ["X", "Y", "Z"],
        }
    )
    rows = pd.DataFrame(
        {
            "current_activity": ["A", "A"],
            "score": [pd.NA, pd.NA],
            "true_next_activity": ["X", "Y"],
        }
    )
    first = EmpiricalAttributeSampler(["score"], seed=7).fit(train)
    second = EmpiricalAttributeSampler(["score"], seed=7).fit(train)

    filled_first, counts_first = first.fill_missing(rows)
    filled_second, counts_second = second.fill_missing(rows)

    assert counts_first["sampled_values"] == 2
    assert counts_first == counts_second
    assert filled_first["score"].tolist() == filled_second["score"].tolist()
    assert not filled_first["score"].map(pd.isna).any()


def test_missingness_sampling_evaluation_reports_delta_metrics():
    rows = pd.DataFrame(
        {
            "current_activity": ["A"] * 10,
            "attr": ["x"] * 10,
            "true_next_activity": ["B"] * 10,
        }
    )

    def predict_fn(frame):
        return ["B"] * len(frame)

    result = evaluate_missingness_sampling(
        rows,
        inject_missingness(rows, attributes=["attr"], rate=0.5, seed=1),
        attributes=["attr"],
        predict_fn=predict_fn,
        rates=[0.5],
        seed=1,
    )

    assert result.loc[0, "missing_rate"] == 0.5
    assert "delta_accuracy_sampled_vs_none" in result.columns


def test_transition_continuation_reports_not_identifiable_when_no_ambiguity():
    log = pd.DataFrame(
        {
            "case:concept:name": ["C1"],
            "concept:name": ["A"],
            "time:timestamp": [pd.Timestamp("2026-01-01", tz="UTC")],
        }
    )
    model = TransitionDisambiguationModel(marking_counts={"m": Counter({"t": 1})})
    result = evaluate_transition_continuation(
        log,
        model,
        bpmn_model="models/v4_replay.bpmn",
        case_limit=1,
    )

    assert result["ambiguous_decisions"] == 0
    assert result["status"] == "not_identifiable_on_this_log_bpmn"
