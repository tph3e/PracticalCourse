from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pm4py
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, f1_score, log_loss

JOAO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = JOAO_ROOT.parent
sys.path.insert(0, str(JOAO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from joao.scripts.branching.run_corrected_branching_evaluation import (  # noqa: E402
    FEATURE_COLUMNS,
    build_model,
    constrained_rf_predictions,
    load_log,
    prepare_X,
    predict_from_priors,
    train_priors,
)
from joao.src.branching.AttributeRuleMining import (  # noqa: E402
    evaluate_rules,
    mine_attribute_rules,
    select_rules_on_validation,
)
from joao.src.branching.AttributeSamplingEvaluation import evaluate_missingness_sampling  # noqa: E402
from joao.src.branching.BranchingUtils import temporal_train_validation_test_split_by_case  # noqa: E402
from joao.src.branching.TransitionContinuationEvaluation import evaluate_transition_continuation  # noqa: E402
from joao.src.resource_allocation.integration.TransitionAwareBranching import load_transition_model  # noqa: E402


CASE_COL = "case:concept:name"
TIME_COL = "time:timestamp"

CATEGORICAL_ATTRIBUTES = [
    "case:ApplicationType",
    "case:LoanGoal",
    "EventOrigin",
    "org:resource",
    "resource",
]
NUMERIC_ATTRIBUTES = [
    "case:RequestedAmount",
    "CreditScore",
]


def add_inner_split(rows: pd.DataFrame, split) -> pd.DataFrame:
    result = rows.copy()
    inner_train = set(str(case_id) for case_id in split.inner_train_cases)
    inner_validation = set(str(case_id) for case_id in split.inner_validation_cases)
    outer_test = set(str(case_id) for case_id in split.outer_test_cases)

    def label(case_id: Any) -> str:
        case_id = str(case_id)
        if case_id in inner_train:
            return "inner_train"
        if case_id in inner_validation:
            return "inner_validation"
        if case_id in outer_test:
            return "outer_test"
        return "unknown"

    result["refinement_split"] = result["case_id"].map(label)
    return result


def evaluate_advanced_i(rows: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    inner_train = rows[rows["refinement_split"] == "inner_train"].reset_index(drop=True)
    inner_validation = rows[rows["refinement_split"] == "inner_validation"].reset_index(drop=True)
    outer_test = rows[rows["refinement_split"] == "outer_test"].reset_index(drop=True)

    candidate_rules = mine_attribute_rules(
        inner_train,
        categorical_attributes=CATEGORICAL_ATTRIBUTES,
        numeric_attributes=NUMERIC_ATTRIBUTES,
        min_support=25,
        min_confidence=0.68,
        min_lift=1.03,
    )
    selected_rules = select_rules_on_validation(
        candidate_rules,
        inner_validation,
        max_rules_per_decision_point=3,
        min_validation_support=8,
        min_validation_accuracy=0.58,
    )
    priors = train_priors(inner_train)
    fallback_predictions, _, _ = predict_from_priors(priors, outer_test, "probability", seed=1)
    metrics = evaluate_rules(selected_rules, outer_test, fallback_predictions=fallback_predictions)

    pd.DataFrame([rule.to_dict() for rule in candidate_rules]).to_csv(
        output_dir / "advanced_i_candidate_rules.csv",
        index=False,
    )
    pd.DataFrame([rule.to_dict() for rule in selected_rules]).to_csv(
        output_dir / "advanced_i_selected_rules.csv",
        index=False,
    )
    pd.DataFrame([metrics]).to_csv(output_dir / "advanced_i_rule_metrics.csv", index=False)
    return {
        "candidate_rule_count": len(candidate_rules),
        "selected_rule_count": len(selected_rules),
        **metrics,
    }


def evaluate_sampling(rows: pd.DataFrame, output_dir: Path, seed: int) -> pd.DataFrame:
    train_rows = rows[rows["split"] == "outer_train"].reset_index(drop=True)
    test_rows = rows[rows["split"] == "outer_test"].reset_index(drop=True)
    model, _, _ = build_model(train_rows, seed)

    def predict_fn(frame: pd.DataFrame) -> list[str]:
        _, constrained, _, _, _ = constrained_rf_predictions(model, frame)
        return constrained

    metrics = evaluate_missingness_sampling(
        train_rows,
        test_rows,
        attributes=CATEGORICAL_ATTRIBUTES + NUMERIC_ATTRIBUTES,
        predict_fn=predict_fn,
        rates=[0.1, 0.3, 0.5],
        seed=seed,
    )
    metrics.to_csv(output_dir / "advanced_i_sampling_missingness_metrics.csv", index=False)
    return metrics


def evaluate_calibration(rows: pd.DataFrame, output_dir: Path, seed: int) -> dict[str, Any]:
    inner_train = rows[rows["refinement_split"] == "inner_train"].reset_index(drop=True)
    inner_validation = rows[rows["refinement_split"] == "inner_validation"].reset_index(drop=True)
    outer_test = rows[rows["refinement_split"] == "outer_test"].reset_index(drop=True)
    base_model, _, _ = build_model(inner_train, seed)
    classes = [str(item) for item in base_model.named_steps["classifier"].classes_]
    y_test = outer_test["true_next_activity"].astype(str).tolist()
    base_raw, base_constrained, base_proba, _, _ = constrained_rf_predictions(base_model, outer_test)
    base_record = {
        "method": "uncalibrated",
        "accuracy": accuracy_score(y_test, base_constrained),
        "macro_f1": f1_score(y_test, base_constrained, average="macro", zero_division=0),
        "log_loss": _safe_log_loss(y_test, base_proba, classes),
        "brier": _brier_multiclass(y_test, base_proba, classes),
        "adopted": False,
        "reason": "baseline",
    }
    records = [base_record]
    for method in ["sigmoid", "isotonic"]:
        try:
            calibrated = CalibratedClassifierCV(FrozenEstimator(base_model), method=method)
            calibrated.fit(prepare_X(inner_validation), inner_validation["true_next_activity"].astype(str))
            proba = calibrated.predict_proba(prepare_X(outer_test))
            cal_classes = [str(item) for item in calibrated.classes_]
            constrained = _constrain_from_probabilities(proba, cal_classes, outer_test)
            record = {
                "method": method,
                "accuracy": accuracy_score(y_test, constrained),
                "macro_f1": f1_score(y_test, constrained, average="macro", zero_division=0),
                "log_loss": _safe_log_loss(y_test, proba, cal_classes),
                "brier": _brier_multiclass(y_test, proba, cal_classes),
                "adopted": False,
                "reason": "not_evaluated_yet",
            }
            improves_probability = record["log_loss"] < base_record["log_loss"] and record["brier"] < base_record["brier"]
            preserves_f1 = record["macro_f1"] >= base_record["macro_f1"] - 0.01
            record["adopted"] = bool(improves_probability and preserves_f1)
            record["reason"] = "meets_adoption_rule" if record["adopted"] else "does_not_meet_adoption_rule"
            records.append(record)
        except Exception as exc:
            records.append(
                {
                    "method": method,
                    "accuracy": math.nan,
                    "macro_f1": math.nan,
                    "log_loss": math.nan,
                    "brier": math.nan,
                    "adopted": False,
                    "reason": f"failed: {type(exc).__name__}: {exc}",
                }
            )
    pd.DataFrame(records).to_csv(output_dir / "rf_calibration_evaluation.csv", index=False)
    adopted = [record for record in records if record.get("adopted")]
    return {
        "evaluated": True,
        "adopted": bool(adopted),
        "adopted_method": adopted[0]["method"] if adopted else None,
        "records": records,
    }


def _constrain_from_probabilities(probabilities: np.ndarray, classes: list[str], rows: pd.DataFrame) -> list[str]:
    predictions: list[str] = []
    for i, row in enumerate(rows.itertuples(index=False)):
        candidates = [item for item in str(row.candidate_activity_labels).split("|") if item]
        pairs = sorted(zip(classes, probabilities[i]), key=lambda item: item[1], reverse=True)
        selected = next((label for label, _ in pairs if label in candidates), None)
        predictions.append(selected or (candidates[0] if candidates else pairs[0][0]))
    return predictions


def _safe_log_loss(y_true: list[str], probabilities: np.ndarray, classes: list[str]) -> float:
    try:
        return float(log_loss(y_true, probabilities, labels=classes))
    except ValueError:
        return math.nan


def _brier_multiclass(y_true: list[str], probabilities: np.ndarray, classes: list[str]) -> float:
    total = 0.0
    for i, label in enumerate(y_true):
        for j, cls in enumerate(classes):
            expected = 1.0 if label == cls else 0.0
            total += (float(probabilities[i, j]) - expected) ** 2
    return total / len(y_true) if y_true else math.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="Final high-value branching refinement.")
    parser.add_argument("--log", default=str(REPO_ROOT / "data/logData.xes"))
    parser.add_argument("--bpmn", default=str(REPO_ROOT / "models/v4_replay.bpmn"))
    parser.add_argument("--corrected-results-dir", default=str(JOAO_ROOT / "results/branching_corrected_20260717"))
    parser.add_argument("--transition-artifact", default=str(JOAO_ROOT / "models/branching/transition_aware_branching_v1_20260715.pkl"))
    parser.add_argument("--output-dir", default=str(JOAO_ROOT / "results/branching_final_refinement_20260717"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--transition-case-limit", type=int, default=500)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log = load_log(args.log)
    split = temporal_train_validation_test_split_by_case(log)
    dataset_path = Path(args.corrected_results_dir) / "datasets/branching_common_dataset_filtered.csv"
    rows = pd.read_csv(dataset_path)
    rows = add_inner_split(rows, split)

    advanced_i = evaluate_advanced_i(rows, output_dir)
    sampling_metrics = evaluate_sampling(rows, output_dir, args.seed)

    transition_model = load_transition_model(args.transition_artifact)
    transition_report = evaluate_transition_continuation(
        log,
        transition_model,
        bpmn_model=args.bpmn,
        case_ids=split.outer_test_cases,
        case_limit=args.transition_case_limit,
    )
    (output_dir / "transition_continuation_consistency.json").write_text(
        json.dumps(transition_report, indent=2, default=str),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "status": transition_report["status"],
                "ambiguous_decisions": transition_report["ambiguous_decisions"],
                "model_invocations": transition_report["model_invocations"],
                "fallback_rate": transition_report["fallback_rate"],
                "model_continuation_1_step": transition_report["model"]["continuation_1_step"],
                "model_continuation_3_steps": transition_report["model"]["continuation_3_steps"],
                "model_continuation_full": transition_report["model"]["continuation_full"],
                "first_continuation_1_step": transition_report["first_candidate"]["continuation_1_step"],
                "random_proxy_continuation_1_step": transition_report["deterministic_random_proxy"]["continuation_1_step"],
            }
        ]
    ).to_csv(output_dir / "transition_continuation_consistency.csv", index=False)

    calibration = evaluate_calibration(rows, output_dir, args.seed)

    requirements_summary = {
        "source": "Repository README only; no detailed Task 1.5 assignment PDF/text was found in the workspace.",
        "advanced_i_requirements": [
            "Attribute-based branching should use case/event attributes to choose among branch candidates.",
            "Attribute sampling should learn attribute distributions from training data and use sampled values when runtime attributes are missing.",
            "Both must be trained/evaluated without test-log leakage and must respect BPMN-valid candidates.",
        ],
    }
    (output_dir / "advanced_i_requirements_audit.json").write_text(json.dumps(requirements_summary, indent=2))

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "pm4py": pm4py.__version__,
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--short"], text=True).strip()),
        "advanced_i": advanced_i,
        "sampling": sampling_metrics.to_dict(orient="records"),
        "transition_continuation": transition_report,
        "calibration": calibration,
        "fixed_replay_rerun_required": False,
        "composite_changed": False,
    }
    (output_dir / "final_refinement_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (output_dir / "README.md").write_text(
        "# Branching Final Refinement 20260717\n\n"
        "This package refines only the remaining branching gaps after the leakage-free correction.\n"
        "It does not replace the corrected evaluation artifact, does not alter the canonical composite, and does not rerun fixed replay.\n\n"
        "Outputs:\n"
        "- `advanced_i_candidate_rules.csv`\n"
        "- `advanced_i_selected_rules.csv`\n"
        "- `advanced_i_rule_metrics.csv`\n"
        "- `advanced_i_sampling_missingness_metrics.csv`\n"
        "- `transition_continuation_consistency.csv`\n"
        "- `rf_calibration_evaluation.csv`\n"
        "- `final_refinement_summary.json`\n"
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
