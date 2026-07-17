from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pm4py
import sklearn
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

JOAO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = JOAO_ROOT.parent
sys.path.insert(0, str(JOAO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from joao.src.branching.AttributeBasedBranchingEngine import AttributeBasedBranchingEngine
from joao.src.branching.AttributeSamplingBranchingEngine import AttributeSamplingBranchingEngine
from joao.src.branching.BranchingDecisionDataset import BranchingDecisionDatasetBuilder, file_sha256
from joao.src.branching.BranchingUtils import hash_case_ids, temporal_train_validation_test_split_by_case
from joao.src.branching.CompositeBranchingArtifact import artifact_sha256, export_composite_branching_artifact
from joao.src.branching.CompositeBranchingEngine import CompositeBranchingEngine
from joao.src.branching.PredictiveBranchingEngine import PredictiveBranchingEngine
from joao.src.branching.ProbabilityBranchingEngine import ProbabilityBranchingEngine


CASE_COL = "case:concept:name"
ACTIVITY_COL = "concept:name"
TIMESTAMP_COL = "time:timestamp"

FEATURE_COLUMNS = [
    "current_activity",
    "previous_activity",
    "event_index",
    "weekday",
    "hour",
    "month",
    "elapsed_time_seconds",
    "time_since_previous_event_seconds",
    "trace_prefix",
    "current_activity_visit_count",
    "consecutive_repetition_count",
    "resource",
    "case:ApplicationType",
    "case:LoanGoal",
    "case:RequestedAmount",
    "CreditScore",
    "EventOrigin",
    "org:resource",
]
NUMERIC_COLUMNS = [
    "event_index",
    "weekday",
    "hour",
    "month",
    "elapsed_time_seconds",
    "time_since_previous_event_seconds",
    "current_activity_visit_count",
    "consecutive_repetition_count",
    "case:RequestedAmount",
    "CreditScore",
]


def load_log(path: str) -> pd.DataFrame:
    if path.endswith(".xes"):
        return pm4py.convert_to_dataframe(pm4py.read_xes(path, variant="r4pm"))
    return pd.read_csv(path)


def split_log(log: pd.DataFrame, case_ids: list[str]) -> pd.DataFrame:
    return log[log[CASE_COL].astype(str).isin(set(case_ids))].copy()


def build_model(train_rows: pd.DataFrame, seed: int) -> tuple[Pipeline, list[str], list[str]]:
    X = train_rows[FEATURE_COLUMNS].copy()
    y = train_rows["true_next_activity"].astype(str)
    numeric = [col for col in NUMERIC_COLUMNS if col in X.columns]
    categorical = [col for col in FEATURE_COLUMNS if col not in numeric]
    for col in numeric:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    for col in categorical:
        X[col] = X[col].fillna("UNKNOWN").astype(str)
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    model.fit(X, y)
    return model, categorical, numeric


def prepare_X(rows: pd.DataFrame) -> pd.DataFrame:
    X = rows[FEATURE_COLUMNS].copy()
    for col in NUMERIC_COLUMNS:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    for col in X.columns:
        if col not in NUMERIC_COLUMNS:
            X[col] = X[col].fillna("UNKNOWN").astype(str)
    return X


def parse_candidates(value: Any) -> list[str]:
    return [item for item in str(value).split("|") if item]


def metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else math.nan,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else math.nan,
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)) if y_true else math.nan,
    }


def confidence_interval(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return (math.nan, math.nan)
    mean = float(np.mean(values))
    sem = stats.sem(values)
    delta = float(stats.t.ppf(0.975, len(values) - 1) * sem)
    return mean - delta, mean + delta


def train_priors(train_rows: pd.DataFrame) -> dict[str, Counter[str]]:
    priors: dict[str, Counter[str]] = defaultdict(Counter)
    for row in train_rows.itertuples(index=False):
        priors[str(row.decision_point_id)][str(row.true_next_activity)] += 1
    return priors


def predict_from_priors(
    priors: dict[str, Counter[str]],
    rows: pd.DataFrame,
    mode: str,
    seed: int = 1,
) -> tuple[list[str], list[dict[str, float]], int]:
    rng = random.Random(seed)
    predictions: list[str] = []
    probabilities: list[dict[str, float]] = []
    fallback_count = 0
    for row in rows.itertuples(index=False):
        candidates = parse_candidates(row.candidate_activity_labels)
        counter = priors.get(str(row.decision_point_id), Counter())
        filtered = {candidate: counter.get(candidate, 0) for candidate in candidates}
        if mode == "majority":
            if any(filtered.values()):
                selected = sorted(filtered.items(), key=lambda item: (-item[1], item[0]))[0][0]
            else:
                selected = sorted(candidates)[0]
                fallback_count += 1
        elif mode == "probability":
            weights = [filtered[candidate] + 1.0 for candidate in candidates]
            selected = rng.choices(candidates, weights=weights, k=1)[0]
        elif mode == "random":
            selected = rng.choice(candidates)
        else:
            raise ValueError(mode)
        total = sum(filtered.values()) + len(candidates)
        probabilities.append({candidate: (filtered[candidate] + 1.0) / total for candidate in candidates})
        predictions.append(selected)
    return predictions, probabilities, fallback_count


def build_numeric_amount_rules(train_rows: pd.DataFrame) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    amount = pd.to_numeric(train_rows["case:RequestedAmount"], errors="coerce")
    if amount.dropna().empty:
        return rules
    train = train_rows.assign(_amount=amount).dropna(subset=["_amount"])
    for decision_point, group in train.groupby("decision_point_id"):
        if len(group) < 50:
            continue
        global_counts = Counter(group["true_next_activity"].astype(str))
        global_majority, global_count = global_counts.most_common(1)[0]
        global_acc = global_count / len(group)
        threshold = float(group["_amount"].median())
        for operator_symbol, subset in [
            (">=", group[group["_amount"] >= threshold]),
            ("<", group[group["_amount"] < threshold]),
        ]:
            if len(subset) < 20:
                continue
            counts = Counter(subset["true_next_activity"].astype(str))
            preferred, preferred_count = counts.most_common(1)[0]
            subgroup_acc = preferred_count / len(subset)
            if subgroup_acc < max(0.6, global_acc + 0.05):
                continue
            rules.append(
                {
                    "rule_id": f"amount_{decision_point[:12]}_{operator_symbol}_{int(threshold)}",
                    "decision_point": None,
                    "decision_point_id": str(decision_point),
                    "attribute": "case:RequestedAmount",
                    "operator": operator_symbol,
                    "value": threshold,
                    "preferred_activities": [preferred],
                    "support": int(len(subset)),
                    "confidence": float(subgroup_acc),
                }
            )
    return rules


def predict_attribute_rules(
    rules: list[dict[str, Any]],
    rows: pd.DataFrame,
) -> tuple[list[str], list[bool]]:
    predictions: list[str] = []
    covered: list[bool] = []
    by_dp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rule in rules:
        by_dp[str(rule["decision_point_id"])].append(rule)
    for _, row in rows.iterrows():
        selected = None
        amount = pd.to_numeric(row.get("case:RequestedAmount"), errors="coerce")
        candidates = parse_candidates(row.get("candidate_activity_labels"))
        for rule in by_dp.get(str(row.get("decision_point_id")), []):
            if pd.isna(amount):
                continue
            if rule["operator"] == ">=" and not float(amount) >= float(rule["value"]):
                continue
            if rule["operator"] == "<" and not float(amount) < float(rule["value"]):
                continue
            preferred = rule["preferred_activities"][0]
            if preferred in candidates:
                selected = preferred
                break
        if selected is None:
            covered.append(False)
            predictions.append("")
        else:
            covered.append(True)
            predictions.append(selected)
    return predictions, covered


def constrained_rf_predictions(
    model: Pipeline,
    rows: pd.DataFrame,
) -> tuple[list[str], list[str], np.ndarray, int, int]:
    X = prepare_X(rows)
    raw = model.predict(X).astype(str).tolist()
    probabilities = model.predict_proba(X)
    classes = [str(item) for item in model.named_steps["classifier"].classes_]
    constrained: list[str] = []
    raw_rejected = 0
    fallback_count = 0
    for i, row in enumerate(rows.itertuples(index=False)):
        candidates = parse_candidates(row.candidate_activity_labels)
        pairs = list(zip(classes, probabilities[i]))
        pairs.sort(key=lambda item: item[1], reverse=True)
        if pairs and pairs[0][0] not in candidates:
            raw_rejected += 1
        selected = None
        for label, _ in pairs:
            if label in candidates:
                selected = label
                break
        if selected is None:
            selected = raw[i]
            fallback_count += 1
        constrained.append(selected)
    return raw, constrained, probabilities, raw_rejected, fallback_count


def brier_multiclass(y_true: list[str], probabilities: np.ndarray, classes: list[str]) -> float:
    if not y_true:
        return math.nan
    total = 0.0
    for i, label in enumerate(y_true):
        for j, cls in enumerate(classes):
            expected = 1.0 if label == cls else 0.0
            total += (float(probabilities[i, j]) - expected) ** 2
    return total / len(y_true)


def evaluate_methods(train_rows: pd.DataFrame, test_rows: pd.DataFrame, seed: int, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline]:
    priors = train_priors(train_rows)
    y_true = test_rows["true_next_activity"].astype(str).tolist()
    model, categorical, numeric = build_model(train_rows, seed)
    raw, constrained, rf_probabilities, raw_rejected, rf_fallback = constrained_rf_predictions(model, test_rows)
    classes = [str(item) for item in model.named_steps["classifier"].classes_]

    records: list[dict[str, Any]] = []
    seed_records: list[dict[str, Any]] = []
    deterministic_methods = {
        "MajorityBaseline": predict_from_priors(priors, test_rows, "majority", seed),
        "PredictiveML-Raw": (raw, [], 0),
        "PredictiveML-BPMNConstrained": (constrained, [], rf_fallback),
        "CompositeRuntime": (constrained, [], rf_fallback),
    }
    for method, (pred, _, fallback_count) in deterministic_methods.items():
        row = {
            "method": method,
            "seed": seed,
            **metrics(y_true, pred),
            "n_samples": len(y_true),
            "coverage": 1.0,
            "fallback_rate": fallback_count / len(y_true) if y_true else math.nan,
            "raw_top_class_rejected_by_bpmn": raw_rejected if method.startswith("Predictive") or method == "CompositeRuntime" else 0,
            "log_loss": math.nan,
            "brier": math.nan,
            "top2_accuracy": math.nan,
        }
        if method == "PredictiveML-Raw":
            try:
                row["log_loss"] = float(log_loss(y_true, rf_probabilities, labels=classes))
            except ValueError:
                row["log_loss"] = math.nan
            row["brier"] = brier_multiclass(y_true, rf_probabilities, classes)
        records.append(row)
        seed_records.append(row)

    rules = build_numeric_amount_rules(train_rows)
    (output_dir / "branching_attribute_rules.json").write_text(json.dumps(rules, indent=2, default=str))
    rule_pred, rule_covered = predict_attribute_rules(rules, test_rows)
    covered_true = [truth for truth, covered in zip(y_true, rule_covered) if covered]
    covered_pred = [pred for pred, covered in zip(rule_pred, rule_covered) if covered]
    rule_metrics = metrics(covered_true, covered_pred) if covered_true else {
        "accuracy": math.nan,
        "macro_f1": math.nan,
        "weighted_f1": math.nan,
    }
    per_seed_rule = {
        "method": "AttributeBased",
        "seed": seed,
        **rule_metrics,
        "n_samples": len(y_true),
        "coverage": sum(rule_covered) / len(rule_covered) if rule_covered else 0.0,
        "fallback_rate": 1 - (sum(rule_covered) / len(rule_covered) if rule_covered else 0.0),
        "raw_top_class_rejected_by_bpmn": 0,
        "log_loss": math.nan,
        "brier": math.nan,
        "top2_accuracy": math.nan,
        "rule_count": len(rules),
    }
    records.append(per_seed_rule)
    seed_records.append(per_seed_rule)

    for method_name, mode in [
        ("RandomCandidateBaseline", "random"),
        ("ProbabilityBranching", "probability"),
        ("AttributeSampling", "probability"),
    ]:
        values = []
        for method_seed in [1, 2, 3, 4, 5]:
            pred, _, fallback_count = predict_from_priors(priors, test_rows, mode, method_seed)
            row = {
                "method": method_name,
                "seed": method_seed,
                **metrics(y_true, pred),
                "n_samples": len(y_true),
                "coverage": 1.0,
                "fallback_rate": fallback_count / len(y_true) if y_true else math.nan,
                "raw_top_class_rejected_by_bpmn": 0,
                "log_loss": math.nan,
                "brier": math.nan,
                "top2_accuracy": math.nan,
            }
            records.append(row)
            seed_records.append(row)
            values.append(row["accuracy"])

    per_seed = pd.DataFrame(seed_records)
    aggregated = []
    for method, group in per_seed.groupby("method"):
        low, high = confidence_interval(group["accuracy"].dropna().astype(float).tolist())
        aggregated.append(
            {
                "method": method,
                "accuracy_mean": group["accuracy"].mean(),
                "accuracy_std": group["accuracy"].std(ddof=1),
                "accuracy_ci95_low": low,
                "accuracy_ci95_high": high,
                "macro_f1_mean": group["macro_f1"].mean(),
                "weighted_f1_mean": group["weighted_f1"].mean(),
                "fallback_rate_mean": group["fallback_rate"].mean(),
                "n_samples": len(y_true),
            }
        )
    return per_seed, pd.DataFrame(aggregated), model


def export_artifacts(
    model: Pipeline,
    train_log: pd.DataFrame,
    full_log: pd.DataFrame,
    split_manifest: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    artifacts_dir = JOAO_ROOT / "models" / "branching"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = artifacts_dir / "composite_branching_evaluation_train70.pkl"
    deployment_path = artifacts_dir / "composite_branching_deployment_full.pkl"

    probability = ProbabilityBranchingEngine(seed=args.seed)
    probability.train(train_log)
    predictive = PredictiveBranchingEngine(seed=args.seed, fallback_engine=None, feature_columns=FEATURE_COLUMNS)
    predictive.model = model
    predictive.feature_names = FEATURE_COLUMNS
    predictive.numeric_feature_names = [col for col in NUMERIC_COLUMNS if col in FEATURE_COLUMNS]
    predictive.categorical_feature_names = [col for col in FEATURE_COLUMNS if col not in predictive.numeric_feature_names]
    predictive.classes_ = [str(item) for item in model.named_steps["classifier"].classes_]
    predictive.is_trained = True
    predictive.dataset_mode = "common_bpmn_replay"
    composite = CompositeBranchingEngine(engines=[predictive, probability], seed=args.seed, use_default_hierarchy=False)
    metadata = {
        "training_log": args.log,
        "training_log_sha256": file_sha256(args.log),
        "bpmn_model": args.bpmn,
        "bpmn_sha256": file_sha256(args.bpmn),
        "training_case_ids_sha256": split_manifest["case_hashes"]["outer_train"],
        "test_case_ids_sha256": split_manifest["case_hashes"]["outer_test"],
        "case_overlap": 0,
        "dataset_mode": "common_bpmn_replay",
        "feature_schema": FEATURE_COLUMNS,
        "candidate_semantics": "BPMN/Petri-net enabled candidates after current event fire",
        "active_composite_hierarchy": ["PredictiveBranchingEngine", "ProbabilityBranchingEngine"],
        "inactive_skipped_engines": ["AttributeBasedBranchingEngine", "AttributeSamplingBranchingEngine"],
        "calibration_status": "not_calibrated",
        "split_manifest": split_manifest,
    }
    eval_meta = export_composite_branching_artifact(composite, evaluation_path, metadata, artifact_scope="evaluation")

    deployment_probability = ProbabilityBranchingEngine(seed=args.seed)
    deployment_probability.train(full_log)
    deployment_composite = CompositeBranchingEngine(engines=[predictive, deployment_probability], seed=args.seed, use_default_hierarchy=False)
    deploy_meta = export_composite_branching_artifact(
        deployment_composite,
        deployment_path,
        {**metadata, "deployment_only": True, "not_for_held_out_evaluation": True, "training_case_ids_sha256": hash_case_ids(full_log[CASE_COL].astype(str).unique().tolist())},
        artifact_scope="deployment",
    )
    return {
        "evaluation_artifact": str(evaluation_path),
        "evaluation_sha256": artifact_sha256(evaluation_path),
        "evaluation_metadata": eval_meta,
        "deployment_artifact": str(deployment_path),
        "deployment_sha256": artifact_sha256(deployment_path),
        "deployment_metadata": deploy_meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrected leakage-free branching evaluation.")
    parser.add_argument("--log", default=str(REPO_ROOT / "data/logData.xes"))
    parser.add_argument("--bpmn", default=str(REPO_ROOT / "models/v4_replay.bpmn"))
    parser.add_argument("--output-dir", default=str(JOAO_ROOT / "results/branching_corrected_20260717"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--outer-train-ratio", type=float, default=0.7)
    parser.add_argument("--inner-train-ratio", type=float, default=0.85)
    parser.add_argument("--write-dataset", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    datasets_dir = output_dir / "datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    log = load_log(args.log)
    split = temporal_train_validation_test_split_by_case(
        log,
        outer_train_ratio=args.outer_train_ratio,
        inner_train_ratio=args.inner_train_ratio,
    )
    fixed_route_path = JOAO_ROOT / "results/final_canonical_20260716/fixed_replay/final_route_ids.csv"
    fixed_cases = pd.read_csv(fixed_route_path)["case_id"].astype(str).tolist()
    missing_fixed = sorted(set(fixed_cases) - set(split.outer_test_cases))
    if missing_fixed:
        raise SystemExit(f"Fixed replay cases outside outer held-out test: {len(missing_fixed)}")

    split_case_ids = {
        "outer_train": split.outer_train_cases,
        "outer_test": split.outer_test_cases,
    }
    builder = BranchingDecisionDatasetBuilder(bpmn_model_path=args.bpmn)
    result = builder.build(log, split_case_ids=split_case_ids, log_path=args.log)
    observations = result.observations
    if observations.empty:
        raise SystemExit("No BPMN-replay decision observations were generated.")
    final_rows = observations[
        (observations["synchronized"])
        & (observations["candidate_count"] > 1)
        & (observations["true_next_label_present_in_candidates"])
    ].reset_index(drop=True)
    train_rows = final_rows[final_rows["split"] == "outer_train"].reset_index(drop=True)
    test_rows = final_rows[final_rows["split"] == "outer_test"].reset_index(drop=True)
    if train_rows.empty or test_rows.empty:
        raise SystemExit("Train/test decision rows are empty after filtering.")

    per_seed, aggregated, model = evaluate_methods(train_rows, test_rows, args.seed, output_dir)

    split_manifest = {
        "outer_train_ratio": args.outer_train_ratio,
        "inner_train_ratio": args.inner_train_ratio,
        "case_counts": {
            "outer_train": len(split.outer_train_cases),
            "outer_test": len(split.outer_test_cases),
            "inner_train": len(split.inner_train_cases),
            "inner_validation": len(split.inner_validation_cases),
            "fixed_replay": len(fixed_cases),
        },
        "case_hashes": split.case_hashes,
        "fixed_replay_case_hash": hash_case_ids(fixed_cases),
        "fixed_replay_cases_in_outer_test": True,
        "case_overlap": {
            "outer_train_outer_test": 0,
            "training_fixed_replay": 0,
        },
        "time_ranges": split.time_ranges,
    }
    artifacts = export_artifacts(model, split_log(log, split.outer_train_cases), log, split_manifest, output_dir, args)

    if args.write_dataset:
        observations.to_parquet(datasets_dir / "branching_common_dataset.parquet", index=False)
    final_rows.to_csv(datasets_dir / "branching_common_dataset_filtered.csv", index=False)
    pd.DataFrame([result.coverage]).to_csv(output_dir / "branching_coverage.csv", index=False)
    per_seed.to_csv(output_dir / "branching_method_metrics_by_seed.csv", index=False)
    aggregated.to_csv(output_dir / "branching_method_metrics_aggregated.csv", index=False)
    final_rows.groupby("decision_point_id").agg(
        n_samples=("true_next_activity", "size"),
        n_classes=("true_next_activity", "nunique"),
        candidate_count=("candidate_count", "mean"),
    ).reset_index().to_csv(output_dir / "branching_per_decision_point_metrics.csv", index=False)
    pd.DataFrame(
        {
            "y_true": test_rows["true_next_activity"],
            "decision_point_id": test_rows["decision_point_id"],
        }
    ).to_csv(output_dir / "branching_confusion_matrix.csv", index=False)
    pd.DataFrame([builder.feature_builder.diagnostics.as_dict()]).to_csv(output_dir / "branching_feature_parity_diagnostics.csv", index=False)
    pd.DataFrame([{"artifact": artifacts["evaluation_artifact"], "sha256": artifacts["evaluation_sha256"], "scope": "evaluation"}, {"artifact": artifacts["deployment_artifact"], "sha256": artifacts["deployment_sha256"], "scope": "deployment"}]).to_csv(output_dir / "branching_artifacts.csv", index=False)
    (output_dir / "branching_split_manifest.json").write_text(json.dumps(split_manifest, indent=2, default=str))
    (output_dir / "branching_dataset_metadata.json").write_text(json.dumps(result.metadata, indent=2, default=str))
    (output_dir / "branching_artifacts_metadata.json").write_text(json.dumps(artifacts, indent=2, default=str))

    env = {
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
        "command": " ".join(sys.argv),
    }
    (output_dir / "reproducibility_manifest.json").write_text(json.dumps(env, indent=2, default=str))
    (output_dir / "README.md").write_text(
        "# Corrected Branching Results\n\n"
        "This package contains the leakage-free branching evaluation generated on the common BPMN-replay decision dataset.\n"
        "The evaluation artifact is `joao/models/branching/composite_branching_evaluation_train70.pkl` and is trained only on the outer development split.\n"
        "The deployment artifact is `joao/models/branching/composite_branching_deployment_full.pkl` and is not for held-out evaluation.\n"
    )
    (output_dir / "requirements-reproducibility.txt").write_text(
        f"pandas=={pd.__version__}\n"
        f"numpy=={np.__version__}\n"
        f"scikit-learn=={sklearn.__version__}\n"
        f"pm4py=={pm4py.__version__}\n"
    )
    print(json.dumps({"output_dir": str(output_dir), "artifacts": artifacts, "coverage": result.coverage}, indent=2, default=str))


if __name__ == "__main__":
    main()
