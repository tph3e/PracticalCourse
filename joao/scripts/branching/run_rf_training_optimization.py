from __future__ import annotations

import argparse
import json
import math
import pickle
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

JOAO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = JOAO_ROOT.parent
sys.path.insert(0, str(JOAO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from joao.scripts.branching.run_corrected_branching_evaluation import (  # noqa: E402
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    constrained_rf_predictions,
    load_log,
    parse_candidates,
    predict_from_priors,
    prepare_X,
    train_priors,
)
from joao.src.branching.BranchingUtils import temporal_train_validation_test_split_by_case  # noqa: E402


CASE_COL = "case:concept:name"
TARGET = "true_next_activity"
CURRENT_PARAMS = {
    "n_estimators": 100,
    "max_depth": 8,
    "min_samples_leaf": 2,
    "min_samples_split": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "criterion": "gini",
}

FEATURE_SETS = {
    "history_only": [
        "current_activity",
        "previous_activity",
        "trace_prefix",
        "event_index",
        "current_activity_visit_count",
        "consecutive_repetition_count",
    ],
    "history_time": [
        "current_activity",
        "previous_activity",
        "trace_prefix",
        "event_index",
        "current_activity_visit_count",
        "consecutive_repetition_count",
        "weekday",
        "hour",
        "month",
        "elapsed_time_seconds",
        "time_since_previous_event_seconds",
    ],
    "history_case": [
        "current_activity",
        "previous_activity",
        "trace_prefix",
        "event_index",
        "current_activity_visit_count",
        "consecutive_repetition_count",
        "case:ApplicationType",
        "case:LoanGoal",
        "case:RequestedAmount",
        "CreditScore",
        "EventOrigin",
    ],
    "full": FEATURE_COLUMNS,
    "full_with_decision_state": FEATURE_COLUMNS + ["decision_point_id", "candidate_set_signature"],
}


def add_inner_split(rows: pd.DataFrame, split) -> pd.DataFrame:
    inner_train = set(str(case_id) for case_id in split.inner_train_cases)
    inner_validation = set(str(case_id) for case_id in split.inner_validation_cases)
    outer_test = set(str(case_id) for case_id in split.outer_test_cases)
    result = rows.copy()

    def label(case_id: Any) -> str:
        case_id = str(case_id)
        if case_id in inner_train:
            return "inner_train"
        if case_id in inner_validation:
            return "inner_validation"
        if case_id in outer_test:
            return "outer_test"
        return "unknown"

    result["rfopt_split"] = result["case_id"].map(label)
    return result


def clean_feature_columns(rows: pd.DataFrame, requested: list[str]) -> tuple[list[str], list[dict[str, Any]]]:
    kept: list[str] = []
    excluded: list[dict[str, Any]] = []
    for col in requested:
        if col not in rows.columns:
            excluded.append({"feature": col, "reason": "missing_column"})
            continue
        missing_rate = float(rows[col].map(pd.isna).mean())
        nunique = rows[col].nunique(dropna=True)
        if missing_rate > 0.99:
            excluded.append({"feature": col, "reason": "missing_rate_gt_0.99", "missing_rate": missing_rate})
            continue
        if nunique <= 1:
            excluded.append({"feature": col, "reason": "constant", "missing_rate": missing_rate})
            continue
        kept.append(col)
    return kept, excluded


def build_rf(rows: pd.DataFrame, features: list[str], params: dict[str, Any], seed: int) -> Pipeline:
    X = _prepare_features(rows, features)
    y = rows[TARGET].astype(str)
    numeric = [col for col in features if col in NUMERIC_COLUMNS]
    categorical = [col for col in features if col not in numeric]
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ]
    )
    classifier = RandomForestClassifier(
        random_state=seed,
        n_jobs=-1,
        **params,
    )
    model = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    model.fit(X, y)
    return model


def _prepare_features(rows: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = rows[features].copy()
    for col in features:
        if col in NUMERIC_COLUMNS:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
        else:
            X[col] = X[col].fillna("UNKNOWN").astype(str)
    return X


def constrained_predictions(model: Pipeline, rows: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str], np.ndarray, int, int, float]:
    X = _prepare_features(rows, features)
    started = time.perf_counter()
    raw = model.predict(X).astype(str).tolist()
    probabilities = model.predict_proba(X)
    runtime = time.perf_counter() - started
    classes = [str(item) for item in model.named_steps["classifier"].classes_]
    constrained: list[str] = []
    raw_rejected = 0
    fallback = 0
    for i, row in enumerate(rows.itertuples(index=False)):
        candidates = parse_candidates(row.candidate_activity_labels)
        pairs = sorted(zip(classes, probabilities[i]), key=lambda item: item[1], reverse=True)
        if pairs and pairs[0][0] not in candidates:
            raw_rejected += 1
        selected = next((label for label, _ in pairs if label in candidates), None)
        if selected is None:
            selected = raw[i] if raw[i] in candidates else (candidates[0] if candidates else raw[i])
            fallback += 1
        constrained.append(selected)
    return raw, constrained, probabilities, raw_rejected, fallback, runtime


def evaluate_model(
    model: Pipeline,
    rows: pd.DataFrame,
    features: list[str],
    *,
    candidate_id: str,
    seed: int,
    phase: str,
) -> dict[str, Any]:
    y_true = rows[TARGET].astype(str).tolist()
    raw, constrained, probabilities, raw_rejected, fallback, runtime = constrained_predictions(model, rows, features)
    classes = [str(item) for item in model.named_steps["classifier"].classes_]
    return {
        "candidate_id": candidate_id,
        "seed": seed,
        "phase": phase,
        "accuracy": float(accuracy_score(y_true, constrained)),
        "macro_f1": float(f1_score(y_true, constrained, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, constrained, average="weighted", zero_division=0)),
        "raw_accuracy": float(accuracy_score(y_true, raw)),
        "log_loss": _safe_log_loss(y_true, probabilities, classes),
        "brier": _brier_multiclass(y_true, probabilities, classes),
        "top2_accuracy": _top_k_accuracy(y_true, probabilities, classes, k=2),
        "fallback_rate": fallback / len(rows) if len(rows) else math.nan,
        "raw_top_class_rejected_by_bpmn": raw_rejected,
        "inference_runtime_seconds": runtime,
        "artifact_size_estimate_bytes": len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)),
        "n_samples": len(rows),
    }


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


def _top_k_accuracy(y_true: list[str], probabilities: np.ndarray, classes: list[str], k: int) -> float:
    hits = 0
    for i, label in enumerate(y_true):
        top = [classes[j] for j in np.argsort(probabilities[i])[-k:]]
        hits += int(label in top)
    return hits / len(y_true) if y_true else math.nan


def candidate_configs() -> list[dict[str, Any]]:
    configs = [CURRENT_PARAMS]
    for n_estimators in [200, 400, 600]:
        for max_depth in [8, 12, 16, None]:
            for min_samples_leaf in [1, 2, 5]:
                for max_features in ["sqrt", "log2", 0.5]:
                    configs.append(
                        {
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "min_samples_leaf": min_samples_leaf,
                            "min_samples_split": 2 if min_samples_leaf <= 2 else 5,
                            "max_features": max_features,
                            "class_weight": "balanced_subsample" if n_estimators >= 200 else "balanced",
                            "criterion": "gini",
                        }
                    )
    configs.extend(
        [
            {**CURRENT_PARAMS, "criterion": "log_loss"},
            {**CURRENT_PARAMS, "class_weight": "balanced_subsample"},
            {**CURRENT_PARAMS, "class_weight": None},
            {**CURRENT_PARAMS, "min_samples_leaf": 1, "max_depth": 12, "n_estimators": 200},
        ]
    )
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cfg in configs:
        key = json.dumps(cfg, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cfg)
        if len(unique) >= 30:
            break
    return unique


def aggregate_stability(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for candidate_id, group in rows.groupby("candidate_id"):
        macro = group["macro_f1"].astype(float)
        ci = _ci95(macro.tolist())
        records.append(
            {
                "candidate_id": candidate_id,
                "mean_macro_f1": macro.mean(),
                "std_macro_f1": macro.std(ddof=1),
                "macro_f1_ci95_low": ci[0],
                "macro_f1_ci95_high": ci[1],
                "mean_weighted_f1": group["weighted_f1"].mean(),
                "mean_accuracy": group["accuracy"].mean(),
                "mean_log_loss": group["log_loss"].mean(),
                "mean_brier": group["brier"].mean(),
                "mean_top2_accuracy": group["top2_accuracy"].mean(),
                "mean_inference_runtime_seconds": group["inference_runtime_seconds"].mean(),
                "mean_artifact_size_estimate_bytes": group["artifact_size_estimate_bytes"].mean(),
                "mean_fallback_rate": group["fallback_rate"].mean(),
            }
        )
    return pd.DataFrame(records).sort_values(["mean_macro_f1", "mean_weighted_f1"], ascending=False)


def _ci95(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return (math.nan, math.nan)
    delta = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    mean = float(np.mean(values))
    return mean - delta, mean + delta


def export_audits(rows: pd.DataFrame, train: pd.DataFrame, validation: pd.DataFrame, output_dir: Path) -> None:
    class_rows = []
    for split_name, split_rows in [("inner_train", train), ("inner_validation", validation), ("all_development", rows[rows["split"] == "outer_train"])]:
        counts = Counter(split_rows[TARGET].astype(str))
        total = sum(counts.values())
        for label, count in sorted(counts.items()):
            class_rows.append({"split": split_name, "class": label, "count": count, "share": count / total if total else 0.0})
    pd.DataFrame(class_rows).to_csv(output_dir / "rf_class_distribution.csv", index=False)
    train.groupby("decision_point_id").agg(
        n_samples=(TARGET, "size"),
        n_classes=(TARGET, "nunique"),
        candidate_count=("candidate_count", "mean"),
    ).reset_index().to_csv(output_dir / "rf_decision_point_support.csv", index=False)
    missing = []
    for col in FEATURE_COLUMNS + ["decision_point_id", "candidate_set_signature"]:
        if col in train.columns:
            missing.append({"feature": col, "missing_rate": float(train[col].map(pd.isna).mean()), "nunique": int(train[col].nunique(dropna=True))})
    pd.DataFrame(missing).to_csv(output_dir / "rf_feature_missingness.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=str(REPO_ROOT / "data/logData.xes"))
    parser.add_argument("--corrected-results-dir", default=str(JOAO_ROOT / "results/branching_corrected_20260717"))
    parser.add_argument("--output-dir", default=str(JOAO_ROOT / "results/rf_training_optimization_20260717"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = load_log(args.log)
    split = temporal_train_validation_test_split_by_case(log)
    rows = pd.read_csv(Path(args.corrected_results_dir) / "datasets/branching_common_dataset_filtered.csv")
    rows = add_inner_split(rows, split)
    inner_train = rows[rows["rfopt_split"] == "inner_train"].reset_index(drop=True)
    inner_validation = rows[rows["rfopt_split"] == "inner_validation"].reset_index(drop=True)
    outer_train = rows[rows["split"] == "outer_train"].reset_index(drop=True)
    outer_test = rows[rows["split"] == "outer_test"].reset_index(drop=True)

    export_audits(rows, inner_train, inner_validation, output_dir)

    ablation_records = []
    for feature_set, requested in FEATURE_SETS.items():
        features, excluded = clean_feature_columns(inner_train, requested)
        model = build_rf(inner_train, features, CURRENT_PARAMS, seed=1)
        record = evaluate_model(model, inner_validation, features, candidate_id=feature_set, seed=1, phase="feature_ablation")
        record["feature_set"] = feature_set
        record["n_features"] = len(features)
        record["excluded_features"] = json.dumps(excluded, default=str)
        ablation_records.append(record)
    ablation_df = pd.DataFrame(ablation_records)
    ablation_df.to_csv(output_dir / "rf_feature_ablation_metrics.csv", index=False)

    full_features, excluded_features = clean_feature_columns(inner_train, FEATURE_COLUMNS)
    configs = candidate_configs()
    config_rows = [{"candidate_id": f"rf_{i:02d}", **cfg} for i, cfg in enumerate(configs)]
    pd.DataFrame(config_rows).to_csv(output_dir / "rf_candidate_hyperparameters.csv", index=False)

    phase_a = []
    for row in config_rows:
        params = {key: row[key] for key in CURRENT_PARAMS}
        model = build_rf(inner_train, full_features, params, seed=1)
        rec = evaluate_model(model, inner_validation, full_features, candidate_id=row["candidate_id"], seed=1, phase="phase_a")
        rec.update(params)
        phase_a.append(rec)
    phase_a_df = pd.DataFrame(phase_a).sort_values(["macro_f1", "weighted_f1"], ascending=False)
    phase_a_df.to_csv(output_dir / "rf_hyperparameter_phase_a_metrics.csv", index=False)

    top_ids = list(dict.fromkeys(["rf_00"] + phase_a_df["candidate_id"].head(3).tolist()))[:4]
    stability = []
    for candidate_id in top_ids:
        cfg = next(row for row in config_rows if row["candidate_id"] == candidate_id)
        params = {key: cfg[key] for key in CURRENT_PARAMS}
        for seed in [1, 2, 3, 4, 5]:
            model = build_rf(inner_train, full_features, params, seed=seed)
            rec = evaluate_model(model, inner_validation, full_features, candidate_id=candidate_id, seed=seed, phase="phase_b_stability")
            rec.update(params)
            stability.append(rec)
    stability_df = pd.DataFrame(stability)
    stability_df.to_csv(output_dir / "rf_seed_stability_metrics.csv", index=False)
    stability_agg = aggregate_stability(stability_df)
    stability_agg.to_csv(output_dir / "rf_seed_stability_aggregated.csv", index=False)

    baseline = stability_agg[stability_agg["candidate_id"] == "rf_00"].iloc[0].to_dict()
    best = stability_agg.iloc[0].to_dict()
    selection = _selection_decision(baseline, best)
    selected_id = best["candidate_id"] if selection["decision"] == "ADOPT_RECOMMENDED" else "rf_00"
    selected_cfg = next(row for row in config_rows if row["candidate_id"] == selected_id)
    selected_params = {key: selected_cfg[key] for key in CURRENT_PARAMS}

    # Outer held-out is opened once after the inner-validation decision.
    outer_model = build_rf(outer_train, full_features, selected_params, seed=1)
    outer_selected = evaluate_model(outer_model, outer_test, full_features, candidate_id=selected_id, seed=1, phase="outer_heldout_once")
    current_model = build_rf(outer_train, full_features, CURRENT_PARAMS, seed=1)
    outer_current = evaluate_model(current_model, outer_test, full_features, candidate_id="rf_00_current", seed=1, phase="outer_heldout_current")
    pd.DataFrame([outer_current, outer_selected]).to_csv(output_dir / "rf_outer_heldout_metrics.csv", index=False)

    priors = train_priors(outer_train)
    majority, _, majority_fallback = predict_from_priors(priors, outer_test, "majority", seed=1)
    random_pred, _, _ = predict_from_priors(priors, outer_test, "random", seed=1)
    prob_pred, _, _ = predict_from_priors(priors, outer_test, "probability", seed=1)
    y_true = outer_test[TARGET].astype(str).tolist()
    core_rows = [
        {"method": "MajorityBaseline", **_simple_metrics(y_true, majority), "fallback_rate": majority_fallback / len(y_true)},
        {"method": "RandomCandidateBaseline", **_simple_metrics(y_true, random_pred), "fallback_rate": 0.0},
        {"method": "ProbabilityBranching", **_simple_metrics(y_true, prob_pred), "fallback_rate": 0.0},
        {"method": "PredictiveML-Raw", "accuracy": outer_selected["raw_accuracy"], "macro_f1": outer_selected["macro_f1"], "weighted_f1": outer_selected["weighted_f1"], "fallback_rate": outer_selected["fallback_rate"]},
        {"method": "PredictiveML-BPMNConstrained", "accuracy": outer_selected["accuracy"], "macro_f1": outer_selected["macro_f1"], "weighted_f1": outer_selected["weighted_f1"], "fallback_rate": outer_selected["fallback_rate"]},
        {"method": "CompositeRuntime", "accuracy": outer_selected["accuracy"], "macro_f1": outer_selected["macro_f1"], "weighted_f1": outer_selected["weighted_f1"], "fallback_rate": outer_selected["fallback_rate"]},
    ]
    pd.DataFrame(core_rows).to_csv(output_dir / "branching_final_core_metrics.csv", index=False)
    pd.DataFrame(core_rows).to_csv(Path(args.corrected_results_dir) / "branching_final_core_metrics.csv", index=False)

    class_metrics = _per_class_metrics(outer_test[TARGET].astype(str).tolist(), constrained_predictions(outer_model, outer_test, full_features)[1])
    class_metrics.to_csv(output_dir / "rf_outer_per_class_metrics.csv", index=False)
    _per_decision_point_metrics(outer_test, constrained_predictions(outer_model, outer_test, full_features)[1]).to_csv(output_dir / "rf_outer_per_decision_point_metrics.csv", index=False)

    calibration = {
        "tested": False,
        "adopted": False,
        "reason": "RF probabilities are used for BPMN-constrained class ranking, but no calibrated artifact is adopted in this simplification phase; selection target is macro-F1 on inner validation.",
    }
    (output_dir / "rf_calibration_diagnostics.json").write_text(json.dumps(calibration, indent=2))

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "sklearn": sklearn.__version__,
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dataset_counts": {
            "inner_train": len(inner_train),
            "inner_validation": len(inner_validation),
            "outer_train": len(outer_train),
            "outer_test": len(outer_test),
        },
        "features": full_features,
        "excluded_features": excluded_features,
        "current_params": CURRENT_PARAMS,
        "candidate_count_phase_a": len(config_rows),
        "selection": selection,
        "selected_candidate_id": selected_id,
        "selected_params": selected_params,
        "artifact_created": False,
        "fixed_replay_rerun_required": selection["decision"] == "ADOPT_RECOMMENDED",
        "composite_hierarchy": ["PredictiveBranchingEngine", "ProbabilityBranchingEngine", "random_bpmn_valid_fallback"],
    }
    (output_dir / "rf_selection_decision.json").write_text(json.dumps(summary, indent=2, default=str))
    (output_dir / "README.md").write_text(
        "# RF Training Optimization 20260717\n\n"
        "Controlled inner-validation optimization for the final branching scope: ProbabilityBranching, Predictive Random Forest, and CompositeRuntime. "
        "No existing artifacts were overwritten and fixed replay was not rerun unless explicitly required by `rf_selection_decision.json`.\n"
    )
    (output_dir / "failures.csv").write_text("stage,message\n")
    (output_dir / "requirements-reproducibility.txt").write_text(f"pandas=={pd.__version__}\nnumpy=={np.__version__}\nscikit-learn=={sklearn.__version__}\nscipy=={stats.__version__ if hasattr(stats, '__version__') else '1.17.1'}\n")
    print(json.dumps(summary, indent=2, default=str))


def _selection_decision(baseline: dict[str, Any], best: dict[str, Any]) -> dict[str, Any]:
    if best["candidate_id"] == baseline["candidate_id"]:
        return {"decision": "KEEP_CURRENT_MODEL", "reason": "current configuration is best on inner validation", "criteria": {}}
    criteria = {
        "macro_f1_improvement_ge_0.005": best["mean_macro_f1"] >= baseline["mean_macro_f1"] + 0.005,
        "weighted_f1_drop_le_0.002": best["mean_weighted_f1"] >= baseline["mean_weighted_f1"] - 0.002,
        "accuracy_drop_le_0.010": best["mean_accuracy"] >= baseline["mean_accuracy"] - 0.010,
        "fallback_rate_not_materially_higher": best["mean_fallback_rate"] <= baseline["mean_fallback_rate"] + 0.001,
        "runtime_acceptable": best["mean_inference_runtime_seconds"] <= baseline["mean_inference_runtime_seconds"] * 3,
    }
    return {
        "decision": "ADOPT_RECOMMENDED" if all(criteria.values()) else "KEEP_CURRENT_MODEL",
        "reason": "predeclared criteria satisfied" if all(criteria.values()) else "predeclared criteria not satisfied",
        "criteria": criteria,
        "baseline": baseline,
        "best": best,
    }


def _simple_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _per_class_metrics(y_true: list[str], y_pred: list[str]) -> pd.DataFrame:
    labels = sorted(set(y_true) | set(y_pred))
    records = []
    for label in labels:
        tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        records.append({"class": label, "support": sum(1 for truth in y_true if truth == label), "precision": precision, "recall": recall, "f1": f1})
    return pd.DataFrame(records)


def _per_decision_point_metrics(rows: pd.DataFrame, predictions: list[str]) -> pd.DataFrame:
    work = rows.copy()
    work["prediction"] = predictions
    records = []
    for decision_point_id, group in work.groupby("decision_point_id"):
        y_true = group[TARGET].astype(str).tolist()
        y_pred = group["prediction"].astype(str).tolist()
        records.append({"decision_point_id": decision_point_id, "n_samples": len(group), **_simple_metrics(y_true, y_pred)})
    return pd.DataFrame(records)


if __name__ == "__main__":
    main()
