from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import unicodedata
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pm4py
from importlib import metadata as importlib_metadata


JOAO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = JOAO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(JOAO_ROOT))


CASE_COL = "case:concept:name"
ACTIVITY_COL = "concept:name"
RESOURCE_COL = "org:resource"
LIFECYCLE_COL = "lifecycle:transition"
TIMESTAMP_COL = "time:timestamp"


def normalize_activity(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    return " ".join(text.split())


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_versions() -> dict[str, str | None]:
    packages = [
        "scikit-learn",
        "scipy",
        "numpy",
        "pandas",
        "joblib",
        "pm4py",
        "rustxes",
        "pyarrow",
        "pytest",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def load_dataframe(path: Path) -> pd.DataFrame:
    log = pm4py.read_xes(str(path), variant="r4pm")
    if isinstance(log, pd.DataFrame):
        df = log.copy()
    else:
        df = pm4py.convert_to_dataframe(log)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    df["_normalized_activity"] = df[ACTIVITY_COL].map(normalize_activity)
    return df


def bpmn_visible_activities(path: Path) -> list[str]:
    bpmn_graph = pm4py.read_bpmn(str(path))
    net, _, _ = pm4py.convert_to_petri_net(bpmn_graph)
    return sorted({transition.label for transition in net.transitions if transition.label})


def pair_activity_occurrences(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stats = Counter()
    sort_cols = [CASE_COL, ACTIVITY_COL, TIMESTAMP_COL, "EventID"]
    existing_sort_cols = [col for col in sort_cols if col in df.columns]

    for (case_id, activity), group in df.sort_values(existing_sort_cols).groupby(
        [CASE_COL, ACTIVITY_COL],
        dropna=False,
    ):
        active_start = None
        active_segments = []
        waiting_start = None
        waiting_seconds = 0.0
        occurrence_index = 0
        occurrence_start_time = pd.NaT
        occurrence_resource = None
        occurrence_origin = None
        missing_lifecycle_events = 0

        for _, event in group.iterrows():
            lifecycle = str(event.get(LIFECYCLE_COL, "") or "").lower()
            timestamp = event.get(TIMESTAMP_COL)
            resource = event.get(RESOURCE_COL)
            origin = event.get("EventOrigin")
            if pd.isna(timestamp):
                stats["missing_timestamps"] += 1
                continue
            if not lifecycle or lifecycle == "nan":
                missing_lifecycle_events += 1
                continue

            if lifecycle == "start":
                if active_start is not None:
                    stats["overlapping_starts"] += 1
                active_start = timestamp
                occurrence_start_time = timestamp
                occurrence_resource = resource
                occurrence_origin = origin
                active_segments = []
                waiting_seconds = 0.0
                waiting_start = None
            elif lifecycle == "resume":
                if waiting_start is not None:
                    wait = (timestamp - waiting_start).total_seconds()
                    if wait >= 0 and math.isfinite(wait):
                        waiting_seconds += wait
                    else:
                        stats["negative_waiting_segments"] += 1
                active_start = timestamp
                if pd.isna(occurrence_start_time):
                    occurrence_start_time = timestamp
                occurrence_resource = resource
                occurrence_origin = origin
                waiting_start = None
            elif lifecycle == "suspend":
                if active_start is not None:
                    segment = (timestamp - active_start).total_seconds()
                    if segment >= 0 and math.isfinite(segment):
                        active_segments.append(segment)
                    else:
                        stats["negative_processing_segments"] += 1
                    active_start = None
                waiting_start = timestamp
            elif lifecycle == "complete":
                if active_start is not None:
                    segment = (timestamp - active_start).total_seconds()
                    if segment >= 0 and math.isfinite(segment):
                        active_segments.append(segment)
                    else:
                        stats["negative_processing_segments"] += 1
                    active_start = None
                processing_seconds = float(sum(active_segments))
                if processing_seconds < 0 or not math.isfinite(processing_seconds):
                    stats["rejected_occurrences"] += 1
                    occurrence_index += 1
                    continue
                rows.append(
                    {
                        CASE_COL: case_id,
                        ACTIVITY_COL: activity,
                        "_normalized_activity": normalize_activity(activity),
                        "occurrence_index": occurrence_index,
                        "start_time": occurrence_start_time,
                        "complete_time": timestamp,
                        "processing_seconds": processing_seconds,
                        "waiting_seconds": waiting_seconds,
                        RESOURCE_COL: resource if pd.notna(resource) else occurrence_resource,
                        "EventOrigin": origin if pd.notna(origin) else occurrence_origin,
                        "has_start": pd.notna(occurrence_start_time),
                        "usable_pair": pd.notna(occurrence_start_time),
                        "missing_lifecycle_events": missing_lifecycle_events,
                    }
                )
                occurrence_index += 1
                active_start = None
                active_segments = []
                waiting_start = None
                waiting_seconds = 0.0
                occurrence_start_time = pd.NaT
                occurrence_resource = None
                occurrence_origin = None
            else:
                stats[f"ignored_lifecycle_{lifecycle}"] += 1

    return pd.DataFrame(rows), dict(stats)


def model_key_indexes(models: dict[str, Any]) -> dict[str, Any]:
    basic = models.get("basic", {}) or {}
    fallback = models.get("fallback_basic", {}) or {}
    exact_by_activity: dict[str, list[str]] = defaultdict(list)
    fallback_by_activity: dict[str, str] = {}

    for key in basic:
        if not key.endswith("_processing"):
            continue
        body = key[: -len("_processing")]
        if "_User_" in body:
            activity = body.rsplit("_User_", 1)[0]
        else:
            activity = body
        exact_by_activity[activity].append(key)

    for key in fallback:
        if not key.endswith("_processing"):
            continue
        activity = key[: -len("_processing")]
        fallback_by_activity[activity] = key

    return {
        "exact_by_activity": exact_by_activity,
        "fallback_by_activity": fallback_by_activity,
    }


def quantile(values: pd.Series, q: float) -> float:
    positives = pd.to_numeric(values, errors="coerce").dropna()
    if positives.empty:
        return float("nan")
    return float(positives.quantile(q))


def summarize_activity(
    activity: str,
    df: pd.DataFrame,
    occurrences: pd.DataFrame,
    model_indexes: dict[str, Any],
) -> dict[str, Any]:
    sub = df[df["_normalized_activity"] == normalize_activity(activity)]
    occ = occurrences[occurrences["_normalized_activity"] == normalize_activity(activity)]
    positive = occ[pd.to_numeric(occ["processing_seconds"], errors="coerce") > 0]
    zero_count = int((pd.to_numeric(occ.get("processing_seconds", pd.Series(dtype=float)), errors="coerce") == 0).sum())
    negative_count = 0
    resources = sorted(
        str(value)
        for value in sub.get(RESOURCE_COL, pd.Series(dtype=object)).dropna().unique()
    )
    exact_keys = sorted(model_indexes["exact_by_activity"].get(activity, []))
    fallback_key = model_indexes["fallback_by_activity"].get(activity, f"{activity}_processing")
    has_empirical = not positive.empty
    has_exact = bool(exact_keys)
    has_fallback = fallback_key in model_indexes["fallback_by_activity"].values()
    if has_exact:
        proposed = "exact_resource_model"
    elif has_fallback:
        proposed = "learned_activity_fallback"
    elif has_empirical:
        proposed = "empirical_activity_fallback"
    else:
        proposed = "category_or_global_fallback_required"

    transitions = (
        sub[LIFECYCLE_COL].astype(str).value_counts(dropna=False).to_dict()
        if LIFECYCLE_COL in sub
        else {}
    )
    return {
        "bpmn_activity_name": activity,
        "normalized_activity_name": normalize_activity(activity),
        "occurrences_in_log": int(len(sub)),
        "lifecycle_transitions_observed": json.dumps(transitions, sort_keys=True),
        "start_events": int((sub[LIFECYCLE_COL].astype(str).str.lower() == "start").sum()) if LIFECYCLE_COL in sub else 0,
        "complete_events": int((sub[LIFECYCLE_COL].astype(str).str.lower() == "complete").sum()) if LIFECYCLE_COL in sub else 0,
        "usable_start_complete_pairs": int(occ["usable_pair"].sum()) if "usable_pair" in occ else 0,
        "positive_observed_durations": int(len(positive)),
        "zero_durations": zero_count,
        "negative_durations": negative_count,
        "missing_timestamp_count": int(sub[TIMESTAMP_COL].isna().sum()) if TIMESTAMP_COL in sub else 0,
        "median_positive_duration": quantile(positive["processing_seconds"], 0.50) if not positive.empty else float("nan"),
        "mean_positive_duration": float(positive["processing_seconds"].mean()) if not positive.empty else float("nan"),
        "std_positive_duration": float(positive["processing_seconds"].std(ddof=1)) if len(positive) > 1 else float("nan"),
        "minimum_positive_duration": float(positive["processing_seconds"].min()) if not positive.empty else float("nan"),
        "p05_positive_duration": quantile(positive["processing_seconds"], 0.05) if not positive.empty else float("nan"),
        "p25_positive_duration": quantile(positive["processing_seconds"], 0.25) if not positive.empty else float("nan"),
        "p50_positive_duration": quantile(positive["processing_seconds"], 0.50) if not positive.empty else float("nan"),
        "p75_positive_duration": quantile(positive["processing_seconds"], 0.75) if not positive.empty else float("nan"),
        "p95_positive_duration": quantile(positive["processing_seconds"], 0.95) if not positive.empty else float("nan"),
        "maximum_positive_duration": float(positive["processing_seconds"].max()) if not positive.empty else float("nan"),
        "distinct_resources": int(len(resources)),
        "exact_resource_specific_model_keys": json.dumps(exact_keys),
        "activity_level_fallback_key": fallback_key,
        "has_exact_model": has_exact,
        "has_activity_fallback": has_fallback,
        "has_empirical_positive_observations": has_empirical,
        "proposed_fallback_source": proposed,
    }


def classify_activity(activity: str, sub: pd.DataFrame, coverage: dict[str, Any]) -> tuple[str, str, str]:
    prefix = activity.split("_", 1)[0] if "_" in activity else ""
    transitions = sub[LIFECYCLE_COL].astype(str).str.lower().value_counts(dropna=False).to_dict()
    origins = sub.get("EventOrigin", pd.Series(dtype=object)).astype(str).value_counts(dropna=False).head(5).to_dict()
    resources = sub.get(RESOURCE_COL, pd.Series(dtype=object)).astype(str).value_counts(dropna=False).head(5).to_dict()
    has_start = transitions.get("start", 0) > 0
    positive = int(coverage.get("positive_observed_durations", 0) or 0)
    if prefix == "W":
        classification = "human task" if has_start or positive else "resource-executed task"
    elif prefix in {"A", "O"} and not has_start:
        classification = "message/event-like visible activity"
    elif prefix in {"A", "O"}:
        classification = "resource-executed task"
    else:
        classification = "uncertain"
    evidence = (
        f"prefix={prefix}; transitions={transitions}; top_event_origins={origins}; "
        f"top_resources={resources}; positive_duration_count={positive}; "
        f"has_exact_model={coverage.get('has_exact_model')}; "
        f"has_activity_fallback={coverage.get('has_activity_fallback')}"
    )
    return classification, evidence, prefix or "uncategorized"


def build_fallback_models(
    occurrences: pd.DataFrame,
    visible_activities: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    empirical_activity: dict[str, Any] = {}
    for activity in visible_activities:
        sub = occurrences[
            (occurrences["_normalized_activity"] == normalize_activity(activity))
            & (pd.to_numeric(occurrences["processing_seconds"], errors="coerce") > 0)
        ]
        samples = [float(value) for value in sub["processing_seconds"].tolist()]
        if not samples:
            continue
        empirical_activity[f"{activity}_processing"] = empirical_model(samples, "activity", activity)

    category_members = {
        "application_event_like": [
            activity for activity in visible_activities if activity.startswith("A_")
        ],
        "offer_event_like": [
            activity for activity in visible_activities if activity.startswith("O_")
        ],
        "work_human_task": [
            activity for activity in visible_activities if activity.startswith("W_")
        ],
    }
    category_fallbacks: dict[str, Any] = {}
    for category, members in category_members.items():
        sub = occurrences[
            occurrences["_normalized_activity"].isin([normalize_activity(member) for member in members])
            & (pd.to_numeric(occurrences["processing_seconds"], errors="coerce") > 0)
        ]
        samples = [float(value) for value in sub["processing_seconds"].tolist()]
        if samples:
            category_fallbacks[category] = empirical_model(samples, "category", category)

    global_samples = [
        float(value)
        for value in occurrences.loc[
            pd.to_numeric(occurrences["processing_seconds"], errors="coerce") > 0,
            "processing_seconds",
        ].tolist()
    ]
    global_fallback = empirical_model(global_samples, "global", "global_positive_processing")
    return empirical_activity, category_fallbacks, global_fallback, category_members


def empirical_model(samples: list[float], source_type: str, source_name: str) -> dict[str, Any]:
    cleaned = sorted(float(value) for value in samples if math.isfinite(float(value)) and float(value) > 0)
    if not cleaned:
        return {
            "distribution": "constant",
            "parameters": {"seconds": 1.0},
            "sample_count": 0,
            "median": 1.0,
            "source_type": source_type,
            "source_name": source_name,
            "fit_diagnostics": {"reason": "no_positive_samples"},
        }
    arr = np.array(cleaned, dtype=float)
    trimmed = arr
    if len(arr) >= 20:
        lo, hi = np.quantile(arr, [0.01, 0.99])
        trimmed = arr[(arr >= lo) & (arr <= hi)]
        if len(trimmed) == 0:
            trimmed = arr
    distribution = "empirical" if len(trimmed) >= 10 else "constant_median"
    return {
        "distribution": distribution,
        "parameters": {
            "samples": [float(value) for value in trimmed.tolist()],
            "seconds": float(np.median(trimmed)),
        },
        "sample_count": int(len(arr)),
        "trimmed_sample_count": int(len(trimmed)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else None,
        "minimum": float(np.min(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "maximum": float(np.max(arr)),
        "source_type": source_type,
        "source_name": source_name,
        "fit_diagnostics": {
            "method": "trimmed_empirical" if distribution == "empirical" else "robust_median",
            "unstable_parametric_fits_avoided": True,
        },
    }


def reconciliation_rows(
    visible_activities: list[str],
    log_activities: list[str],
    model_indexes: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    normalized_log = {normalize_activity(activity): activity for activity in log_activities}
    all_model_activities = set(model_indexes["exact_by_activity"]) | set(
        model_indexes["fallback_by_activity"]
    )
    normalized_models = defaultdict(list)
    for activity in all_model_activities:
        normalized_models[normalize_activity(activity)].append(activity)

    for bpmn_activity in visible_activities:
        normalized = normalize_activity(bpmn_activity)
        log_activity = normalized_log.get(normalized, "")
        model_matches = normalized_models.get(normalized, [])
        fallback_key = f"{bpmn_activity}_processing"
        if log_activity and model_matches:
            reason = ""
            normalized_match = True
        elif log_activity and not model_matches:
            reason = "no_processing_model_for_matching_activity"
            normalized_match = False
        elif not log_activity and model_matches:
            reason = "model_activity_not_observed_in_log"
            normalized_match = False
        else:
            reason = "activity_missing_from_log_and_model"
            normalized_match = False
        rows.append(
            {
                "log_activity": log_activity,
                "bpmn_activity": bpmn_activity,
                "model_activity_key": json.dumps(sorted(model_matches)),
                "fallback_key": fallback_key,
                "normalized_match": normalized_match,
                "mismatch_reason": reason,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/logData.xes")
    parser.add_argument("--bpmn-path", default="models/v4_replay.bpmn")
    parser.add_argument("--existing-artifact", default="processTimes/processing_time_models.pkl")
    parser.add_argument("--output-dir", default="joao/results/processing_time_coverage_v2")
    parser.add_argument("--artifact-path", default="joao/models/process_time/final_process_time_coverage_v2.pkl")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = Path(args.artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data_path)
    bpmn_path = Path(args.bpmn_path)
    existing_artifact_path = Path(args.existing_artifact)

    df = load_dataframe(data_path)
    visible_activities = bpmn_visible_activities(bpmn_path)
    existing_models = joblib.load(existing_artifact_path)
    model_indexes = model_key_indexes(existing_models)
    occurrences, pairing_stats = pair_activity_occurrences(df)

    coverage_rows = [
        summarize_activity(activity, df, occurrences, model_indexes)
        for activity in visible_activities
    ]
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(output_dir / "activity_coverage_audit.csv", index=False)

    semantics_rows = []
    for row in coverage_rows:
        activity = row["bpmn_activity_name"]
        sub = df[df["_normalized_activity"] == normalize_activity(activity)]
        classification, evidence, category = classify_activity(activity, sub, row)
        semantics_rows.append(
            {
                "activity": activity,
                "classification": classification,
                "category": category,
                "evidence": evidence,
            }
        )
    pd.DataFrame(semantics_rows).to_csv(output_dir / "activity_semantics.csv", index=False)

    reconciliation = reconciliation_rows(
        visible_activities=visible_activities,
        log_activities=sorted(df[ACTIVITY_COL].dropna().astype(str).unique()),
        model_indexes=model_indexes,
    )
    pd.DataFrame(reconciliation).to_csv(
        output_dir / "activity_key_reconciliation.csv",
        index=False,
    )

    empirical_activity, category_fallbacks, global_fallback, category_members = (
        build_fallback_models(occurrences, visible_activities)
    )

    uncovered = [
        row["bpmn_activity_name"]
        for row in coverage_rows
        if not row["has_exact_model"]
        and not row["has_activity_fallback"]
        and not row["has_empirical_positive_observations"]
    ]
    summary = {
        "visible_bpmn_activity_count": len(visible_activities),
        "source_log_path": str(data_path),
        "source_log_sha256": source_sha256(data_path),
        "existing_processing_time_artifact": str(existing_artifact_path),
        "existing_processing_time_artifact_sha256": source_sha256(existing_artifact_path),
        "exact_model_activity_count": int(coverage_df["has_exact_model"].sum()),
        "learned_activity_fallback_count": int(coverage_df["has_activity_fallback"].sum()),
        "empirical_activity_fallback_count": len(empirical_activity),
        "category_fallback_count": len(category_fallbacks),
        "uncovered_activity_count_before_global": len(uncovered),
        "uncovered_activities_before_global": uncovered,
        "pairing_algorithm": (
            "Events are grouped by case and activity, sorted by timestamp/EventID. "
            "start opens an occurrence, suspend closes an active segment, resume opens "
            "a new active segment after waiting, and complete closes the occurrence. "
            "Completions without a preceding start are retained as zero-duration "
            "unpaired observations and are not used as positive empirical evidence."
        ),
        "pairing_stats": pairing_stats,
        "priority_activities": {
            activity: next(row for row in coverage_rows if row["bpmn_activity_name"] == activity)
            for activity in [
                "A_Create Application",
                "A_Submitted",
                "O_Sent (mail and online)",
            ]
            if activity in visible_activities
        },
    }
    (output_dir / "activity_coverage_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    artifact = {
        "basic": existing_models.get("basic", {}),
        "quantiles": existing_models.get("quantiles", {}),
        "advanced": existing_models.get("advanced", {}),
        "fallback_basic": existing_models.get("fallback_basic", {}),
        "empirical_activity_fallback": empirical_activity,
        "category_fallback": category_fallbacks,
        "activity_to_category": {
            activity: (
                "application_event_like"
                if activity.startswith("A_")
                else "offer_event_like"
                if activity.startswith("O_")
                else "work_human_task"
                if activity.startswith("W_")
                else "uncategorized"
            )
            for activity in visible_activities
        },
        "category_definitions": category_members,
        "global_processing_fallback": global_fallback,
        "metadata": {
            "artifact_kind": "process_time_coverage_v2",
            "creation_timestamp": datetime.now(timezone.utc).isoformat(),
            "source_log_path": str(data_path),
            "source_log_sha256": source_sha256(data_path),
            "environment_versions": package_versions(),
            "seed": args.seed,
            "lifecycle_pairing_method": summary["pairing_algorithm"],
            "exact_model_activity_coverage": int(coverage_df["has_exact_model"].sum()),
            "learned_fallback_activity_coverage": int(coverage_df["has_activity_fallback"].sum()),
            "empirical_activity_fallback_coverage": len(empirical_activity),
            "category_fallback_coverage": len(category_fallbacks),
            "global_fallback_definition": global_fallback,
            "emergency_guard_definition": "MIN_VISIBLE_PROCESSING_DURATION = timedelta(seconds=1), final layer only",
            "uncovered_activities_before_global": uncovered,
            "activity_sample_counts": {
                row["bpmn_activity_name"]: int(row["positive_observed_durations"])
                for row in coverage_rows
            },
            "category_definitions": category_members,
            "model_parameters": {
                "empirical_activity_fallback": empirical_activity,
                "category_fallback": category_fallbacks,
                "global_processing_fallback": global_fallback,
            },
            "artifact_sha256": "computed_after_serialization_in_report",
        },
    }
    joblib.dump(artifact, artifact_path)
    artifact_hash = source_sha256(artifact_path)
    report = [
        "# Processing-Time Coverage v2 Artifact",
        "",
        f"Created: {artifact['metadata']['creation_timestamp']}",
        f"Source log: `{data_path}`",
        f"Source log SHA-256: `{summary['source_log_sha256']}`",
        f"Existing artifact: `{existing_artifact_path}`",
        f"Existing artifact SHA-256: `{summary['existing_processing_time_artifact_sha256']}`",
        f"New artifact: `{artifact_path}`",
        f"New artifact SHA-256: `{artifact_hash}`",
        "",
        "## Pairing Method",
        summary["pairing_algorithm"],
        "",
        "## Coverage",
        f"Visible BPMN activities: {len(visible_activities)}",
        f"Activities with exact resource models: {int(coverage_df['has_exact_model'].sum())}",
        f"Activities with learned activity fallback: {int(coverage_df['has_activity_fallback'].sum())}",
        f"Activities with empirical activity fallback: {len(empirical_activity)}",
        f"Activities uncovered before category/global fallback: {len(uncovered)}",
        "",
        "## Categories",
    ]
    for category, members in category_members.items():
        report.append(f"- `{category}`: {', '.join(members)}")
    report.extend(
        [
            "",
            "## Priority Activities",
        ]
    )
    for activity, row in summary["priority_activities"].items():
        report.append(
            f"- `{activity}`: exact={row['has_exact_model']}, learned_fallback={row['has_activity_fallback']}, "
            f"positive_pairs={row['positive_observed_durations']}, proposed={row['proposed_fallback_source']}"
        )
    (output_dir / "processing_time_artifact_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({**summary, "new_artifact_path": str(artifact_path), "new_artifact_sha256": artifact_hash}, indent=2, default=str))


if __name__ == "__main__":
    main()
