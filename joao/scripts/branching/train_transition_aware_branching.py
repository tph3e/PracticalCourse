from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import pandas as pd
import pm4py

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BPMN_engine import BPMNEngine
from joao.src.resource_allocation.integration.TransitionAwareBranching import (
    TransitionDisambiguationModel,
    save_transition_model,
)


CASE_COL = "case:concept:name"
ACTIVITY_COL = "concept:name"
TIME_COL = "time:timestamp"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_log(log: pd.DataFrame):
    case_times = (
        log.groupby(CASE_COL)[TIME_COL]
        .agg(["min", "max"])
        .sort_values(["min", "max"])
    )
    split_index = int(len(case_times) * 0.7)
    train_cases = set(case_times.index[:split_index])
    test_cases = set(case_times.index[split_index:])
    return (
        log[log[CASE_COL].isin(train_cases)].copy(),
        log[log[CASE_COL].isin(test_cases)].copy(),
        {
            "train_cases": len(train_cases),
            "test_cases": len(test_cases),
            "case_overlap": len(train_cases & test_cases),
            "train_time_range": [
                str(log[log[CASE_COL].isin(train_cases)][TIME_COL].min()),
                str(log[log[CASE_COL].isin(train_cases)][TIME_COL].max()),
            ],
            "test_time_range": [
                str(log[log[CASE_COL].isin(test_cases)][TIME_COL].min()),
                str(log[log[CASE_COL].isin(test_cases)][TIME_COL].max()),
            ],
        },
    )


def ordered_case_ids(log: pd.DataFrame, limit: int) -> list[str]:
    case_ids = list(log.groupby(CASE_COL)[TIME_COL].min().sort_values().index)
    return case_ids[:limit] if limit > 0 else case_ids


def bucket(value: int) -> int:
    if value <= 0:
        return 0
    if value == 1:
        return 1
    if value <= 3:
        return 3
    if value <= 10:
        return 10
    return 99


def replay_observations(
    log: pd.DataFrame,
    case_limit: int,
    learn: bool,
    model: TransitionDisambiguationModel | None = None,
) -> dict:
    engine = BPMNEngine(model_filename="models/v4_replay.bpmn")
    metrics = Counter()
    class_distribution = Counter()
    started = perf_counter()

    for case_id in ordered_case_ids(log, case_limit):
        case_events = log[log[CASE_COL] == case_id].sort_values(TIME_COL)
        activities = [str(activity) for activity in case_events[ACTIVITY_COL]]
        engine.initialize_case(case_id)
        history: list[str] = []
        for index, activity in enumerate(activities):
            candidates = engine.getPossibleNextTransitionCandidates(case_id)
            if not candidates:
                metrics["deadlocked_before_event"] += 1
                break
            metrics["decision_observations"] += 1
            if len(candidates) > 1:
                metrics["multi_candidate_observations"] += 1
            matches = [
                candidate
                for candidate in candidates
                if str(candidate.activity_label) == activity
            ]
            if len(matches) == 1:
                candidate = matches[0]
                metrics["synchronized_observations"] += 1
                class_distribution[str(candidate.transition_id)] += 1
                previous_activity = history[-1] if history else "START"
                prior = history
                visit_count = sum(1 for item in prior if item == activity)
                consecutive = 0
                for item in reversed(prior):
                    if item != activity:
                        break
                    consecutive += 1
                if learn and model is not None:
                    model.observe(
                        transition_id=str(candidate.transition_id),
                        activity_label=str(candidate.activity_label),
                        marking_signature=str(candidate.source_marking),
                        previous_activity=previous_activity,
                        current_activity=activity,
                        visit_count_bucket=bucket(visit_count),
                        repetition_bucket=bucket(consecutive),
                    )
                fired = engine.fire_transition_candidate(case_id, candidate)
                metrics["exact_transition_fires"] += int(fired)
                if not fired:
                    metrics["fire_failures"] += 1
                    break
            elif len(matches) > 1:
                metrics["ambiguous_observations_skipped"] += 1
                break
            else:
                metrics["nonconformant_observations_skipped"] += 1
                break
            history.append(activity)
        if engine.is_final_marking(case_id) or engine.can_reach_final_by_silent_path(case_id):
            metrics["cases_final_reachable"] += 1

    metrics["runtime_seconds"] = round(perf_counter() - started, 3)
    return {
        "metrics": dict(metrics),
        "class_distribution": dict(class_distribution),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="data/logData.xes")
    parser.add_argument("--train-case-limit", type=int, default=1500)
    parser.add_argument("--test-case-limit", type=int, default=500)
    parser.add_argument(
        "--artifact",
        default="joao/models/branching/transition_aware_branching_v1_20260715.pkl",
    )
    parser.add_argument(
        "--output-dir",
        default="joao/results/transition_aware_branching_20260715",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    log = pm4py.read_xes(str(log_path), variant="r4pm")
    log[TIME_COL] = pd.to_datetime(log[TIME_COL], utc=True)
    log = log.sort_values([CASE_COL, TIME_COL]).reset_index(drop=True)
    train_log, test_log, split = split_log(log)

    model = TransitionDisambiguationModel(
        metadata={
            "created_at": datetime.utcnow().isoformat() + "Z",
            "source_model": "models/v4_replay.bpmn",
            "source_log_sha256": sha256(log_path),
            "train_case_limit": args.train_case_limit,
            "test_case_limit": args.test_case_limit,
            "split": split,
            "uses_test_data_for_training": False,
        }
    )
    train_report = replay_observations(
        train_log,
        case_limit=args.train_case_limit,
        learn=True,
        model=model,
    )
    test_report = replay_observations(
        test_log,
        case_limit=args.test_case_limit,
        learn=False,
        model=model,
    )
    artifact = Path(args.artifact)
    save_transition_model(model, artifact)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "artifact": str(artifact),
        "artifact_sha256": sha256(artifact),
        "split": split,
        "train_report": train_report,
        "held_out_report": test_report,
        "fixed_replay_outputs_modified": False,
    }
    (output_dir / "transition_training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
