from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pm4py
from scipy.stats import t as student_t

JOAO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = JOAO_ROOT.parent
sys.path.insert(0, str(JOAO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from joao.src.branching.CompositeBranchingArtifact import (  # noqa: E402
    artifact_sha256,
    export_composite_branching_artifact,
    load_artifact_payload,
    load_composite_branching_artifact,
)
from joao.src.branching.BranchingUtils import temporal_train_test_split_by_case  # noqa: E402
from joao.src.branching.CompositeBranchingEngine import CompositeBranchingEngine  # noqa: E402
from joao.src.resource_allocation.BatchAllocationAdapter import BatchAllocationAdapter  # noqa: E402
from joao.src.resource_allocation.KunklerAllocationAdapter import KunklerAllocationAdapter  # noqa: E402
from joao.src.resource_allocation.ParkSongAllocation import ParkSongAllocation  # noqa: E402
from joao.src.resource_allocation.PickInterfaceAllocationAdapter import (  # noqa: E402
    PickInterfaceAllocationAdapter,
)
from joao.src.resource_allocation.RandomResourceAllocation import RandomResourceAllocation  # noqa: E402
from joao.src.resource_allocation.RoundRobinResourceAllocation import (  # noqa: E402
    RoundRobinResourceAllocation,
)
from joao.src.resource_allocation.ShortestQueueAllocation import ShortestQueueAllocation  # noqa: E402
from joao.src.resource_allocation.integration.IntegratedAllocationEngine import (  # noqa: E402
    IntegratedAllocationEngine,
)
from joao.src.resource_allocation.integration.WeightedFairnessAdapter import (  # noqa: E402
    compute_weighted_fairness_from_engine,
)

CASE_COL = "case:concept:name"
ACTIVITY_COL = "concept:name"
RESOURCE_COL = "org:resource"
TIMESTAMP_COL = "time:timestamp"
LIFECYCLE_COL = "lifecycle:transition"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def method_inventory(repo_root: Path) -> list[dict[str, str]]:
    rl_policy = repo_root / "results" / "rl_policy.json"
    rl_policy_sim = repo_root / "results" / "rl_policy_sim.json"
    ppo_model = repo_root / "results" / "ppo_model.zip"
    return [
        {
            "canonical_method": "Random",
            "aliases": "RandomAllocation;random",
            "source_path": "joao/src/resource_allocation/RandomResourceAllocation.py;resources/allocation.py",
            "owner": "Joao wrapper plus group baseline",
            "allocation_type": "pull/global baseline",
            "required_inputs": "eligible resources,current waiting tasks,seed",
            "required_artifacts": "none",
            "simulator_path": "IntegratedAllocationEngine",
            "tests_available": "joao/tests/resource_allocation/test_random_resource_allocation.py",
            "fairly_comparable": "yes",
            "distinct_method": "yes",
            "final_group": "primary",
            "status": "included",
            "notes": "Uses the same global allocation lifecycle as the other final methods.",
        },
        {
            "canonical_method": "RoundRobin",
            "aliases": "R-RRA",
            "source_path": "joao/src/resource_allocation/RoundRobinResourceAllocation.py",
            "owner": "Joao",
            "allocation_type": "pull/global deterministic rotation",
            "required_inputs": "eligible resources,current waiting tasks",
            "required_artifacts": "none",
            "simulator_path": "IntegratedAllocationEngine",
            "tests_available": "joao/tests/resource_allocation/test_round_robin_resource_allocation.py",
            "fairly_comparable": "yes",
            "distinct_method": "yes",
            "final_group": "primary",
            "status": "included",
            "notes": "Deterministic resource-id rotation over currently available eligible resources.",
        },
        {
            "canonical_method": "ShortestQueue",
            "aliases": "R-SHQ",
            "source_path": "joao/src/resource_allocation/ShortestQueueAllocation.py",
            "owner": "Joao",
            "allocation_type": "pull/global load-balancing",
            "required_inputs": "eligible resources,current waiting tasks,resource loads",
            "required_artifacts": "none",
            "simulator_path": "IntegratedAllocationEngine",
            "tests_available": "joao/tests/resource_allocation/test_shortest_queue_allocation.py",
            "fairly_comparable": "yes",
            "distinct_method": "yes",
            "final_group": "primary",
            "status": "included",
            "notes": "Uses ResourceEngine cumulative load as the current queue/load proxy.",
        },
        {
            "canonical_method": "ParkSong-NoPrediction",
            "aliases": "plain ParkSong",
            "source_path": "joao/src/resource_allocation/ParkSongAllocation.py",
            "owner": "Joao",
            "allocation_type": "pull/global cost heuristic",
            "required_inputs": "eligible resources,current waiting tasks",
            "required_artifacts": "none",
            "simulator_path": "IntegratedAllocationEngine",
            "tests_available": "joao/tests/resource_allocation/test_park_song_allocation.py",
            "fairly_comparable": "yes",
            "distinct_method": "yes",
            "final_group": "ablation",
            "status": "included",
            "notes": "Same cost heuristic with strategic idling disabled; no reservations expected.",
        },
        {
            "canonical_method": "ParkSong-Composite",
            "aliases": "ParkSong;prediction-aware ParkSong",
            "source_path": "joao/src/resource_allocation/ParkSongAllocation.py;joao/src/resource_allocation/integration/CompositeBranchingAdapter.py",
            "owner": "Joao",
            "allocation_type": "prediction-aware,reservation-based",
            "required_inputs": "eligible resources,current waiting tasks,Composite predictions",
            "required_artifacts": "joao/models/branching/final_composite_branching.pkl",
            "simulator_path": "IntegratedAllocationEngine",
            "tests_available": "joao/tests/resource_allocation/test_integrated_allocation_engine.py",
            "fairly_comparable": "yes",
            "distinct_method": "yes",
            "final_group": "primary",
            "status": "included",
            "notes": "Consumes current CompositeBranchingAdapter predictions and creates reservations.",
        },
        {
            "canonical_method": "ParkSongML",
            "aliases": "MLPredictionAdapter;ParkSongMLIntegration",
            "source_path": "joao/src/resource_allocation/ParkSongMLIntegration.py;joao/src/resource_allocation/MLPredictionAdapter.py",
            "owner": "Joao",
            "allocation_type": "prediction supplier plus ParkSong allocator",
            "required_inputs": "event,possible activities,ML prediction model",
            "required_artifacts": "branching model if configured",
            "simulator_path": "not separate from ParkSong-Composite in final IntegratedAllocationEngine",
            "tests_available": "joao/tests/resource_allocation/test_parksong_ml_integration.py",
            "fairly_comparable": "no as separate final strategy",
            "distinct_method": "no",
            "final_group": "documented variant",
            "status": "excluded",
            "notes": "Repository evidence shows it as an integration layer/prediction supplier, not a separate simulator strategy.",
        },
        {
            "canonical_method": "Batch",
            "aliases": "BatchAllocation;BatchAllocationEngine",
            "source_path": "BatchAllocationEngine.py;joao/src/resource_allocation/BatchAllocationAdapter.py",
            "owner": "group implementation with Joao adapter",
            "allocation_type": "batch/current-queue snapshot",
            "required_inputs": "eligible resources,current waiting tasks,k_limit",
            "required_artifacts": "none",
            "simulator_path": "IntegratedAllocationEngine via BatchAllocationAdapter",
            "tests_available": "joao/tests/resource_allocation/test_batch_allocation_adapter.py",
            "fairly_comparable": "yes with noted decision-epoch limitation",
            "distinct_method": "yes",
            "final_group": "primary",
            "status": "included",
            "notes": "Adapter preserves group engine assignment rule but fires per integrated decision epoch.",
        },
        {
            "canonical_method": "Kunkler-Rinderle-Ma",
            "aliases": "Künstler;Kunkler;AnticipatoryAssignmentAllocator",
            "source_path": "resourceAllocation_KunklerRinderleMa.py;joao/src/resource_allocation/KunklerAllocationAdapter.py;notebooks/2.3.1_formalization_kunkler.ipynb",
            "owner": "group/reference implementation with Joao adapter",
            "allocation_type": "anticipatory assignment/cost matrix",
            "required_inputs": "eligible resources,current waiting tasks,processing-time quantile shim",
            "required_artifacts": "none identified",
            "simulator_path": "IntegratedAllocationEngine via KunklerAllocationAdapter",
            "tests_available": "joao/tests/resource_allocation/test_kunkler_allocation_adapter.py",
            "fairly_comparable": "yes with implementation limitations noted",
            "distinct_method": "yes",
            "final_group": "primary",
            "status": "included",
            "notes": "Adapter invokes the original allocator and enforces integrated eligibility/assignment semantics; root class cost-matrix limitations are diagnosed.",
        },
        {
            "canonical_method": "RL-REINFORCE",
            "aliases": "RLAllocation;rl_policy.json",
            "source_path": "optimization/rl_agent.py;results/rl_policy.json",
            "owner": "group creative/experimental",
            "allocation_type": "experimental policy-gradient pick strategy",
            "required_inputs": "current activity,candidates,load,busy fraction",
            "required_artifacts": "results/rl_policy.json",
            "simulator_path": "IntegratedAllocationEngine via PickInterfaceAllocationAdapter",
            "tests_available": "none found",
            "fairly_comparable": "yes if artifact loads",
            "distinct_method": "yes",
            "final_group": "creative appendix",
            "status": "included" if rl_policy.exists() else "excluded",
            "notes": "Uses current state only; no retraining is performed.",
        },
        {
            "canonical_method": "RL-Sim",
            "aliases": "rl_policy_sim.json",
            "source_path": "optimization/rl_agent.py;results/rl_policy_sim.json",
            "owner": "group creative/experimental",
            "allocation_type": "experimental sim-trained policy-gradient pick strategy",
            "required_inputs": "current activity,candidates,load,busy fraction",
            "required_artifacts": "results/rl_policy_sim.json",
            "simulator_path": "IntegratedAllocationEngine via PickInterfaceAllocationAdapter",
            "tests_available": "none found",
            "fairly_comparable": "yes if artifact loads",
            "distinct_method": "yes",
            "final_group": "creative appendix",
            "status": "included" if rl_policy_sim.exists() else "excluded",
            "notes": "Uses current state only; no retraining is performed.",
        },
        {
            "canonical_method": "PPO",
            "aliases": "MaskablePPO;PPOAllocation",
            "source_path": "optimization/ppo_agent.py;results/ppo_model.zip",
            "owner": "group creative/experimental",
            "allocation_type": "experimental PPO pick strategy",
            "required_inputs": "Gym-style observation and action mask",
            "required_artifacts": "results/ppo_model.zip plus sb3_contrib runtime",
            "simulator_path": "group Engine path; adapter possible only if dependencies load",
            "tests_available": "none found",
            "fairly_comparable": "not in default run",
            "distinct_method": "yes",
            "final_group": "creative appendix",
            "status": "excluded",
            "notes": "Artifact exists but requires optional sb3_contrib stack; not included by default to avoid dependency-driven failure.",
        },
    ]


def canonical_strategy_names(inventory: list[dict[str, str]]) -> list[str]:
    return [
        row["canonical_method"]
        for row in inventory
        if row["status"] == "included"
        and row["canonical_method"] != "ParkSongML"
    ]


def canonical_inventory_frame(inventory: list[dict[str, str]], selected: list[str]) -> pd.DataFrame:
    selected_set = set(selected)
    rows = []
    for row in inventory:
        method = row["canonical_method"]
        is_primary = method in selected_set
        is_parksong_ml = method == "ParkSongML"
        is_experimental = method in {"RL-REINFORCE", "RL-Sim", "PPO", "Random", "ParkSong-NoPrediction"}
        implemented = row.get("source_path", "") != ""
        artifact_required = row.get("required_artifacts", "none") != "none"
        artifact_available = (not artifact_required) or all(
            (REPO_ROOT / part.strip()).exists()
            for part in row.get("required_artifacts", "").split(" plus ")[0].split(";")
            if part.strip() and not part.strip().startswith("none")
        )
        rows.append(
            {
                "canonical_method": method,
                "aliases": row.get("aliases", ""),
                "source_path": row.get("source_path", ""),
                "owner": row.get("owner", ""),
                "implemented": implemented,
                "artifact_available": artifact_available,
                "adapter_available": "Adapter" in row.get("source_path", "") or "adapter" in row.get("simulator_path", "").lower(),
                "tested_unit": "test_" in row.get("tests_available", ""),
                "tested_integration": row.get("simulator_path", "") == "IntegratedAllocationEngine" or "IntegratedAllocationEngine" in row.get("simulator_path", ""),
                "evaluated_fixed_replay_final": is_primary,
                "evaluated_generative_final": is_primary,
                "reported_main_table": is_primary,
                "reported_appendix": is_primary or is_experimental or is_parksong_ml,
                "final_role": (
                    "integration_layer" if is_parksong_ml
                    else "primary" if is_primary and method == "ParkSong-Composite"
                    else "baseline" if is_primary and method in {"RoundRobin", "ShortestQueue"}
                    else "group_owned_reference" if is_primary and method in {"Kunkler-Rinderle-Ma", "Batch"}
                    else "ablation" if method == "ParkSong-NoPrediction"
                    else "experimental" if is_experimental
                    else "excluded_dependency"
                ),
                "notes": row.get("notes", ""),
            }
        )
    return pd.DataFrame(rows)


def parse_strategy_filter(strategy_filter: str | None, available: list[str]) -> list[str]:
    if not strategy_filter:
        return list(available)
    requested = [part.strip() for part in strategy_filter.split(",") if part.strip()]
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise ValueError(
            f"Unknown strategy name(s): {', '.join(unknown)}. "
            f"Valid strategies: {', '.join(available)}"
        )
    return requested


PARKSONG_PARAM_TYPES = {
    "cost_time_scale": float,
    "no_show_penalty_weight": float,
    "processing_time_weight": float,
    "prediction_probability_threshold": float,
    "uncertainty_weight": float,
    "idling_weight": float,
    "waiting_weight": float,
    "priority_weight": float,
    "planning_horizon": float,
    "future_delay_weight": float,
    "reservation_margin": float,
}


def parse_parksong_params(raw_params: str | None) -> dict[str, float]:
    if not raw_params:
        return {}

    params: dict[str, float] = {}
    for item in raw_params.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "ParkSong params must use key=value pairs separated by commas."
            )
        key, raw_value = [part.strip() for part in item.split("=", 1)]
        if key not in PARKSONG_PARAM_TYPES:
            valid = ", ".join(sorted(PARKSONG_PARAM_TYPES))
            raise ValueError(f"Unknown ParkSong parameter {key!r}. Valid: {valid}")
        params[key] = PARKSONG_PARAM_TYPES[key](raw_value)
    return params


def _string_or_none(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def parksong_processing_time_estimates_from_log(
    log: pd.DataFrame,
) -> tuple[dict[tuple[str, str], float], float, dict[str, Any]]:
    df = log.copy()
    if df.empty or not {CASE_COL, ACTIVITY_COL, RESOURCE_COL, LIFECYCLE_COL, TIMESTAMP_COL}.issubset(df.columns):
        return {}, 1.0, {
            "processing_time_estimate_mode": "empty_or_missing_columns",
            "estimate_count": 0,
            "positive_occurrence_count": 0,
            "default_processing_time": 1.0,
        }

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    df = df.dropna(subset=[TIMESTAMP_COL])
    rows: list[dict[str, Any]] = []
    sort_cols = [CASE_COL, ACTIVITY_COL, TIMESTAMP_COL, "EventID"]
    existing_sort_cols = [column for column in sort_cols if column in df.columns]

    for (_case_id, activity), group in df.sort_values(existing_sort_cols).groupby(
        [CASE_COL, ACTIVITY_COL],
        dropna=False,
    ):
        active_start = None
        processing_seconds = 0.0
        occurrence_resource = None

        for _, event in group.iterrows():
            lifecycle = str(event.get(LIFECYCLE_COL, "") or "").lower()
            timestamp = event[TIMESTAMP_COL]
            resource = _string_or_none(event.get(RESOURCE_COL))

            if lifecycle == "start":
                active_start = timestamp
                processing_seconds = 0.0
                occurrence_resource = resource
            elif lifecycle == "resume":
                active_start = timestamp
                if resource is not None:
                    occurrence_resource = resource
            elif lifecycle == "suspend":
                if active_start is not None:
                    segment = (timestamp - active_start).total_seconds()
                    if math.isfinite(segment) and segment >= 0:
                        processing_seconds += segment
                    active_start = None
            elif lifecycle == "complete":
                if active_start is not None:
                    segment = (timestamp - active_start).total_seconds()
                    if math.isfinite(segment) and segment >= 0:
                        processing_seconds += segment
                    active_start = None
                final_resource = resource or occurrence_resource
                if final_resource and processing_seconds > 0 and math.isfinite(processing_seconds):
                    rows.append(
                        {
                            RESOURCE_COL: final_resource,
                            ACTIVITY_COL: str(activity),
                            "processing_seconds": float(processing_seconds),
                        }
                    )
                processing_seconds = 0.0
                occurrence_resource = None

    occurrences = pd.DataFrame(rows)
    if occurrences.empty:
        return {}, 1.0, {
            "processing_time_estimate_mode": "no_positive_pairs",
            "estimate_count": 0,
            "positive_occurrence_count": 0,
            "default_processing_time": 1.0,
        }

    grouped = occurrences.groupby([RESOURCE_COL, ACTIVITY_COL])["processing_seconds"].median()
    estimates = {
        (str(resource), str(activity)): float(seconds)
        for (resource, activity), seconds in grouped.items()
        if math.isfinite(float(seconds)) and float(seconds) > 0
    }
    default_processing_time = float(occurrences["processing_seconds"].median())
    if not math.isfinite(default_processing_time) or default_processing_time <= 0:
        default_processing_time = 1.0
    summary = {
        "processing_time_estimate_mode": "train_median_by_resource_activity",
        "estimate_count": len(estimates),
        "positive_occurrence_count": int(len(occurrences)),
        "resource_count": int(occurrences[RESOURCE_COL].nunique()),
        "activity_count": int(occurrences[ACTIVITY_COL].nunique()),
        "default_processing_time": default_processing_time,
        "median_distribution_seconds": describe_numeric(list(estimates.values())),
    }
    return estimates, default_processing_time, summary


def build_parksong_processing_time_estimates(
    data_path: str | Path,
    split_ratio: float,
) -> tuple[dict[tuple[str, str], float], float, dict[str, Any]]:
    log = load_event_log(data_path)
    train_log, _test_log = temporal_train_test_split_by_case(
        log,
        train_ratio=split_ratio,
        case_col=CASE_COL,
        timestamp_col=TIMESTAMP_COL,
    )
    estimates, default_processing_time, summary = parksong_processing_time_estimates_from_log(train_log)
    summary.update(
        {
            "split_ratio": split_ratio,
            "training_event_count": int(len(train_log)),
            "evaluation_log_used_for_training_constraints": False,
        }
    )
    return estimates, default_processing_time, summary


def build_strategy(
    name: str,
    seed: int,
    parksong_params: dict[str, float] | None = None,
    parksong_processing_time_estimates: dict[tuple[str, str], float] | None = None,
    parksong_default_processing_time: float = 1.0,
):
    parksong_params = parksong_params or {}
    if name == "Random":
        return RandomResourceAllocation(seed=seed)
    if name == "RoundRobin":
        return RoundRobinResourceAllocation()
    if name == "ShortestQueue":
        return ShortestQueueAllocation()
    if name == "ParkSong-NoPrediction":
        return ParkSongAllocation(
            allow_strategic_idling=False,
            processing_time_estimates=parksong_processing_time_estimates,
            default_processing_time=parksong_default_processing_time,
            **parksong_params,
        )
    if name == "ParkSong-Composite":
        composite_defaults = {"waiting_weight": 0.5}
        composite_defaults.update(parksong_params)
        return ParkSongAllocation(
            allow_strategic_idling=True,
            processing_time_estimates=parksong_processing_time_estimates,
            default_processing_time=parksong_default_processing_time,
            **composite_defaults,
        )
    if name == "Kunkler-Rinderle-Ma":
        return KunklerAllocationAdapter(seed=seed)
    if name == "Batch":
        return BatchAllocationAdapter(k_limit=5)
    if name in {"RL-REINFORCE", "RL-Sim"}:
        from optimization.rl_agent import RLAllocation

        artifact = "rl_policy.json" if name == "RL-REINFORCE" else "rl_policy_sim.json"
        return PickInterfaceAllocationAdapter(
            RLAllocation.load(str(REPO_ROOT / "results" / artifact)),
            name,
        )
    raise ValueError(f"No builder registered for strategy {name}")


def lifecycle_metrics(
    log: pd.DataFrame,
    engine: IntegratedAllocationEngine,
    start: datetime,
    end: datetime,
    observation_end: datetime | None = None,
) -> dict[str, Any]:
    if log.empty:
        return empty_metrics()
    df = log.copy()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    df = df.dropna(subset=[TIMESTAMP_COL])
    if df.empty:
        return empty_metrics()

    observation_end = observation_end or end
    admitted_case_ids = set(getattr(engine, "admitted_case_ids", set()))
    completed_case_ids = set(getattr(engine, "completed_case_ids", set()))
    deadlocked_case_ids = set(getattr(engine, "deadlocked_case_ids", set()))
    cyclic_case_ids = set(getattr(engine, "cyclic_case_ids", set()))
    censored_case_ids = set(getattr(engine, "censored_case_ids", set()))
    terminal_ids = completed_case_ids | deadlocked_case_ids | cyclic_case_ids | censored_case_ids
    active_case_ids = admitted_case_ids - terminal_ids

    cycle = completed_case_cycle_durations(engine, completed_case_ids)
    wait = lifecycle_waiting_durations(engine)
    resource_intervals = busy_intervals(df)
    busy_by_resource = union_busy_seconds(resource_intervals)
    run_seconds = max((observation_end - start).total_seconds(), 0.0)
    assigned = df[RESOURCE_COL].fillna("").astype(str).str.len() > 0
    admitted_cases = max(len(admitted_case_ids), int(df[CASE_COL].nunique()))
    completed_cases = len(completed_case_ids)
    incomplete_cases = max(0, admitted_cases - completed_cases)
    weighted = compute_weighted_fairness_from_engine(engine=engine, start_time=start, end_time=observation_end)
    utilization_values = [
        seconds / run_seconds
        for seconds in busy_by_resource.values()
        if run_seconds > 0
    ]
    metrics = {
        "n_events": int(len(df)),
        "cases_admitted": int(admitted_cases),
        "cases_observed": int(df[CASE_COL].nunique()),
        "fixed_routes_completed": int(completed_cases),
        "cases_completed": int(completed_cases),
        "cases_incomplete": int(incomplete_cases),
        "cases_deadlocked": int(len(deadlocked_case_ids)),
        "cases_cyclic": int(len(cyclic_case_ids)),
        "cases_censored": int(len(censored_case_ids)),
        "cases_active": int(len(active_case_ids)),
        "completed_case_ids": ";".join(sorted(completed_case_ids)),
        "deadlocked_case_ids": ";".join(sorted(deadlocked_case_ids)),
        "cyclic_case_ids": ";".join(sorted(cyclic_case_ids)),
        "censored_case_ids": ";".join(sorted(censored_case_ids)),
        "active_case_ids": ";".join(sorted(active_case_ids)),
        "fixed_route_completion_rate": completed_cases / admitted_cases if admitted_cases else float("nan"),
        "completion_rate": completed_cases / admitted_cases if admitted_cases else float("nan"),
        "tasks_assigned": int(assigned.sum()),
        "cycle_time_mean_s": safe_stat(cycle, "mean"),
        "cycle_time_median_s": safe_stat(cycle, "median"),
        "cycle_time_std_s": safe_stat(cycle, "std"),
        "cycle_time_p90_s": safe_quantile(cycle, 0.90),
        "cycle_time_p95_s": safe_quantile(cycle, 0.95),
        "waiting_time_mean_s": safe_stat(wait, "mean"),
        "waiting_time_median_s": safe_stat(wait, "median"),
        "waiting_time_p90_s": safe_quantile(wait, 0.90),
        "waiting_time_p95_s": safe_quantile(wait, 0.95),
        "horizon_normalized_throughput_cases_per_hour": completed_cases / (run_seconds / 3600.0) if run_seconds > 0 else float("nan"),
        "throughput_cases_per_hour": completed_cases / (run_seconds / 3600.0) if run_seconds > 0 else float("nan"),
        "horizon_normalized_resource_occupation_mean": float(np.mean(utilization_values)) if utilization_values else float("nan"),
        "resource_occupation_mean": float(np.mean(utilization_values)) if utilization_values else float("nan"),
        "horizon_normalized_resource_occupation_median": float(np.median(utilization_values)) if utilization_values else float("nan"),
        "resource_occupation_median": float(np.median(utilization_values)) if utilization_values else float("nan"),
        "resource_fairness_gini": gini(list(busy_by_resource.values())),
        "weighted_resource_fairness": weighted.get("weighted_resource_fairness", float("nan")),
        "weighted_resource_fairness_status": weighted.get("weighted_resource_fairness_status", ""),
        "weighted_fairness_resource_count": weighted.get("weighted_fairness_resource_count", 0),
        "active_cases_remaining": int(len(active_case_ids)),
        "waiting_queue_remaining": int(len(getattr(engine, "waiting_processes", []))),
        "drain_stopped_by_limit": int(bool(getattr(engine, "drain_stopped_by_limit", False))),
        "run_window_seconds": run_seconds,
        "throughput_denominator": "arrival_plus_drain_horizon",
    }
    return metrics


def empty_metrics() -> dict[str, Any]:
    return {
        "n_events": 0,
        "cases_admitted": 0,
        "cases_observed": 0,
        "fixed_routes_completed": 0,
        "cases_completed": 0,
        "cases_incomplete": 0,
        "cases_deadlocked": 0,
        "cases_cyclic": 0,
        "cases_censored": 0,
        "cases_active": 0,
        "completed_case_ids": "",
        "deadlocked_case_ids": "",
        "cyclic_case_ids": "",
        "censored_case_ids": "",
        "active_case_ids": "",
        "fixed_route_completion_rate": float("nan"),
        "completion_rate": float("nan"),
        "tasks_assigned": 0,
        "cycle_time_mean_s": float("nan"),
        "cycle_time_median_s": float("nan"),
        "cycle_time_std_s": float("nan"),
        "cycle_time_p90_s": float("nan"),
        "cycle_time_p95_s": float("nan"),
        "waiting_time_mean_s": float("nan"),
        "waiting_time_median_s": float("nan"),
        "waiting_time_p90_s": float("nan"),
        "waiting_time_p95_s": float("nan"),
        "horizon_normalized_throughput_cases_per_hour": 0.0,
        "throughput_cases_per_hour": 0.0,
        "horizon_normalized_resource_occupation_mean": float("nan"),
        "resource_occupation_mean": float("nan"),
        "horizon_normalized_resource_occupation_median": float("nan"),
        "resource_occupation_median": float("nan"),
        "resource_fairness_gini": float("nan"),
        "weighted_resource_fairness": float("nan"),
        "weighted_resource_fairness_status": "empty_log",
        "weighted_fairness_resource_count": 0,
        "active_cases_remaining": 0,
        "waiting_queue_remaining": 0,
        "drain_stopped_by_limit": 0,
        "run_window_seconds": 0.0,
        "throughput_denominator": "arrival_plus_drain_horizon",
    }


def completed_case_cycle_durations(engine: IntegratedAllocationEngine, completed_case_ids: set[str]) -> pd.Series:
    durations: list[float] = []
    lifecycles = list(getattr(engine, "task_lifecycle", {}).values())
    for case_id in completed_case_ids:
        case_lifecycles = [lifecycle for lifecycle in lifecycles if lifecycle.case_id == case_id]
        start_candidates = [
            lifecycle.enabled_time
            for lifecycle in case_lifecycles
            if getattr(lifecycle, "enabled_time", None) is not None
        ]
        end_candidates = [
            lifecycle.processing_end_time
            for lifecycle in case_lifecycles
            if getattr(lifecycle, "processing_end_time", None) is not None
        ]
        if not start_candidates or not end_candidates:
            continue
        seconds = (max(end_candidates) - min(start_candidates)).total_seconds()
        if math.isfinite(seconds) and seconds >= 0:
            durations.append(seconds)
    return pd.Series(durations, dtype=float)


def lifecycle_waiting_durations(engine: IntegratedAllocationEngine) -> pd.Series:
    waits: list[float] = []
    for lifecycle in getattr(engine, "task_lifecycle", {}).values():
        start = getattr(lifecycle, "resource_queue_entry_time", None)
        end = getattr(lifecycle, "resource_assignment_time", None)
        if start is None:
            start = getattr(lifecycle, "process_wait_start", None)
        if end is None:
            end = getattr(lifecycle, "process_wait_end", None)
        if start is None or end is None:
            continue
        seconds = (end - start).total_seconds()
        if math.isfinite(seconds) and seconds >= 0:
            waits.append(seconds)
    return pd.Series(waits, dtype=float)


def waiting_durations(df: pd.DataFrame) -> pd.Series:
    if LIFECYCLE_COL not in df:
        return pd.Series(dtype=float)
    waits = []
    for _, group in df.sort_values(TIMESTAMP_COL).groupby([CASE_COL, ACTIVITY_COL], dropna=False):
        starts = group[group[LIFECYCLE_COL].isin(["start", "suspend"])][TIMESTAMP_COL].tolist()
        resumes = group[group[LIFECYCLE_COL].isin(["resume", "complete"])][TIMESTAMP_COL].tolist()
        for start, resume in zip(starts, resumes):
            seconds = (resume - start).total_seconds()
            if math.isfinite(seconds) and seconds >= 0:
                waits.append(seconds)
    return pd.Series(waits, dtype=float)


def busy_intervals(df: pd.DataFrame) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    intervals = []
    starts = df[df[LIFECYCLE_COL].isin(["start", "resume"])].copy()
    ends = df[df[LIFECYCLE_COL] == "complete"].copy()
    starts["occ"] = starts.groupby([CASE_COL, ACTIVITY_COL]).cumcount()
    ends["occ"] = ends.groupby([CASE_COL, ACTIVITY_COL]).cumcount()
    merged = starts.merge(ends, on=[CASE_COL, ACTIVITY_COL, "occ"], suffixes=("_s", "_e"))
    for _, row in merged.iterrows():
        resource = str(row.get(f"{RESOURCE_COL}_s", "") or "")
        start = row[f"{TIMESTAMP_COL}_s"]
        end = row[f"{TIMESTAMP_COL}_e"]
        if resource and pd.notna(start) and pd.notna(end) and end >= start:
            intervals.append((resource, start, end))
    return intervals


def union_busy_seconds(intervals: list[tuple[str, pd.Timestamp, pd.Timestamp]]) -> dict[str, float]:
    by_resource: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for resource, start, end in intervals:
        by_resource.setdefault(resource, []).append((start, end))
    totals = {}
    for resource, values in by_resource.items():
        total = 0.0
        current_start = current_end = None
        for start, end in sorted(values):
            if current_end is None:
                current_start, current_end = start, end
            elif start <= current_end:
                current_end = max(current_end, end)
            else:
                total += (current_end - current_start).total_seconds()
                current_start, current_end = start, end
        if current_end is not None:
            total += (current_end - current_start).total_seconds()
        totals[resource] = total
    return totals


def safe_stat(series: pd.Series, name: str) -> float:
    if series.empty:
        return float("nan")
    value = getattr(series, name)()
    return float(value) if pd.notna(value) else float("nan")


def safe_quantile(series: pd.Series, q: float) -> float:
    if series.empty:
        return float("nan")
    value = series.quantile(q)
    return float(value) if pd.notna(value) else float("nan")


def describe_numeric(values: list[float] | list[int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    series = pd.Series(values, dtype=float)
    return {
        "count": int(series.count()),
        "min": float(series.min()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
    }


def gini(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0 or array.sum() == 0:
        return float("nan")
    array = np.sort(array)
    cumulative = np.cumsum(array)
    return float((array.size + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / array.size)


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    numeric = raw.select_dtypes(include=[np.number]).columns.tolist()
    excluded = {
        "seed",
        "completed_case_ids",
        "deadlocked_case_ids",
        "cyclic_case_ids",
        "censored_case_ids",
        "active_case_ids",
    }
    numeric = [column for column in numeric if column not in excluded and not column.endswith("_id")]
    grouped = raw.groupby("strategy")[numeric].agg(["mean", "std", "median", "min", "max"])
    grouped.columns = ["_".join(column).strip("_") for column in grouped.columns]
    grouped = grouped.reset_index()
    grouped["seed_count"] = raw.groupby("strategy")["seed"].nunique().values
    for metric in [
        "cycle_time_mean_s",
        "waiting_time_mean_s",
        "horizon_normalized_throughput_cases_per_hour",
        "horizon_normalized_resource_occupation_mean",
        "resource_fairness_gini",
        "weighted_resource_fairness",
        "fixed_route_completion_rate",
    ]:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col in grouped and std_col in grouped:
            n = grouped["seed_count"].replace(0, np.nan)
            critical = n.apply(lambda value: student_t.ppf(0.975, df=int(value) - 1) if pd.notna(value) and value >= 2 else np.nan)
            grouped[f"{metric}_ci95_half_width"] = critical * grouped[std_col] / np.sqrt(n)
    grouped["ci_method"] = "student_t"
    grouped["ci_level"] = 0.95
    grouped["n_runs"] = grouped["seed_count"]
    return grouped


def paired_comparisons(raw: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("RoundRobin", "Random"),
        ("ShortestQueue", "Random"),
        ("ParkSong-Composite", "RoundRobin"),
        ("ParkSong-Composite", "ShortestQueue"),
        ("Batch", "Random"),
    ]
    metrics = [
        "cycle_time_mean_s",
        "waiting_time_mean_s",
        "horizon_normalized_throughput_cases_per_hour",
        "horizon_normalized_resource_occupation_mean",
        "resource_fairness_gini",
        "weighted_resource_fairness",
    ]
    rows = []
    for left, right in comparisons:
        for metric in metrics:
            pivot = raw.pivot_table(index="seed", columns="strategy", values=metric, aggfunc="first")
            if left not in pivot or right not in pivot:
                continue
            diff = (pivot[left] - pivot[right]).dropna()
            pct = ((pivot[left] - pivot[right]) / pivot[right].replace(0, np.nan) * 100.0).dropna()
            rows.append(
                {
                    "comparison": f"{left} - {right}",
                    "metric": metric,
                    "paired_seed_count": int(diff.size),
                    "mean_difference": float(diff.mean()) if not diff.empty else float("nan"),
                    "median_difference": float(diff.median()) if not diff.empty else float("nan"),
                    "std_difference": float(diff.std()) if diff.size > 1 else float("nan"),
                    "ci_method": "student_t",
                    "ci_level": 0.95,
                    "difference_ci95_half_width": (
                        float(student_t.ppf(0.975, df=diff.size - 1) * diff.std() / math.sqrt(diff.size))
                        if diff.size > 1 else float("nan")
                    ),
                    "wins": int((diff < 0).sum()),
                    "ties": int((diff == 0).sum()),
                    "losses": int((diff > 0).sum()),
                    "mean_percent_difference": float(pct.mean()) if not pct.empty else float("nan"),
                    "median_percent_difference": float(pct.median()) if not pct.empty else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def run_one(
    strategy_name: str,
    seed: int,
    data_path: str,
    branching_artifact: str,
    processing_time_artifact: str,
    start: datetime,
    end: datetime,
    drain_until: datetime | None = None,
    diagnostic_cycle_guard: bool = False,
    cycle_repetition_limit: int = 50,
    fixed_routes: list[list[str]] | None = None,
    fixed_route_case_ids: list[str] | None = None,
    fixed_route_arrival_times: list[datetime] | None = None,
    parksong_params: dict[str, float] | None = None,
    parksong_processing_time_estimates: dict[tuple[str, str], float] | None = None,
    parksong_default_processing_time: float = 1.0,
    reservation_expiration_multiplier: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    strategy = build_strategy(
        strategy_name,
        seed,
        parksong_params=parksong_params,
        parksong_processing_time_estimates=parksong_processing_time_estimates,
        parksong_default_processing_time=parksong_default_processing_time,
    )
    branching_engine = load_composite_branching_artifact(branching_artifact)
    init_start = time.perf_counter()
    engine = IntegratedAllocationEngine(
        dataPath=data_path,
        seed=seed,
        allocation_strategy=strategy,
        branching_engine=branching_engine,
        processing_time_artifact=processing_time_artifact,
        diagnostic_cycle_guard=diagnostic_cycle_guard,
        cycle_repetition_limit=cycle_repetition_limit,
        fixed_routes=fixed_routes,
        fixed_route_case_ids=fixed_route_case_ids,
        fixed_route_arrival_times=fixed_route_arrival_times,
        reservation_expiration_multiplier=reservation_expiration_multiplier,
    )
    init_seconds = time.perf_counter() - init_start
    run_start = time.perf_counter()
    engine.run(start_time=start, end_time=end, format_type=[], drain_until=drain_until)
    run_seconds = time.perf_counter() - run_start
    log = engine.logger.get_log()
    metrics = lifecycle_metrics(log, engine, start, end, observation_end=drain_until or end)
    metrics.update(
        {
            "strategy": strategy_name,
            "seed": seed,
            "simulation_start": start.isoformat(),
            "simulation_end": end.isoformat(),
            "drain_until": drain_until.isoformat() if drain_until else "",
            "diagnostic_cycle_guard": int(diagnostic_cycle_guard),
            "cycle_repetition_limit": cycle_repetition_limit,
            "engine_initialization_seconds": init_seconds,
            "simulation_runtime_seconds": run_seconds,
        }
    )
    diagnostics = engine.get_integration_diagnostics()
    diagnostics.update(
        {
            "strategy": strategy_name,
            "seed": seed,
            "events_processed": len(log),
            "drain_until": drain_until.isoformat() if drain_until else "",
            "diagnostic_cycle_guard": int(diagnostic_cycle_guard),
            "cycle_repetition_limit": cycle_repetition_limit,
            "engine_initialization_seconds": init_seconds,
            "simulation_runtime_seconds": run_seconds,
        }
    )
    return metrics, diagnostics


def read_existing(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def append_csv(path: Path, row: dict[str, Any]) -> None:
    df = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True, sort=False)
    df.to_csv(path, index=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def command_output(args: list[str]) -> str:
    try:
        result = subprocess.run(
            args,
            cwd=REPO_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def load_event_log(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".xes" or path.name.lower().endswith(".xes.gz"):
        return pm4py.read_xes(str(path))
    return pd.read_csv(path)


def case_id_digest(case_ids: list[str]) -> str:
    payload = "\n".join(sorted(map(str, case_ids))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepare_branching_artifact(args, output_dir: Path, seeds: list[int]) -> str:
    summary_path = output_dir / "branching_split_summary.json"
    if not args.train_branching_from_split:
        write_json(
            summary_path,
            {
                "mode": "pretrained_artifact",
                "branching_artifact": args.branching_artifact,
                "artifact_sha256": artifact_sha256(args.branching_artifact),
                "evaluation_log_used_for_training_constraints": False,
            },
        )
        return args.branching_artifact

    log = load_event_log(args.data_path)
    train_log, test_log = temporal_train_test_split_by_case(
        log,
        train_ratio=args.split_ratio,
        case_col=CASE_COL,
        timestamp_col=TIMESTAMP_COL,
    )
    seed = seeds[0] if seeds else 1
    composite = CompositeBranchingEngine(
        log=train_log,
        seed=seed,
        use_default_hierarchy=True,
        train_on_init=True,
    )
    model_path = output_dir / "models" / "composite_branching_temporal_split.pkl"
    train_cases = sorted(map(str, train_log[CASE_COL].dropna().unique()))
    test_cases = sorted(map(str, test_log[CASE_COL].dropna().unique()))
    metadata = export_composite_branching_artifact(
        composite,
        model_path,
        metadata={
            "training_mode": "temporal_case_split",
            "split_ratio": args.split_ratio,
            "seed": seed,
            "data_path": args.data_path,
            "data_sha256": sha256_file(args.data_path),
            "train_case_count": len(train_cases),
            "test_case_count": len(test_cases),
            "train_event_count": int(len(train_log)),
            "test_event_count": int(len(test_log)),
            "train_case_id_sha256": case_id_digest(train_cases),
            "test_case_id_sha256": case_id_digest(test_cases),
            "evaluation_log_used_for_training_constraints": False,
            "stateful_features": [
                "current_activity",
                "previous_activity",
                "trace_prefix",
                "current_activity_visit_count",
                "consecutive_repetition_count",
                "elapsed_case_time_seconds",
                "time_since_previous_event_seconds",
                "case_attributes",
            ],
        },
    )
    train_times = pd.to_datetime(train_log[TIMESTAMP_COL], errors="coerce")
    test_times = pd.to_datetime(test_log[TIMESTAMP_COL], errors="coerce")
    write_json(
        summary_path,
        {
            "mode": "temporal_case_split",
            "split_ratio": args.split_ratio,
            "seed": seed,
            "artifact": str(model_path),
            "artifact_sha256": metadata.get("artifact_sha256"),
            "train_case_count": len(train_cases),
            "test_case_count": len(test_cases),
            "train_event_count": int(len(train_log)),
            "test_event_count": int(len(test_log)),
            "train_case_id_sha256": case_id_digest(train_cases),
            "test_case_id_sha256": case_id_digest(test_cases),
            "train_time_min": train_times.min(),
            "train_time_max": train_times.max(),
            "test_time_min": test_times.min(),
            "test_time_max": test_times.max(),
            "evaluation_log_used_for_training_constraints": False,
        },
    )
    return str(model_path)


def build_fixed_routes_from_heldout(
    data_path: str,
    start: datetime,
    end: datetime,
    split_ratio: float,
    limit: int | None = None,
) -> tuple[list[list[str]], list[str], list[datetime], dict[str, Any]]:
    log = load_event_log(data_path)
    log[TIMESTAMP_COL] = pd.to_datetime(log[TIMESTAMP_COL], errors="coerce")
    train_log, test_log = temporal_train_test_split_by_case(
        log,
        train_ratio=split_ratio,
        case_col=CASE_COL,
        timestamp_col=TIMESTAMP_COL,
    )
    case_starts = test_log.groupby(CASE_COL)[TIMESTAMP_COL].min().sort_values()
    selected_case_ids = [
        str(case_id)
        for case_id, case_start in case_starts.items()
        if start <= case_start.to_pydatetime() <= end
    ]
    if limit is not None:
        selected_case_ids = selected_case_ids[:limit]

    complete_log = test_log
    if LIFECYCLE_COL in complete_log.columns:
        complete_log = complete_log[complete_log[LIFECYCLE_COL] == "complete"]

    routes: list[list[str]] = []
    route_case_ids: list[str] = []
    route_arrival_times: list[datetime] = []
    route_durations_s: list[float] = []
    terminal_activities: list[str] = []
    activity_counts: Counter[str] = Counter()
    for case_id in selected_case_ids:
        case_events = complete_log[complete_log[CASE_COL].astype(str) == case_id]
        case_events = case_events.sort_values(TIMESTAMP_COL)
        route = [
            str(activity)
            for activity in case_events[ACTIVITY_COL].tolist()
            if pd.notna(activity)
        ]
        if route:
            routes.append(route)
            route_case_ids.append(case_id)
            route_arrival_times.append(case_starts.loc[case_id].to_pydatetime())
            terminal_activities.append(route[-1])
            activity_counts.update(route)
            all_case_events = test_log[test_log[CASE_COL].astype(str) == case_id]
            start_ts = all_case_events[TIMESTAMP_COL].min()
            end_ts = all_case_events[TIMESTAMP_COL].max()
            route_durations_s.append(float((end_ts - start_ts).total_seconds()))

    summary = {
        "route_mode": "fixed_replay",
        "split_ratio": split_ratio,
        "selected_case_count": len(route_case_ids),
        "selected_case_ids": route_case_ids,
        "arrival_times": [arrival.isoformat() for arrival in route_arrival_times],
        "route_lengths": [len(route) for route in routes],
        "route_length_distribution": describe_numeric([len(route) for route in routes]),
        "historical_duration_seconds_distribution": describe_numeric(route_durations_s),
        "historical_duration_days_distribution": describe_numeric(
            [duration / 86400.0 for duration in route_durations_s]
        ),
        "terminal_activity_distribution": dict(Counter(terminal_activities).most_common()),
        "activity_distribution": dict(activity_counts.most_common()),
        "selection_start": start.isoformat(),
        "selection_end": end.isoformat(),
        "lifecycle_filter": "complete" if LIFECYCLE_COL in log.columns else "all_events",
        "future_route_visible_to_strategy": False,
    }
    return routes, route_case_ids, route_arrival_times, summary


def write_static_outputs(args, output_dir: Path, inventory: list[dict[str, str]], selected: list[str]) -> None:
    inventory_frame = canonical_inventory_frame(inventory, selected)
    inventory_frame.to_csv(output_dir / "method_inventory.csv", index=False)
    inventory_frame.to_csv(output_dir / "final_method_inventory.csv", index=False)
    run_config = {
            "data_path": args.data_path,
            "branching_artifact": args.branching_artifact,
            "processing_time_artifact": args.processing_time_artifact,
            "start": args.start,
            "end": args.end,
            "seeds": [int(part) for part in args.seeds.split(",") if part.strip()],
            "strategies": selected,
            "engine": "joao.src.resource_allocation.integration.IntegratedAllocationEngine",
            "diagnostic_cycle_guard": bool(args.diagnostic_cycle_guard),
            "cycle_repetition_limit": args.cycle_repetition_limit,
            "train_branching_from_split": bool(args.train_branching_from_split),
            "split_ratio": args.split_ratio,
            "route_mode": args.route_mode,
            "fixed_route_limit": args.fixed_route_limit,
            "parksong_processing_times": args.parksong_processing_times,
            "parksong_params": parse_parksong_params(args.parksong_params),
            "reservation_expiration_multiplier": args.reservation_expiration_multiplier,
            "drain": (
                "arrivals stop at --end; when --drain-until is set, queued "
                "events continue until that horizon"
            ),
            "drain_until": args.drain_until,
        }
    write_json(output_dir / "final_run_config.json", run_config)
    write_json(output_dir / "fixed_replay_config.json", run_config)
    write_json(output_dir / "joao_method_run_config.json", run_config)
    environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "git_branch": command_output(["git", "branch", "--show-current"]),
            "git_status_short": command_output(["git", "status", "--short"]),
            "git_log_head": command_output(["git", "log", "-1", "--oneline", "--decorate"]),
        }
    write_json(output_dir / "final_environment.json", environment)
    write_json(output_dir / "joao_method_environment.json", environment)
    payload = load_artifact_payload(args.branching_artifact)
    artifact_hashes = {
            "data_path": args.data_path,
            "data_sha256": sha256_file(args.data_path),
            "branching_artifact": args.branching_artifact,
            "branching_sha256": artifact_sha256(args.branching_artifact),
            "branching_metadata": payload.get("metadata", {}),
            "processing_time_artifact": args.processing_time_artifact,
            "processing_time_sha256": sha256_file(args.processing_time_artifact),
        }
    write_json(output_dir / "final_artifact_hashes.json", artifact_hashes)
    write_json(output_dir / "joao_method_artifact_hashes.json", artifact_hashes)


def processing_time_coverage(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if diagnostics.empty:
        return pd.DataFrame(rows)
    for _, row in diagnostics.iterrows():
        exact = int(row.get("processing_time_model_hits", 0) or 0)
        activity = int(row.get("processing_time_activity_fallback_hits", 0) or 0) + int(row.get("processing_time_empirical_activity_fallback_hits", 0) or 0)
        category = int(row.get("processing_time_category_fallback_hits", 0) or 0)
        global_fallback = int(row.get("processing_time_global_fallback_hits", 0) or 0)
        emergency = int(row.get("processing_time_emergency_guard_hits", 0) or 0)
        total = exact + activity + category + global_fallback + emergency
        rows.append(
            {
                "strategy": row.get("strategy", ""),
                "seed": row.get("seed", ""),
                "exact_resource_activity_model_calls": exact,
                "activity_level_fallback_calls": activity,
                "category_fallback_calls": category,
                "global_fallback_calls": global_fallback,
                "emergency_guard_calls": emergency,
                "total_processing_time_sampling_calls": total,
                "exact_resource_activity_model_rate": exact / total if total else float("nan"),
                "activity_level_fallback_rate": activity / total if total else float("nan"),
                "global_fallback_rate": global_fallback / total if total else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def resource_pressure_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if diagnostics.empty:
        return pd.DataFrame(rows)
    for _, row in diagnostics.iterrows():
        samples = int(row.get("allocation_call_samples", 0) or row.get("waiting_queue_size_samples", 0) or 0)
        waiting_sum = float(row.get("waiting_queue_size_sum", 0) or 0)
        resource_sum = float(row.get("resources_converted_sum", 0) or 0)
        waiting_tasks_seen = float(row.get("waiting_tasks_seen", 0) or 0)
        rows.append(
            {
                "strategy": row.get("strategy", ""),
                "seed": row.get("seed", ""),
                "allocation_epochs": int(row.get("global_strategy_calls", 0) or row.get("allocation_strategy_calls", 0) or 0),
                "epochs_with_waiting_tasks": samples,
                "epochs_with_zero_eligible_resource": int(row.get("waiting_retry_no_future_availability", 0) or 0),
                "epochs_with_less_eligible_resources_than_waiting_tasks": "",
                "epochs_with_all_eligible_resources_occupied": int(row.get("event_queue_empty_with_waiting_tasks", 0) or 0),
                "epochs_with_resources_unavailable_by_calendar": "",
                "average_waiting_tasks_per_allocation_epoch": waiting_sum / samples if samples else float("nan"),
                "average_eligible_resources_per_waiting_task": resource_sum / waiting_tasks_seen if waiting_tasks_seen else float("nan"),
                "fraction_tasks_waited_no_available_eligible_resource": (
                    float(row.get("waiting_retry_no_future_availability", 0) or 0) / waiting_tasks_seen
                    if waiting_tasks_seen else float("nan")
                ),
                "max_waiting_queue_length": int(row.get("max_queue_size", 0) or 0),
                "mean_waiting_queue_length": float(row.get("waiting_queue_size_mean", float("nan"))),
                "fraction_epochs_strategic_idling_feasible": (
                    float(row.get("active_prediction_count_sum", 0) or 0) / samples
                    if samples else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def reservation_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if diagnostics.empty:
        return pd.DataFrame(rows)
    def finite_int(value: Any) -> int:
        numeric = pd.to_numeric(value, errors="coerce")
        return int(numeric) if pd.notna(numeric) else 0

    def finite_float(value: Any) -> float:
        numeric = pd.to_numeric(value, errors="coerce")
        return float(numeric) if pd.notna(numeric) else float("nan")

    for _, row in diagnostics.iterrows():
        created = finite_int(row.get("reservations_created", 0))
        used = finite_int(row.get("reservations_used", 0))
        assignment_calls = finite_float(row.get("global_assignment_calls", 0))
        rows.append(
            {
                "strategy": row.get("strategy", ""),
                "seed": row.get("seed", ""),
                "predictions_consumed": finite_int(row.get("park_song_predictions_consumed", 0)),
                "prediction_execution_matches": finite_int(row.get("prediction_execution_matches", 0)),
                "prediction_execution_mismatches": finite_int(row.get("prediction_execution_mismatches", 0)),
                "reservations_created": created,
                "reservations_used": used,
                "reservations_expired": finite_int(row.get("reservations_expired", 0)),
                "reservations_cancelled": finite_int(row.get("reservations_cancelled", 0)),
                "unresolved_reservations": finite_int(row.get("unresolved_reservations", 0)),
                "reservation_utilization_rate": used / created if created else float("nan"),
                "reservations_rejected_by_probability_threshold": finite_int(row.get("predictions_below_threshold", 0)),
                "reservations_filtered_by_reservation_margin": finite_int(row.get("reservation_margin_filtered", 0)),
                "reservations_rejected_by_no_show_penalty": finite_int(row.get("no_show_penalty_filtered", 0)),
                "global_assignment_calls": finite_int(row.get("global_assignment_calls", 0)),
                "feasible_candidate_pairs": finite_int(row.get("global_assignment_feasible_pairs_total", 0)),
                "average_candidate_count": (
                    finite_float(row.get("global_assignment_candidates_total", 0)) / assignment_calls
                    if assignment_calls and math.isfinite(assignment_calls) else float("nan")
                ),
                "simulation_runtime_seconds": finite_float(row.get("simulation_runtime_seconds", float("nan"))),
            }
        )
    return pd.DataFrame(rows)


def write_report(output_dir: Path, raw: pd.DataFrame, aggregated: pd.DataFrame, failures: pd.DataFrame) -> None:
    def table_text(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "No rows."
        return "```csv\n" + frame.to_csv(index=False) + "```"

    lines = [
        "# Final Resource Allocation Evaluation",
        "",
        "This report is generated from the common IntegratedAllocationEngine lifecycle.",
        "Kunkler/Rinderle-Ma is included through the adapter that invokes the repository implementation and enforces simulator eligibility.",
        "ParkSongML is inventoried as a prediction integration layer, not a separate runnable strategy identity.",
        "",
        "## Raw Runs",
        table_text(raw),
        "",
        "## Aggregated Metrics",
        table_text(aggregated),
        "",
        "## Failures",
        table_text(failures),
    ]
    report_text = "\n".join(lines)
    (output_dir / "final_evaluation_report.md").write_text(report_text, encoding="utf-8")
    (output_dir / "joao_final_readiness.md").write_text(report_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final common-lifecycle resource-allocation evaluation.")
    parser.add_argument("--data-path", default=str(REPO_ROOT / "data" / "logData.xes"))
    parser.add_argument(
        "--branching-artifact",
        default=str(REPO_ROOT / "joao" / "models" / "branching" / "final_composite_branching.pkl"),
    )
    parser.add_argument(
        "--processing-time-artifact",
        default=str(REPO_ROOT / "joao" / "models" / "process_time" / "final_process_time_coverage_v2.pkl"),
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--drain-until",
        default=None,
        help="Optional ISO timestamp. Arrivals stop at --end; existing events may run until this horizon.",
    )
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--strategies", default=None)
    parser.add_argument(
        "--diagnostic-cycle-guard",
        action="store_true",
        help="Enable training-log empirical successor/visit filters for diagnostics only.",
    )
    parser.add_argument(
        "--cycle-repetition-limit",
        type=int,
        default=50,
        help="Non-mutating repeated-activity classification threshold.",
    )
    parser.add_argument(
        "--train-branching-from-split",
        action="store_true",
        help="Train a new local Composite artifact on the temporal training split.",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.7,
        help="Temporal case split ratio used with --train-branching-from-split.",
    )
    parser.add_argument(
        "--route-mode",
        choices=["generative", "fixed-replay"],
        default="generative",
        help="Use Composite to generate routes or replay held-out historical complete-event routes.",
    )
    parser.add_argument(
        "--fixed-route-limit",
        type=int,
        default=None,
        help="Optional maximum held-out routes admitted in fixed-replay mode.",
    )
    parser.add_argument(
        "--parksong-params",
        default=None,
        help=(
            "Optional comma-separated ParkSong key=value overrides, e.g. "
            "prediction_probability_threshold=0.8,idling_weight=2.0. "
            "Only affects ParkSong strategies."
        ),
    )
    parser.add_argument(
        "--reservation-expiration-multiplier",
        type=float,
        default=1.0,
        help=(
            "Multiplier for prediction expected_delay before a reservation expires. "
            "Default 1.0 preserves the canonical reservation horizon."
        ),
    )
    parser.add_argument(
        "--parksong-processing-times",
        choices=["train-median", "none"],
        default="train-median",
        help=(
            "How ParkSong obtains resource/activity processing-time estimates. "
            "train-median uses positive start/resume/suspend/complete durations "
            "from the temporal training split; none uses ParkSong's constant fallback."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(JOAO_ROOT / "results" / "final_resource_allocation" / datetime.now().strftime("%Y%m%d_%H%M%S")),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = method_inventory(REPO_ROOT)
    available = canonical_strategy_names(inventory)
    selected = parse_strategy_filter(args.strategies, available)
    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    parksong_params = parse_parksong_params(args.parksong_params)
    parksong_params = {
        "cost_time_scale": 3600.0,
        "no_show_penalty_weight": 1.0,
        "future_delay_weight": 0.0,
        "reservation_margin": 0.0,
        **parksong_params,
    }
    args.parksong_params = ",".join(
        f"{key}={value}" for key, value in sorted(parksong_params.items())
    )
    args.branching_artifact = prepare_branching_artifact(args, output_dir, seeds)
    parksong_processing_time_estimates = None
    parksong_default_processing_time = 1.0
    if args.parksong_processing_times == "train-median":
        (
            parksong_processing_time_estimates,
            parksong_default_processing_time,
            parksong_processing_summary,
        ) = build_parksong_processing_time_estimates(args.data_path, args.split_ratio)
    else:
        parksong_processing_summary = {
            "processing_time_estimate_mode": "constant_fallback",
            "estimate_count": 0,
            "default_processing_time": parksong_default_processing_time,
        }
    write_json(output_dir / "parksong_processing_time_estimates_summary.json", parksong_processing_summary)
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    drain_until = datetime.fromisoformat(args.drain_until) if args.drain_until else None
    fixed_routes = None
    fixed_route_case_ids = None
    fixed_route_arrival_times = None
    if args.route_mode == "fixed-replay":
        (
            fixed_routes,
            fixed_route_case_ids,
            fixed_route_arrival_times,
            route_summary,
        ) = build_fixed_routes_from_heldout(
            data_path=args.data_path,
            start=start,
            end=end,
            split_ratio=args.split_ratio,
            limit=args.fixed_route_limit,
        )
        write_json(output_dir / "fixed_route_workload_summary.json", route_summary)
        write_json(output_dir / "final_workload_summary.json", route_summary)
        pd.DataFrame(
            {
                "case_id": fixed_route_case_ids,
                "arrival_time": [
                    arrival.isoformat() for arrival in fixed_route_arrival_times
                ],
                "route_length": [len(route) for route in fixed_routes],
            }
        ).to_csv(output_dir / "final_route_ids.csv", index=False)
        if not fixed_routes:
            raise ValueError("No held-out complete-event routes selected for fixed-replay mode.")
    write_static_outputs(args, output_dir, inventory, selected)

    raw_path = output_dir / "final_raw_metrics.csv"
    diag_path = output_dir / "final_strategy_diagnostics.csv"
    failure_path = output_dir / "failures.csv"
    completed = set()
    if args.resume:
        existing = read_existing(raw_path)
        if not existing.empty:
            completed = {
                (str(row["strategy"]), int(row["seed"]))
                for _, row in existing.iterrows()
            }

    for strategy_name in selected:
        for seed in seeds:
            combo = (strategy_name, seed)
            if combo in completed:
                print(f"[skip] {strategy_name} seed={seed} already completed")
                continue
            print(f"[run] {strategy_name} seed={seed} start={args.start} end={args.end}", flush=True)
            started = time.perf_counter()
            try:
                metrics, diagnostics = run_one(
                    strategy_name=strategy_name,
                    seed=seed,
                    data_path=args.data_path,
                    branching_artifact=args.branching_artifact,
                    processing_time_artifact=args.processing_time_artifact,
                    start=start,
                    end=end,
                    drain_until=drain_until,
                    diagnostic_cycle_guard=args.diagnostic_cycle_guard,
                    cycle_repetition_limit=args.cycle_repetition_limit,
                    fixed_routes=fixed_routes,
                    fixed_route_case_ids=fixed_route_case_ids,
                    fixed_route_arrival_times=fixed_route_arrival_times,
                    parksong_params=parksong_params,
                    parksong_processing_time_estimates=parksong_processing_time_estimates,
                    parksong_default_processing_time=parksong_default_processing_time,
                    reservation_expiration_multiplier=args.reservation_expiration_multiplier,
                )
            except Exception as exc:
                append_csv(
                    failure_path,
                    {
                        "strategy": strategy_name,
                        "seed": seed,
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                        "runtime_seconds_before_failure": time.perf_counter() - started,
                    },
                )
                print(f"[fail] {strategy_name} seed={seed}: {exc}", flush=True)
                continue
            append_csv(raw_path, metrics)
            append_csv(diag_path, diagnostics)
            print(
                f"[done] {strategy_name} seed={seed} "
                f"runtime={metrics['simulation_runtime_seconds']:.2f}s "
                f"events={metrics['n_events']} completed={metrics['cases_completed']}",
                flush=True,
            )

    raw = read_existing(raw_path)
    diagnostics = read_existing(diag_path)
    failures = read_existing(failure_path)
    if not raw.empty:
        aggregated = aggregate(raw)
        paired = paired_comparisons(raw)
        aggregated.to_csv(output_dir / "final_aggregated_metrics.csv", index=False)
        aggregated.to_csv(output_dir / "fixed_replay_aggregated_metrics.csv", index=False)
        paired.to_csv(output_dir / "final_paired_comparisons.csv", index=False)
        paired.to_csv(output_dir / "final_paired_comparisons.csv", index=False)
        paired.to_csv(output_dir / "statistical_comparison.csv", index=False)
        if not diagnostics.empty:
            diagnostics.to_csv(output_dir / "final_strategy_diagnostics.csv", index=False)
            diagnostics.to_csv(output_dir / "fixed_replay_method_diagnostics.csv", index=False)
            processing_time_coverage(diagnostics).to_csv(
                output_dir / "processing_time_coverage.csv",
                index=False,
            )
            resource_pressure_diagnostics(diagnostics).to_csv(
                output_dir / "resource_pressure_diagnostics.csv",
                index=False,
            )
            reservation_diagnostics(diagnostics).to_csv(
                output_dir / "parksong_reservation_diagnostics.csv",
                index=False,
            )
        resource_rows = raw[
            [
                column
                for column in raw.columns
                if column
                in {
                    "strategy",
                    "seed",
                    "resource_occupation_mean",
                    "resource_occupation_median",
                    "horizon_normalized_resource_occupation_mean",
                    "horizon_normalized_resource_occupation_median",
                    "resource_fairness_gini",
                    "weighted_resource_fairness",
                    "weighted_resource_fairness_status",
                    "weighted_fairness_resource_count",
                }
            ]
        ]
        resource_rows.to_csv(output_dir / "final_resource_metrics.csv", index=False)
        resource_rows.to_csv(output_dir / "fixed_replay_resource_metrics.csv", index=False)
        resource_rows.to_csv(output_dir / "resource_utilization.csv", index=False)
        case_rows = raw[
            [
                column
                for column in raw.columns
                if column
                in {
                    "strategy",
                    "seed",
                    "cases_admitted",
                    "cases_observed",
                    "cases_completed",
                    "fixed_routes_completed",
                    "cases_incomplete",
                    "cases_deadlocked",
                    "cases_cyclic",
                    "cases_censored",
                    "completion_rate",
                    "fixed_route_completion_rate",
                    "active_cases_remaining",
                    "waiting_queue_remaining",
                }
            ]
        ]
        case_rows.to_csv(output_dir / "final_case_completion.csv", index=False)
        case_rows.to_csv(output_dir / "fixed_replay_case_completion.csv", index=False)
        case_rows.to_csv(output_dir / "final_case_classification.csv", index=False)
        case_rows.to_csv(output_dir / "joao_case_classification.csv", index=False)
        raw.to_csv(output_dir / "joao_raw_metrics.csv", index=False)
        raw.to_csv(output_dir / "fixed_replay_raw_metrics.csv", index=False)
        aggregated.to_csv(output_dir / "joao_aggregated_metrics.csv", index=False)
        diagnostics.to_csv(output_dir / "joao_strategy_diagnostics.csv", index=False)
        resource_rows.to_csv(output_dir / "joao_resource_metrics.csv", index=False)
        report_columns = {
            "strategy": "Method",
            "fixed_route_completion_rate_mean": "Fixed-route completion",
            "cycle_time_mean_s_mean": "Mean cycle time",
            "waiting_time_mean_s_mean": "Mean waiting time",
            "horizon_normalized_throughput_cases_per_hour_mean": "Horizon-normalized throughput",
            "horizon_normalized_resource_occupation_mean_mean": "Horizon-normalized occupation",
            "resource_fairness_gini_mean": "Fairness",
            "weighted_resource_fairness_mean": "Weighted fairness",
        }
        available_report_columns = [
            column for column in report_columns if column in aggregated.columns
        ]
        aggregated[available_report_columns].rename(columns=report_columns).to_csv(
            output_dir / "final_report_table.csv",
            index=False,
        )
        aggregated[available_report_columns].rename(columns=report_columns).to_csv(
            output_dir / "fixed_replay_report_table.csv",
            index=False,
        )
        appendix_columns = [
            column
            for column in aggregated.columns
            if column == "strategy"
            or column.endswith("_mean")
            or column.endswith("_std")
            or column.endswith("_median")
            or column.endswith("_min")
            or column.endswith("_max")
            or column.endswith("_ci95_half_width")
        ]
        aggregated[appendix_columns].to_csv(
            output_dir / "final_appendix_table.csv",
            index=False,
        )
    else:
        aggregated = pd.DataFrame()
    if not failures.empty:
        failures.to_csv(failure_path, index=False)
        failures.to_csv(output_dir / "joao_failures.csv", index=False)
    else:
        empty_failures = pd.DataFrame(
            columns=["strategy", "seed", "error", "traceback"]
        )
        empty_failures.to_csv(failure_path, index=False)
        empty_failures.to_csv(output_dir / "final_failures.csv", index=False)
        empty_failures.to_csv(output_dir / "joao_failures.csv", index=False)
    write_report(output_dir, raw, aggregated, failures)
    print(f"[output] {output_dir}")


if __name__ == "__main__":
    main()
