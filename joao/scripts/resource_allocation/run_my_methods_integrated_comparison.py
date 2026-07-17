from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pm4py

JOAO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = JOAO_ROOT.parent
sys.path.insert(0, str(JOAO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from joao.src.branching.CompositeBranchingArtifact import (
    artifact_sha256,
    load_artifact_payload,
    load_composite_branching_artifact,
)
from joao.src.branching.CompositeBranchingEngine import CompositeBranchingEngine
from joao.src.resource_allocation.integration.AllocationStrategyFactory import (
    build_my_allocation_strategies,
)
from joao.src.resource_allocation.integration.IntegratedAllocationEngine import (
    IntegratedAllocationEngine,
)
from joao.src.resource_allocation.integration.WeightedFairnessAdapter import (
    compute_weighted_fairness_from_engine,
)
from joao.scripts.resource_allocation.run_integrated_allocation_comparison import (
    average_cycle_time,
    average_resource_occupation,
    average_waiting_time,
    compute_log_diagnostics,
    gini,
)


CASE_COL = "case:concept:name"
RESOURCE_COL = "org:resource"
TIMESTAMP_COL = "time:timestamp"


def available_strategy_names(seed: int = 1) -> list[str]:
    return list(build_my_allocation_strategies(seed).keys())


def parse_strategy_filter(
    strategy_filter: str | None,
    available_names: list[str] | None = None,
) -> list[str]:
    available_names = available_names or available_strategy_names()
    if strategy_filter is None or not strategy_filter.strip():
        return list(available_names)

    requested = [name.strip() for name in strategy_filter.split(",") if name.strip()]
    unknown = [name for name in requested if name not in available_names]
    if unknown:
        valid = ", ".join(available_names)
        invalid = ", ".join(unknown)
        raise ValueError(f"Unknown strategy name(s): {invalid}. Valid strategies: {valid}")
    return requested


def build_trained_composite_branching(log: pd.DataFrame, seed: int):
    return CompositeBranchingEngine(
        log=log,
        seed=seed,
        use_default_hierarchy=True,
        train_on_init=True,
    )


def run_one_seed(
    data_path: str,
    start_time: datetime,
    end_time: datetime,
    seed: int,
    branching_artifact: str | None = None,
    processing_time_artifact: str | None = None,
    strategy_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    diagnostics_rows = []
    artifact_hash = ""
    artifact_metadata = {}

    if branching_artifact:
        trained_branching_engine = None
        composite_training_count = 0
        branching_training_performed = 0
        artifact_hash = artifact_sha256(branching_artifact)
        artifact_metadata = load_artifact_payload(branching_artifact).get("metadata", {})
    else:
        source_log = pm4py.read_xes(data_path, variant="r4pm")
        trained_branching_engine = build_trained_composite_branching(source_log, seed)
        composite_training_count = 1
        branching_training_performed = 1

    strategies = build_my_allocation_strategies(seed)
    selected_strategy_names = strategy_names or list(strategies.keys())

    for strategy_name in selected_strategy_names:
        strategy = strategies[strategy_name]
        if branching_artifact:
            branching_engine = load_composite_branching_artifact(branching_artifact)
        else:
            branching_engine = copy.deepcopy(trained_branching_engine)

        engine_init_start = time.perf_counter()
        engine = IntegratedAllocationEngine(
            dataPath=data_path,
            seed=seed,
            allocation_strategy=strategy,
            branching_engine=branching_engine,
            processing_time_artifact=processing_time_artifact,
        )
        engine_initialization_seconds = time.perf_counter() - engine_init_start
        simulation_start = time.perf_counter()
        engine.run(start_time=start_time, end_time=end_time, format_type=[])
        simulation_run_seconds = time.perf_counter() - simulation_start

        log = engine.logger.get_log()
        metrics = compute_metrics(
            log,
            engine=engine,
            start_time=start_time,
            end_time=end_time,
        )
        metrics.update(compute_log_diagnostics(log))
        metrics.update(
            {
                "strategy": strategy_name,
                "seed": seed,
                "simulation_start": start_time.isoformat(),
                "simulation_end": end_time.isoformat(),
            }
        )
        rows.append(metrics)

        diagnostics = engine.get_integration_diagnostics()
        simulated_seconds = max((end_time - start_time).total_seconds(), 0.0)
        diagnostics.update(
            {
                "strategy": strategy_name,
                "seed": seed,
                "engine_initialization_seconds": engine_initialization_seconds,
                "simulation_run_seconds": simulation_run_seconds,
                "events_processed": len(log),
                "simulated_seconds": simulated_seconds,
                "calls_per_simulated_second": (
                    diagnostics.get("_allocate_enabled_events_calls", 0)
                    / simulated_seconds
                    if simulated_seconds > 0
                    else 0.0
                ),
                "composite_training_count": composite_training_count,
                "composite_model_reuse_count": 1,
                "branching_artifact_loaded": int(bool(branching_artifact)),
                "branching_artifact_path": branching_artifact or "",
                "branching_training_performed": branching_training_performed,
                "composite_runtime_instances": 1,
                "artifact_model_hash": artifact_hash,
                "artifact_training_mode": artifact_metadata.get(
                    "deployment_training_mode", ""
                ),
                "configured_branching_engines": ",".join(
                    branching_engine.get_statistics().get("configured_engines", [])
                ),
                "processing_time_artifact_path": processing_time_artifact or "",
                "processing_time_artifact_hash": (
                    hashlib.sha256(Path(processing_time_artifact).read_bytes()).hexdigest()
                    if processing_time_artifact
                    else ""
                ),
            }
        )
        diagnostics_rows.append(diagnostics)

    return pd.DataFrame(rows), pd.DataFrame(diagnostics_rows)


def run_comparison(
    data_path: str,
    start_time: datetime,
    end_time: datetime,
    seeds: list[int],
    output_dir: Path,
    branching_artifact: str | None = None,
    processing_time_artifact: str | None = None,
    strategy_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_runs = []
    diagnostics = []
    selected_strategy_names = strategy_names or available_strategy_names(seeds[0])
    for seed in seeds:
        seed_raw, seed_diagnostics = run_one_seed(
            data_path=data_path,
            start_time=start_time,
            end_time=end_time,
            seed=seed,
            branching_artifact=branching_artifact,
            processing_time_artifact=processing_time_artifact,
            strategy_names=selected_strategy_names,
        )
        raw_runs.append(seed_raw)
        diagnostics.append(seed_diagnostics)

    raw = pd.concat(raw_runs, ignore_index=True)
    diag = pd.concat(diagnostics, ignore_index=True)
    summary = summarize(raw, diag)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_dir / "my_methods_raw_runs.csv", index=False)
    diag.to_csv(output_dir / "my_methods_diagnostics.csv", index=False)
    summary.to_csv(output_dir / "my_methods_summary.csv", index=False)
    diag.to_csv(output_dir / "branch_prediction_diagnostics.csv", index=False)
    write_configuration(
        output_dir=output_dir,
        data_path=data_path,
        start_time=start_time,
        end_time=end_time,
        seeds=seeds,
        branching_artifact=branching_artifact,
        processing_time_artifact=processing_time_artifact,
        strategy_names=selected_strategy_names,
    )
    return raw, diag, summary


def compute_metrics(
    log: pd.DataFrame,
    engine=None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> dict[str, float | int | str]:
    if log.empty:
        return {
            "n_events": 0,
            "n_cases": 0,
            "assigned_events": 0,
            "average_cycle_time": float("nan"),
            "average_waiting_time": float("nan"),
            "average_resource_occupation": float("nan"),
            "resource_fairness": float("nan"),
            "weighted_resource_fairness": float("nan"),
            "weighted_resource_fairness_status": "empty_log",
            "completed_cases": 0,
            "unfinished_cases": 0,
            "throughput": 0,
        }

    prepared = log.copy()
    prepared[TIMESTAMP_COL] = pd.to_datetime(prepared[TIMESTAMP_COL], errors="coerce")
    prepared = prepared.dropna(subset=[TIMESTAMP_COL])
    assigned = prepared[RESOURCE_COL].fillna("").astype(str).str.len() > 0
    resource_counts = prepared.loc[assigned, RESOURCE_COL].astype(str).value_counts()

    metrics: dict[str, float | int | str] = {
        "n_events": int(len(prepared)),
        "n_cases": int(prepared[CASE_COL].nunique()) if CASE_COL in prepared else 0,
        "assigned_events": int(assigned.sum()),
        "average_cycle_time": average_cycle_time(prepared),
        "average_waiting_time": average_waiting_time(prepared),
        "average_resource_occupation": average_resource_occupation(prepared),
        "resource_fairness": gini(resource_counts.tolist()),
        "weighted_resource_fairness": float("nan"),
        "weighted_resource_fairness_status": "missing_busy_and_availability_intervals",
    }

    if engine is not None and start_time is not None and end_time is not None:
        metrics.update(
            compute_weighted_fairness_from_engine(
                engine=engine,
                start_time=start_time,
                end_time=end_time,
            )
        )
        case_markings = getattr(engine.bpmnEngine, "case_markings", {})
        final_marking = getattr(engine.bpmnEngine, "final_marking", None)
        completed_cases = sum(
            1
            for marking in case_markings.values()
            if final_marking is not None and marking == final_marking
        )
        metrics["completed_cases"] = int(completed_cases)
        metrics["unfinished_cases"] = max(0, int(metrics["n_cases"]) - completed_cases)
        duration_hours = max((end_time - start_time).total_seconds() / 3600.0, 0.0)
        metrics["throughput"] = (
            completed_cases / duration_hours if duration_hours > 0 else float("nan")
        )
    else:
        metrics["completed_cases"] = 0
        metrics["unfinished_cases"] = int(metrics["n_cases"])
        metrics["throughput"] = 0

    return metrics


def summarize(raw: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "n_events",
        "n_cases",
        "assigned_events",
        "average_cycle_time",
        "average_waiting_time",
        "average_resource_occupation",
        "resource_fairness",
        "weighted_resource_fairness",
        "completed_cases",
        "unfinished_cases",
        "throughput",
        "busy_interval_count",
        "availability_interval_count",
        "weighted_fairness_resource_count",
        "suspended_events",
        "resumed_events",
    ]
    diagnostic_columns = [
        "global_strategy_calls",
        "strategy_assignments",
        "waiting_tasks_seen",
        "max_queue_size",
        "branch_predictions",
        "branch_predictions_executed",
        "prediction_execution_matches",
        "prediction_execution_mismatches",
        "park_song_predictions_consumed",
        "reservation_decisions",
        "reservations_used",
        "duplicate_assignment_errors",
        "composite_training_count",
        "composite_model_reuse_count",
        "resources_allocated",
        "resources_released",
        "unique_predictions_executed",
        "park_song_prediction_reuse_count",
        "reservations_created",
        "reservations_expired",
        "reservations_cancelled",
        "reservations_overwritten",
        "reservations_rejected_existing_kept",
        "reservations_unresolved_at_horizon",
        "reservations_active_after_cleanup",
        "stale_predictions",
        "branching_artifact_loaded",
        "branching_training_performed",
        "composite_runtime_instances",
        "unequal_resource_load_comparisons",
        "equal_resource_load_ties",
        "resource_load_unequal_decisions",
        "resource_load_tie_break_decisions",
        "resource_load_assignment_decisions",
        "engine_initialization_seconds",
        "simulation_run_seconds",
        "events_processed",
        "_allocate_enabled_events_calls",
        "_allocate_enabled_events_time_seconds",
        "_build_tasks_calls",
        "_build_tasks_time_seconds",
        "_build_resources_calls",
        "_build_resources_time_seconds",
        "branch_prediction_calls",
        "branch_prediction_time_seconds",
        "allocation_strategy_calls",
        "allocation_strategy_time_seconds",
        "waiting_queue_size_mean",
        "max_queue_size",
        "tasks_converted_per_call_mean",
        "resources_converted_per_call_mean",
        "active_prediction_count_mean",
        "reservation_count_mean",
        "calls_per_simulated_second",
        "task_cache_hits",
        "task_cache_misses",
        "task_cache_created",
        "task_cache_removed",
        "task_cache_max_size",
        "resource_build_calls",
        "permission_cache_hits",
        "permission_cache_misses",
        "resource_objects_created",
        "processing_time_model_hits",
        "processing_time_activity_fallback_hits",
        "processing_time_missing_model_count",
        "processing_time_invalid_value_count",
        "minimum_visible_duration_applications",
        "minimum_visible_duration_application_rate",
        "visible_activity_processing_starts",
        "final_processing_duration_min",
        "final_processing_duration_median",
        "final_processing_duration_mean",
        "final_processing_duration_max",
        "zero_duration_silent_transitions",
        "processing_time_empirical_activity_fallback_hits",
        "processing_time_category_fallback_hits",
        "processing_time_global_fallback_hits",
        "processing_time_emergency_guard_hits",
        "processing_time_data_backed_coverage_rate",
        "processing_time_exact_model_coverage_rate",
        "processing_time_any_non_emergency_coverage_rate",
        "processing_time_cached_duration_uses",
        "processing_time_resampling_attempts",
        "final_positive_duration_count",
        "final_zero_visible_duration_count",
    ]
    available_metric_columns = [column for column in metric_columns if column in raw.columns]
    metric_summary = (
        raw.groupby("strategy")[available_metric_columns]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    metric_summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in metric_summary.columns
    ]
    seed_counts = raw.groupby("strategy", as_index=False)["seed"].nunique()
    seed_counts = seed_counts.rename(columns={"seed": "seed_count"})
    metric_summary = metric_summary.merge(seed_counts, on="strategy", how="left")

    dynamic_diagnostic_columns = [
        column
        for column in diagnostics.columns
        if column.startswith("processing_time_missing_model_activity_")
        or column.startswith("minimum_visible_duration_activity_")
        or column.startswith("processing_time_source_activity_")
        or column.startswith("processing_time_source_category_")
    ]
    selected_diagnostic_columns = [
        column for column in diagnostic_columns if column in diagnostics.columns
    ]
    selected_diagnostic_columns.extend(
        column
        for column in dynamic_diagnostic_columns
        if column not in selected_diagnostic_columns
    )
    diagnostic_summary = diagnostics.groupby("strategy", as_index=False)[
        selected_diagnostic_columns
    ].mean(numeric_only=True)
    return metric_summary.merge(diagnostic_summary, on="strategy", how="left")


def write_configuration(
    output_dir: Path,
    data_path: str,
    start_time: datetime,
    end_time: datetime,
    seeds: list[int],
    branching_artifact: str | None = None,
    processing_time_artifact: str | None = None,
    strategy_names: list[str] | None = None,
) -> None:
    selected_strategy_names = strategy_names or available_strategy_names(seeds[0])
    config = {
        "data_path": data_path,
        "simulation_start": start_time.isoformat(),
        "simulation_end": end_time.isoformat(),
        "seeds": seeds,
        "methods": selected_strategy_names,
        "selected_strategies": selected_strategy_names,
        "branching": (
            f"artifact:{branching_artifact}"
            if branching_artifact
            else "CompositeBranchingEngine(log=source_log, train_on_init=True)"
        ),
        "branching_artifact": branching_artifact,
        "processing_time_artifact": processing_time_artifact,
        "engine": "joao.src.resource_allocation.integration.IntegratedAllocationEngine",
    }
    (output_dir / "my_methods_configuration.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run João's allocation methods through the integrated group simulator."
    )
    parser.add_argument("--data-path", default=str(REPO_ROOT / "data" / "logData.xes"))
    parser.add_argument("--branching-artifact", default=None)
    parser.add_argument("--processing-time-artifact", default=None)
    parser.add_argument("--start", default="2000-01-03T09:00:00")
    parser.add_argument("--end", default="2000-01-03T12:00:00")
    parser.add_argument("--seeds", default="1")
    parser.add_argument(
        "--output-dir",
        default=str(JOAO_ROOT / "results" / "my_methods_integrated"),
    )
    parser.add_argument(
        "--strategies",
        default=None,
        help=(
            "Comma-separated strategy names to run. "
            "Valid names: Random, RoundRobin, ShortestQueue, ParkSong."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    try:
        strategy_names = parse_strategy_filter(args.strategies)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    raw, diagnostics, summary = run_comparison(
        data_path=args.data_path,
        start_time=datetime.fromisoformat(args.start),
        end_time=datetime.fromisoformat(args.end),
        seeds=seeds,
        output_dir=Path(args.output_dir),
        branching_artifact=args.branching_artifact,
        processing_time_artifact=args.processing_time_artifact,
        strategy_names=strategy_names,
    )
    print(raw.to_string(index=False))
    print("\nDiagnostics:")
    print(diagnostics.to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
