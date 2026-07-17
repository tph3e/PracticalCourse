from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from joao.src.branching.CompositeBranchingEngine import CompositeBranchingEngine
from joao.src.resource_allocation.BatchAllocationAdapter import BatchAllocationAdapter
from joao.src.resource_allocation.KunklerAllocationAdapter import KunklerAllocationAdapter
from joao.src.resource_allocation.ParkSongAllocation import ParkSongAllocation
from joao.src.resource_allocation.RoundRobinResourceAllocation import (
    RoundRobinResourceAllocation,
)
from joao.src.resource_allocation.ShortestQueueAllocation import ShortestQueueAllocation
from joao.src.resource_allocation.integration.IntegratedAllocationEngine import (
    IntegratedAllocationEngine,
)
from joao.scripts.resource_allocation.run_final_resource_allocation_evaluation import (
    build_parksong_processing_time_estimates,
    sha256_file,
)


def build_strategy(name: str, args):
    if name == "round_robin":
        return RoundRobinResourceAllocation()
    if name == "shortest_queue":
        return ShortestQueueAllocation()
    if name == "parksong_composite":
        estimates = getattr(args, "_parksong_processing_time_estimates", None)
        default_processing_time = getattr(args, "_parksong_default_processing_time", 1.0)
        return ParkSongAllocation(
            prediction_probability_threshold=0.0,
            processing_time_estimates=estimates,
            default_processing_time=default_processing_time,
            cost_time_scale=3600.0,
            no_show_penalty_weight=1.0,
            future_delay_weight=0.0,
            reservation_margin=0.0,
            allow_strategic_idling=True,
        )
    if name == "kunkler":
        return KunklerAllocationAdapter()
    if name == "batch":
        return BatchAllocationAdapter(k_limit=5)
    raise ValueError(f"Unknown strategy: {name}")


def run_strategy(args, strategy_name: str, seed: int) -> dict:
    start = datetime(2016, 1, 4, 9, 0)
    end = start + timedelta(hours=args.hours)
    drain = end + timedelta(hours=args.drain_hours)
    branching = CompositeBranchingEngine(
        engines=[],
        seed=seed,
        use_default_hierarchy=False,
    )
    engine = IntegratedAllocationEngine(
        dataPath=args.log,
        seed=seed,
        allocation_strategy=build_strategy(strategy_name, args),
        branching_engine=branching,
        processing_time_artifact=args.processing_time_artifact,
        transition_model_path=args.transition_artifact,
        diagnostic_cycle_guard=False,
        cycle_repetition_limit=args.event_cap,
    )
    started = perf_counter()
    engine.run(start, end, format_type=[], drain_until=drain)
    runtime = round(perf_counter() - started, 3)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    event_log_path = output_dir / f"{strategy_name}_seed{seed}_event_log.csv"
    engine.logger.to_csv(str(event_log_path))
    diagnostics = engine.get_integration_diagnostics()
    admitted = len(engine.admitted_case_ids)
    completed = len(engine.completed_case_ids)
    return {
        "mode": "transition-aware BPMN and allocation integration smoke",
        "strategy": strategy_name,
        "seed": seed,
        "runtime_seconds": runtime,
        "event_log_path": str(event_log_path),
        "admitted": admitted,
        "completed": completed,
        "completion_rate": completed / admitted if admitted else 0.0,
        "deadlocked": len(engine.deadlocked_case_ids),
        "cyclic": len(engine.cyclic_case_ids),
        "censored": len(engine.censored_case_ids),
        "final_marking_rate": completed / admitted if admitted else 0.0,
        "events": len(engine.logger.records),
        "diagnostics": {
            "exact_transition_fires": diagnostics.get("exact_transition_fires", 0),
            "legacy_label_fires": diagnostics.get("legacy_label_fires", 0),
            "ambiguous_legacy_label_rejections": diagnostics.get(
                "ambiguous_legacy_label_rejections",
                0,
            ),
            "branch_predictions": diagnostics.get("branch_predictions", 0),
            "branch_transition_ambiguities": diagnostics.get(
                "branch_transition_ambiguities",
                0,
            ),
            "branch_invalid_predictions_rejected": diagnostics.get(
                "branch_invalid_predictions_rejected",
                0,
            ),
            "resources_allocated": diagnostics.get("resources_allocated", 0),
            "resources_released": diagnostics.get("resources_released", 0),
            "resource_overlap_violations": diagnostics.get("resource_overlap_violations", 0),
            "permission_violations": diagnostics.get("permission_violations", 0),
            "invalid_transition_fires": diagnostics.get("invalid_transition_fires", 0),
            "reservation_decisions": diagnostics.get("reservation_decisions", 0),
            "reservations_created": diagnostics.get("reservations_created", 0),
            "reservations_used": diagnostics.get("reservations_used", 0),
            "diagnostic_cycle_guard_enabled": diagnostics.get(
                "diagnostic_cycle_guard_enabled",
                0,
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="data/logData.xes")
    parser.add_argument(
        "--transition-artifact",
        default="joao/models/branching/transition_aware_branching_v1_20260715.pkl",
    )
    parser.add_argument(
        "--processing-time-artifact",
        default="joao/models/process_time/final_process_time_coverage_v2.pkl",
    )
    parser.add_argument(
        "--output-dir",
        default="joao/results/transition_aware_branching_20260715",
    )
    parser.add_argument("--hours", type=float, default=3.0)
    parser.add_argument("--drain-hours", type=float, default=8.0)
    parser.add_argument("--event-cap", type=int, default=100)
    parser.add_argument("--seeds", default="1")
    parser.add_argument(
        "--strategies",
        default="round_robin,shortest_queue,parksong_composite",
    )
    parser.add_argument("--split-ratio", type=float, default=0.7)
    args = parser.parse_args()

    (
        args._parksong_processing_time_estimates,
        args._parksong_default_processing_time,
        parksong_processing_summary,
    ) = build_parksong_processing_time_estimates(args.log, args.split_ratio)

    rows = []
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    for seed in [int(item) for item in args.seeds.split(",") if item.strip()]:
        for strategy_name in strategies:
            rows.append(run_strategy(args, strategy_name, seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "generative_integration_smoke.json"
    report_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    flat_rows = []
    diagnostics_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items() if key != "diagnostics"}
        flat.update(row["diagnostics"])
        flat_rows.append(flat)
        diagnostics_rows.append(
            {
                "strategy": row["strategy"],
                "seed": row["seed"],
                **row["diagnostics"],
            }
        )
    runs = pd.DataFrame(flat_rows)
    runs.to_csv(output_dir / "generative_runs.csv", index=False)
    pd.DataFrame(diagnostics_rows).to_csv(
        output_dir / "generative_method_diagnostics.csv",
        index=False,
    )
    summary_rows = []
    for strategy_name, group in runs.groupby("strategy"):
        admitted_total = int(group["admitted"].sum())
        completed_total = int(group["completed"].sum())
        summary_rows.append(
            {
                "strategy": strategy_name,
                "runs": int(len(group)),
                "admitted_total": admitted_total,
                "completed_total": completed_total,
                "pooled_completion_rate": completed_total / admitted_total if admitted_total else 0.0,
                "mean_seed_completion_rate": float(group["completion_rate"].mean()),
                "censored_total": int(group["censored"].sum()),
                "deadlocked_total": int(group["deadlocked"].sum()),
                "exact_transition_fires": int(group["exact_transition_fires"].sum()),
                "legacy_label_fires": int(group["legacy_label_fires"].sum()),
                "invalid_transition_fires": int(group["invalid_transition_fires"].sum()),
                "resource_violations": int(group["resource_overlap_violations"].sum()),
                "permission_violations": int(group["permission_violations"].sum()),
                "reservations_created": int(group["reservations_created"].sum()),
                "reservations_used": int(group["reservations_used"].sum()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_dir / "generative_summary.csv", index=False)
    config = {
        "mode": "transition-aware BPMN and allocation integration smoke",
        "log": args.log,
        "log_sha256": sha256_file(args.log),
        "transition_artifact": args.transition_artifact,
        "transition_artifact_sha256": sha256_file(args.transition_artifact),
        "processing_time_artifact": args.processing_time_artifact,
        "processing_time_artifact_sha256": sha256_file(args.processing_time_artifact),
        "hours": args.hours,
        "drain_hours": args.drain_hours,
        "event_cap": args.event_cap,
        "seeds": [int(item) for item in args.seeds.split(",") if item.strip()],
        "strategies": strategies,
        "parksong": {
            "cost_time_scale": 3600.0,
            "no_show_penalty_weight": 1.0,
            "future_delay_weight": 0.0,
            "reservation_margin": 0.0,
            "reservation_expiration_multiplier": 1.0,
            "processing_time_estimates": parksong_processing_summary,
        },
        "classifier_note": (
            "This smoke uses CompositeBranchingEngine(engines=[], use_default_hierarchy=False); "
            "branching choices are transition-aware integration choices, not the Random Forest composite classifier."
        ),
    }
    (output_dir / "generative_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "fixed_vs_generative.md").write_text(
        "\n".join(
            [
                "# Fixed replay vs generative smoke",
                "",
                "Fixed replay is the ranking benchmark for resource allocation on held-out historical routes.",
                "The generative run is a transition-aware BPMN and allocation integration smoke, not a ranking benchmark.",
                "It uses the transition-aware BPMN path with an empty CompositeBranchingEngine hierarchy, so it does not evaluate the Random Forest composite classifier.",
                "",
                "Completion is reported both as pooled completion and as mean seed-level completion.",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"report": str(report_path), "runs": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
