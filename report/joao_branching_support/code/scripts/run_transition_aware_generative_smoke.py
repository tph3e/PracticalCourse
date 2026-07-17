from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter

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


def build_strategy(name: str):
    if name == "round_robin":
        return RoundRobinResourceAllocation()
    if name == "shortest_queue":
        return ShortestQueueAllocation()
    if name == "parksong_composite":
        return ParkSongAllocation(prediction_probability_threshold=0.0)
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
        allocation_strategy=build_strategy(strategy_name),
        branching_engine=branching,
        processing_time_artifact=args.processing_time_artifact,
        transition_model_path=args.transition_artifact,
        diagnostic_cycle_guard=False,
        cycle_repetition_limit=args.event_cap,
    )
    started = perf_counter()
    engine.run(start, end, format_type=[], drain_until=drain)
    runtime = round(perf_counter() - started, 3)
    output_dir = Path(args.output_dir) / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    event_log_path = output_dir / f"{strategy_name}_seed{seed}_event_log.csv"
    engine.logger.to_csv(str(event_log_path))
    diagnostics = engine.get_integration_diagnostics()
    return {
        "strategy": strategy_name,
        "seed": seed,
        "runtime_seconds": runtime,
        "event_log_path": str(event_log_path),
        "admitted": len(engine.admitted_case_ids),
        "completed": len(engine.completed_case_ids),
        "deadlocked": len(engine.deadlocked_case_ids),
        "cyclic": len(engine.cyclic_case_ids),
        "censored": len(engine.censored_case_ids),
        "final_marking_rate": (
            len(engine.completed_case_ids) / len(engine.admitted_case_ids)
            if engine.admitted_case_ids
            else 0.0
        ),
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
            "reservation_decisions": diagnostics.get("reservation_decisions", 0),
            "reservations_created": diagnostics.get("reservations_created", 0),
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
    args = parser.parse_args()

    rows = []
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    for seed in [int(item) for item in args.seeds.split(",") if item.strip()]:
        for strategy_name in strategies:
            rows.append(run_strategy(args, strategy_name, seed))
    output_dir = Path(args.output_dir) / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "generative_integration_smoke.json"
    report_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "runs": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
