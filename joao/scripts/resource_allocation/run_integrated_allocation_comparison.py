from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from datetime import datetime
from typing import Any
from types import MethodType

import pandas as pd

JOAO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = JOAO_ROOT.parent
sys.path.insert(0, str(JOAO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from SimulationEngineCore import Engine
from src.resource_allocation.ParkSongAllocation import ParkSongAllocation
from src.resource_allocation.RandomResourceAllocation import RandomResourceAllocation
from src.resource_allocation.ShortestQueueAllocation import ShortestQueueAllocation


CASE_COL = "case:concept:name"
RESOURCE_COL = "org:resource"
TIMESTAMP_COL = "time:timestamp"
LIFECYCLE_COL = "lifecycle:transition"
ACTIVITY_COL = "concept:name"


def build_strategies(seed: int) -> dict[str, Any]:
    return {
        "Random": RandomResourceAllocation(seed=seed),
        "ShortestQueue": ShortestQueueAllocation(),
        "ParkSong": ParkSongAllocation(allow_strategic_idling=False),
    }


def run_integrated_comparison(
    data_path: str,
    start_time: datetime,
    end_time: datetime,
    seed: int,
    output_path: Path,
) -> pd.DataFrame:
    rows = []

    for strategy_name, strategy in build_strategies(seed).items():
        engine = Engine(dataPath=data_path, seed=seed)
        engine.resourceEngine.global_allocation_strategy = strategy
        diagnostics = attach_allocation_diagnostics(engine)

        engine.run(
            start_time=start_time,
            end_time=end_time,
            format_type=[],
        )

        log = engine.logger.get_log()
        metrics = compute_metrics(log)
        metrics.update(compute_log_diagnostics(log))
        metrics.update(diagnostics)
        metrics.update(
            {
                "strategy": strategy_name,
                "seed": seed,
                "simulation_start": start_time.isoformat(),
                "simulation_end": end_time.isoformat(),
            }
        )
        rows.append(metrics)

    result = pd.DataFrame(rows)
    ordered_columns = [
        "strategy",
        "seed",
        "simulation_start",
        "simulation_end",
        "n_events",
        "n_cases",
        "assigned_events",
        "average_cycle_time",
        "average_waiting_time",
        "average_resource_occupation",
        "resource_fairness",
        "weighted_resource_fairness",
        "allocate_waiting_task_calls",
        "global_strategy_calls",
        "waiting_events_seen",
        "global_assignments",
        "old_path_assignments",
        "suspended_events",
        "resumed_events",
        "max_waiting_queue_length",
    ]
    result = result[ordered_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def compute_metrics(log: pd.DataFrame) -> dict[str, float | int]:
    if log.empty:
        return {
            "n_events": 0,
            "n_cases": 0,
            "assigned_events": 0,
            "average_cycle_time": math.nan,
            "average_waiting_time": math.nan,
            "average_resource_occupation": math.nan,
            "resource_fairness": math.nan,
            "weighted_resource_fairness": math.nan,
        }

    prepared = log.copy()
    prepared[TIMESTAMP_COL] = pd.to_datetime(
        prepared[TIMESTAMP_COL],
        errors="coerce",
    )
    prepared = prepared.dropna(subset=[TIMESTAMP_COL])

    assigned = prepared[RESOURCE_COL].fillna("").astype(str).str.len() > 0
    resource_counts = prepared.loc[assigned, RESOURCE_COL].astype(str).value_counts()

    return {
        "n_events": int(len(prepared)),
        "n_cases": int(prepared[CASE_COL].nunique()) if CASE_COL in prepared else 0,
        "assigned_events": int(assigned.sum()),
        "average_cycle_time": average_cycle_time(prepared),
        "average_waiting_time": average_waiting_time(prepared),
        "average_resource_occupation": average_resource_occupation(prepared),
        "resource_fairness": gini(resource_counts.tolist()),
        "weighted_resource_fairness": math.nan,
    }


def attach_allocation_diagnostics(engine) -> dict[str, int]:
    diagnostics = {
        "allocate_waiting_task_calls": 0,
        "global_strategy_calls": 0,
        "waiting_events_seen": 0,
        "global_assignments": 0,
        "old_path_assignments": 0,
        "max_waiting_queue_length": 0,
    }

    resource_engine = engine.resourceEngine
    original_allocate_resource = resource_engine.allocateResource
    original_allocate_waiting_tasks = resource_engine.allocate_waiting_tasks
    strategy = resource_engine.global_allocation_strategy
    original_strategy_allocate = getattr(strategy, "allocate", None)

    def counted_allocate_resource(self, event):
        allocated = original_allocate_resource(event)
        if allocated:
            diagnostics["old_path_assignments"] += 1
        return allocated

    def counted_allocate_waiting_tasks(self, waiting_events, current_time, predictions=None):
        waiting_count = len(waiting_events)
        diagnostics["allocate_waiting_task_calls"] += 1
        diagnostics["waiting_events_seen"] += waiting_count
        diagnostics["max_waiting_queue_length"] = max(
            diagnostics["max_waiting_queue_length"],
            waiting_count,
        )

        decisions = original_allocate_waiting_tasks(
            waiting_events=waiting_events,
            current_time=current_time,
            predictions=predictions,
        )
        diagnostics["global_assignments"] += sum(
            1
            for decision in decisions
            if getattr(decision, "decision_type", None) == "assignment"
        )
        return decisions

    def counted_strategy_allocate(self, *args, **kwargs):
        diagnostics["global_strategy_calls"] += 1
        return original_strategy_allocate(*args, **kwargs)

    resource_engine.allocateResource = MethodType(
        counted_allocate_resource,
        resource_engine,
    )
    resource_engine.allocate_waiting_tasks = MethodType(
        counted_allocate_waiting_tasks,
        resource_engine,
    )
    if original_strategy_allocate is not None:
        strategy.allocate = MethodType(
            counted_strategy_allocate,
            strategy,
        )

    return diagnostics


def compute_log_diagnostics(log: pd.DataFrame) -> dict[str, int]:
    if log.empty or LIFECYCLE_COL not in log:
        return {
            "suspended_events": 0,
            "resumed_events": 0,
        }

    lifecycle = log[LIFECYCLE_COL].astype(str)
    return {
        "suspended_events": int((lifecycle == "suspend").sum()),
        "resumed_events": int((lifecycle == "resume").sum()),
    }


def average_cycle_time(log: pd.DataFrame) -> float:
    if CASE_COL not in log or log.empty:
        return math.nan

    spans = (
        log.groupby(CASE_COL)[TIMESTAMP_COL]
        .agg(lambda values: values.max() - values.min())
        .dt.total_seconds()
    )
    spans = spans[spans >= 0]
    return float(spans.mean()) if not spans.empty else math.nan


def average_waiting_time(log: pd.DataFrame) -> float:
    required = {CASE_COL, ACTIVITY_COL, TIMESTAMP_COL, LIFECYCLE_COL}
    if not required.issubset(log.columns):
        return math.nan

    waiting_times = []

    for _, group in log.sort_values(TIMESTAMP_COL).groupby([CASE_COL, ACTIVITY_COL]):
        suspended_at = None

        for _, row in group.iterrows():
            lifecycle = str(row[LIFECYCLE_COL])
            timestamp = row[TIMESTAMP_COL]

            if lifecycle == "suspend":
                suspended_at = timestamp
            elif lifecycle == "resume" and suspended_at is not None:
                duration = (timestamp - suspended_at).total_seconds()
                if duration >= 0:
                    waiting_times.append(duration)
                suspended_at = None

    return float(sum(waiting_times) / len(waiting_times)) if waiting_times else math.nan


def average_resource_occupation(log: pd.DataFrame) -> float:
    required = {RESOURCE_COL, TIMESTAMP_COL, LIFECYCLE_COL}
    if not required.issubset(log.columns) or log.empty:
        return math.nan

    window = (log[TIMESTAMP_COL].max() - log[TIMESTAMP_COL].min()).total_seconds()
    if window <= 0:
        return math.nan

    occupations = []

    for _, group in log.sort_values(TIMESTAMP_COL).groupby(RESOURCE_COL):
        busy_start = None
        busy_seconds = 0.0

        for _, row in group.iterrows():
            lifecycle = str(row[LIFECYCLE_COL])
            timestamp = row[TIMESTAMP_COL]

            if lifecycle in {"start", "resume"}:
                busy_start = timestamp
            elif lifecycle in {"complete", "suspend"} and busy_start is not None:
                duration = (timestamp - busy_start).total_seconds()
                if duration >= 0:
                    busy_seconds += duration
                busy_start = None

        occupations.append(busy_seconds / window)

    return float(sum(occupations) / len(occupations)) if occupations else math.nan


def gini(values: list[int | float]) -> float:
    if not values:
        return math.nan

    sorted_values = sorted(float(value) for value in values)
    total = sum(sorted_values)
    count = len(sorted_values)

    if total <= 0:
        return 0.0

    weighted_sum = sum((index + 1) * value for index, value in enumerate(sorted_values))
    return float((2 * weighted_sum) / (count * total) - (count + 1) / count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run integrated resource allocation comparison in the main simulator."
    )
    parser.add_argument("--data-path", default=str(REPO_ROOT / "data" / "logData.xes"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--start", default="2000-01-03T09:00:00")
    parser.add_argument("--end", default="2000-01-03T12:00:00")
    parser.add_argument(
        "--output",
        default=str(JOAO_ROOT / "results" / "integrated_resource_allocation_comparison.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_integrated_comparison(
        data_path=args.data_path,
        start_time=datetime.fromisoformat(args.start),
        end_time=datetime.fromisoformat(args.end),
        seed=args.seed,
        output_path=Path(args.output),
    )
    print(result.to_string(index=False))
    print(f"\nSaved integrated comparison to: {args.output}")


if __name__ == "__main__":
    main()
