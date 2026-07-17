"""
Controlled/synthetic sanity check; not the source of final quantitative claims.

Final Part II-A claims come from
joao/results/final_canonical_rfopt_candidate_20260717/fixed_replay.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.resource_allocation.AllocationStrategy import Resource, Task, Prediction
from src.resource_allocation.RandomResourceAllocation import RandomResourceAllocation
from src.resource_allocation.ShortestQueueAllocation import ShortestQueueAllocation
from src.resource_allocation.ParkSongAllocation import ParkSongAllocation
from src.resource_allocation.ResourceAllocationEvaluator import ResourceAllocationEvaluator


def clone_resources(resources):
    return [
        Resource(
            resource_id=r.resource_id,
            available=r.available,
            skills=list(r.skills) if r.skills is not None else None,
        )
        for r in resources
    ]


def clone_tasks(tasks):
    return [
        Task(
            task_id=t.task_id,
            case_id=t.case_id,
            activity=t.activity,
            enabled_time=t.enabled_time,
            assigned=t.assigned,
            blocked=t.blocked,
            priority=t.priority,
        )
        for t in tasks
    ]


def build_predictions():
    return [
        Prediction(
            case_id="C_FUTURE_1",
            activity="A",
            probability=0.85,
            expected_delay=1.0,
            source="ControlledMLPrediction",
            confidence=0.85,
        ),
        Prediction(
            case_id="C_FUTURE_2",
            activity="B",
            probability=0.40,
            expected_delay=2.0,
            source="ControlledMLPrediction",
            confidence=0.40,
        ),
    ]


def build_metric_inputs(decisions, current_time):
    case_times = []
    task_times = []
    resource_intervals = []
    availability_times = {}

    for idx, decision in enumerate(decisions):
        availability_times[decision.resource_id] = 10.0

        if decision.decision_type == "assignment":
            case_times.append(
                {
                    "case_id": decision.case_id,
                    "arrival_time": 0.0,
                    "completion_time": current_time + 3.0 + idx,
                }
            )

            task_times.append(
                {
                    "task_id": decision.task_id,
                    "enabled_time": max(0.0, current_time - 2.0 - idx),
                    "start_time": current_time,
                }
            )

            resource_intervals.append(
                {
                    "resource_id": decision.resource_id,
                    "start_time": current_time,
                    "end_time": current_time + 3.0,
                }
            )

        elif decision.decision_type == "reservation":
            # Reservation is strategic idling; it does not create a completed task.
            availability_times[decision.resource_id] = 10.0
            resource_intervals.append(
                {
                    "resource_id": decision.resource_id,
                    "start_time": current_time,
                    "end_time": current_time,
                }
            )

    return {
        "case_times": case_times,
        "task_times": task_times,
        "resource_intervals": resource_intervals,
        "availability_times": availability_times,
    }


def build_scenarios():
    return {
        "balanced_tasks": {
            "resources": [
                Resource("R1", True, ["A", "B", "C"]),
                Resource("R2", True, ["A", "B", "C"]),
            ],
            "tasks": [
                Task("T1", "C1", "A", enabled_time=0.0, priority=0.0),
                Task("T2", "C2", "B", enabled_time=1.0, priority=0.0),
            ],
            "current_time": 4.0,
        },
        "scarce_resources": {
            "resources": [
                Resource("R1", True, ["A", "B", "C"]),
            ],
            "tasks": [
                Task("T1", "C1", "A", enabled_time=0.0, priority=0.0),
                Task("T2", "C2", "B", enabled_time=1.0, priority=0.0),
                Task("T3", "C3", "C", enabled_time=2.0, priority=0.0),
            ],
            "current_time": 5.0,
        },
        "unavailable_resource": {
            "resources": [
                Resource("R1", False, ["A", "B", "C"]),
                Resource("R2", True, ["A", "B", "C"]),
            ],
            "tasks": [
                Task("T1", "C1", "A", enabled_time=0.0, priority=0.0),
                Task("T2", "C2", "B", enabled_time=1.0, priority=0.0),
            ],
            "current_time": 4.0,
        },
        "priority_old_task": {
            "resources": [
                Resource("R1", True, ["A", "B", "C"]),
                Resource("R2", True, ["A", "B", "C"]),
            ],
            "tasks": [
                Task("T1", "C1", "A", enabled_time=0.0, priority=0.0),
                Task("T2", "C2", "B", enabled_time=3.5, priority=10.0),
                Task("T3", "C3", "B", enabled_time=3.8, priority=0.0),
            ],
            "current_time": 6.0,
        },
        "future_task_only": {
            "resources": [
                Resource("R1", True, ["A", "B", "C"]),
                Resource("R2", True, ["A", "B", "C"]),
            ],
            "tasks": [],
            "current_time": 5.0,
        },
    }


def main():
    scenarios = build_scenarios()
    predictions = build_predictions()

    strategies = {
        "Random": RandomResourceAllocation(seed=1),
        "ShortestQueue": ShortestQueueAllocation(),
        "ParkSong": ParkSongAllocation(
            prediction_probability_threshold=0.5,
            allow_strategic_idling=False,
        ),
        "ParkSongML": ParkSongAllocation(
            prediction_probability_threshold=0.5,
            allow_strategic_idling=True,
        ),
    }

    evaluator = ResourceAllocationEvaluator()

    metric_rows = []
    decision_rows = []

    for scenario_name, scenario in scenarios.items():
        current_time = scenario["current_time"]

        for strategy_name, strategy in strategies.items():
            resources = clone_resources(scenario["resources"])
            tasks = clone_tasks(scenario["tasks"])

            if strategy_name == "ParkSongML":
                decisions = strategy.allocate(
                    resources=resources,
                    waiting_tasks=tasks,
                    current_time=current_time,
                    predictions=predictions,
                )
            else:
                decisions = strategy.allocate(
                    resources=resources,
                    waiting_tasks=tasks,
                    current_time=current_time,
                )

            metric_input = build_metric_inputs(decisions, current_time)

            metrics = evaluator.evaluate(
                strategy_name=strategy_name,
                case_times=metric_input["case_times"],
                task_times=metric_input["task_times"],
                resource_intervals=metric_input["resource_intervals"],
                availability_times=metric_input["availability_times"],
            )

            metrics["scenario"] = scenario_name
            metric_rows.append(metrics)

            for decision in decisions:
                decision_rows.append(
                    {
                        "scenario": scenario_name,
                        "strategy": strategy_name,
                        "resource_id": decision.resource_id,
                        "task_id": decision.task_id,
                        "activity": decision.activity,
                        "case_id": decision.case_id,
                        "decision_type": decision.decision_type,
                        "reason": decision.reason,
                    }
                )

    metrics_df = pd.DataFrame(metric_rows)
    decisions_df = pd.DataFrame(decision_rows)

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    metrics_path = output_dir / "resource_allocation_scenario_comparison.csv"
    decisions_path = output_dir / "resource_allocation_scenario_decisions.csv"

    metrics_df.to_csv(metrics_path, index=False)
    decisions_df.to_csv(decisions_path, index=False)

    print("\n=== Resource Allocation Scenario Metrics ===")
    print(
        metrics_df[
            [
                "scenario",
                "strategy",
                "average_cycle_time",
                "average_waiting_time",
                "average_resource_occupation",
                "resource_fairness",
                "weighted_resource_fairness",
            ]
        ].to_string(index=False)
    )

    print("\n=== Decisions Preview ===")
    print(decisions_df.head(40).to_string(index=False))

    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved decisions to: {decisions_path}")


if __name__ == "__main__":
    main()
