import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.branching.PredictiveBranchingEngine import PredictiveBranchingEngine
from src.resource_allocation.AllocationStrategy import Resource, Task
from src.resource_allocation.RandomResourceAllocation import RandomResourceAllocation
from src.resource_allocation.ShortestQueueAllocation import ShortestQueueAllocation
from src.resource_allocation.ParkSongAllocation import ParkSongAllocation
from src.resource_allocation.MLPredictionAdapter import MLPredictionAdapter
from src.resource_allocation.ParkSongMLIntegration import ParkSongMLIntegration
from src.resource_allocation.ResourceAllocationEvaluator import ResourceAllocationEvaluator


def build_demo_log():
    data = [
        {
            "case:concept:name": "C1",
            "concept:name": "A_START",
            "time:timestamp": "2026-01-01 08:00:00",
            "case:ApplicationType": "new",
            "case:LoanGoal": "car",
            "case:RequestedAmount": 10000,
            "CreditScore": 720,
            "EventOrigin": "online",
            "org:resource": "R1",
        },
        {
            "case:concept:name": "C1",
            "concept:name": "A_APPROVED",
            "time:timestamp": "2026-01-01 09:00:00",
            "case:ApplicationType": "new",
            "case:LoanGoal": "car",
            "case:RequestedAmount": 10000,
            "CreditScore": 720,
            "EventOrigin": "online",
            "org:resource": "R1",
        },
        {
            "case:concept:name": "C2",
            "concept:name": "A_START",
            "time:timestamp": "2026-01-01 08:10:00",
            "case:ApplicationType": "new",
            "case:LoanGoal": "home",
            "case:RequestedAmount": 70000,
            "CreditScore": 450,
            "EventOrigin": "branch",
            "org:resource": "R2",
        },
        {
            "case:concept:name": "C2",
            "concept:name": "A_REJECTED",
            "time:timestamp": "2026-01-01 09:20:00",
            "case:ApplicationType": "new",
            "case:LoanGoal": "home",
            "case:RequestedAmount": 70000,
            "CreditScore": 450,
            "EventOrigin": "branch",
            "org:resource": "R2",
        },
        {
            "case:concept:name": "C3",
            "concept:name": "A_START",
            "time:timestamp": "2026-01-01 08:20:00",
            "case:ApplicationType": "repeat",
            "case:LoanGoal": "car",
            "case:RequestedAmount": 15000,
            "CreditScore": 690,
            "EventOrigin": "online",
            "org:resource": "R1",
        },
        {
            "case:concept:name": "C3",
            "concept:name": "A_APPROVED",
            "time:timestamp": "2026-01-01 09:30:00",
            "case:ApplicationType": "repeat",
            "case:LoanGoal": "car",
            "case:RequestedAmount": 15000,
            "CreditScore": 690,
            "EventOrigin": "online",
            "org:resource": "R1",
        },
        {
            "case:concept:name": "C4",
            "concept:name": "A_START",
            "time:timestamp": "2026-01-01 08:30:00",
            "case:ApplicationType": "new",
            "case:LoanGoal": "business",
            "case:RequestedAmount": 90000,
            "CreditScore": 400,
            "EventOrigin": "branch",
            "org:resource": "R2",
        },
        {
            "case:concept:name": "C4",
            "concept:name": "A_REJECTED",
            "time:timestamp": "2026-01-01 09:40:00",
            "case:ApplicationType": "new",
            "case:LoanGoal": "business",
            "case:RequestedAmount": 90000,
            "CreditScore": 400,
            "EventOrigin": "branch",
            "org:resource": "R2",
        },
    ]

    return pd.DataFrame(data)


def build_resources():
    return [
        Resource(
            resource_id="R1",
            available=True,
            skills=["A_CURRENT", "A_APPROVED", "A_REJECTED"],
        ),
        Resource(
            resource_id="R2",
            available=True,
            skills=["A_CURRENT", "A_APPROVED", "A_REJECTED"],
        ),
    ]


def build_waiting_tasks():
    return [
        Task(
            task_id="T1",
            case_id="C_WAIT_1",
            activity="A_CURRENT",
            enabled_time=0.0,
            priority=0.0,
        ),
        Task(
            task_id="T2",
            case_id="C_WAIT_2",
            activity="A_CURRENT",
            enabled_time=2.0,
            priority=1.0,
        ),
    ]


def clone_tasks(tasks):
    return [
        Task(
            task_id=task.task_id,
            case_id=task.case_id,
            activity=task.activity,
            enabled_time=task.enabled_time,
            assigned=task.assigned,
            blocked=task.blocked,
            priority=task.priority,
        )
        for task in tasks
    ]


def clone_resources(resources):
    return [
        Resource(
            resource_id=resource.resource_id,
            available=resource.available,
            skills=list(resource.skills) if resource.skills is not None else None,
        )
        for resource in resources
    ]


def build_predictive_integration(train_log):
    predictive_engine = PredictiveBranchingEngine(
        feature_columns=[
            "case:ApplicationType",
            "case:LoanGoal",
            "case:RequestedAmount",
            "CreditScore",
            "EventOrigin",
            "org:resource",
        ],
        n_estimators=50,
        max_depth=6,
        seed=1,
    )

    predictive_engine.train(train_log)

    adapter = MLPredictionAdapter(
        predictive_engine=predictive_engine,
        default_expected_delay=1.0,
    )

    allocator = ParkSongAllocation(
        prediction_probability_threshold=0.5,
        allow_strategic_idling=True,
    )

    return ParkSongMLIntegration(
        prediction_adapter=adapter,
        allocator=allocator,
    ), adapter


def decisions_to_metric_inputs(strategy_name, decisions):
    """
    Creates simple synthetic metric inputs from allocation decisions.

    This is not the full simulator evaluation yet.
    It is an integration sanity check for the 1.2 metrics pipeline.
    """

    case_times = []
    task_times = []
    resource_intervals = []
    availability_times = {}

    for index, decision in enumerate(decisions):
        availability_times[decision.resource_id] = 10.0

        if decision.decision_type == "assignment":
            case_times.append(
                {
                    "case_id": decision.case_id,
                    "arrival_time": 0.0,
                    "completion_time": 5.0 + index,
                }
            )

            task_times.append(
                {
                    "task_id": decision.task_id,
                    "enabled_time": 0.0,
                    "start_time": 1.0 + index,
                }
            )

            resource_intervals.append(
                {
                    "resource_id": decision.resource_id,
                    "start_time": 1.0 + index,
                    "end_time": 4.0 + index,
                }
            )

        elif decision.decision_type == "reservation":
            resource_intervals.append(
                {
                    "resource_id": decision.resource_id,
                    "start_time": 0.0,
                    "end_time": 0.0,
                }
            )

    return {
        "case_times": case_times,
        "task_times": task_times,
        "resource_intervals": resource_intervals,
        "availability_times": availability_times,
    }


def main():
    train_log = build_demo_log()

    current_event = {
        "case:concept:name": "C5",
        "concept:name": "A_START",
        "time:timestamp": "2026-01-01 10:00:00",
        "case:ApplicationType": "new",
        "case:LoanGoal": "car",
        "case:RequestedAmount": 12000,
        "CreditScore": 710,
        "EventOrigin": "online",
        "org:resource": "R1",
        "event_index": 0,
    }

    possible_activities = ["A_APPROVED", "A_REJECTED"]

    resources = build_resources()
    waiting_tasks = build_waiting_tasks()
    current_time = 3.0

    strategies = {
        "Random": RandomResourceAllocation(seed=1),
        "ShortestQueue": ShortestQueueAllocation(),
        "ParkSong": ParkSongAllocation(
            prediction_probability_threshold=0.5,
            allow_strategic_idling=True,
        ),
    }

    parksong_ml_integration, adapter = build_predictive_integration(train_log)

    simulation_results = {}

    print("\n=== Resource Allocation Strategy Comparison ===")

    for strategy_name, strategy in strategies.items():
        local_resources = clone_resources(resources)
        local_tasks = clone_tasks(waiting_tasks)

        if strategy_name == "ParkSong":
            predictions = adapter.predict_for_event(
                event=current_event,
                possible_activities=possible_activities,
            )

            decisions = strategy.allocate(
                resources=local_resources,
                waiting_tasks=local_tasks,
                current_time=current_time,
                predictions=predictions,
            )
        else:
            decisions = strategy.allocate(
                resources=local_resources,
                waiting_tasks=local_tasks,
                current_time=current_time,
            )

        print(f"\n--- {strategy_name} decisions ---")
        for decision in decisions:
            print(decision)

        simulation_results[strategy_name] = decisions_to_metric_inputs(
            strategy_name=strategy_name,
            decisions=decisions,
        )

    local_resources = clone_resources(resources)
    local_tasks = []

    ml_decisions = parksong_ml_integration.allocate_with_ml_predictions(
        event=current_event,
        possible_activities=possible_activities,
        resources=local_resources,
        waiting_tasks=local_tasks,
        current_time=current_time,
    )

    print("\n--- ParkSongML decisions ---")
    for decision in ml_decisions:
        print(decision)

    simulation_results["ParkSongML"] = decisions_to_metric_inputs(
        strategy_name="ParkSongML",
        decisions=ml_decisions,
    )

    evaluator = ResourceAllocationEvaluator()
    evaluation_results = evaluator.evaluate_multiple(simulation_results)

    results_df = pd.DataFrame(evaluation_results)

    print("\n=== Evaluation Metrics ===")
    print(results_df.to_string(index=False))

    output_path = PROJECT_ROOT / "results" / "resource_allocation_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\nSaved metrics to: {output_path}")


if __name__ == "__main__":
    main()
