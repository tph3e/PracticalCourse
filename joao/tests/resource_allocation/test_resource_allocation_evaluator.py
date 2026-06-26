# tests/resource_allocation/test_resource_allocation_evaluator.py

from src.resource_allocation.ResourceAllocationEvaluator import ResourceAllocationEvaluator


def test_evaluator_computes_metrics_for_one_strategy():
    evaluator = ResourceAllocationEvaluator()

    case_times = [
        {"case_id": "C1", "arrival_time": 0.0, "completion_time": 10.0},
        {"case_id": "C2", "arrival_time": 2.0, "completion_time": 8.0},
    ]

    task_times = [
        {"task_id": "T1", "enabled_time": 0.0, "start_time": 2.0},
        {"task_id": "T2", "enabled_time": 1.0, "start_time": 4.0},
    ]

    resource_intervals = [
        {"resource_id": "R1", "start_time": 0.0, "end_time": 5.0},
        {"resource_id": "R2", "start_time": 0.0, "end_time": 10.0},
    ]

    availability_times = {
        "R1": 10.0,
        "R2": 10.0,
    }

    result = evaluator.evaluate(
        strategy_name="R-RRA",
        case_times=case_times,
        task_times=task_times,
        resource_intervals=resource_intervals,
        availability_times=availability_times,
    )

    assert result["strategy"] == "R-RRA"
    assert result["average_cycle_time"] == 8.0
    assert result["average_waiting_time"] == 2.5
    assert result["average_resource_occupation"] == 0.75
    assert result["resource_fairness"] == 0.25
    assert result["weighted_resource_fairness"] == 0.25


def test_evaluator_computes_metrics_for_multiple_strategies():
    evaluator = ResourceAllocationEvaluator()

    simulation_results = {
        "R-RRA": {
            "case_times": [
                {"case_id": "C1", "arrival_time": 0.0, "completion_time": 10.0},
            ],
            "task_times": [
                {"task_id": "T1", "enabled_time": 0.0, "start_time": 2.0},
            ],
            "resource_intervals": [
                {"resource_id": "R1", "start_time": 0.0, "end_time": 5.0},
            ],
            "availability_times": {
                "R1": 10.0,
            },
        },
        "R-SHQ": {
            "case_times": [
                {"case_id": "C2", "arrival_time": 0.0, "completion_time": 8.0},
            ],
            "task_times": [
                {"task_id": "T2", "enabled_time": 0.0, "start_time": 1.0},
            ],
            "resource_intervals": [
                {"resource_id": "R1", "start_time": 0.0, "end_time": 6.0},
            ],
            "availability_times": {
                "R1": 10.0,
            },
        },
    }

    results = evaluator.evaluate_multiple(simulation_results)

    assert len(results) == 2

    strategies = [result["strategy"] for result in results]

    assert "R-RRA" in strategies
    assert "R-SHQ" in strategies

    r_rra_result = next(result for result in results if result["strategy"] == "R-RRA")
    r_shq_result = next(result for result in results if result["strategy"] == "R-SHQ")

    assert r_rra_result["average_cycle_time"] == 10.0
    assert r_rra_result["average_waiting_time"] == 2.0
    assert r_rra_result["average_resource_occupation"] == 0.5

    assert r_shq_result["average_cycle_time"] == 8.0
    assert r_shq_result["average_waiting_time"] == 1.0
    assert r_shq_result["average_resource_occupation"] == 0.6