# tests/resource_allocation/test_resource_allocation_metrics.py

from src.resource_allocation.ResourceAllocationMetrics import (
    average_cycle_time,
    average_resource_occupation,
    average_waiting_time,
    resource_fairness,
    resource_occupation,
    weighted_resource_fairness,
)


def test_average_cycle_time():
    case_times = [
        {"case_id": "C1", "arrival_time": 0.0, "completion_time": 10.0},
        {"case_id": "C2", "arrival_time": 2.0, "completion_time": 8.0},
    ]

    assert average_cycle_time(case_times) == 8.0


def test_average_cycle_time_ignores_incomplete_cases():
    case_times = [
        {"case_id": "C1", "arrival_time": 0.0, "completion_time": 10.0},
        {"case_id": "C2", "arrival_time": 2.0, "completion_time": None},
    ]

    assert average_cycle_time(case_times) == 10.0


def test_average_cycle_time_returns_none_for_empty_valid_input():
    case_times = [
        {"case_id": "C1", "arrival_time": None, "completion_time": 10.0},
        {"case_id": "C2", "arrival_time": 2.0, "completion_time": None},
    ]

    assert average_cycle_time(case_times) is None


def test_average_waiting_time():
    task_times = [
        {"task_id": "T1", "enabled_time": 0.0, "start_time": 2.0},
        {"task_id": "T2", "enabled_time": 1.0, "start_time": 4.0},
    ]

    assert average_waiting_time(task_times) == 2.5


def test_average_waiting_time_ignores_incomplete_tasks():
    task_times = [
        {"task_id": "T1", "enabled_time": 0.0, "start_time": 2.0},
        {"task_id": "T2", "enabled_time": 1.0, "start_time": None},
    ]

    assert average_waiting_time(task_times) == 2.0


def test_resource_occupation_per_resource():
    intervals = [
        {"resource_id": "R1", "start_time": 0.0, "end_time": 5.0},
        {"resource_id": "R1", "start_time": 6.0, "end_time": 9.0},
        {"resource_id": "R2", "start_time": 0.0, "end_time": 10.0},
    ]

    availability = {
        "R1": 20.0,
        "R2": 20.0,
    }

    occupation = resource_occupation(intervals, availability)

    assert occupation["R1"] == 0.4
    assert occupation["R2"] == 0.5


def test_average_resource_occupation():
    intervals = [
        {"resource_id": "R1", "start_time": 0.0, "end_time": 5.0},
        {"resource_id": "R2", "start_time": 0.0, "end_time": 10.0},
    ]

    availability = {
        "R1": 10.0,
        "R2": 10.0,
    }

    assert average_resource_occupation(intervals, availability) == 0.75


def test_resource_fairness():
    intervals = [
        {"resource_id": "R1", "start_time": 0.0, "end_time": 5.0},
        {"resource_id": "R2", "start_time": 0.0, "end_time": 10.0},
    ]

    availability = {
        "R1": 10.0,
        "R2": 10.0,
    }

    assert resource_fairness(intervals, availability) == 0.25


def test_weighted_resource_fairness():
    intervals = [
        {"resource_id": "R1", "start_time": 0.0, "end_time": 5.0},
        {"resource_id": "R2", "start_time": 0.0, "end_time": 10.0},
    ]

    availability = {
        "R1": 10.0,
        "R2": 30.0,
    }

    result = weighted_resource_fairness(intervals, availability)

    assert round(result, 4) == 0.0625
    

def test_resource_occupation_handles_zero_availability():
    intervals = [
        {"resource_id": "R1", "start_time": 0.0, "end_time": 5.0},
    ]

    availability = {
        "R1": 0.0,
    }

    occupation = resource_occupation(intervals, availability)

    assert occupation["R1"] == 0.0