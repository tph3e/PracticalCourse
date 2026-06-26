# src/resource_allocation/ResourceAllocationMetrics.py

from collections import defaultdict
from typing import Dict, List, Optional


def average_cycle_time(case_times: List[Dict]) -> Optional[float]:
    """
    Compute the average cycle time of completed cases.

    Cycle time = completion_time - arrival_time

    Expected input:
    [
        {"case_id": "C1", "arrival_time": 0.0, "completion_time": 10.0},
        {"case_id": "C2", "arrival_time": 2.0, "completion_time": 8.0},
    ]
    """
    cycle_times = []

    for case in case_times:
        arrival_time = case.get("arrival_time")
        completion_time = case.get("completion_time")

        if arrival_time is None or completion_time is None:
            continue

        cycle_times.append(completion_time - arrival_time)

    if not cycle_times:
        return None

    return sum(cycle_times) / len(cycle_times)


def average_waiting_time(task_times: List[Dict]) -> Optional[float]:
    """
    Compute the average waiting time of enabled tasks.

    Waiting time = start_time - enabled_time

    Expected input:
    [
        {"task_id": "T1", "enabled_time": 1.0, "start_time": 4.0},
        {"task_id": "T2", "enabled_time": 2.0, "start_time": 3.0},
    ]
    """
    waiting_times = []

    for task in task_times:
        enabled_time = task.get("enabled_time")
        start_time = task.get("start_time")

        if enabled_time is None or start_time is None:
            continue

        waiting_times.append(start_time - enabled_time)

    if not waiting_times:
        return None

    return sum(waiting_times) / len(waiting_times)


def resource_occupation(
    resource_intervals: List[Dict],
    availability_times: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute the occupation rate of each resource.

    Occupation = busy_time / available_time

    Expected resource_intervals:
    [
        {"resource_id": "R1", "start_time": 0.0, "end_time": 5.0},
        {"resource_id": "R2", "start_time": 2.0, "end_time": 8.0},
    ]

    Expected availability_times:
    {
        "R1": 10.0,
        "R2": 10.0,
    }
    """
    busy_time = defaultdict(float)

    for interval in resource_intervals:
        resource_id = interval["resource_id"]
        start_time = interval["start_time"]
        end_time = interval["end_time"]

        busy_time[resource_id] += max(0.0, end_time - start_time)

    occupation = {}

    for resource_id, available_time in availability_times.items():
        if available_time <= 0:
            occupation[resource_id] = 0.0
        else:
            occupation[resource_id] = busy_time[resource_id] / available_time

    return occupation


def average_resource_occupation(
    resource_intervals: List[Dict],
    availability_times: Dict[str, float],
) -> Optional[float]:
    """
    Compute average occupation across all resources.
    """
    occupation = resource_occupation(
        resource_intervals=resource_intervals,
        availability_times=availability_times,
    )

    if not occupation:
        return None

    return sum(occupation.values()) / len(occupation)


def resource_fairness(
    resource_intervals: List[Dict],
    availability_times: Dict[str, float],
) -> Optional[float]:
    """
    Compute simple resource fairness.

    This metric is the mean absolute deviation from the average occupation.

    Lower is better.
    A value close to 0 means resources are used similarly.
    """
    occupation = resource_occupation(
        resource_intervals=resource_intervals,
        availability_times=availability_times,
    )

    if not occupation:
        return None

    average_occupation = sum(occupation.values()) / len(occupation)

    deviations = [
        abs(value - average_occupation)
        for value in occupation.values()
    ]

    return sum(deviations) / len(deviations)


def weighted_resource_fairness(
    resource_intervals: List[Dict],
    availability_times: Dict[str, float],
) -> Optional[float]:
    """
    Compute weighted resource fairness.

    The deviation is weighted by available time.

    This is useful when some resources are available for longer periods than others.

    Lower is better.
    """
    occupation = resource_occupation(
        resource_intervals=resource_intervals,
        availability_times=availability_times,
    )

    if not occupation:
        return None

    total_available_time = sum(availability_times.values())

    if total_available_time <= 0:
        return None

    weighted_average_occupation = sum(
        occupation[resource_id] * availability_times[resource_id]
        for resource_id in occupation
    ) / total_available_time

    weighted_deviation = sum(
        availability_times[resource_id]
        * abs(occupation[resource_id] - weighted_average_occupation)
        for resource_id in occupation
    ) / total_available_time

    return weighted_deviation