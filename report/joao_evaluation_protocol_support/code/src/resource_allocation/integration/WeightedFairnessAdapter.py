from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from joao.src.resource_allocation.ResourceAllocationMetrics import (
    weighted_resource_fairness,
)


def busy_intervals_from_lifecycle(task_lifecycle: dict[str, Any]) -> list[dict]:
    intervals = []
    for lifecycle in task_lifecycle.values():
        if (
            lifecycle.resource_id is None
            or lifecycle.processing_start_time is None
            or lifecycle.processing_end_time is None
        ):
            continue
        intervals.append(
            {
                "resource_id": str(lifecycle.resource_id),
                "start_time": lifecycle.processing_start_time.timestamp(),
                "end_time": lifecycle.processing_end_time.timestamp(),
            }
        )
    return intervals


def availability_seconds_by_resource(
    availability_model,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, float]:
    resources = sorted(getattr(availability_model, "_all_resources", set()))
    availability = {resource_id: 0.0 for resource_id in resources}
    if end_time <= start_time:
        return availability

    cursor = start_time
    while cursor < end_time:
        next_hour = (cursor.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
        segment_end = min(next_hour, end_time)
        seconds = max(0.0, (segment_end - cursor).total_seconds())
        for resource_id in availability_model.who_is_available(cursor):
            availability.setdefault(str(resource_id), 0.0)
            availability[str(resource_id)] += seconds
        cursor = segment_end
    return availability


def compute_weighted_fairness_from_engine(engine, start_time: datetime, end_time: datetime):
    busy_intervals = busy_intervals_from_lifecycle(engine.task_lifecycle)
    availability_times = availability_seconds_by_resource(
        engine.resourceEngine.availability,
        start_time,
        end_time,
    )
    included_resources = [
        resource_id
        for resource_id, seconds in availability_times.items()
        if seconds > 0
    ]
    if not busy_intervals:
        return {
            "weighted_resource_fairness": float("nan"),
            "weighted_resource_fairness_status": "no_busy_intervals",
            "busy_interval_count": 0,
            "availability_interval_count": len(included_resources),
            "weighted_fairness_resource_count": len(included_resources),
        }
    if not included_resources:
        return {
            "weighted_resource_fairness": float("nan"),
            "weighted_resource_fairness_status": "no_available_resource_intervals",
            "busy_interval_count": len(busy_intervals),
            "availability_interval_count": 0,
            "weighted_fairness_resource_count": 0,
        }

    value = weighted_resource_fairness(
        resource_intervals=busy_intervals,
        availability_times={
            resource_id: availability_times[resource_id]
            for resource_id in included_resources
        },
    )
    return {
        "weighted_resource_fairness": (
            float(value) if value is not None else float("nan")
        ),
        "weighted_resource_fairness_status": (
            "computed" if value is not None else "metric_returned_none"
        ),
        "busy_interval_count": len(busy_intervals),
        "availability_interval_count": len(included_resources),
        "weighted_fairness_resource_count": len(included_resources),
    }
