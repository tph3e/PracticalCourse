from datetime import datetime
from types import SimpleNamespace

from src.resource_allocation.integration.TaskLifecycleContext import TaskLifecycle
from src.resource_allocation.integration.WeightedFairnessAdapter import (
    availability_seconds_by_resource,
    busy_intervals_from_lifecycle,
    compute_weighted_fairness_from_engine,
)


class Availability:
    _all_resources = {"R1", "R2"}

    def who_is_available(self, time):
        return {"R1", "R2"}


def test_busy_intervals_from_task_lifecycle():
    lifecycle = TaskLifecycle(
        task_id="T1",
        case_id="C1",
        activity="A",
        enabled_time=datetime(2026, 1, 1, 9, 0),
        resource_id="R1",
        processing_start_time=datetime(2026, 1, 1, 9, 0),
        processing_end_time=datetime(2026, 1, 1, 9, 5),
    )

    intervals = busy_intervals_from_lifecycle({"T1": lifecycle})

    assert intervals == [
        {
            "resource_id": "R1",
            "start_time": datetime(2026, 1, 1, 9, 0).timestamp(),
            "end_time": datetime(2026, 1, 1, 9, 5).timestamp(),
        }
    ]


def test_availability_seconds_by_resource():
    availability = availability_seconds_by_resource(
        Availability(),
        datetime(2026, 1, 1, 9, 0),
        datetime(2026, 1, 1, 9, 15),
    )

    assert availability == {"R1": 900.0, "R2": 900.0}


def test_compute_weighted_fairness_from_engine_uses_existing_metric():
    lifecycle = TaskLifecycle(
        task_id="T1",
        case_id="C1",
        activity="A",
        enabled_time=datetime(2026, 1, 1, 9, 0),
        resource_id="R1",
        processing_start_time=datetime(2026, 1, 1, 9, 0),
        processing_end_time=datetime(2026, 1, 1, 9, 5),
    )
    engine = SimpleNamespace(
        task_lifecycle={"T1": lifecycle},
        resourceEngine=SimpleNamespace(availability=Availability()),
    )

    result = compute_weighted_fairness_from_engine(
        engine,
        datetime(2026, 1, 1, 9, 0),
        datetime(2026, 1, 1, 9, 10),
    )

    assert result["weighted_resource_fairness_status"] == "computed"
    assert result["busy_interval_count"] == 1
    assert result["availability_interval_count"] == 2
    assert round(result["weighted_resource_fairness"], 6) == 0.25
