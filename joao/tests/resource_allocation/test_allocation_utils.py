# tests/resource_allocation/test_allocation_utils.py

from src.resource_allocation.AllocationStrategy import Resource, Task
from src.resource_allocation.AllocationUtils import (
    compute_activity_queue_lengths,
    get_available_resources,
    get_eligible_tasks,
    is_resource_eligible,
    mark_task_assigned,
)


def test_get_available_resources_returns_only_available_resources():
    resources = [
        Resource(resource_id="R1", available=True),
        Resource(resource_id="R2", available=False),
        Resource(resource_id="R3", available=True),
    ]

    available_resources = get_available_resources(resources)

    assert len(available_resources) == 2
    assert available_resources[0].resource_id == "R1"
    assert available_resources[1].resource_id == "R3"


def test_resource_is_eligible_when_available_and_has_skill():
    resource = Resource(resource_id="R1", available=True, skills=["A", "B"])
    task = Task(
        task_id="T1",
        case_id="C1",
        activity="A",
        enabled_time=0.0,
    )

    assert is_resource_eligible(resource, task) is True


def test_resource_is_not_eligible_when_unavailable():
    resource = Resource(resource_id="R1", available=False, skills=["A"])
    task = Task(
        task_id="T1",
        case_id="C1",
        activity="A",
        enabled_time=0.0,
    )

    assert is_resource_eligible(resource, task) is False


def test_resource_is_not_eligible_when_skill_missing():
    resource = Resource(resource_id="R1", available=True, skills=["B"])
    task = Task(
        task_id="T1",
        case_id="C1",
        activity="A",
        enabled_time=0.0,
    )

    assert is_resource_eligible(resource, task) is False


def test_resource_is_not_eligible_when_task_already_assigned():
    resource = Resource(resource_id="R1", available=True, skills=["A"])
    task = Task(
        task_id="T1",
        case_id="C1",
        activity="A",
        enabled_time=0.0,
        assigned=True,
    )

    assert is_resource_eligible(resource, task) is False


def test_resource_is_not_eligible_when_task_blocked():
    resource = Resource(resource_id="R1", available=True, skills=["A"])
    task = Task(
        task_id="T1",
        case_id="C1",
        activity="A",
        enabled_time=0.0,
        blocked=True,
    )

    assert is_resource_eligible(resource, task) is False


def test_resource_without_skills_can_execute_all_tasks():
    resource = Resource(resource_id="R1", available=True, skills=None)
    task = Task(
        task_id="T1",
        case_id="C1",
        activity="A",
        enabled_time=0.0,
    )

    assert is_resource_eligible(resource, task) is True


def test_get_eligible_tasks_returns_only_feasible_tasks():
    resource = Resource(resource_id="R1", available=True, skills=["A"])

    tasks = [
        Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0),
        Task(task_id="T2", case_id="C2", activity="B", enabled_time=0.0),
        Task(task_id="T3", case_id="C3", activity="A", enabled_time=0.0, assigned=True),
        Task(task_id="T4", case_id="C4", activity="A", enabled_time=0.0, blocked=True),
    ]

    eligible_tasks = get_eligible_tasks(resource, tasks)

    assert len(eligible_tasks) == 1
    assert eligible_tasks[0].task_id == "T1"


def test_compute_activity_queue_lengths_ignores_assigned_and_blocked_tasks():
    tasks = [
        Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0),
        Task(task_id="T2", case_id="C2", activity="A", enabled_time=0.0),
        Task(task_id="T3", case_id="C3", activity="B", enabled_time=0.0),
        Task(task_id="T4", case_id="C4", activity="B", enabled_time=0.0, assigned=True),
        Task(task_id="T5", case_id="C5", activity="C", enabled_time=0.0, blocked=True),
    ]

    queue_lengths = compute_activity_queue_lengths(tasks)

    assert queue_lengths == {
        "A": 2,
        "B": 1,
    }


def test_mark_task_assigned_marks_correct_task():
    tasks = [
        Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0),
        Task(task_id="T2", case_id="C2", activity="B", enabled_time=0.0),
    ]

    mark_task_assigned(tasks, "T2")

    assert tasks[0].assigned is False
    assert tasks[1].assigned is True