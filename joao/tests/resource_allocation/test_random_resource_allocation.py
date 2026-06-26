# tests/resource_allocation/test_random_resource_allocation.py

from src.resource_allocation.AllocationStrategy import Resource, Task
from src.resource_allocation.RandomResourceAllocation import RandomResourceAllocation


def test_random_allocation_assigns_available_resource_to_eligible_task():
    resources = [
        Resource(resource_id="R1", available=True, skills=["A"]),
    ]

    tasks = [
        Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0),
    ]

    strategy = RandomResourceAllocation(seed=1)
    decisions = strategy.allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=1.0,
    )

    assert len(decisions) == 1
    assert decisions[0].resource_id == "R1"
    assert decisions[0].task_id == "T1"
    assert decisions[0].decision_type == "assignment"
    assert tasks[0].assigned is True


def test_random_allocation_ignores_unavailable_resources():
    resources = [
        Resource(resource_id="R1", available=False, skills=["A"]),
    ]

    tasks = [
        Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0),
    ]

    strategy = RandomResourceAllocation(seed=1)
    decisions = strategy.allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=1.0,
    )

    assert len(decisions) == 0
    assert tasks[0].assigned is False


def test_random_allocation_respects_resource_skills():
    resources = [
        Resource(resource_id="R1", available=True, skills=["B"]),
    ]

    tasks = [
        Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0),
    ]

    strategy = RandomResourceAllocation(seed=1)
    decisions = strategy.allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=1.0,
    )

    assert len(decisions) == 1
    assert decisions[0].decision_type == "idle"
    assert decisions[0].task_id is None
    assert tasks[0].assigned is False


def test_random_allocation_does_not_assign_same_task_twice():
    resources = [
        Resource(resource_id="R1", available=True, skills=["A"]),
        Resource(resource_id="R2", available=True, skills=["A"]),
    ]

    tasks = [
        Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0),
    ]

    strategy = RandomResourceAllocation(seed=1)
    decisions = strategy.allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=1.0,
    )

    assignments = [
        decision for decision in decisions
        if decision.decision_type == "assignment"
    ]

    idle_decisions = [
        decision for decision in decisions
        if decision.decision_type == "idle"
    ]

    assert len(assignments) == 1
    assert len(idle_decisions) == 1
    assert assignments[0].task_id == "T1"
    assert tasks[0].assigned is True


def test_random_allocation_ignores_assigned_and_blocked_tasks():
    resources = [
        Resource(resource_id="R1", available=True, skills=["A"]),
    ]

    tasks = [
        Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0, assigned=True),
        Task(task_id="T2", case_id="C2", activity="A", enabled_time=0.0, blocked=True),
    ]

    strategy = RandomResourceAllocation(seed=1)
    decisions = strategy.allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=1.0,
    )

    assert len(decisions) == 1
    assert decisions[0].decision_type == "idle"
    assert decisions[0].task_id is None