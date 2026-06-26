# tests/resource_allocation/test_shortest_queue_allocation.py

from src.resource_allocation.AllocationStrategy import Resource, Task
from src.resource_allocation.ShortestQueueAllocation import ShortestQueueAllocation


def test_shortest_queue_selects_task_from_shortest_activity_queue():
    resources = [
        Resource(resource_id="R1", available=True, skills=["A", "B"]),
    ]

    tasks = [
        Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0),
        Task(task_id="T2", case_id="C2", activity="A", enabled_time=1.0),
        Task(task_id="T3", case_id="C3", activity="B", enabled_time=2.0),
    ]

    strategy = ShortestQueueAllocation()
    decisions = strategy.allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=5.0,
    )

    assert len(decisions) == 1
    assert decisions[0].decision_type == "assignment"
    assert decisions[0].task_id == "T3"
    assert decisions[0].activity == "B"
    assert tasks[2].assigned is True


def test_shortest_queue_respects_resource_skills():
    resources = [
        Resource(resource_id="R1", available=True, skills=["A"]),
    ]

    tasks = [
        Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0),
        Task(task_id="T2", case_id="C2", activity="B", enabled_time=1.0),
    ]

    strategy = ShortestQueueAllocation()
    decisions = strategy.allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=5.0,
    )

    assert len(decisions) == 1
    assert decisions[0].decision_type == "assignment"
    assert decisions[0].activity == "A"
    assert decisions[0].task_id == "T1"


def test_shortest_queue_uses_oldest_task_as_tie_breaker():
    resources = [
        Resource(resource_id="R1", available=True, skills=["A", "B"]),
    ]

    tasks = [
        Task(task_id="T1", case_id="C1", activity="A", enabled_time=5.0),
        Task(task_id="T2", case_id="C2", activity="B", enabled_time=2.0),
    ]

    strategy = ShortestQueueAllocation()
    decisions = strategy.allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=10.0,
    )

    assert len(decisions) == 1
    assert decisions[0].task_id == "T2"
    assert decisions[0].activity == "B"


def test_shortest_queue_uses_priority_after_queue_and_time_tie():
    resources = [
        Resource(resource_id="R1", available=True, skills=["A", "B"]),
    ]

    tasks = [
        Task(task_id="T1", case_id="C1", activity="A", enabled_time=1.0, priority=1.0),
        Task(task_id="T2", case_id="C2", activity="B", enabled_time=1.0, priority=5.0),
    ]

    strategy = ShortestQueueAllocation()
    decisions = strategy.allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=10.0,
    )

    assert len(decisions) == 1
    assert decisions[0].task_id == "T2"
    assert decisions[0].activity == "B"


def test_shortest_queue_ignores_unavailable_resources():
    resources = [
        Resource(resource_id="R1", available=False, skills=["A"]),
    ]

    tasks = [
        Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0),
    ]

    strategy = ShortestQueueAllocation()
    decisions = strategy.allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=1.0,
    )

    assert len(decisions) == 0
    assert tasks[0].assigned is False


def test_shortest_queue_returns_idle_when_no_eligible_task_exists():
    resources = [
        Resource(resource_id="R1", available=True, skills=["A"]),
    ]

    tasks = [
        Task(task_id="T1", case_id="C1", activity="B", enabled_time=0.0),
    ]

    strategy = ShortestQueueAllocation()
    decisions = strategy.allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=1.0,
    )

    assert len(decisions) == 1
    assert decisions[0].decision_type == "idle"
    assert decisions[0].task_id is None
    assert tasks[0].assigned is False


def test_shortest_queue_does_not_assign_same_task_twice():
    resources = [
        Resource(resource_id="R1", available=True, skills=["A"]),
        Resource(resource_id="R2", available=True, skills=["A"]),
    ]

    tasks = [
        Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0),
    ]

    strategy = ShortestQueueAllocation()
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