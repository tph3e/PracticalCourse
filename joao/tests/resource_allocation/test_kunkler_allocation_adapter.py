from datetime import timedelta

from src.resource_allocation.AllocationStrategy import Resource, Task
from src.resource_allocation.KunklerAllocationAdapter import KunklerAllocationAdapter
from src.resource_allocation.integration.AllocationStrategyFactory import (
    build_my_allocation_strategies,
)


class FakeProcessTimeEngine:
    def sampleTime_basic(self, activity, resource="", kind="processing"):
        values = {"A": 3, "B": 7}
        return timedelta(seconds=values.get(activity, 1))


def assignments(decisions):
    return [decision for decision in decisions if decision.decision_type == "assignment"]


def test_kunkler_adapter_registered_in_strategy_factory():
    strategies = build_my_allocation_strategies(seed=1)

    assert "Kunkler" in strategies
    assert strategies["Kunkler"].__class__.__name__ == "KunklerAllocationAdapter"


def test_kunkler_adapter_assigns_and_marks_eligible_tasks():
    resources = [
        Resource(resource_id="R1", available=True, skills=["A"]),
        Resource(resource_id="R2", available=True, skills=["B"]),
    ]
    tasks = [
        Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0),
        Task(task_id="T2", case_id="C2", activity="B", enabled_time=0.0),
    ]

    decisions = KunklerAllocationAdapter(FakeProcessTimeEngine()).allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=1.0,
    )

    selected = assignments(decisions)
    assert {(decision.resource_id, decision.task_id) for decision in selected} == {
        ("R1", "T1"),
        ("R2", "T2"),
    }
    assert all(task.assigned for task in tasks)


def test_kunkler_adapter_filters_permissions_and_availability():
    resources = [
        Resource(resource_id="R1", available=False, skills=["A"]),
        Resource(resource_id="R2", available=True, skills=["B"]),
    ]
    tasks = [
        Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0),
        Task(task_id="T2", case_id="C2", activity="B", enabled_time=0.0),
    ]

    decisions = KunklerAllocationAdapter(FakeProcessTimeEngine()).allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=1.0,
    )

    selected = assignments(decisions)
    assert len(selected) == 1
    assert selected[0].resource_id == "R2"
    assert selected[0].task_id == "T2"
    assert tasks[0].assigned is False
    assert tasks[1].assigned is True


def test_kunkler_adapter_handles_empty_and_no_eligible_cases():
    adapter = KunklerAllocationAdapter(FakeProcessTimeEngine())

    assert adapter.allocate(resources=[], waiting_tasks=[], current_time=1.0) == []

    decisions = adapter.allocate(
        resources=[Resource(resource_id="R1", available=True, skills=["B"])],
        waiting_tasks=[Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0)],
        current_time=1.0,
    )

    assert assignments(decisions) == []
    assert decisions[0].decision_type == "idle"


def test_kunkler_adapter_diagnostics_are_exposed():
    adapter = KunklerAllocationAdapter(FakeProcessTimeEngine())
    adapter.allocate(
        resources=[Resource(resource_id="R1", available=True, skills=["A"])],
        waiting_tasks=[Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0)],
        current_time=1.0,
    )

    diagnostics = adapter.get_diagnostics()
    assert diagnostics["kunkler_calls"] == 1
    assert diagnostics["kunkler_scoring_decisions"] == 1
    assert diagnostics["kunkler_assignments"] == 1
    assert diagnostics["kunkler_selected_score_mean"] > 0


def test_kunkler_adapter_accepts_process_time_engine_kwarg():
    adapter = KunklerAllocationAdapter()

    decisions = adapter.allocate(
        resources=[Resource(resource_id="R1", available=True, skills=["A"])],
        waiting_tasks=[Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0)],
        current_time=1.0,
        process_time_engine=FakeProcessTimeEngine(),
    )

    assert len(assignments(decisions)) == 1
    assert adapter.get_diagnostics()["kunkler_selected_score_mean"] == 3
