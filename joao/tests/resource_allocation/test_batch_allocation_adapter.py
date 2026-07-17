from src.resource_allocation.AllocationStrategy import Resource, Task
from src.resource_allocation.BatchAllocationAdapter import BatchAllocationAdapter


def test_batch_adapter_assigns_current_waiting_queue_snapshot():
    resources = [
        Resource(resource_id="R1", available=True, skills=["A"]),
        Resource(resource_id="R2", available=True, skills=["B"]),
    ]
    tasks = [
        Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0),
        Task(task_id="T2", case_id="C2", activity="B", enabled_time=0.0),
    ]

    decisions = BatchAllocationAdapter(k_limit=5).allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=1.0,
    )

    assignments = [
        decision
        for decision in decisions
        if decision.decision_type == "assignment"
    ]

    assert {(d.resource_id, d.task_id) for d in assignments} == {
        ("R1", "T1"),
        ("R2", "T2"),
    }
    assert all(task.assigned for task in tasks)


def test_batch_adapter_respects_skills_and_availability():
    resources = [
        Resource(resource_id="R1", available=False, skills=["A"]),
        Resource(resource_id="R2", available=True, skills=["B"]),
    ]
    tasks = [
        Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0),
        Task(task_id="T2", case_id="C2", activity="B", enabled_time=0.0),
    ]

    decisions = BatchAllocationAdapter().allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=1.0,
    )

    assignments = [
        decision
        for decision in decisions
        if decision.decision_type == "assignment"
    ]

    assert len(assignments) == 1
    assert assignments[0].resource_id == "R2"
    assert assignments[0].task_id == "T2"
    assert tasks[0].assigned is False
    assert tasks[1].assigned is True


def test_batch_adapter_exports_batch_diagnostics():
    resources = [Resource(resource_id="R1", available=True, skills=["A"])]
    tasks = [Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0)]
    adapter = BatchAllocationAdapter()

    adapter.allocate(resources=resources, waiting_tasks=tasks, current_time=1.0)

    diagnostics = adapter.get_diagnostics()
    assert diagnostics["batch_calls"] == 1
    assert diagnostics["batches_executed"] == 1
    assert diagnostics["batch_assignments"] == 1
    assert diagnostics["batch_average_size"] == 1
