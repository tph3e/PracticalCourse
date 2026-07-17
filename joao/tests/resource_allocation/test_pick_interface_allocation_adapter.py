from resources.allocation import AllocationStrategy as PickStrategy

from src.resource_allocation.AllocationStrategy import Resource, Task
from src.resource_allocation.PickInterfaceAllocationAdapter import (
    PickInterfaceAllocationAdapter,
)


class FirstCandidate(PickStrategy):
    def __init__(self):
        self.contexts = []

    def pick(self, candidates, context=None):
        self.contexts.append(context)
        return sorted(candidates)[0]


class Postpone(PickStrategy):
    def pick(self, candidates, context=None):
        return None


def test_pick_adapter_uses_only_eligible_candidates_and_marks_assignment():
    pick = FirstCandidate()
    adapter = PickInterfaceAllocationAdapter(pick, "First")
    resources = [
        Resource(resource_id="R1", available=True, skills=["B"]),
        Resource(resource_id="R2", available=True, skills=["A"]),
    ]
    tasks = [Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0)]

    decisions = adapter.allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=1.0,
        resource_loads={"R1": 0, "R2": 2},
    )

    assignments = [d for d in decisions if d.decision_type == "assignment"]
    assert len(assignments) == 1
    assert assignments[0].resource_id == "R2"
    assert tasks[0].assigned is True
    assert pick.contexts[0].event.activity == "A"


def test_pick_adapter_records_postpones():
    adapter = PickInterfaceAllocationAdapter(Postpone(), "Postpone")
    resources = [Resource(resource_id="R1", available=True, skills=["A"])]
    tasks = [Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0)]

    decisions = adapter.allocate(resources, tasks, current_time=1.0)

    assert not [d for d in decisions if d.decision_type == "assignment"]
    assert adapter.get_diagnostics()["postpone_postpones"] == 1
