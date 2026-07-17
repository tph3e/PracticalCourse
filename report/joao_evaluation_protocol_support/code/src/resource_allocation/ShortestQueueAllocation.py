from typing import Any, List

from .AllocationStrategy import AllocationDecision, AllocationStrategy, Resource, Task
from .AllocationUtils import (
    get_available_resources,
    get_eligible_tasks,
    mark_task_assigned,
)


class ShortestQueueAllocation(AllocationStrategy):
    """
    R-SHQ: Resource-based Shortest Queue Heuristic

    This is a pull-based allocation heuristic using the group simulator's
    cumulative ResourceEngine.load as the resource queue/load proxy.
    Each enabled task is assigned to the eligible available resource with the
    smallest cumulative load.

    Tie-breaking:
    1. smallest cumulative resource load
    2. deterministic resource id
    3. oldest enabled task
    4. highest task priority
    """

    def __init__(self):
        self.diagnostics: dict[str, int] = {}
        self.last_resource_loads: dict[str, float] = {}
        self.last_min_candidate_resource_load: float | None = None
        self.last_selected_resource_load: float | None = None

    def allocate(
            self,
            resources: List[Resource],
            waiting_tasks: List[Task],
            current_time: float,
            **kwargs: Any
    ) -> List[AllocationDecision]:
        decisions: List[AllocationDecision] = []
        available_resources = get_available_resources(resources)
        resource_loads = {
            str(resource_id): float(load)
            for resource_id, load in kwargs.get("resource_loads", {}).items()
        }
        remaining_tasks = sorted(
            [task for task in waiting_tasks if not task.assigned and not task.blocked],
            key=lambda task: (task.enabled_time, -task.priority, task.task_id),
        )

        for task in remaining_tasks:
            feasible_resources = [
                resource
                for resource in available_resources
                if task in get_eligible_tasks(resource, waiting_tasks)
                and resource.resource_id not in {
                    decision.resource_id
                    for decision in decisions
                    if decision.decision_type == "assignment"
                }
            ]
            if not feasible_resources:
                continue

            candidate_loads = {
                resource.resource_id: resource_loads.get(str(resource.resource_id), 0.0)
                for resource in feasible_resources
            }
            self.last_resource_loads = candidate_loads
            self.last_min_candidate_resource_load = min(candidate_loads.values())

            unique_loads = set(candidate_loads.values())
            if len(unique_loads) > 1:
                self._increment("unequal_resource_load_comparisons")
                self._increment("resource_load_unequal_decisions")
            else:
                self._increment("equal_resource_load_ties")
                if len(feasible_resources) > 1:
                    self._increment("resource_load_tie_break_decisions")

            selected_resource = min(
                feasible_resources,
                key=lambda resource: (
                    candidate_loads[resource.resource_id],
                    resource.resource_id,
                ),
            )
            self.last_selected_resource_load = candidate_loads[
                selected_resource.resource_id
            ]

            mark_task_assigned(waiting_tasks, task.task_id)
            self._increment("resource_load_assignment_decisions")

            decisions.append(
                AllocationDecision(
                    resource_id=selected_resource.resource_id,
                    task_id=task.task_id,
                    activity=task.activity,
                    case_id=task.case_id,
                    decision_type="assignment",
                    reason="Selected resource with smallest cumulative resource load",
                )
            )

        assigned_resources = {
            decision.resource_id
            for decision in decisions
            if decision.decision_type == "assignment"
        }
        for resource in available_resources:
            if resource.resource_id not in assigned_resources:
                decisions.append(
                    AllocationDecision(
                        resource_id=resource.resource_id,
                        task_id=None,
                        activity=None,
                        case_id=None,
                        decision_type="idle",
                        reason="No eligible waiting task available",
                    )
                )

        return decisions

    def get_diagnostics(self) -> dict[str, int]:
        return dict(self.diagnostics)

    def _increment(self, key: str) -> None:
        self.diagnostics[key] = self.diagnostics.get(key, 0) + 1
