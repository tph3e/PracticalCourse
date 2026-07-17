from __future__ import annotations

from typing import Any, List

from .AllocationStrategy import AllocationDecision, AllocationStrategy, Resource, Task
from .AllocationUtils import get_available_resources, get_eligible_tasks, mark_task_assigned


class RoundRobinResourceAllocation(AllocationStrategy):
    """
    R-RRA: Resource-based Round Robin Allocation.

    Resources are considered in deterministic resource-id order with a persistent
    rotation pointer. Each selected resource receives the oldest eligible task.
    Availability, busy-resource filtering, and permissions are expected to be
    reflected in the Resource objects passed by the integration layer.
    """

    def __init__(self):
        self._next_resource_id: str | None = None
        self.call_count = 0

    def allocate(
        self,
        resources: List[Resource],
        waiting_tasks: List[Task],
        current_time: float,
        **kwargs: Any,
    ) -> List[AllocationDecision]:
        self.call_count += 1
        available_resources = sorted(
            get_available_resources(resources),
            key=lambda resource: resource.resource_id,
        )

        if not available_resources:
            return []

        ordered_resources = self._rotate_resources(available_resources)
        decisions: List[AllocationDecision] = []
        assigned_resource_ids: set[str] = set()

        for resource in ordered_resources:
            eligible_tasks = sorted(
                get_eligible_tasks(resource, waiting_tasks),
                key=lambda task: (task.enabled_time, task.task_id),
            )

            if not eligible_tasks:
                decisions.append(
                    AllocationDecision(
                        resource_id=resource.resource_id,
                        task_id=None,
                        activity=None,
                        case_id=None,
                        decision_type="idle",
                        reason="No eligible waiting task available.",
                    )
                )
                continue

            selected_task = eligible_tasks[0]
            mark_task_assigned(waiting_tasks, selected_task.task_id)
            assigned_resource_ids.add(resource.resource_id)
            decisions.append(
                AllocationDecision(
                    resource_id=resource.resource_id,
                    task_id=selected_task.task_id,
                    activity=selected_task.activity,
                    case_id=selected_task.case_id,
                    decision_type="assignment",
                    reason="Assigned by resource-based round robin.",
                )
            )

        if assigned_resource_ids:
            last_assigned = max(
                assigned_resource_ids,
                key=lambda resource_id: [
                    resource.resource_id for resource in available_resources
                ].index(resource_id),
            )
            self._next_resource_id = self._successor_resource_id(
                available_resources,
                last_assigned,
            )

        return decisions

    def _rotate_resources(self, resources: List[Resource]) -> List[Resource]:
        if self._next_resource_id is None:
            return resources

        resource_ids = [resource.resource_id for resource in resources]
        if self._next_resource_id in resource_ids:
            start_index = resource_ids.index(self._next_resource_id)
        else:
            start_index = next(
                (
                    index
                    for index, resource_id in enumerate(resource_ids)
                    if resource_id > self._next_resource_id
                ),
                0,
            )

        return resources[start_index:] + resources[:start_index]

    def _successor_resource_id(
        self,
        resources: List[Resource],
        resource_id: str,
    ) -> str:
        resource_ids = [resource.resource_id for resource in resources]
        index = resource_ids.index(resource_id)
        return resource_ids[(index + 1) % len(resource_ids)]
