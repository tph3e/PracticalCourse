from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, List

from resources.allocation import AllocationContext

from .AllocationStrategy import AllocationDecision, AllocationStrategy, Resource, Task
from .AllocationUtils import get_available_resources, get_eligible_tasks, mark_task_assigned


class PickInterfaceAllocationAdapter(AllocationStrategy):
    """
    Runs a group-owned resources.allocation strategy in João's global allocator.

    The wrapped strategy sees only candidates that are currently available,
    permitted for the task, and not already used in the current decision epoch.
    """

    def __init__(self, pick_strategy: Any, label: str):
        self.pick_strategy = pick_strategy
        self.label = label
        self.diagnostics = {
            f"{self._key_prefix()}_pick_calls": 0,
            f"{self._key_prefix()}_assignments": 0,
            f"{self._key_prefix()}_postpones": 0,
            f"{self._key_prefix()}_empty_candidate_sets": 0,
        }

    def allocate(
        self,
        resources: List[Resource],
        waiting_tasks: List[Task],
        current_time: float,
        **kwargs: Any,
    ) -> List[AllocationDecision]:
        available_resources = get_available_resources(resources)
        remaining_tasks = sorted(
            [task for task in waiting_tasks if not task.assigned and not task.blocked],
            key=lambda task: (task.enabled_time, -task.priority, task.task_id),
        )
        resource_loads = {
            str(resource_id): float(load)
            for resource_id, load in kwargs.get("resource_loads", {}).items()
        }
        used_resources: set[str] = set()
        decisions: list[AllocationDecision] = []

        for task in remaining_tasks:
            eligible_ids = {
                resource.resource_id
                for resource in get_eligible_tasks_inverse(available_resources, task)
                if resource.resource_id not in used_resources
            }
            if not eligible_ids:
                self.diagnostics[f"{self._key_prefix()}_empty_candidate_sets"] += 1
                continue

            self.diagnostics[f"{self._key_prefix()}_pick_calls"] += 1
            selected = self.pick_strategy.pick(
                eligible_ids,
                AllocationContext(
                    time=self._datetime_from_value(current_time),
                    event=SimpleNamespace(
                        activity=task.activity,
                        case_id=task.case_id,
                        task_id=task.task_id,
                    ),
                    busy=set(used_resources),
                    load=resource_loads,
                ),
            )
            if selected is None:
                self.diagnostics[f"{self._key_prefix()}_postpones"] += 1
                continue
            if selected not in eligible_ids:
                self.diagnostics[f"{self._key_prefix()}_postpones"] += 1
                continue

            mark_task_assigned(waiting_tasks, task.task_id)
            used_resources.add(str(selected))
            self.diagnostics[f"{self._key_prefix()}_assignments"] += 1
            decisions.append(
                AllocationDecision(
                    resource_id=str(selected),
                    task_id=task.task_id,
                    activity=task.activity,
                    case_id=task.case_id,
                    decision_type="assignment",
                    reason=f"Selected by {self.label} pick-interface strategy.",
                )
            )

        for resource in available_resources:
            if resource.resource_id in used_resources:
                continue
            decisions.append(
                AllocationDecision(
                    resource_id=resource.resource_id,
                    task_id=None,
                    activity=None,
                    case_id=None,
                    decision_type="idle",
                    reason=f"No {self.label} assignment selected.",
                )
            )

        return decisions

    def get_diagnostics(self) -> dict[str, int]:
        return dict(self.diagnostics)

    def _key_prefix(self) -> str:
        return "".join(
            char.lower() if char.isalnum() else "_"
            for char in self.label
        ).strip("_")

    def _datetime_from_value(self, value: float) -> datetime | None:
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).replace(tzinfo=None)
        except (TypeError, ValueError, OSError):
            return None


def get_eligible_tasks_inverse(resources: List[Resource], task: Task) -> list[Resource]:
    eligible_resources = []
    for resource in resources:
        if not resource.available:
            continue
        if resource.skills is not None and task.activity not in resource.skills:
            continue
        eligible_resources.append(resource)
    return eligible_resources
