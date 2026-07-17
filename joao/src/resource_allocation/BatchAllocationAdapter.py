from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Any, List

from BatchAllocationEngine import BatchAllocationEngine

from .AllocationStrategy import AllocationDecision, AllocationStrategy, Resource, Task
from .AllocationUtils import get_available_resources, mark_task_assigned


@dataclass(frozen=True)
class _BatchResource:
    resource_id: str
    authorized_tasks: tuple[str, ...] | None


@dataclass(frozen=True)
class _BatchTask:
    task_id: str
    case_id: str
    task_type: str


class BatchAllocationAdapter(AllocationStrategy):
    """
    Adapter around the repository-level BatchAllocationEngine.

    The original BatchAllocationEngine owns a buffer and fires when the buffer
    reaches k_limit. The integrated simulator already owns the waiting queue and
    retries it after every release, so this adapter fires the current queue
    snapshot once per global allocation decision. This keeps Batch in the same
    simulator lifecycle as the other methods without modifying the group-owned
    implementation.
    """

    def __init__(self, k_limit: int = 5):
        self.k_limit = k_limit
        self.diagnostics = {
            "batch_calls": 0,
            "batches_executed": 0,
            "batch_assignments": 0,
            "batch_unassigned_tasks": 0,
            "batch_size_sum": 0,
            "batch_size_max": 0,
        }

    def allocate(
        self,
        resources: List[Resource],
        waiting_tasks: List[Task],
        current_time: float,
        **kwargs: Any,
    ) -> List[AllocationDecision]:
        self.diagnostics["batch_calls"] += 1
        available_resources = get_available_resources(resources)
        unassigned_tasks = [
            task
            for task in waiting_tasks
            if not task.assigned and not task.blocked
        ]
        self.diagnostics["batch_size_sum"] += len(unassigned_tasks)
        self.diagnostics["batch_size_max"] = max(
            self.diagnostics["batch_size_max"],
            len(unassigned_tasks),
        )

        if not available_resources:
            return []

        if not unassigned_tasks:
            return [
                AllocationDecision(
                    resource_id=resource.resource_id,
                    task_id=None,
                    activity=None,
                    case_id=None,
                    decision_type="idle",
                    reason="No waiting task available for batch snapshot.",
                )
                for resource in available_resources
            ]

        batch_resources = [
            _BatchResource(
                resource_id=resource.resource_id,
                authorized_tasks=(
                    tuple(resource.skills)
                    if resource.skills is not None
                    else None
                ),
            )
            for resource in available_resources
        ]
        resource_by_wrapper = {
            batch_resource: resource
            for batch_resource, resource in zip(batch_resources, available_resources)
        }

        batch_tasks = [
            _BatchTask(
                task_id=task.task_id,
                case_id=task.case_id,
                task_type=task.activity,
            )
            for task in unassigned_tasks
        ]
        task_by_wrapper = {
            batch_task: task
            for batch_task, task in zip(batch_tasks, unassigned_tasks)
        }

        engine = BatchAllocationEngine(k_limit=self.k_limit)
        engine.batch_buffer = list(batch_tasks)

        with contextlib.redirect_stdout(io.StringIO()):
            assignments = engine.fire_batch(batch_resources, current_time)
        self.diagnostics["batches_executed"] += 1

        assigned_task_ids: set[str] = set()
        assigned_resource_ids: set[str] = set()
        decisions: List[AllocationDecision] = []

        for assignment in assignments:
            batch_task = assignment["task"]
            batch_resource = assignment["resource"]
            task = task_by_wrapper[batch_task]
            resource = resource_by_wrapper[batch_resource]

            if task.task_id in assigned_task_ids:
                continue
            if resource.resource_id in assigned_resource_ids:
                continue

            assigned_task_ids.add(task.task_id)
            assigned_resource_ids.add(resource.resource_id)
            mark_task_assigned(waiting_tasks, task.task_id)
            decisions.append(
                AllocationDecision(
                    resource_id=resource.resource_id,
                    task_id=task.task_id,
                    activity=task.activity,
                    case_id=task.case_id,
                    decision_type="assignment",
                    reason="Assigned by batch current-queue snapshot.",
                )
            )

        self.diagnostics["batch_assignments"] += len(assigned_task_ids)
        self.diagnostics["batch_unassigned_tasks"] += (
            len(unassigned_tasks) - len(assigned_task_ids)
        )

        for resource in available_resources:
            if resource.resource_id in assigned_resource_ids:
                continue

            decisions.append(
                AllocationDecision(
                    resource_id=resource.resource_id,
                    task_id=None,
                    activity=None,
                    case_id=None,
                    decision_type="idle",
                    reason="No batch-feasible waiting task available.",
                )
            )

        return decisions

    def get_diagnostics(self) -> dict[str, float | int]:
        diagnostics = dict(self.diagnostics)
        calls = max(1, int(diagnostics.get("batch_calls", 0)))
        diagnostics["batch_average_size"] = diagnostics["batch_size_sum"] / calls
        return diagnostics
