from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, List

from resourceAllocation_KunklerRinderleMa import AnticipatoryAssignmentAllocator

from .AllocationStrategy import AllocationDecision, AllocationStrategy, Resource, Task
from .AllocationUtils import get_available_resources, get_eligible_tasks, mark_task_assigned


@dataclass
class _QuantileProcessTimeShim:
    """
    Minimal compatibility layer for the original Kunkler/Rinderle-Ma allocator.

    The repository implementation asks for getQuantileValue(task, q), while the
    shared processing-time engine exposes sampled durations. For allocation
    ranking we use the available simulator-side estimate and keep the fallback
    explicit in adapter diagnostics.
    """

    processing_time_engine: Any | None = None
    default_seconds: float = 1.0
    fallback_count: int = 0

    def getMedian(self, activity, resource):
        engine = self.processing_time_engine
        value = engine.getMedian(activity, resource)
        return value


    def getQuantileValue(self, task: Task, q_value: float) -> float:
        engine = self.processing_time_engine
        if engine is None:
            self.fallback_count += 1
            return self.default_seconds

        if hasattr(engine, "getQuantileValue"):
            value = engine.getQuantileValue(task, q_value)
            return self._seconds(value)

        if hasattr(engine, "sampleTime_basic"):
            for resource_id in ("", getattr(task, "resource_id", "")):
                value = engine.sampleTime_basic(task.activity, resource_id, "processing")
                seconds = self._seconds(value)
                if seconds > 0:
                    return seconds

        self.fallback_count += 1
        return self.default_seconds

    @staticmethod
    def _seconds(value: Any) -> float:
        if isinstance(value, timedelta):
            seconds = value.total_seconds()
        else:
            try:
                seconds = float(value)
            except (TypeError, ValueError):
                seconds = 0.0
        if not math.isfinite(seconds) or seconds <= 0:
            return 0.0
        return seconds


class KunklerAllocationAdapter(AllocationStrategy):
    """
    Integrated-simulator adapter for the original Kunkler/Rinderle-Ma allocator.

    The root allocator is preserved and invoked directly. This adapter supplies
    the quantile method expected by that implementation, filters to currently
    eligible simulator resources/tasks, validates returned assignments, marks
    assigned tasks, and records transparent diagnostics.
    """

    def __init__(
        self,
        processing_time_engine: Any | None = None,
        task_prediction_model: Any | None = None,
        resource_model: Any | None = None,
        wait_penalty_weight: float = 1.0,
        delta: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.processing_time_engine = processing_time_engine
        self.seed = seed
        self.quantile_shim = _QuantileProcessTimeShim(processing_time_engine)
        self.allocator = AnticipatoryAssignmentAllocator(
            processing_time_model=self.quantile_shim,
            task_prediction_model=task_prediction_model,
            resource_model=resource_model,
            wait_penalty_weight=wait_penalty_weight,
            delta=delta,
        )
        self.diagnostics = {
            "kunkler_calls": 0,
            "kunkler_scoring_decisions": 0,
            "kunkler_candidate_resources_total": 0,
            "kunkler_candidate_tasks_total": 0,
            "kunkler_assignments": 0,
            "kunkler_idle_decisions": 0,
            "kunkler_invalid_decisions_filtered": 0,
            "kunkler_fallback_count": 0,
            "kunkler_sequential_repair_decisions": 0,
            "kunkler_tie_count": 0,
            "kunkler_selected_score_sum": 0.0,
            "kunkler_selected_score_min": math.inf,
            "kunkler_selected_score_max": 0.0,
            "kunkler_selected_score_count": 0,
        }

    def allocate(
        self,
        resources: List[Resource],
        waiting_tasks: List[Task],
        current_time: float,
        **kwargs: Any,
    ) -> List[AllocationDecision]:
        self.diagnostics["kunkler_calls"] += 1
        process_time_engine = kwargs.get("process_time_engine")
        if process_time_engine is not None and process_time_engine is not self.processing_time_engine:
            self.processing_time_engine = process_time_engine
            self.quantile_shim.processing_time_engine = process_time_engine

        available_resources = get_available_resources(resources)
        eligible_task_ids = {
            task.task_id
            for resource in available_resources
            for task in get_eligible_tasks(resource, waiting_tasks)
        }
        eligible_tasks = [
            task
            for task in waiting_tasks
            if task.task_id in eligible_task_ids and not task.assigned and not task.blocked
        ]

        self.diagnostics["kunkler_candidate_resources_total"] += len(available_resources)
        self.diagnostics["kunkler_candidate_tasks_total"] += len(eligible_tasks)
        if not available_resources or not eligible_tasks:
            self.diagnostics["kunkler_idle_decisions"] += len(available_resources)
            return [
                AllocationDecision(
                    resource_id=resource.resource_id,
                    task_id=None,
                    activity=None,
                    case_id=None,
                    decision_type="idle",
                    reason="No eligible Kunkler task available.",
                )
                for resource in available_resources
            ]

        score_by_task_id = {
            task.task_id: self.quantile_shim.getQuantileValue(task, 0.5)
            for task in eligible_tasks
        }
        if len(set(score_by_task_id.values())) < len(score_by_task_id):
            self.diagnostics["kunkler_tie_count"] += 1

        raw_decisions = self.allocator.allocate(
            resource=available_resources,
            waiting_tasks=eligible_tasks,
            current_time=current_time,
            **kwargs,
        )
        self.diagnostics["kunkler_scoring_decisions"] += 1

        assigned_resource_ids: set[str] = set()
        assigned_task_ids: set[str] = set()
        decisions: list[AllocationDecision] = []
        task_by_id = {task.task_id: task for task in eligible_tasks}
        resource_by_id = {resource.resource_id: resource for resource in available_resources}

        for decision in raw_decisions:
            if getattr(decision, "decision_type", None) != "assignment":
                continue
            resource = resource_by_id.get(str(decision.resource_id))
            task = task_by_id.get(str(decision.task_id))
            if resource is None or task is None:
                self.diagnostics["kunkler_invalid_decisions_filtered"] += 1
                continue
            if resource.resource_id in assigned_resource_ids or task.task_id in assigned_task_ids:
                self.diagnostics["kunkler_invalid_decisions_filtered"] += 1
                continue
            if task not in get_eligible_tasks(resource, eligible_tasks):
                self.diagnostics["kunkler_invalid_decisions_filtered"] += 1
                continue

            mark_task_assigned(waiting_tasks, task.task_id)
            assigned_resource_ids.add(resource.resource_id)
            assigned_task_ids.add(task.task_id)
            score = float(score_by_task_id.get(task.task_id, 0.0))
            self.diagnostics["kunkler_selected_score_sum"] += score
            self.diagnostics["kunkler_selected_score_min"] = min(
                self.diagnostics["kunkler_selected_score_min"],
                score,
            )
            self.diagnostics["kunkler_selected_score_max"] = max(
                self.diagnostics["kunkler_selected_score_max"],
                score,
            )
            self.diagnostics["kunkler_selected_score_count"] += 1
            decisions.append(
                AllocationDecision(
                    resource_id=resource.resource_id,
                    task_id=task.task_id,
                    activity=task.activity,
                    case_id=task.case_id,
                    decision_type="assignment",
                    reason="Selected by Kunkler/Rinderle-Ma allocator through integrated adapter.",
                )
            )

        for resource in available_resources:
            if resource.resource_id in assigned_resource_ids:
                continue
            remaining_tasks = [
                task
                for task in get_eligible_tasks(resource, eligible_tasks)
                if task.task_id not in assigned_task_ids
            ]
            if not remaining_tasks:
                continue
            repaired = self.allocator.allocate(
                resource=[resource],
                waiting_tasks=remaining_tasks,
                current_time=current_time,
                **kwargs,
            )
            self.diagnostics["kunkler_sequential_repair_decisions"] += 1
            for decision in repaired:
                if getattr(decision, "decision_type", None) != "assignment":
                    continue
                task = task_by_id.get(str(decision.task_id))
                if task is None or task.task_id in assigned_task_ids:
                    self.diagnostics["kunkler_invalid_decisions_filtered"] += 1
                    continue
                if task not in get_eligible_tasks(resource, remaining_tasks):
                    self.diagnostics["kunkler_invalid_decisions_filtered"] += 1
                    continue
                mark_task_assigned(waiting_tasks, task.task_id)
                assigned_resource_ids.add(resource.resource_id)
                assigned_task_ids.add(task.task_id)
                score = float(score_by_task_id.get(task.task_id, 0.0))
                self.diagnostics["kunkler_selected_score_sum"] += score
                self.diagnostics["kunkler_selected_score_min"] = min(
                    self.diagnostics["kunkler_selected_score_min"],
                    score,
                )
                self.diagnostics["kunkler_selected_score_max"] = max(
                    self.diagnostics["kunkler_selected_score_max"],
                    score,
                )
                self.diagnostics["kunkler_selected_score_count"] += 1
                decisions.append(
                    AllocationDecision(
                        resource_id=resource.resource_id,
                        task_id=task.task_id,
                        activity=task.activity,
                        case_id=task.case_id,
                        decision_type="assignment",
                        reason=(
                            "Selected by Kunkler/Rinderle-Ma sequential "
                            "eligibility repair."
                        ),
                    )
                )
                break

        self.diagnostics["kunkler_assignments"] += len(assigned_task_ids)
        idle_count = 0
        for resource in available_resources:
            if resource.resource_id in assigned_resource_ids:
                continue
            idle_count += 1
            decisions.append(
                AllocationDecision(
                    resource_id=resource.resource_id,
                    task_id=None,
                    activity=None,
                    case_id=None,
                    decision_type="idle",
                    reason="No Kunkler assignment selected for this resource.",
                )
            )
        self.diagnostics["kunkler_idle_decisions"] += idle_count
        self.diagnostics["kunkler_fallback_count"] += self.quantile_shim.fallback_count
        self.quantile_shim.fallback_count = 0
        return decisions

    def get_diagnostics(self) -> dict[str, float | int]:
        diagnostics = dict(self.diagnostics)
        count = int(diagnostics.get("kunkler_selected_score_count", 0))
        diagnostics["kunkler_selected_score_mean"] = (
            diagnostics["kunkler_selected_score_sum"] / count if count else 0.0
        )
        if diagnostics["kunkler_selected_score_min"] == math.inf:
            diagnostics["kunkler_selected_score_min"] = 0.0
        return diagnostics
