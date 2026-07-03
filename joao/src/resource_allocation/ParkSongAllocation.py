# src/resource_allocation/ParkSongAllocation.py

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .AllocationStrategy import (
    AllocationDecision,
    AllocationStrategy,
    Prediction,
    Resource,
    Task,
)
from .AllocationUtils import (
    get_available_resources,
    is_resource_eligible,
    mark_task_assigned,
)


@dataclass
class CandidateTask:
    """
    Internal representation of a task candidate.

    A candidate can be:
    - current: a real currently enabled task;
    - predicted: a future task that is expected to become enabled soon.
    """

    candidate_id: str
    case_id: str
    activity: str
    candidate_type: str  # "current" or "predicted"
    task_id: Optional[str] = None
    probability: float = 1.0
    expected_delay: float = 0.0
    enabled_time: Optional[float] = None
    priority: float = 0.0


class ParkSongAllocation(AllocationStrategy):
    """
    Conservative Park & Song inspired allocation strategy.

    The strategy builds a candidate set consisting of:
    - currently enabled waiting tasks;
    - predicted future tasks.

    Current tasks can be assigned immediately.
    Predicted tasks cannot be executed yet, so selecting a predicted task creates
    a reservation decision. This corresponds to conservative strategic idling.
    """

    def __init__(
        self,
        processing_time_estimates: Optional[Dict[Tuple[str, str], float]] = None,
        default_processing_time: float = 1.0,
        prediction_probability_threshold: float = 0.5,
        uncertainty_weight: float = 5.0,
        idling_weight: float = 1.0,
        waiting_weight: float = 0.2,
        priority_weight: float = 0.1,
        allow_strategic_idling: bool = True,
    ):
        self.processing_time_estimates = processing_time_estimates or {}
        self.default_processing_time = default_processing_time
        self.prediction_probability_threshold = prediction_probability_threshold
        self.uncertainty_weight = uncertainty_weight
        self.idling_weight = idling_weight
        self.waiting_weight = waiting_weight
        self.priority_weight = priority_weight
        self.allow_strategic_idling = allow_strategic_idling

    def allocate(
        self,
        resources: List[Resource],
        waiting_tasks: List[Task],
        current_time: float,
        **kwargs: Any,
    ) -> List[AllocationDecision]:
        """
        Allocate available resources using current and predicted task candidates.

        Optional kwargs:
            predictions: List[Prediction]
        """

        predictions: List[Prediction] = kwargs.get("predictions", [])

        available_resources = get_available_resources(resources)

        current_candidates = self._build_current_candidates(waiting_tasks)
        predicted_candidates = self._build_predicted_candidates(predictions)

        if self.allow_strategic_idling:
            candidates = current_candidates + predicted_candidates
        else:
            candidates = current_candidates

        decisions: List[AllocationDecision] = []
        used_candidates = set()

        for resource in available_resources:
            feasible_candidates = [
                candidate
                for candidate in candidates
                if candidate.candidate_id not in used_candidates
                and self._is_candidate_feasible(
                    resource=resource,
                    candidate=candidate,
                    waiting_tasks=waiting_tasks,
                )
            ]

            if not feasible_candidates:
                decisions.append(
                    AllocationDecision(
                        resource_id=resource.resource_id,
                        task_id=None,
                        activity=None,
                        case_id=None,
                        decision_type="idle",
                        reason="No feasible current or predicted candidate.",
                    )
                )
                continue

            selected_candidate = min(
                feasible_candidates,
                key=lambda candidate: self._compute_cost(
                    resource=resource,
                    candidate=candidate,
                    current_time=current_time,
                ),
            )

            used_candidates.add(selected_candidate.candidate_id)

            if selected_candidate.candidate_type == "current":
                if selected_candidate.task_id is not None:
                    mark_task_assigned(waiting_tasks, selected_candidate.task_id)

                decisions.append(
                    AllocationDecision(
                        resource_id=resource.resource_id,
                        task_id=selected_candidate.task_id,
                        activity=selected_candidate.activity,
                        case_id=selected_candidate.case_id,
                        decision_type="assignment",
                        reason="Assigned current task using Park-Song cost approximation.",
                    )
                )

            elif selected_candidate.candidate_type == "predicted":
                decisions.append(
                    AllocationDecision(
                        resource_id=resource.resource_id,
                        task_id=None,
                        activity=selected_candidate.activity,
                        case_id=selected_candidate.case_id,
                        decision_type="reservation",
                        reason=(
                            "Selected predicted task. Resource remains idle/reserved "
                            "as conservative strategic idling."
                        ),
                    )
                )

        return decisions

    def _build_current_candidates(self, waiting_tasks: List[Task]) -> List[CandidateTask]:
        """
        Convert currently waiting tasks into current candidates.
        """

        candidates: List[CandidateTask] = []

        for task in waiting_tasks:
            if task.assigned or task.blocked:
                continue

            candidates.append(
                CandidateTask(
                    candidate_id=f"current::{task.task_id}",
                    case_id=task.case_id,
                    activity=task.activity,
                    candidate_type="current",
                    task_id=task.task_id,
                    probability=1.0,
                    expected_delay=0.0,
                    enabled_time=task.enabled_time,
                    priority=task.priority,
                )
            )

        return candidates

    def _build_predicted_candidates(
        self,
        predictions: List[Prediction],
    ) -> List[CandidateTask]:
        """
        Convert predictions into predicted candidates.
        """

        candidates: List[CandidateTask] = []

        for index, prediction in enumerate(predictions):
            if prediction.probability < self.prediction_probability_threshold:
                continue

            candidates.append(
                CandidateTask(
                    candidate_id=(
                        f"predicted::{prediction.case_id}::"
                        f"{prediction.activity}::{index}"
                    ),
                    case_id=prediction.case_id,
                    activity=prediction.activity,
                    candidate_type="predicted",
                    task_id=None,
                    probability=prediction.probability,
                    expected_delay=max(0.0, prediction.expected_delay),
                    enabled_time=None,
                    priority=0.0,
                )
            )

        return candidates

    def _is_candidate_feasible(
        self,
        resource: Resource,
        candidate: CandidateTask,
        waiting_tasks: List[Task],
    ) -> bool:
        """
        Check if a resource-candidate pair is feasible.
        """

        if not resource.available:
            return False

        if resource.skills is not None and candidate.activity not in resource.skills:
            return False

        if candidate.candidate_type == "current":
            original_task = next(
                (
                    task
                    for task in waiting_tasks
                    if task.task_id == candidate.task_id
                ),
                None,
            )

            if original_task is None:
                return False

            return is_resource_eligible(resource, original_task)

        if candidate.candidate_type == "predicted":
            return self.allow_strategic_idling

        return False

    def _estimate_processing_time(self, resource: Resource, activity: str) -> float:
        """
        Estimate processing time for a resource-activity pair.
        """

        return self.processing_time_estimates.get(
            (resource.resource_id, activity),
            self.default_processing_time,
        )

    def _compute_cost(
        self,
        resource: Resource,
        candidate: CandidateTask,
        current_time: float,
    ) -> float:
        """
        Compute expected cost for assigning a resource to a candidate.

        Cost components:
        - processing time cost;
        - waiting time reward for current tasks;
        - prediction uncertainty penalty for predicted tasks;
        - idling penalty for predicted tasks;
        - priority reward.
        """

        processing_time_cost = self._estimate_processing_time(
            resource=resource,
            activity=candidate.activity,
        )

        if candidate.candidate_type == "current":
            waiting_time = 0.0

            if candidate.enabled_time is not None:
                waiting_time = max(0.0, current_time - candidate.enabled_time)

            waiting_time_reward = self.waiting_weight * waiting_time
            prediction_uncertainty_penalty = 0.0
            idling_penalty = 0.0

        else:
            waiting_time_reward = 0.0
            prediction_uncertainty_penalty = (
                self.uncertainty_weight * (1.0 - candidate.probability)
            )
            idling_penalty = self.idling_weight * candidate.expected_delay

        priority_reward = self.priority_weight * candidate.priority

        return (
            processing_time_cost
            - waiting_time_reward
            - priority_reward
            + prediction_uncertainty_penalty
            + idling_penalty
        )