# src/resource_allocation/ParkSongAllocation.py

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

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

    The allocation decision is solved as a global min-cost assignment over the
    current allocation epoch. This keeps the implementation close to an
    assignment/flow-style anticipatory allocation formulation while still using
    the simulator's lightweight task/prediction abstractions.
    """

    def __init__(
        self,
        processing_time_estimates: Optional[Dict[Tuple[str, str], float]] = None,
        default_processing_time: float = 1.0,
        processing_time_weight: float = 1.0,
        cost_time_scale: float = 1.0,
        prediction_probability_threshold: float = 0.5,
        uncertainty_weight: float = 5.0,
        idling_weight: float = 1.0,
        waiting_weight: float = 0.2,
        priority_weight: float = 0.1,
        no_show_penalty_weight: float = 0.0,
        allow_strategic_idling: bool = True,
        planning_horizon: float = 3600.0,
        future_delay_weight: float = 0.0,
        reservation_margin: float = 0.0,
    ):
        self.processing_time_estimates = processing_time_estimates or {}
        self.default_processing_time = default_processing_time
        self.processing_time_weight = processing_time_weight
        self.cost_time_scale = max(1.0, cost_time_scale)
        self.prediction_probability_threshold = prediction_probability_threshold
        self.uncertainty_weight = uncertainty_weight
        self.idling_weight = idling_weight
        self.waiting_weight = waiting_weight
        self.priority_weight = priority_weight
        self.no_show_penalty_weight = no_show_penalty_weight
        self.allow_strategic_idling = allow_strategic_idling
        self.planning_horizon = planning_horizon
        self.future_delay_weight = future_delay_weight
        self.reservation_margin = max(0.0, reservation_margin)
        self.diagnostics: dict[str, int] = {}

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
        assignments = self._solve_global_assignment(
            resources=available_resources,
            candidates=candidates,
            waiting_tasks=waiting_tasks,
            current_time=current_time,
        )

        for resource, selected_candidate in assignments:
            if selected_candidate is None:
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

    def get_diagnostics(self) -> dict[str, int]:
        return dict(self.diagnostics)

    def _solve_global_assignment(
        self,
        resources: List[Resource],
        candidates: List[CandidateTask],
        waiting_tasks: List[Task],
        current_time: float,
    ) -> list[tuple[Resource, CandidateTask | None]]:
        """
        Solve a min-cost assignment with idle dummy columns.

        One candidate can be used at most once. Each resource is assigned either
        to one feasible candidate or to its own idle dummy option.
        """

        if not resources:
            return []

        if not candidates:
            return [(resource, None) for resource in resources]

        self.diagnostics["global_assignment_calls"] = (
            self.diagnostics.get("global_assignment_calls", 0) + 1
        )
        self.diagnostics["global_assignment_resources_total"] = (
            self.diagnostics.get("global_assignment_resources_total", 0)
            + len(resources)
        )
        self.diagnostics["global_assignment_candidates_total"] = (
            self.diagnostics.get("global_assignment_candidates_total", 0)
            + len(candidates)
        )

        infeasible_cost = 1_000_000_000_000_000.0
        idle_cost = 1_000_000_000_000.0
        candidate_count = len(candidates)
        matrix = np.full(
            (len(resources), candidate_count + len(resources)),
            infeasible_cost,
            dtype=float,
        )

        feasible_pairs = 0
        for row, resource in enumerate(resources):
            best_current_cost = self._best_current_candidate_cost(
                resource=resource,
                candidates=candidates,
                waiting_tasks=waiting_tasks,
                current_time=current_time,
            )
            for col, candidate in enumerate(candidates):
                if not self._is_candidate_feasible(resource, candidate, waiting_tasks):
                    continue
                candidate_cost = self._compute_temporal_cost(
                    resource=resource,
                    candidate=candidate,
                    candidates=candidates,
                    current_time=current_time,
                )
                if (
                    candidate.candidate_type == "predicted"
                    and best_current_cost is not None
                    and candidate_cost + self.reservation_margin >= best_current_cost
                ):
                    self.diagnostics["reservation_margin_filtered"] = (
                        self.diagnostics.get("reservation_margin_filtered", 0) + 1
                    )
                    continue
                matrix[row, col] = candidate_cost
                feasible_pairs += 1
            matrix[row, candidate_count + row] = idle_cost

        self.diagnostics["global_assignment_feasible_pairs_total"] = (
            self.diagnostics.get("global_assignment_feasible_pairs_total", 0)
            + feasible_pairs
        )

        row_indices, col_indices = linear_sum_assignment(matrix)
        by_row = {int(row): int(col) for row, col in zip(row_indices, col_indices)}
        assignments: list[tuple[Resource, CandidateTask | None]] = []

        for row, resource in enumerate(resources):
            col = by_row.get(row)
            if col is None or col >= candidate_count or matrix[row, col] >= idle_cost:
                assignments.append((resource, None))
                continue
            assignments.append((resource, candidates[col]))

        return assignments

    def _best_current_candidate_cost(
        self,
        resource: Resource,
        candidates: List[CandidateTask],
        waiting_tasks: List[Task],
        current_time: float,
    ) -> float | None:
        best_cost: float | None = None
        for candidate in candidates:
            if candidate.candidate_type != "current":
                continue
            if not self._is_candidate_feasible(resource, candidate, waiting_tasks):
                continue
            cost = self._compute_temporal_cost(
                resource=resource,
                candidate=candidate,
                candidates=candidates,
                current_time=current_time,
                record_diagnostics=False,
            )
            if best_cost is None or cost < best_cost:
                best_cost = cost
        return best_cost

    def _compute_temporal_cost(
        self,
        resource: Resource,
        candidate: CandidateTask,
        candidates: List[CandidateTask],
        current_time: float,
        record_diagnostics: bool = True,
    ) -> float:
        """
        Compute the pair cost with a one-step temporal lookahead.

        The direct cost decides whether a resource should work on a current task
        or reserve for a predicted one. The lookahead penalty makes a current
        assignment account for high-confidence future candidates that would be
        delayed by the resource staying busy beyond their expected arrival time.
        This is a lightweight temporal approximation of anticipatory allocation:
        it does not commit a full future schedule, but the current choice is no
        longer evaluated as if future predictions did not occupy time.
        """

        direct_cost = self._compute_cost(
            resource=resource,
            candidate=candidate,
            current_time=current_time,
        )

        if (
            candidate.candidate_type != "current"
            or self.planning_horizon <= 0.0
            or self.future_delay_weight <= 0.0
        ):
            return direct_cost

        processing_time = self._estimate_processing_time(
            resource=resource,
            activity=candidate.activity,
        )
        candidate_finish_time = current_time + max(0.0, processing_time)
        lookahead_penalty = 0.0

        for future_candidate in candidates:
            if future_candidate.candidate_type != "predicted":
                continue
            if not self._is_future_candidate_compatible(resource, future_candidate):
                continue
            if future_candidate.expected_delay > self.planning_horizon:
                continue

            expected_arrival_time = current_time + future_candidate.expected_delay
            induced_delay = max(0.0, candidate_finish_time - expected_arrival_time)
            if induced_delay <= 0.0:
                continue

            lookahead_penalty += (
                self.future_delay_weight
                * future_candidate.probability
                * (induced_delay / self.cost_time_scale)
            )

        if lookahead_penalty > 0.0 and record_diagnostics:
            self.diagnostics["temporal_lookahead_penalties"] = (
                self.diagnostics.get("temporal_lookahead_penalties", 0) + 1
            )

        return direct_cost + lookahead_penalty

    def _is_future_candidate_compatible(
        self,
        resource: Resource,
        candidate: CandidateTask,
    ) -> bool:
        if not resource.available:
            return False
        if resource.skills is not None and candidate.activity not in resource.skills:
            return False
        return True

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

        processing_time_cost = (
            self.processing_time_weight
            * self._estimate_processing_time(
                resource=resource,
                activity=candidate.activity,
            )
            / self.cost_time_scale
        )

        if candidate.candidate_type == "current":
            waiting_time = 0.0

            if candidate.enabled_time is not None:
                waiting_time = max(0.0, current_time - candidate.enabled_time)

            waiting_time_reward = self.waiting_weight * (
                waiting_time / self.cost_time_scale
            )
            prediction_uncertainty_penalty = 0.0
            idling_penalty = 0.0
            no_show_penalty = 0.0

        else:
            waiting_time_reward = 0.0
            prediction_uncertainty_penalty = (
                self.uncertainty_weight * (1.0 - candidate.probability)
            )
            idling_penalty = self.idling_weight * (
                candidate.expected_delay / self.cost_time_scale
            )
            no_show_penalty = (
                self.no_show_penalty_weight
                * (1.0 - candidate.probability)
                * (candidate.expected_delay / self.cost_time_scale)
            )

        priority_reward = self.priority_weight * candidate.priority

        return (
            processing_time_cost
            - waiting_time_reward
            - priority_reward
            + prediction_uncertainty_penalty
            + idling_penalty
            + no_show_penalty
        )
