from __future__ import annotations

import heapq
import math
import statistics
import time
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import numpy as np
import scipy.stats as stats
from pm4py.objects.petri_net.semantics import enabled_transitions, execute

from Helper import Case, EventType
from SimulationEngineCore import Engine
from joao.src.resource_allocation.AllocationStrategy import Prediction, Resource, Task
from joao.src.resource_allocation.integration.CompositeBranchingAdapter import (
    CompositeBranchingAdapter,
)
from joao.src.resource_allocation.integration.RequestedAmountCompatibility import (
    fit_requested_amount_distributions,
)
from joao.src.resource_allocation.integration.TaskLifecycleContext import (
    ResourceReservation,
    TaskLifecycle,
)
from joao.src.resource_allocation.integration.TransitionAwareBranching import (
    TransitionDisambiguationModel,
    load_transition_model,
)

MIN_VISIBLE_PROCESSING_DURATION = timedelta(seconds=1)
WAITING_RETRY_ACTIVITY = "__JOAO_WAITING_RETRY__"


class IntegratedAllocationEngine(Engine):
    """
    João-owned integration subclass for Part II allocation strategies.

    The class reuses the group simulator components created by Engine.__init__.
    It overrides only case-amount fitting and the event loop hooks needed to
    compare João's global allocation strategies and consume Composite branching
    predictions consistently.
    """

    def __init__(
        self,
        dataPath: str = "data/BPI Challenge 2017.xes",
        seed: int = 1,
        allocation_strategy: Any | None = None,
        branching_engine: Any | None = None,
        processing_time_artifact: str | None = None,
        diagnostic_cycle_guard: bool = False,
        cycle_repetition_limit: int = 50,
        fixed_routes: list[list[str]] | None = None,
        fixed_route_case_ids: list[str] | None = None,
        fixed_route_arrival_times: list[datetime] | None = None,
        transition_model: TransitionDisambiguationModel | None = None,
        transition_model_path: str | None = None,
        reservation_expiration_multiplier: float = 1.0,
    ) -> None:
        self.allocation_strategy = allocation_strategy
        self.seed = seed
        self.requested_amount_rng = np.random.default_rng(seed)
        self.requested_amount_bounds: tuple[float, float] | None = None
        self.diagnostic_cycle_guard = diagnostic_cycle_guard
        self.cycle_repetition_limit = cycle_repetition_limit
        self.fixed_routes = [list(route) for route in fixed_routes or []]
        self.fixed_route_case_ids = list(fixed_route_case_ids or [])
        self.fixed_route_arrival_times = list(fixed_route_arrival_times or [])
        self._fixed_route_cursor = 0
        self._fixed_route_by_case_id: dict[str, list[str]] = {}
        self._fixed_route_position_by_case_id: dict[str, int] = {}
        self._fixed_route_source_case_by_case_id: dict[str, str] = {}
        self._waiting_retry_times: set[datetime] = set()
        self._event_horizon: datetime | None = None
        self.branching_adapter: CompositeBranchingAdapter | None = None
        self.future_predictions_by_task_id: dict[str, Prediction] = {}
        self.branch_prediction_by_task_id: dict[str, Any] = {}
        self.prediction_id_by_task_id: dict[str, str] = {}
        self.reservations_by_resource_id: dict[str, ResourceReservation] = {}
        self.reservation_by_target_task_id: dict[str, ResourceReservation] = {}
        self.reservation_history: list[ResourceReservation] = []
        self.reservation_expiration_multiplier = max(
            1.0,
            float(reservation_expiration_multiplier),
        )
        self.task_lifecycle: dict[str, TaskLifecycle] = {}
        self._task_cache: dict[str, Task] = {}
        self._event_id_to_task_id: dict[str, str] = {}
        self._transition_id_by_task_id: dict[str, str] = {}
        self._permission_cache: dict[str, set[str]] = {}
        self._suspended_task_ids: set[str] = set()
        self._consumed_prediction_ids: set[str] = set()
        self._executed_prediction_ids: set[str] = set()
        self.diagnostics: Counter[str] = Counter()
        self.prediction_source_counts: Counter[str] = Counter()
        self.processing_time_missing_model_by_activity: Counter[str] = Counter()
        self.minimum_visible_duration_by_activity: Counter[str] = Counter()
        self._processing_duration_diagnostic_task_ids: set[str] = set()
        self._final_processing_duration_seconds: list[float] = []
        self._reservation_counter = 0
        self.arrival_cutoff: datetime | None = None
        self.admitted_case_ids: set[str] = set()
        self.completed_case_ids: set[str] = set()
        self.deadlocked_case_ids: set[str] = set()
        self.cyclic_case_ids: set[str] = set()
        self.censored_case_ids: set[str] = set()
        self._completed_activity_counts_by_case: dict[str, Counter[str]] = {}
        self.activity_visit_limits: dict[str, int] = {}
        self.empirical_successors: dict[str, set[str]] = {}
        self.drain_stopped_by_limit = False
        self.transition_model = transition_model
        if self.transition_model is None and transition_model_path:
            self.transition_model = load_transition_model(transition_model_path)

        super().__init__(
            dataPath=dataPath,
            seed=seed,
            processing_time_artifact=processing_time_artifact,
        )

        if branching_engine is not None:
            self.branchingEngine = branching_engine
        self.branching_adapter = CompositeBranchingAdapter(
            self.branchingEngine,
            seed=seed,
            transition_model=self.transition_model,
        )

    def train(self, log: pd.DataFrame) -> None:
        self.freq, self.amount_dists, self.global_params = (
            fit_requested_amount_distributions(log)
        )
        requested_amounts = pd.to_numeric(
            log["case:RequestedAmount"],
            errors="coerce",
        ).dropna()
        if not requested_amounts.empty:
            self.requested_amount_bounds = (
                float(requested_amounts.min()),
                float(requested_amounts.max()),
            )
        if (
            self.diagnostic_cycle_guard
            and "case:concept:name" in log
            and "concept:name" in log
        ):
            counts = (
                log.groupby(["case:concept:name", "concept:name"])
                .size()
                .groupby("concept:name")
                .max()
            )
            self.activity_visit_limits = {
                str(activity): max(1, int(limit))
                for activity, limit in counts.items()
            }
            ordered = log.sort_values(["case:concept:name", "time:timestamp"])
            successors: dict[str, set[str]] = {}
            for _, group in ordered.groupby("case:concept:name"):
                activities = [str(activity) for activity in group["concept:name"]]
                for current, nxt in zip(activities, activities[1:]):
                    successors.setdefault(current, set()).add(nxt)
            self.empirical_successors = successors

    def run(
        self,
        start_time: datetime,
        end_time: datetime,
        format_type=["csv", "xes"],
        drain_until: datetime | None = None,
    ) -> None:
        self.arrival_cutoff = end_time
        event_horizon = drain_until or end_time
        self._event_horizon = event_horizon
        if self.fixed_routes and self.fixed_route_arrival_times:
            for arrival_time in self.fixed_route_arrival_times:
                if start_time <= arrival_time <= end_time:
                    self.push_event(
                        arrival_time,
                        EventType.CASE_ARRIVAL,
                        "",
                        self.sample_case_data(),
                        Case(self.case_counter),
                    )
                    self.case_counter += 1
            self.diagnostics["fixed_route_historical_arrivals_scheduled"] += len(
                self.event_queue
            )
        else:
            self.push_event(
                start_time,
                EventType.CASE_ARRIVAL,
                "",
                self.sample_case_data(),
                Case(self.case_counter),
            )
            self.case_counter += 1

        while self.event_queue:
            event = self.pop_event()
            self.simulation_time = event.time
            if self.simulation_time > event_horizon:
                self.drain_stopped_by_limit = bool(drain_until)
                break

            if event.eventType == EventType.CASE_ARRIVAL:
                if self.simulation_time > end_time:
                    continue
                self._handle_case_arrival(event)
                continue

            if (
                event.eventType == EventType.ACTIVITY_RESUME
                and event.activity == WAITING_RETRY_ACTIVITY
            ):
                self._handle_waiting_retry_event(event)
                continue

            if event.eventType == EventType.ACTIVITY_START:
                self._handle_activity_start(event)
                if event.eventType != EventType.ACTIVITY_SUSPEND:
                    self._check_prediction_execution(event)
            elif event.eventType == EventType.ACTIVITY_END:
                self._handle_activity_end(event)

            self.logger.log_event(event)

        self._cleanup_reservations_at_horizon()
        self._record_waiting_drain_feasibility()
        self._classify_censored_cases()

        if "csv" in format_type:
            self.logger.to_csv()
        if "xes" in format_type:
            self.logger.to_xes()

    def sample_case_data(self) -> dict[str, Any]:
        sampled_id = self.random_choices_index()
        row = self.freq.loc[sampled_id]
        lookup_key = (row["case:ApplicationType"], row["case:LoanGoal"])
        shape, loc, scale = self.amount_dists.get(lookup_key, self.global_params)
        requested_amount = stats.lognorm.rvs(
            shape,
            loc,
            scale,
            random_state=self.requested_amount_rng,
        )
        if self.requested_amount_bounds is not None:
            lower, upper = self.requested_amount_bounds
            requested_amount = min(max(float(requested_amount), lower), upper)
        return {
            "case:ApplicationType": row["case:ApplicationType"],
            "case:LoanGoal": row["case:LoanGoal"],
            "case:RequestedAmount": round(requested_amount, 1),
            "EventOrigin": "Application",
        }

    def random_choices_index(self):
        import random

        return random.choices(self.freq.index, weights=self.freq["prob"])[0]

    def _handle_case_arrival(self, event) -> None:
        if self.fixed_routes and self._fixed_route_cursor >= len(self.fixed_routes):
            self.diagnostics["fixed_route_arrivals_skipped_after_routes_exhausted"] += 1
            return
        self.admitted_case_ids.add(self.case_id_for_event(event))
        data = self.sample_case_data()
        first_candidate = (
            self.bpmnEngine.getStartTransitionCandidate()
            if hasattr(self.bpmnEngine, "getStartTransitionCandidate")
            else None
        )
        first_activity = (
            first_candidate.activity_label
            if first_candidate is not None
            else self.bpmnEngine.getStartActivity(data)
        )
        fixed_route = self._assign_fixed_route(event)
        if fixed_route:
            first_activity = fixed_route[0]
            self.diagnostics["fixed_route_cases_admitted"] += 1
            first_candidate = None
        target_task_id = str(self.event_counter)
        self.push_event(
            event.time,
            EventType.ACTIVITY_START,
            first_activity,
            data,
            event.eventCase,
        )
        if first_candidate is not None:
            self._transition_id_by_task_id[target_task_id] = str(
                first_candidate.transition_id
            )
        self.bpmnEngine.initialize_case(event.eventCase.caseId)

        if not self.fixed_routes:
            new_case = Case(self.case_counter)
            self.case_counter += 1
            self.cases.append(new_case)
            next_arrival_time = self.arrivalEngine.nextArrivalTime(event.time) + event.time
            if self.arrival_cutoff is None or next_arrival_time <= self.arrival_cutoff:
                self.push_event(next_arrival_time, EventType.CASE_ARRIVAL, "", dict(), new_case)

    def _assign_fixed_route(self, event) -> list[str] | None:
        if not self.fixed_routes:
            return None
        case_id = self.case_id_for_event(event)
        if case_id in self._fixed_route_by_case_id:
            return self._fixed_route_by_case_id[case_id]
        if self._fixed_route_cursor >= len(self.fixed_routes):
            self.diagnostics["fixed_route_no_route_available"] += 1
            return None
        route = self.fixed_routes[self._fixed_route_cursor]
        source_case_id = (
            self.fixed_route_case_ids[self._fixed_route_cursor]
            if self._fixed_route_cursor < len(self.fixed_route_case_ids)
            else str(self._fixed_route_cursor)
        )
        self._fixed_route_cursor += 1
        self._fixed_route_by_case_id[case_id] = route
        self._fixed_route_position_by_case_id[case_id] = 0
        self._fixed_route_source_case_by_case_id[case_id] = source_case_id
        return route

    def _handle_activity_start(self, event) -> None:
        self._ensure_task_lifecycle(event)
        if self.allocation_strategy is None:
            self.diagnostics["immediate_allocation_calls"] += 1
            allocated = self.resourceEngine.allocateResource(event)
            if allocated:
                self.diagnostics["strategy_assignments"] += 1
                self.diagnostics["resources_allocated"] += 1
                self._mark_resource_assignment(event)
                self._schedule_activity_end(event)
            else:
                self._suspend_event(event)
            self.future_predictions_by_task_id.pop(self.task_id_for_event(event), None)
            return

        reserved_resource = self._pop_matching_reservation(event)
        if reserved_resource is not None:
            event.resource = reserved_resource
            self.resourceEngine.busy.add(reserved_resource)
            self.resourceEngine.load[reserved_resource] = (
                self.resourceEngine.load.get(reserved_resource, 0) + 1
            )
            self.diagnostics["strategy_assignments"] += 1
            self.diagnostics["resources_allocated"] += 1
            self._mark_resource_assignment(event)
            self._schedule_activity_end(event)
            return

        self.diagnostics["immediate_allocation_calls"] += 1
        assigned_events = self._allocate_enabled_events(
            [event],
            include_waiting=False,
        )
        if event in assigned_events:
            self._cancel_reservation_for_started_task(event)
            self._schedule_activity_end(event)
        else:
            self._suspend_event(event)
            self._schedule_waiting_retry("allocation_failure")
        self.future_predictions_by_task_id.pop(self.task_id_for_event(event), None)

    def _handle_activity_end(self, event) -> None:
        self._mark_processing_end(event)
        task_id = self.task_id_for_event(event)
        if hasattr(self.allocation_strategy, "release_task"):
            self.allocation_strategy.release_task(task_id)
        if self._fixed_route_by_case_id:
            self._handle_fixed_route_activity_end(event)
            return
        fired = self._fire_scheduled_transition(event)
        if not fired:
            self.diagnostics["bpmn_fire_failures"] += 1
        self.resourceEngine.releaseResource(event)
        self.diagnostics["resources_released"] += 1

        transition_candidates = self._transition_candidates(event)
        possible_next = sorted(
            {str(candidate.activity_label) for candidate in transition_candidates}
        )
        if not transition_candidates:
            self._classify_no_next_case(event)
            self._cancel_reservations_for_case(self.case_id_for_event(event))
            self._mark_activity_completed(event)
            self._retry_waiting_after_resource_release()
            return
        self._mark_activity_completed(event)
        if self._detect_repetition_cycle(event):
            self._classify_cyclic_case(event)
            self._cancel_reservations_for_case(self.case_id_for_event(event))
            self._retry_waiting_after_resource_release()
            return
        if self.diagnostic_cycle_guard:
            possible_next = self._filter_cycle_candidates(event, possible_next)
            transition_candidates = [
                candidate
                for candidate in transition_candidates
                if str(candidate.activity_label) in possible_next
            ]
            if not possible_next:
                self._classify_cyclic_case(event)
                self._cancel_reservations_for_case(self.case_id_for_event(event))
                self._retry_waiting_after_resource_release()
                return
        prediction = self._predict_branch(
            event,
            possible_next,
            transition_candidates=transition_candidates,
        )
        if prediction.selected_activity is not None:
            wait_time = self.processTimeEngine.getWaitingTime(
                event,
                prediction.selected_activity,
            )
            target_task_id = str(self.event_counter)
            prediction = self._replace_prediction_target(
                prediction,
                target_task_id=target_task_id,
                expected_delay=wait_time.total_seconds(),
            )
            self._remember_prediction(prediction)
            if prediction.selected_transition_id is not None:
                self._transition_id_by_task_id[target_task_id] = str(
                    prediction.selected_transition_id
                )
            self.push_event(
                wait_time + event.time,
                EventType.ACTIVITY_START,
                prediction.selected_activity,
                event.getAttribs(),
                event.eventCase,
            )

        self._retry_waiting_after_resource_release()

    def _handle_fixed_route_activity_end(self, event) -> None:
        case_id = self.case_id_for_event(event)
        route = self._fixed_route_by_case_id.get(case_id)
        if route is None:
            self.bpmnEngine.fire_activity(event.activity, event.eventCase.caseId)
            self.resourceEngine.releaseResource(event)
            self.diagnostics["resources_released"] += 1
            return

        fired = self.bpmnEngine.fire_activity(event.activity, event.eventCase.caseId)
        if not fired:
            self.diagnostics["fixed_route_bpmn_fire_failures"] += 1
        self.resourceEngine.releaseResource(event)
        self.diagnostics["resources_released"] += 1
        self._mark_activity_completed(event)

        position = self._fixed_route_position_by_case_id.get(case_id, 0)
        next_position = position + 1
        self._fixed_route_position_by_case_id[case_id] = next_position
        if next_position >= len(route):
            self.completed_case_ids.add(case_id)
            self.diagnostics["fixed_route_cases_completed"] += 1
            self._cancel_reservations_for_case(case_id)
            self._retry_waiting_after_resource_release()
            return

        next_activity = route[next_position]
        if self._detect_repetition_cycle(event):
            self._classify_cyclic_case(event)
            self._cancel_reservations_for_case(case_id)
            self._retry_waiting_after_resource_release()
            return

        possible_next = self._prediction_candidates_for_fixed_route(event)
        prediction = self._predict_branch(event, possible_next)
        wait_time = self.processTimeEngine.getWaitingTime(event, next_activity)
        target_task_id = str(self.event_counter)
        if prediction.selected_activity is not None:
            prediction = self._replace_prediction_target(
                prediction,
                target_task_id=target_task_id,
                expected_delay=wait_time.total_seconds(),
            )
            self._remember_prediction(prediction)
        self.push_event(
            wait_time + event.time,
            EventType.ACTIVITY_START,
            next_activity,
            event.getAttribs(),
            event.eventCase,
        )
        self._retry_waiting_after_resource_release()

    def _prediction_candidates_for_fixed_route(self, event) -> list[str]:
        candidates = self.bpmnEngine.getPossibleNextActivities(
            event.activity,
            case_id=event.eventCase.caseId,
        )
        if candidates:
            return candidates
        labels = sorted(
            {
                transition.label
                for transition in getattr(self.bpmnEngine, "net", None).transitions
                if transition.label is not None
            }
        ) if getattr(self.bpmnEngine, "net", None) is not None else []
        self.diagnostics["fixed_route_prediction_global_candidate_fallbacks"] += 1
        return labels

    def _fire_scheduled_transition(self, event) -> bool:
        task_id = self.task_id_for_event(event)
        transition_id = self._transition_id_by_task_id.pop(task_id, None)
        if transition_id is not None and hasattr(self.bpmnEngine, "fire_transition"):
            fired = self.bpmnEngine.fire_transition(
                transition_id,
                event.eventCase.caseId,
            )
            if fired:
                self.diagnostics["bpmn_transition_identity_fires"] += 1
                self.diagnostics["exact_transition_fires"] += 1
                return True
            self.diagnostics["bpmn_transition_identity_fire_failures"] += 1

        fired = self.bpmnEngine.fire_activity(event.activity, event.eventCase.caseId)
        if fired:
            self.diagnostics["bpmn_label_compatibility_fires"] += 1
            self.diagnostics["legacy_label_fires"] += 1
        else:
            self.diagnostics["ambiguous_legacy_label_rejections"] += 1
        return fired

    def _transition_candidates(self, event) -> list[Any]:
        if hasattr(self.bpmnEngine, "get_enabled_transition_alternatives"):
            candidates = self.bpmnEngine.get_enabled_transition_alternatives(
                event.eventCase.caseId
            )
            self.diagnostics["bpmn_transition_candidate_sets"] += 1
            self.diagnostics["bpmn_transition_candidates_total"] += len(candidates)
            return candidates

        labels = self.bpmnEngine.getPossibleNextActivities(
            event.activity,
            case_id=event.eventCase.caseId,
        )
        from types import SimpleNamespace

        return [
            SimpleNamespace(
                transition_id=str(label),
                activity_label=str(label),
                marking_before=None,
                resulting_marking=None,
                silent_transition_path=(),
                duplicate_label_count=1,
            )
            for label in labels
        ]

    def _mark_activity_completed(self, event) -> None:
        case_id = self.case_id_for_event(event)
        activity = str(getattr(event, "activity", "") or "")
        if not activity:
            return
        self._completed_activity_counts_by_case.setdefault(case_id, Counter())[activity] += 1

    def _filter_cycle_candidates(self, event, possible_next: list[str]) -> list[str]:
        case_id = self.case_id_for_event(event)
        counts = self._completed_activity_counts_by_case.get(case_id, Counter())
        filtered = []
        for activity in possible_next:
            limit = self.activity_visit_limits.get(str(activity))
            if limit is not None and counts.get(str(activity), 0) >= limit:
                self.diagnostics["cycle_candidate_filtered"] += 1
                self.diagnostics[
                    f"cycle_candidate_filtered_activity_{self._safe_activity_key(activity)}"
                ] += 1
                continue
            filtered.append(activity)
        if len(filtered) != len(possible_next):
            self.diagnostics["cycle_guard_filter_events"] += 1
        return filtered

    def _detect_repetition_cycle(self, event) -> bool:
        activity = str(getattr(event, "activity", "") or "")
        if not activity or self.cycle_repetition_limit <= 0:
            return False
        case_id = self.case_id_for_event(event)
        counts = self._completed_activity_counts_by_case.get(case_id, Counter())
        visit_count = counts.get(activity, 0)
        if visit_count >= self.cycle_repetition_limit:
            self.diagnostics["cycle_repetition_limit_hits"] += 1
            self.diagnostics[
                f"cycle_repetition_limit_activity_{self._safe_activity_key(activity)}"
            ] += 1
            return True
        return False

    def _filter_empirical_successors(self, event, possible_next: list[str]) -> list[str]:
        current_activity = str(getattr(event, "activity", "") or "")
        observed = self.empirical_successors.get(current_activity)
        if not observed:
            return possible_next
        supported = [
            activity
            for activity in possible_next
            if str(activity) in observed
        ]
        if supported:
            removed = len(possible_next) - len(supported)
            if removed:
                self.diagnostics["empirical_successor_candidates_filtered"] += removed
                self.diagnostics["empirical_successor_filter_events"] += 1
            return supported
        return possible_next

    def _classify_no_next_case(self, event) -> None:
        case_id = self.case_id_for_event(event)
        if hasattr(self.bpmnEngine, "can_reach_final_by_silent_path"):
            final_reachable = self.bpmnEngine.can_reach_final_by_silent_path(
                event.eventCase.caseId
            )
        else:
            marking = getattr(self.bpmnEngine, "case_markings", {}).get(
                event.eventCase.caseId
            )
            final_marking = getattr(self.bpmnEngine, "final_marking", None)
            final_reachable = self._is_final_or_silent_final_reachable(
                marking,
                final_marking,
            )
        if final_reachable:
            self.completed_case_ids.add(case_id)
            self.diagnostics["cases_completed_final_marking"] += 1
            return
        self.deadlocked_case_ids.add(case_id)
        self.diagnostics["cases_deadlocked"] += 1

    def _is_final_or_silent_final_reachable(self, marking, final_marking) -> bool:
        if marking is None or final_marking is None:
            return False
        if marking == final_marking:
            return True
        net = getattr(self.bpmnEngine, "net", None)
        if net is None:
            return False
        queue = [marking]
        visited = []
        while queue:
            current = queue.pop(0)
            if current == final_marking:
                return True
            if current in visited:
                continue
            visited.append(current)
            for transition in enabled_transitions(net, current):
                if transition.label is not None:
                    continue
                next_marking = execute(transition, net, current)
                if next_marking == final_marking:
                    return True
                if next_marking not in visited:
                    queue.append(next_marking)
        return False

    def _classify_cyclic_case(self, event) -> None:
        case_id = self.case_id_for_event(event)
        self.cyclic_case_ids.add(case_id)
        self.diagnostics["cases_cyclic"] += 1

    def _classify_censored_cases(self) -> None:
        unresolved = (
            set(self.admitted_case_ids)
            - set(self.completed_case_ids)
            - set(self.deadlocked_case_ids)
            - set(self.cyclic_case_ids)
        )
        self.censored_case_ids = unresolved
        self.diagnostics["cases_censored"] = len(unresolved)

    def _predict_branch(
        self,
        event,
        possible_next: list[str],
        transition_candidates: list[Any] | None = None,
    ):
        start = time.perf_counter()
        if self.branching_adapter is None:
            selected = self.branchingEngine.getNextActivities(event, possible_next)
            selected_activity = selected[0] if selected else None
            from joao.src.resource_allocation.integration.BranchPredictionContext import (
                BranchPrediction,
            )

            prediction = BranchPrediction(
                prediction_id=f"BP_FALLBACK_{self.diagnostics['branch_predictions']}",
                case_id=str(event.eventCase.caseId),
                current_activity=event.activity,
                decision_point=event.activity,
                candidate_activities=tuple(possible_next),
                selected_activity=selected_activity,
                prediction_time=self.simulation_time,
            )
            self.diagnostics["branch_prediction_calls"] += 1
            self.diagnostics["branch_prediction_time_seconds"] += (
                time.perf_counter() - start
            )
            return prediction

        if transition_candidates is not None and hasattr(
            self.branching_adapter,
            "predict_transition",
        ):
            prediction = self.branching_adapter.predict_transition(
                event=event,
                transition_candidates=transition_candidates,
                prediction_time=self.simulation_time,
            )
        else:
            prediction = self.branching_adapter.predict(
                event=event,
                possible_activities=possible_next,
                prediction_time=self.simulation_time,
            )
        self.diagnostics["branch_predictions"] += 1
        self.diagnostics["branch_prediction_calls"] += 1
        self.diagnostics["branch_prediction_time_seconds"] += (
            time.perf_counter() - start
        )
        self.prediction_source_counts[prediction.prediction_source] += 1
        if getattr(prediction, "transition_ambiguity", False):
            self.diagnostics["branch_transition_ambiguities"] += 1
        if getattr(prediction, "fallback_source", None):
            self.diagnostics[
                f"branch_transition_fallback_{prediction.fallback_source}"
            ] += 1
        if getattr(prediction, "rejected_activity", None):
            self.diagnostics["branch_invalid_predictions_rejected"] += 1
        return prediction

    def _replace_prediction_target(
        self,
        prediction,
        target_task_id: str,
        expected_delay: float,
    ):
        from dataclasses import replace

        return replace(
            prediction,
            target_task_id=target_task_id,
            expected_delay=expected_delay,
            scheduled_activity=prediction.selected_activity,
            status="branch_scheduled",
        )

    def _remember_prediction(self, prediction) -> None:
        if prediction.target_task_id is None or prediction.selected_activity is None:
            return
        self.branch_prediction_by_task_id[prediction.target_task_id] = prediction
        self.prediction_id_by_task_id[prediction.target_task_id] = prediction.prediction_id
        self.future_predictions_by_task_id[prediction.target_task_id] = Prediction(
            case_id=prediction.case_id,
            activity=prediction.selected_activity,
            probability=prediction.confidence if prediction.confidence is not None else 1.0,
            expected_delay=prediction.expected_delay,
            source=prediction.prediction_source,
            confidence=prediction.confidence,
        )

    def _check_prediction_execution(self, event) -> None:
        task_id = self.task_id_for_event(event)
        prediction = self.branch_prediction_by_task_id.pop(task_id, None)
        if prediction is None:
            return
        self.future_predictions_by_task_id.pop(task_id, None)
        prediction_id = self.prediction_id_by_task_id.pop(task_id, prediction.prediction_id)
        if prediction_id not in self._executed_prediction_ids:
            self._executed_prediction_ids.add(prediction_id)
            self.diagnostics["unique_predictions_executed"] += 1
        self.diagnostics["branch_predictions_executed"] += 1
        if prediction.selected_activity == event.activity:
            self.diagnostics["prediction_execution_matches"] += 1
        else:
            self.diagnostics["prediction_execution_mismatches"] += 1

    def _retry_waiting_events(self) -> None:
        if not self.waiting_processes:
            return
        waiting_events = []
        while self.waiting_processes:
            event = heapq.heappop(self.waiting_processes)
            if self.case_id_for_event(event) in self._terminal_case_ids():
                self.diagnostics["waiting_retry_skipped_terminal_case"] += 1
                continue
            waiting_events.append(event)

        assigned_events = self._allocate_enabled_events(
            waiting_events,
            include_waiting=False,
        )
        assigned_ids = {id(event) for event in assigned_events}
        for event in waiting_events:
            if id(event) in assigned_ids:
                stable_task_id = self.task_id_for_event(event)
                setattr(event, "stable_task_id", stable_task_id)
                event.update(
                    {
                        "EventID": self.event_counter,
                        "lifecycle:transition": EventType.ACTIVITY_RESUME,
                        "time:timestamp": self.simulation_time,
                    }
                )
                self.event_counter += 1
                self._suspended_task_ids.discard(stable_task_id)
                self._mark_resource_assignment(event)
                self._cancel_reservation_for_started_task(event)
                self._schedule_activity_end(event)
                self._check_prediction_execution(event)
                self.logger.log_event(event)
            else:
                heapq.heappush(self.waiting_processes, event)
        if self.waiting_processes:
            self._schedule_waiting_retry("post_retry_unassigned")

    def _retry_waiting_after_resource_release(self) -> None:
        if self.waiting_processes:
            self.diagnostics["waiting_retry_after_resource_release"] += 1
        self._retry_waiting_events()

    def _terminal_case_ids(self) -> set[str]:
        return (
            set(self.completed_case_ids)
            | set(self.deadlocked_case_ids)
            | set(self.cyclic_case_ids)
            | set(self.censored_case_ids)
        )

    def _handle_waiting_retry_event(self, event) -> None:
        self._waiting_retry_times.discard(event.time)
        self.diagnostics["waiting_retry_events_processed"] += 1
        self._retry_waiting_events()

    def _schedule_waiting_retry(self, reason: str) -> None:
        if not self.waiting_processes:
            return
        retry_time = self._next_waiting_retry_time()
        if retry_time is None:
            self.diagnostics["waiting_retry_no_future_availability"] += 1
            return
        if retry_time in self._waiting_retry_times:
            self.diagnostics["waiting_retry_duplicate_suppressed"] += 1
            return
        self._waiting_retry_times.add(retry_time)
        self.push_event(
            retry_time,
            EventType.ACTIVITY_RESUME,
            WAITING_RETRY_ACTIVITY,
            {},
            Case(f"waiting-retry-{len(self._waiting_retry_times)}"),
        )
        self.diagnostics["waiting_retry_events_scheduled"] += 1
        self.diagnostics[f"waiting_retry_scheduled_reason_{reason}"] += 1

    def _next_waiting_retry_time(self) -> datetime | None:
        if self.simulation_time is None or self._event_horizon is None:
            return None
        current_time = self.simulation_time
        if self._any_waiting_task_feasible_at(current_time):
            return current_time + timedelta(microseconds=1)

        next_times = [
            retry_time
            for event in list(self.waiting_processes)
            if (retry_time := self._next_available_time_for_event(event, current_time)) is not None
        ]
        return min(next_times) if next_times else None

    def _next_available_time_for_event(self, event, after_time: datetime) -> datetime | None:
        eligible = self._permitted_resources_for_activity(event.activity)
        blocked_by_reservation = set(self.reservations_by_resource_id)
        candidate_resources = eligible - set(self.resourceEngine.busy) - blocked_by_reservation
        if not candidate_resources:
            return None

        availability = self.resourceEngine.availability
        start = after_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        if availability.calendars is not None:
            best: datetime | None = None
            for resource_id in candidate_resources:
                for weekday, hour in availability.calendars.get(resource_id, set()):
                    days_ahead = (weekday - start.weekday()) % 7
                    candidate = (start + timedelta(days=days_ahead)).replace(
                        hour=hour, minute=0, second=0, microsecond=0
                    )
                    if candidate < start:
                        candidate += timedelta(days=7)
                    if candidate <= self._event_horizon and (best is None or candidate < best):
                        best = candidate
            return best

        cursor = start
        while cursor <= self._event_horizon:
            if candidate_resources & set(availability.who_is_available(cursor)):
                return cursor
            cursor += timedelta(hours=1)
        return None

    def _any_waiting_task_feasible_at(self, at_time: datetime) -> bool:
        for event in list(self.waiting_processes):
            if self._eligible_available_resources_for_event_at(event, at_time):
                return True
        return False

    def _eligible_available_resources_for_event_at(self, event, at_time: datetime) -> set[str]:
        eligible = self._permitted_resources_for_activity(event.activity)
        available = self.resourceEngine.availability.who_is_available(at_time)
        blocked_by_reservation = set(self.reservations_by_resource_id)
        return (eligible & available) - set(self.resourceEngine.busy) - blocked_by_reservation

    def _record_waiting_drain_feasibility(self) -> None:
        if not self.waiting_processes:
            return
        self.diagnostics["event_queue_empty_with_waiting_tasks"] += int(not self.event_queue)
        feasible = 0
        infeasible = 0
        at_time = self.simulation_time or self._event_horizon
        for event in list(self.waiting_processes):
            if self._eligible_available_resources_for_event_at(event, at_time):
                feasible += 1
            else:
                infeasible += 1
        self.diagnostics["feasible_waiting_tasks_at_drain_end"] += feasible
        self.diagnostics["infeasible_waiting_tasks_at_drain_end"] += infeasible

    def _allocate_enabled_events(
        self,
        enabled_events,
        include_waiting: bool,
    ) -> list:
        allocation_start = time.perf_counter()
        if self.allocation_strategy is None or not enabled_events:
            return []

        waiting_events = list(enabled_events)
        if include_waiting:
            waiting_events.extend(list(self.waiting_processes))

        self.diagnostics["global_strategy_calls"] += 1
        self.diagnostics["waiting_tasks_seen"] += len(waiting_events)
        self.diagnostics["waiting_queue_size_sum"] += len(waiting_events)
        self.diagnostics["waiting_queue_size_samples"] += 1
        self.diagnostics["max_queue_size"] = max(
            self.diagnostics["max_queue_size"],
            len(waiting_events),
        )

        self._expire_due_reservations(self.simulation_time)
        predictions = self._active_future_predictions()
        self.diagnostics["active_prediction_count_sum"] += len(predictions)
        self.diagnostics["active_reservation_count_sum"] += len(
            self.reservations_by_resource_id
        )
        self.diagnostics["allocation_call_samples"] += 1
        resources = self._build_resources(waiting_events)
        tasks = self._build_tasks(waiting_events)
        self.diagnostics["tasks_converted_sum"] += len(tasks)
        self.diagnostics["resources_converted_sum"] += len(resources)

        if self.allocation_strategy.__class__.__name__.startswith("ParkSong"):
            consumed_ids = [
                self.prediction_id_by_task_id[task_id]
                for task_id in self.future_predictions_by_task_id
                if task_id in self.prediction_id_by_task_id
            ]
            new_ids = [
                prediction_id
                for prediction_id in consumed_ids
                if prediction_id not in self._consumed_prediction_ids
            ]
            self._consumed_prediction_ids.update(new_ids)
            self.diagnostics["park_song_predictions_consumed"] += len(new_ids)
            self.diagnostics["park_song_prediction_reuse_count"] += (
                len(consumed_ids) - len(new_ids)
            )

        strategy_start = time.perf_counter()
        decisions = self.allocation_strategy.allocate(
            resources=resources,
            waiting_tasks=tasks,
            current_time=self._time_value(self.simulation_time),
            predictions=predictions,
            process_time_engine=self.processTimeEngine,
            resource_loads={
                str(resource_id): float(load)
                for resource_id, load in self.resourceEngine.load.items()
            },
        )
        self.diagnostics["allocation_strategy_calls"] += 1
        self.diagnostics["allocation_strategy_time_seconds"] += (
            time.perf_counter() - strategy_start
        )

        event_by_task_id = {
            self.task_id_for_event(event): event
            for event in waiting_events
        }
        assigned_events = []
        assigned_resource_ids: set[str] = set()
        assigned_task_ids: set[str] = set()

        for decision in decisions:
            if getattr(decision, "decision_type", None) == "reservation":
                prediction = self._prediction_for_reservation(decision)
                if prediction is not None:
                    reservation = self._create_reservation(decision, prediction)
                    old_reservation = self.reservations_by_resource_id.get(
                        decision.resource_id
                    )
                    if old_reservation is not None:
                        if not self._reservation_is_preferred(
                            reservation,
                            old_reservation,
                        ):
                            self.diagnostics["reservations_rejected_existing_kept"] += 1
                            continue
                        self._finalize_reservation(
                            old_reservation,
                            status="overwritten",
                        )
                    self._store_reservation(reservation)
                    self.diagnostics["reservation_decisions"] += 1
                    self.diagnostics["reservations_created"] += 1
                continue

            if getattr(decision, "decision_type", None) != "assignment":
                continue
            if decision.resource_id in assigned_resource_ids:
                self.diagnostics["duplicate_assignment_errors"] += 1
                continue
            if decision.task_id in assigned_task_ids:
                self.diagnostics["duplicate_assignment_errors"] += 1
                continue

            event = event_by_task_id.get(str(decision.task_id))
            if event is None:
                continue

            assigned_resource_ids.add(decision.resource_id)
            assigned_task_ids.add(str(decision.task_id))
            event.resource = decision.resource_id
            self.resourceEngine.busy.add(decision.resource_id)
            self.resourceEngine.load[decision.resource_id] = (
                self.resourceEngine.load.get(decision.resource_id, 0) + 1
            )
            self.diagnostics["strategy_assignments"] += 1
            self.diagnostics["resources_allocated"] += 1
            self._mark_resource_assignment(event)
            assigned_events.append(event)

        self.diagnostics["_allocate_enabled_events_calls"] += 1
        self.diagnostics["_allocate_enabled_events_time_seconds"] += (
            time.perf_counter() - allocation_start
        )
        return assigned_events

    def _build_resources(self, events) -> list[Resource]:
        start = time.perf_counter()
        self.diagnostics["resource_build_calls"] += 1
        current_time = self.simulation_time
        candidate_activities = {event.activity for event in events}
        candidate_activities.update(
            prediction.activity
            for prediction in self.future_predictions_by_task_id.values()
        )
        available_resources = (
            self.resourceEngine.availability.who_is_available(current_time)
            - self.resourceEngine.busy
            - set(self.reservations_by_resource_id)
        )

        resource_to_skills: dict[str, list[str]] = {
            resource_id: []
            for resource_id in available_resources
        }
        for activity in sorted(candidate_activities):
            permitted = self._permitted_resources_for_activity(activity)
            for resource_id in permitted & available_resources:
                resource_to_skills[resource_id].append(activity)

        resources = [
            Resource(resource_id=resource_id, available=True, skills=skills)
            for resource_id, skills in sorted(resource_to_skills.items())
            if skills
        ]
        self.diagnostics["resource_objects_created"] += len(resources)
        self.diagnostics["_build_resources_calls"] += 1
        self.diagnostics["_build_resources_time_seconds"] += (
            time.perf_counter() - start
        )
        return resources

    def _permitted_resources_for_activity(self, activity: str) -> set[str]:
        key = str(activity)
        if key not in self._permission_cache:
            self.diagnostics["permission_cache_misses"] += 1
            self._permission_cache[key] = set(self.resourceEngine.permissions.who_can(key))
        else:
            self.diagnostics["permission_cache_hits"] += 1
        return self._permission_cache[key]

    def _build_tasks(self, events) -> list[Task]:
        start = time.perf_counter()
        tasks = []
        for event in events:
            task_id = self.task_id_for_event(event)
            task = self._task_cache.get(task_id)
            if task is None:
                task = Task(
                    task_id=task_id,
                    case_id=self.case_id_for_event(event),
                    activity=event.activity,
                    enabled_time=self._time_value(
                        getattr(event, "time", self.simulation_time)
                    ),
                    priority=float(getattr(event, "priority", 0.0) or 0.0),
                )
                self._task_cache[task_id] = task
                self.diagnostics["task_cache_misses"] += 1
                self.diagnostics["task_cache_created"] += 1
            else:
                task.assigned = False
                task.blocked = False
                self.diagnostics["task_cache_hits"] += 1
            tasks.append(task)

        self.diagnostics["task_cache_max_size"] = max(
            self.diagnostics["task_cache_max_size"],
            len(self._task_cache),
        )
        self.diagnostics["_build_tasks_calls"] += 1
        self.diagnostics["_build_tasks_time_seconds"] += time.perf_counter() - start
        return tasks

    def _schedule_activity_end(self, event) -> None:
        lifecycle = self._ensure_task_lifecycle(event)
        if lifecycle.processing_end_event_scheduled:
            return
        should_record_diagnostics = (
            lifecycle.task_id not in self._processing_duration_diagnostic_task_ids
        )
        if lifecycle.sampled_processing_duration is None:
            lifecycle.sampled_processing_duration = self.processTimeEngine.getProcessingTime(
                event
            )
        elif should_record_diagnostics:
            self.diagnostics["processing_time_cached_duration_uses"] += 1
        lifecycle.sampled_processing_duration = self._normalized_processing_duration(
            event,
            lifecycle.sampled_processing_duration,
            record_diagnostics=should_record_diagnostics,
        )
        if should_record_diagnostics:
            self._record_final_processing_duration(event, lifecycle.sampled_processing_duration)
            self._processing_duration_diagnostic_task_ids.add(lifecycle.task_id)
        lifecycle.processing_start_time = self.simulation_time
        end_time = lifecycle.sampled_processing_duration + self.simulation_time
        end_event_id = str(self.event_counter)
        self._event_id_to_task_id[end_event_id] = lifecycle.task_id
        lifecycle.processing_end_event_scheduled = True
        lifecycle.processing_end_event_id = end_event_id
        self.push_event(
            end_time,
            EventType.ACTIVITY_END,
            event.activity,
            event.getAttribs(),
            event.eventCase,
        )
        self._remove_task_cache_entry(lifecycle.task_id)

    def _suspend_event(self, event) -> None:
        lifecycle = self._ensure_task_lifecycle(event)
        if lifecycle.resource_queue_entry_time is None:
            lifecycle.resource_queue_entry_time = self.simulation_time
        event.eventType = EventType.ACTIVITY_SUSPEND
        self._suspended_task_ids.add(self.task_id_for_event(event))
        self.diagnostics["suspended_events"] += 1
        heapq.heappush(self.waiting_processes, event)

    def _prediction_for_reservation(self, decision):
        for task_id, prediction in self.branch_prediction_by_task_id.items():
            if (
                prediction.case_id == str(decision.case_id)
                and prediction.selected_activity == decision.activity
                and task_id in self.future_predictions_by_task_id
            ):
                return prediction
        return None

    def _store_reservation(self, reservation: ResourceReservation) -> None:
        self.reservations_by_resource_id[reservation.resource_id] = reservation
        if reservation.target_task_id is not None:
            self.reservation_by_target_task_id[reservation.target_task_id] = reservation

    def _create_reservation(self, decision, prediction) -> ResourceReservation:
        self._reservation_counter += 1
        expected_delay = max(
            0.0,
            float(getattr(prediction, "expected_delay", 0.0) or 0.0),
        )
        expiration_time = None
        if expected_delay > 0.0:
            expiration_time = self.simulation_time + timedelta(
                seconds=expected_delay * self.reservation_expiration_multiplier
            )
        return ResourceReservation(
            reservation_id=f"RES{self._reservation_counter}",
            resource_id=str(decision.resource_id),
            case_id=str(decision.case_id),
            target_activity=str(decision.activity),
            target_task_id=prediction.target_task_id,
            source_prediction_id=prediction.prediction_id,
            creation_time=self.simulation_time,
            valid_from=self.simulation_time,
            expiration_time=expiration_time,
            status="created",
        )

    def _pop_matching_reservation(self, event) -> str | None:
        case_id = self.case_id_for_event(event)
        activity = getattr(event, "activity", None)
        task_id = self.task_id_for_event(event)
        self._expire_due_reservations(getattr(event, "time", self.simulation_time))
        reservation = self.reservation_by_target_task_id.get(task_id)
        candidates = (
            [reservation]
            if reservation is not None
            else list(self.reservations_by_resource_id.values())
        )
        for reservation in candidates:
            if reservation.case_id != case_id or reservation.target_activity != activity:
                continue
            if reservation.target_task_id is not None and reservation.target_task_id != task_id:
                continue
            resource_id = reservation.resource_id
            if resource_id not in self.resourceEngine.availability.who_is_available(event.time):
                self._finalize_reservation(reservation, status="expired")
                return None
            if resource_id not in self.resourceEngine.permissions.who_can(activity):
                self._finalize_reservation(reservation, status="cancelled")
                return None
            self._finalize_reservation(reservation, status="consumed")
            return resource_id
        return None

    def _active_future_predictions(self) -> list[Prediction]:
        active_predictions: list[Prediction] = []
        for task_id, prediction in list(self.future_predictions_by_task_id.items()):
            branch_prediction = self.branch_prediction_by_task_id.get(task_id)
            if branch_prediction is None:
                self._cancel_reservations_for_task(task_id)
                self.future_predictions_by_task_id.pop(task_id, None)
                self.prediction_id_by_task_id.pop(task_id, None)
                self.diagnostics["stale_predictions"] += 1
                continue
            if branch_prediction.status in {"branch_executed", "invalidated", "stale"}:
                self._cancel_reservations_for_task(task_id)
                self.future_predictions_by_task_id.pop(task_id, None)
                self.prediction_id_by_task_id.pop(task_id, None)
                self.diagnostics["stale_predictions"] += 1
                continue
            active_predictions.append(prediction)
        return active_predictions

    def _reservation_is_preferred(
        self,
        new_reservation: ResourceReservation,
        old_reservation: ResourceReservation,
    ) -> bool:
        if old_reservation.expiration_time is None:
            return False
        if new_reservation.expiration_time is None:
            return False
        return new_reservation.expiration_time < old_reservation.expiration_time

    def _expire_due_reservations(self, current_time) -> None:
        for reservation in list(self.reservations_by_resource_id.values()):
            if (
                reservation.expiration_time is not None
                and current_time > reservation.expiration_time
            ):
                self._finalize_reservation(reservation, status="expired")

    def _cancel_reservation_for_started_task(self, event) -> None:
        task_id = self.task_id_for_event(event)
        resource_id = str(getattr(event, "resource", ""))
        reservation = self.reservation_by_target_task_id.get(task_id)
        candidates = (
            [reservation]
            if reservation is not None
            else list(self.reservations_by_resource_id.values())
        )
        for reservation in candidates:
            if reservation.target_task_id != task_id:
                continue
            if reservation.resource_id != resource_id:
                self._finalize_reservation(reservation, status="cancelled")

    def _cancel_reservations_for_task(self, task_id: str) -> None:
        reservation = self.reservation_by_target_task_id.get(str(task_id))
        candidates = (
            [reservation]
            if reservation is not None
            else list(self.reservations_by_resource_id.values())
        )
        for reservation in candidates:
            if reservation.target_task_id == str(task_id):
                self._finalize_reservation(reservation, status="cancelled")

    def _cancel_reservations_for_case(self, case_id: str) -> None:
        for reservation in list(self.reservations_by_resource_id.values()):
            if reservation.case_id == str(case_id):
                self._finalize_reservation(reservation, status="cancelled")

    def _cleanup_reservations_at_horizon(self) -> None:
        for reservation in list(self.reservations_by_resource_id.values()):
            self._finalize_reservation(
                reservation,
                status="unresolved_at_horizon",
            )
        self.diagnostics["reservations_active_after_cleanup"] = len(
            self.reservations_by_resource_id
        )

    def _finalize_reservation(
        self,
        reservation: ResourceReservation,
        status: str,
    ) -> None:
        current = self.reservations_by_resource_id.get(reservation.resource_id)
        if current is not reservation:
            return

        self.reservations_by_resource_id.pop(reservation.resource_id, None)
        if reservation.target_task_id is not None:
            mapped = self.reservation_by_target_task_id.get(reservation.target_task_id)
            if mapped is reservation:
                self.reservation_by_target_task_id.pop(reservation.target_task_id, None)
        reservation.status = status
        self.reservation_history.append(reservation)

        diagnostic_key = {
            "consumed": "reservations_used",
            "expired": "reservations_expired",
            "cancelled": "reservations_cancelled",
            "overwritten": "reservations_overwritten",
            "unresolved_at_horizon": "reservations_unresolved_at_horizon",
        }.get(status)
        if diagnostic_key is not None:
            self.diagnostics[diagnostic_key] += 1

        task_id = reservation.target_task_id
        if task_id is None or status == "consumed":
            return

        self.future_predictions_by_task_id.pop(task_id, None)
        self.prediction_id_by_task_id.pop(task_id, None)
        prediction = self.branch_prediction_by_task_id.get(task_id)
        if prediction is not None:
            from dataclasses import replace

            self.branch_prediction_by_task_id[task_id] = replace(
                prediction,
                status=status,
            )

    def _ensure_task_lifecycle(self, event) -> TaskLifecycle:
        task_id = self.task_id_for_event(event)
        lifecycle = self.task_lifecycle.get(task_id)
        if lifecycle is None:
            lifecycle = TaskLifecycle(
                task_id=task_id,
                case_id=self.case_id_for_event(event),
                activity=event.activity,
                enabled_time=getattr(event, "time", self.simulation_time),
                process_wait_start=getattr(event, "time", self.simulation_time),
            )
            self.task_lifecycle[task_id] = lifecycle
        return lifecycle

    def _mark_resource_assignment(self, event) -> None:
        lifecycle = self._ensure_task_lifecycle(event)
        lifecycle.resource_assignment_time = self.simulation_time
        lifecycle.process_wait_end = self.simulation_time
        lifecycle.resource_id = getattr(event, "resource", None)

    def _mark_processing_end(self, event) -> None:
        lifecycle = self.task_lifecycle.get(self.task_id_for_event(event))
        if lifecycle is not None:
            lifecycle.processing_end_time = self.simulation_time
            lifecycle.processing_end_event_scheduled = False

    def _normalized_processing_duration(
        self,
        event,
        duration,
        record_diagnostics: bool = True,
    ) -> timedelta:
        activity = str(getattr(event, "activity", "") or "")
        visible_activity = bool(activity)
        if record_diagnostics:
            self._record_processing_time_source(event, duration)

        if not isinstance(duration, timedelta):
            if record_diagnostics:
                self.diagnostics["processing_time_invalid_value_count"] += 1
            duration = self._fallback_processing_duration(event)

        seconds = duration.total_seconds()
        if not math.isfinite(seconds) or seconds < 0:
            if record_diagnostics:
                self.diagnostics["processing_time_invalid_value_count"] += 1
            duration = self._fallback_processing_duration(event)
            seconds = duration.total_seconds()

        if visible_activity and (not math.isfinite(seconds) or seconds <= 0):
            if record_diagnostics:
                self.diagnostics["minimum_visible_duration_applications"] += 1
                self.diagnostics["processing_time_emergency_guard_hits"] += 1
                self.minimum_visible_duration_by_activity[activity] += 1
            return MIN_VISIBLE_PROCESSING_DURATION

        if (
            record_diagnostics
            and not visible_activity
            and isinstance(duration, timedelta)
            and duration.total_seconds() == 0
        ):
            self.diagnostics["zero_duration_silent_transitions"] += 1

        return duration

    def _fallback_processing_duration(self, event) -> timedelta:
        sample_basic = getattr(self.processTimeEngine, "sampleTime_basic", None)
        if sample_basic is None:
            return timedelta(0)
        try:
            duration = sample_basic(event.activity, "", "processing")
        except Exception:
            return timedelta(0)
        if isinstance(duration, timedelta):
            return duration
        return timedelta(0)

    def _record_processing_time_source(self, event, duration) -> None:
        activity = str(getattr(event, "activity", "") or "")
        if activity:
            self.diagnostics["visible_activity_processing_starts"] += 1

        sample_diagnostics = getattr(
            self.processTimeEngine,
            "last_sample_diagnostics",
            {},
        ) or {}
        self.diagnostics["processing_time_resampling_attempts"] += int(
            sample_diagnostics.get("resampling_attempts", 0) or 0
        )
        source = sample_diagnostics.get("source", "missing_model")
        category = sample_diagnostics.get("category", "") or ""
        source_to_counter = {
            "exact_model": "processing_time_model_hits",
            "learned_activity_fallback": "processing_time_activity_fallback_hits",
            "empirical_activity_fallback": "processing_time_empirical_activity_fallback_hits",
            "category_fallback": "processing_time_category_fallback_hits",
            "global_fallback": "processing_time_global_fallback_hits",
        }
        counter = source_to_counter.get(source)
        if counter is None:
            self.diagnostics["processing_time_missing_model_count"] += 1
            if activity:
                self.processing_time_missing_model_by_activity[activity] += 1
        else:
            self.diagnostics[counter] += 1
            if activity:
                self.diagnostics[
                    f"processing_time_source_activity_{self._safe_activity_key(activity)}_{source}"
                ] += 1
            if category:
                self.diagnostics[
                    f"processing_time_source_category_{self._safe_activity_key(category)}_{source}"
                ] += 1

        if not isinstance(duration, timedelta):
            self.diagnostics["processing_time_missing_model_count"] += 1
            if activity:
                self.processing_time_missing_model_by_activity[activity] += 1

    def _record_final_processing_duration(self, event, duration: timedelta) -> None:
        if not getattr(event, "activity", None):
            return
        seconds = duration.total_seconds()
        if math.isfinite(seconds):
            self._final_processing_duration_seconds.append(float(seconds))
            if seconds > 0:
                self.diagnostics["final_positive_duration_count"] += 1
            elif seconds == 0:
                self.diagnostics["final_zero_visible_duration_count"] += 1

    def _safe_activity_key(self, activity: str) -> str:
        return "".join(
            char if char.isalnum() else "_"
            for char in str(activity).strip()
        ).strip("_") or "unknown"

    def _remove_task_cache_entry(self, task_id: str) -> None:
        if self._task_cache.pop(str(task_id), None) is not None:
            self.diagnostics["task_cache_removed"] += 1

    def task_id_for_event(self, event) -> str:
        stable_task_id = getattr(event, "stable_task_id", None)
        if stable_task_id is not None:
            return str(stable_task_id)
        event_id = getattr(event, "eventId", None)
        if event_id is None and hasattr(event, "getAttribs"):
            event_id = event.getAttribs().get("EventID")
        if event_id is not None:
            return self._event_id_to_task_id.get(str(event_id), str(event_id))
        raise ValueError("Cannot derive a stable task id for an event without EventID")

    def case_id_for_event(self, event) -> str:
        event_case = getattr(event, "eventCase", None)
        case_id = getattr(event_case, "caseId", None)
        if case_id is None and hasattr(event, "getAttribs"):
            case_id = event.getAttribs().get("case:concept:name")
        return str(case_id if case_id is not None else "UNKNOWN_CASE")

    def _time_value(self, value) -> float:
        if hasattr(value, "timestamp"):
            return float(value.timestamp())
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def get_integration_diagnostics(self) -> dict[str, Any]:
        diagnostics = dict(self.diagnostics)
        diagnostics.update(
            {
                f"branch_prediction_source_{source}": count
                for source, count in self.prediction_source_counts.items()
            }
        )
        diagnostics.update(
            {
                f"processing_time_missing_model_activity_{self._safe_activity_key(activity)}": count
                for activity, count in self.processing_time_missing_model_by_activity.items()
            }
        )
        diagnostics.update(
            {
                f"minimum_visible_duration_activity_{self._safe_activity_key(activity)}": count
                for activity, count in self.minimum_visible_duration_by_activity.items()
            }
        )
        diagnostics.setdefault("branch_predictions", 0)
        diagnostics["diagnostic_cycle_guard_enabled"] = int(self.diagnostic_cycle_guard)
        diagnostics["cycle_repetition_limit"] = int(self.cycle_repetition_limit)
        diagnostics.setdefault("cycle_repetition_limit_hits", 0)
        diagnostics.setdefault("cycle_candidate_filtered", 0)
        diagnostics.setdefault("cycle_guard_filter_events", 0)
        diagnostics.setdefault("empirical_successor_candidates_filtered", 0)
        diagnostics.setdefault("empirical_successor_filter_events", 0)
        diagnostics.setdefault("fixed_route_cases_admitted", 0)
        diagnostics.setdefault("fixed_route_cases_completed", 0)
        diagnostics.setdefault("fixed_route_bpmn_fire_failures", 0)
        diagnostics.setdefault("fixed_route_prediction_global_candidate_fallbacks", 0)
        diagnostics.setdefault("fixed_route_no_route_available", 0)
        diagnostics.setdefault("fixed_route_arrivals_skipped_after_routes_exhausted", 0)
        diagnostics.setdefault("waiting_retry_events_scheduled", 0)
        diagnostics.setdefault("waiting_retry_events_processed", 0)
        diagnostics.setdefault("waiting_retry_duplicate_suppressed", 0)
        diagnostics.setdefault("waiting_retry_no_future_availability", 0)
        diagnostics.setdefault("waiting_retry_after_resource_release", 0)
        diagnostics.setdefault("waiting_retry_after_reservation_change", 0)
        diagnostics.setdefault("event_queue_empty_with_waiting_tasks", 0)
        diagnostics.setdefault("feasible_waiting_tasks_at_drain_end", 0)
        diagnostics.setdefault("infeasible_waiting_tasks_at_drain_end", 0)
        diagnostics.setdefault("branch_predictions_executed", 0)
        diagnostics.setdefault("prediction_execution_matches", 0)
        diagnostics.setdefault("prediction_execution_mismatches", 0)
        diagnostics.setdefault("park_song_predictions_consumed", 0)
        diagnostics.setdefault("duplicate_assignment_errors", 0)
        diagnostics.setdefault("reservation_decisions", 0)
        diagnostics.setdefault("reservations_created", 0)
        diagnostics.setdefault("reservations_used", 0)
        diagnostics.setdefault("reservations_expired", 0)
        diagnostics.setdefault("reservations_cancelled", 0)
        diagnostics.setdefault("reservations_overwritten", 0)
        diagnostics.setdefault("reservations_rejected_existing_kept", 0)
        diagnostics.setdefault("reservations_unresolved_at_horizon", 0)
        diagnostics.setdefault("reservations_active_after_cleanup", 0)
        diagnostics.setdefault("resources_allocated", 0)
        diagnostics.setdefault("resources_released", 0)
        diagnostics.setdefault("unique_predictions_executed", 0)
        diagnostics.setdefault("park_song_prediction_reuse_count", 0)
        diagnostics.setdefault("stale_predictions", 0)
        diagnostics.setdefault("unresolved_reservations", len(self.reservations_by_resource_id))
        diagnostics.setdefault(
            "unresolved_predictions",
            len(self.branch_prediction_by_task_id),
        )
        if hasattr(self.allocation_strategy, "get_diagnostics"):
            for key, value in self.allocation_strategy.get_diagnostics().items():
                diagnostics[key] = value
        if hasattr(self.allocation_strategy, "last_resource_loads"):
            resource_loads = getattr(self.allocation_strategy, "last_resource_loads")
            diagnostics["last_resource_load_candidate_count"] = len(resource_loads)
            if resource_loads:
                diagnostics["last_min_candidate_resource_load"] = min(
                    resource_loads.values()
                )
                diagnostics["last_selected_resource_load"] = getattr(
                    self.allocation_strategy,
                    "last_selected_resource_load",
                    min(resource_loads.values()),
                )
        diagnostics.setdefault("unequal_resource_load_comparisons", 0)
        diagnostics.setdefault("equal_resource_load_ties", 0)
        diagnostics.setdefault("resource_load_unequal_decisions", 0)
        diagnostics.setdefault("resource_load_tie_break_decisions", 0)
        diagnostics.setdefault("resource_load_assignment_decisions", 0)
        diagnostics.setdefault("_allocate_enabled_events_calls", 0)
        diagnostics.setdefault("_allocate_enabled_events_time_seconds", 0.0)
        diagnostics.setdefault("_build_tasks_calls", 0)
        diagnostics.setdefault("_build_tasks_time_seconds", 0.0)
        diagnostics.setdefault("_build_resources_calls", 0)
        diagnostics.setdefault("_build_resources_time_seconds", 0.0)
        diagnostics.setdefault("branch_prediction_calls", 0)
        diagnostics.setdefault("branch_prediction_time_seconds", 0.0)
        diagnostics.setdefault("allocation_strategy_calls", 0)
        diagnostics.setdefault("allocation_strategy_time_seconds", 0.0)
        diagnostics.setdefault("allocation_call_samples", 0)
        diagnostics.setdefault("waiting_queue_size_sum", 0)
        diagnostics.setdefault("tasks_converted_sum", 0)
        diagnostics.setdefault("resources_converted_sum", 0)
        diagnostics.setdefault("active_prediction_count_sum", 0)
        diagnostics.setdefault("active_reservation_count_sum", 0)
        diagnostics.setdefault("task_cache_hits", 0)
        diagnostics.setdefault("task_cache_misses", 0)
        diagnostics.setdefault("task_cache_created", 0)
        diagnostics.setdefault("task_cache_removed", 0)
        diagnostics.setdefault("task_cache_max_size", 0)
        diagnostics.setdefault("resource_build_calls", 0)
        diagnostics.setdefault("permission_cache_hits", 0)
        diagnostics.setdefault("permission_cache_misses", 0)
        diagnostics.setdefault("resource_objects_created", 0)
        diagnostics.setdefault("processing_time_model_hits", 0)
        diagnostics.setdefault("processing_time_activity_fallback_hits", 0)
        diagnostics.setdefault("processing_time_empirical_activity_fallback_hits", 0)
        diagnostics.setdefault("processing_time_category_fallback_hits", 0)
        diagnostics.setdefault("processing_time_global_fallback_hits", 0)
        diagnostics.setdefault("processing_time_emergency_guard_hits", 0)
        diagnostics.setdefault("processing_time_missing_model_count", 0)
        diagnostics.setdefault("processing_time_invalid_value_count", 0)
        diagnostics.setdefault("processing_time_cached_duration_uses", 0)
        diagnostics.setdefault("processing_time_resampling_attempts", 0)
        diagnostics["processing_model_loaded"] = int(
            bool(getattr(self.processTimeEngine, "model_loaded", False))
        )
        diagnostics["processing_model_retrained"] = int(
            bool(getattr(self.processTimeEngine, "model_retrained", False))
        )
        diagnostics["processing_model_load_error"] = (
            getattr(self.processTimeEngine, "model_load_error", None) or ""
        )
        diagnostics["processing_model_artifact_path"] = (
            getattr(self.processTimeEngine, "model_path", "") or ""
        )
        diagnostics.setdefault("minimum_visible_duration_applications", 0)
        diagnostics.setdefault("visible_activity_processing_starts", 0)
        diagnostics.setdefault("zero_duration_silent_transitions", 0)
        diagnostics.setdefault("final_positive_duration_count", 0)
        diagnostics.setdefault("final_zero_visible_duration_count", 0)
        visible_starts = diagnostics.get("visible_activity_processing_starts", 0)
        diagnostics["minimum_visible_duration_application_rate"] = (
            diagnostics.get("minimum_visible_duration_applications", 0) / visible_starts
            if visible_starts
            else 0.0
        )
        data_backed_hits = (
            diagnostics.get("processing_time_model_hits", 0)
            + diagnostics.get("processing_time_activity_fallback_hits", 0)
            + diagnostics.get("processing_time_empirical_activity_fallback_hits", 0)
            + diagnostics.get("processing_time_category_fallback_hits", 0)
        )
        diagnostics["processing_time_data_backed_coverage_rate"] = (
            data_backed_hits / visible_starts if visible_starts else 0.0
        )
        diagnostics["processing_time_exact_model_coverage_rate"] = (
            diagnostics.get("processing_time_model_hits", 0) / visible_starts
            if visible_starts
            else 0.0
        )
        diagnostics["processing_time_any_non_emergency_coverage_rate"] = (
            (
                visible_starts
                - diagnostics.get("processing_time_emergency_guard_hits", 0)
            )
            / visible_starts
            if visible_starts
            else 0.0
        )
        duration_seconds = self._final_processing_duration_seconds
        diagnostics["final_processing_duration_min"] = (
            min(duration_seconds) if duration_seconds else 0.0
        )
        diagnostics["final_processing_duration_median"] = (
            statistics.median(duration_seconds) if duration_seconds else 0.0
        )
        diagnostics["final_processing_duration_mean"] = (
            statistics.mean(duration_seconds) if duration_seconds else 0.0
        )
        diagnostics["final_processing_duration_max"] = (
            max(duration_seconds) if duration_seconds else 0.0
        )
        samples = max(int(diagnostics.get("allocation_call_samples", 0)), 1)
        diagnostics["waiting_queue_size_mean"] = (
            diagnostics.get("waiting_queue_size_sum", 0) / samples
        )
        diagnostics["tasks_converted_per_call_mean"] = (
            diagnostics.get("tasks_converted_sum", 0) / samples
        )
        diagnostics["resources_converted_per_call_mean"] = (
            diagnostics.get("resources_converted_sum", 0) / samples
        )
        diagnostics["active_prediction_count_mean"] = (
            diagnostics.get("active_prediction_count_sum", 0) / samples
        )
        diagnostics["reservation_count_mean"] = (
            diagnostics.get("active_reservation_count_sum", 0) / samples
        )
        return diagnostics
