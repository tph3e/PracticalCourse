from __future__ import annotations

import argparse
from copy import copy
from dataclasses import dataclass
import heapq
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import MethodType
from typing import Any

import pandas as pd
import pm4py

JOAO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = JOAO_ROOT.parent
sys.path.insert(0, str(JOAO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from Helper import Case, EventType
from SimulationEngineCore import Engine
from src.branching.PredictiveBranchingEngine import PredictiveBranchingEngine
from src.resource_allocation.AllocationStrategy import AllocationDecision, Prediction
from src.resource_allocation.MLPredictionAdapter import MLPredictionAdapter
from src.resource_allocation.ParkSongAllocation import ParkSongAllocation
from src.resource_allocation.RandomResourceAllocation import RandomResourceAllocation
from src.resource_allocation.RoundRobinResourceAllocation import RoundRobinResourceAllocation
from src.resource_allocation.ShortestQueueAllocation import ShortestQueueAllocation
from joao.scripts.resource_allocation.run_integrated_allocation_comparison import (
    average_cycle_time,
    average_resource_occupation,
    average_waiting_time,
    compute_log_diagnostics,
    gini,
)

try:
    from src.resource_allocation.BatchAllocationAdapter import BatchAllocationAdapter
except ModuleNotFoundError:
    BatchAllocationAdapter = None


CASE_COL = "case:concept:name"
RESOURCE_COL = "org:resource"
TIMESTAMP_COL = "time:timestamp"


def build_strategies(seed: int) -> dict[str, Any]:
    return {
        "Random": RandomResourceAllocation(seed=seed),
        "RoundRobin": RoundRobinResourceAllocation(),
        "ShortestQueue": ShortestQueueAllocation(),
        "ParkSong": ParkSongAllocation(allow_strategic_idling=False),
    }


def build_group_strategies(seed: int) -> dict[str, Any]:
    strategies = build_strategies(seed)
    if BatchAllocationAdapter is not None:
        strategies["Batch"] = BatchAllocationAdapter()
    return strategies


def build_parksong_ml_strategy() -> ParkSongAllocation:
    return ParkSongAllocation(
        prediction_probability_threshold=0.5,
        uncertainty_weight=5.0,
        idling_weight=1.0,
        waiting_weight=0.2,
        priority_weight=0.1,
        allow_strategic_idling=True,
    )


def build_prediction_adapter(data_path: str, seed: int) -> MLPredictionAdapter:
    log = pm4py.read_xes(data_path, variant="r4pm")
    feature_columns = [
        column
        for column in [
            "case:ApplicationType",
            "case:LoanGoal",
            "case:RequestedAmount",
            "CreditScore",
            "EventOrigin",
            "org:resource",
        ]
        if column in log.columns
    ]
    predictive_engine = PredictiveBranchingEngine(
        feature_columns=feature_columns,
        seed=seed,
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=2,
    )
    predictive_engine.train(log)
    return MLPredictionAdapter(
        predictive_engine=predictive_engine,
        default_expected_delay=1.0,
        source_name="PredictiveBranchingEngine",
    )


def run_full_simulation_comparison(
    data_path: str,
    start_time: datetime,
    end_time: datetime,
    seed: int,
    output_path: Path,
    min_processing_seconds: float = 1.0,
    strategies: dict[str, Any] | None = None,
    include_parksong_ml: bool = False,
) -> pd.DataFrame:
    rows = []

    selected_strategies = dict(strategies or build_strategies(seed))
    prediction_adapter = None
    if include_parksong_ml:
        selected_strategies["ParkSongML"] = build_parksong_ml_strategy()
        prediction_adapter = build_prediction_adapter(data_path, seed)

    for strategy_name, strategy in selected_strategies.items():
        engine = Engine(dataPath=data_path, seed=seed)
        engine.resourceEngine.global_allocation_strategy = strategy
        diagnostics = attach_full_global_diagnostics(engine)

        runner = FullGlobalAllocationRunner(
            engine=engine,
            min_processing_seconds=min_processing_seconds,
            prediction_adapter=(
                prediction_adapter
                if strategy_name == "ParkSongML"
                else None
            ),
            enable_reservations=strategy_name == "ParkSongML",
        )
        runner.run(start_time=start_time, end_time=end_time)

        log = engine.logger.get_log()
        metrics = compute_metrics(log)
        metrics.update(compute_log_diagnostics(log))
        metrics.update(diagnostics)
        metrics.update(runner.get_diagnostics())
        metrics.update(
            {
                "strategy": strategy_name,
                "seed": seed,
                "simulation_start": start_time.isoformat(),
                "simulation_end": end_time.isoformat(),
            }
        )
        rows.append(metrics)

    result = pd.DataFrame(rows)
    ordered_columns = [
        "strategy",
        "seed",
        "simulation_start",
        "simulation_end",
        "n_events",
        "n_cases",
        "throughput",
        "assigned_events",
        "average_cycle_time",
        "average_waiting_time",
        "average_resource_occupation",
        "resource_fairness",
        "weighted_resource_fairness",
        "global_strategy_calls",
        "waiting_events_seen",
        "global_assignments",
        "old_path_assignments",
        "suspended_events",
        "resumed_events",
        "max_waiting_queue_length",
        "ml_prediction_calls",
        "predictions_generated",
        "reservation_decisions",
        "reservations_created",
        "reservations_used",
        "reservations_expired",
    ]
    result = result[ordered_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def run_multi_seed_full_simulation_comparison(
    data_path: str,
    start_time: datetime,
    end_time: datetime,
    seeds: list[int],
    output_path: Path,
    summary_output_path: Path,
    min_processing_seconds: float = 1.0,
    group_methods: bool = False,
    include_parksong_ml: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_results = []

    for seed in seeds:
        seed_output_path = output_path.parent / f".{output_path.stem}_seed_{seed}.csv"
        strategies = (
            build_group_strategies(seed)
            if group_methods
            else build_strategies(seed)
        )
        seed_results.append(
            run_full_simulation_comparison(
                data_path=data_path,
                start_time=start_time,
                end_time=end_time,
                seed=seed,
                output_path=seed_output_path,
                min_processing_seconds=min_processing_seconds,
                strategies=strategies,
                include_parksong_ml=include_parksong_ml,
            )
        )
        seed_output_path.unlink(missing_ok=True)

    all_results = pd.concat(seed_results, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(output_path, index=False)

    metric_columns = [
        "n_events",
        "n_cases",
        "throughput",
        "assigned_events",
        "average_cycle_time",
        "average_resource_occupation",
        "resource_fairness",
        "global_strategy_calls",
        "waiting_events_seen",
        "global_assignments",
        "old_path_assignments",
        "suspended_events",
        "resumed_events",
        "max_waiting_queue_length",
        "ml_prediction_calls",
        "predictions_generated",
        "reservation_decisions",
        "reservations_created",
        "reservations_used",
        "reservations_expired",
    ]
    summary = (
        all_results.groupby("strategy", as_index=False)[metric_columns]
        .mean(numeric_only=True)
        .sort_values("strategy")
    )
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_output_path, index=False)
    return all_results, summary


@dataclass
class ResourceReservation:
    resource_id: str
    case_id: str
    activity: str
    created_time: float
    expected_time: float
    expires_at: float


class ReservationManager:
    def __init__(self, expiry_tolerance_seconds: float = 300.0):
        self.expiry_tolerance_seconds = max(0.0, float(expiry_tolerance_seconds))
        self._reservations: dict[str, ResourceReservation] = {}
        self.created = 0
        self.used = 0
        self.expired = 0

    def create_from_decision(
        self,
        decision: AllocationDecision,
        current_time: float,
    ) -> bool:
        if decision.decision_type != "reservation":
            return False
        if decision.task_id is not None:
            return False
        if decision.activity is None or decision.case_id is None:
            return False

        expected_delay = self._expected_delay_from_reasonless_decision(decision)
        self._reservations[decision.resource_id] = ResourceReservation(
            resource_id=decision.resource_id,
            case_id=str(decision.case_id),
            activity=str(decision.activity),
            created_time=current_time,
            expected_time=current_time + expected_delay,
            expires_at=current_time + expected_delay + self.expiry_tolerance_seconds,
        )
        self.created += 1
        return True

    def reserved_resource_ids(self, current_time: float) -> set[str]:
        self.expire(current_time)
        return set(self._reservations)

    def pop_matching(self, event, current_time: float) -> ResourceReservation | None:
        self.expire(current_time)
        case_id = self._case_id_for_event(event)
        activity = getattr(event, "activity", None)

        for resource_id, reservation in list(self._reservations.items()):
            if reservation.case_id == case_id and reservation.activity == activity:
                self.used += 1
                return self._reservations.pop(resource_id)

        return None

    def expire(self, current_time: float) -> None:
        expired_ids = [
            resource_id
            for resource_id, reservation in self._reservations.items()
            if current_time > reservation.expires_at
        ]
        for resource_id in expired_ids:
            self._reservations.pop(resource_id, None)
            self.expired += 1

    def diagnostics(self) -> dict[str, int]:
        return {
            "reservations_created": self.created,
            "reservations_used": self.used,
            "reservations_expired": self.expired,
        }

    def _expected_delay_from_reasonless_decision(
        self,
        decision: AllocationDecision,
    ) -> float:
        return 1.0

    def _case_id_for_event(self, event) -> str:
        event_case = getattr(event, "eventCase", None)
        case_id = getattr(event_case, "caseId", None)
        if case_id is None and hasattr(event, "getAttribs"):
            case_id = event.getAttribs().get("case:concept:name")
        return str(case_id if case_id is not None else "UNKNOWN_CASE")


class OnlineParkSongPredictionProvider:
    def __init__(self, prediction_adapter: MLPredictionAdapter):
        self.prediction_adapter = prediction_adapter
        self.calls = 0
        self.generated = 0

    def predict_for_waiting_events(self, engine: Engine, waiting_events) -> list[Prediction]:
        predictions: list[Prediction] = []

        for event in waiting_events:
            possible_activities = self._possible_after_current_activity(engine, event)
            if not possible_activities:
                continue

            self.calls += 1
            event_predictions = self.prediction_adapter.predict_for_event(
                event=event,
                possible_activities=possible_activities,
            )
            self.generated += len(event_predictions)
            predictions.extend(event_predictions)

        return predictions

    def diagnostics(self) -> dict[str, int]:
        return {
            "ml_prediction_calls": self.calls,
            "predictions_generated": self.generated,
        }

    def _possible_after_current_activity(self, engine: Engine, event) -> list[str]:
        case_id = getattr(getattr(event, "eventCase", None), "caseId", None)
        if case_id is None:
            return []

        bpmn_engine = engine.bpmnEngine
        had_marking = case_id in bpmn_engine.case_markings
        previous_marking = (
            copy(bpmn_engine.case_markings[case_id])
            if had_marking
            else None
        )

        try:
            if not bpmn_engine.fire_activity(event.activity, case_id):
                return []
            return bpmn_engine.getPossibleNextActivities(
                event.activity,
                case_id=case_id,
            )
        finally:
            if had_marking:
                bpmn_engine.case_markings[case_id] = previous_marking
            else:
                bpmn_engine.case_markings.pop(case_id, None)


class FullGlobalAllocationRunner:
    """
    Experiment-only simulator loop that routes activity-start allocation through
    ResourceEngine.allocate_waiting_tasks instead of the legacy per-event path.
    """

    def __init__(
        self,
        engine: Engine,
        min_processing_seconds: float = 1.0,
        prediction_adapter: MLPredictionAdapter | None = None,
        enable_reservations: bool = False,
    ):
        self.engine = engine
        self.min_processing_seconds = max(0.0, float(min_processing_seconds))
        self.prediction_provider = (
            OnlineParkSongPredictionProvider(prediction_adapter)
            if prediction_adapter is not None
            else None
        )
        self.reservation_manager = (
            ReservationManager()
            if enable_reservations
            else None
        )
        self.reservation_decisions = 0

    def get_diagnostics(self) -> dict[str, int]:
        diagnostics = {
            "ml_prediction_calls": 0,
            "predictions_generated": 0,
            "reservation_decisions": self.reservation_decisions,
            "reservations_created": 0,
            "reservations_used": 0,
            "reservations_expired": 0,
        }
        if self.prediction_provider is not None:
            diagnostics.update(self.prediction_provider.diagnostics())
        if self.reservation_manager is not None:
            diagnostics.update(self.reservation_manager.diagnostics())
        return diagnostics

    def run(self, start_time: datetime, end_time: datetime) -> None:
        engine = self.engine

        engine.push_event(
            start_time,
            EventType.CASE_ARRIVAL,
            "",
            engine.sample_case_data(),
            Case(engine.case_counter),
        )
        engine.case_counter += 1

        while engine.event_queue:
            event = engine.pop_event()
            engine.simulation_time = event.time
            if engine.simulation_time > end_time:
                break

            if event.eventType == EventType.CASE_ARRIVAL:
                self._handle_case_arrival(event)
                continue

            if event.eventType == EventType.ACTIVITY_START:
                self._handle_activity_start(event)
            elif event.eventType == EventType.ACTIVITY_END:
                self._handle_activity_end(event)

        remaining = []
        while engine.waiting_processes:
            remaining.append(heapq.heappop(engine.waiting_processes))
        for event in remaining:
            heapq.heappush(engine.waiting_processes, event)

    def _handle_case_arrival(self, event) -> None:
        engine = self.engine
        data = engine.sample_case_data()
        first_activity = engine.bpmnEngine.getStartActivity(data)

        engine.push_event(
            event.time,
            EventType.ACTIVITY_START,
            first_activity,
            data,
            event.eventCase,
        )
        engine.bpmnEngine.initialize_case(event.eventCase)

        new_case = Case(engine.case_counter)
        engine.case_counter += 1
        engine.cases.append(new_case)

        next_arrival_time = engine.arrivalEngine.nextArrivalTime(event.time) + event.time
        engine.push_event(next_arrival_time, EventType.CASE_ARRIVAL, "", dict(), new_case)

    def _handle_activity_start(self, event) -> None:
        assigned = self._try_allocate_events([event], resume=False)
        if not assigned:
            event.eventType = EventType.ACTIVITY_SUSPEND
            heapq.heappush(self.engine.waiting_processes, event)
            self.engine.logger.log_event(event)

    def _handle_activity_end(self, event) -> None:
        engine = self.engine
        engine.bpmnEngine.fire_activity(event.activity, event.eventCase.caseId)
        engine.resourceEngine.releaseResource(event)
        possible_next = engine.bpmnEngine.getPossibleNextActivities(
            event.activity,
            case_id=event.eventCase.caseId,
        )
        new_activities = engine.branchingEngine.getNextActivities(event, possible_next)
        if not (new_activities is None or new_activities == []):
            if isinstance(new_activities, str):
                new_activities = [new_activities]
            for new_activity in new_activities:
                if new_activity is None:
                    continue
                wait_time = engine.processTimeEngine.getWaitingTime(event, new_activity)
                engine.push_event(
                    wait_time + event.time,
                    EventType.ACTIVITY_START,
                    new_activity,
                    event.getAttribs(),
                    event.eventCase,
                )

        engine.logger.log_event(event)
        self._retry_waiting_events()

    def _retry_waiting_events(self) -> None:
        waiting_events = []
        while self.engine.waiting_processes:
            event = heapq.heappop(self.engine.waiting_processes)
            event.time = self.engine.simulation_time
            waiting_events.append(event)

        if not waiting_events:
            return

        assigned_events = self._try_allocate_events(waiting_events, resume=True)
        assigned_ids = {id(event) for event in assigned_events}

        for event in waiting_events:
            if id(event) not in assigned_ids:
                heapq.heappush(self.engine.waiting_processes, event)

    def _try_allocate_events(self, events, resume: bool):
        if not events:
            return []

        current_time_value = self.engine.resourceEngine._time_value(
            self.engine.simulation_time
        )
        if self.reservation_manager is not None:
            self.reservation_manager.expire(current_time_value)

        reserved_assignments = self._assign_matching_reservations(
            events=events,
            current_time_value=current_time_value,
        )
        reserved_event_ids = {
            id(event)
            for event, _decision in reserved_assignments
        }
        remaining_events = [
            event
            for event in events
            if id(event) not in reserved_event_ids
        ]

        predictions = []
        if self.prediction_provider is not None and remaining_events:
            predictions = self.prediction_provider.predict_for_waiting_events(
                engine=self.engine,
                waiting_events=remaining_events,
            )

        temporarily_reserved_resources = self._hold_unmatched_reservations(
            current_time_value=current_time_value,
        )

        decisions = self.engine.resourceEngine.allocate_waiting_tasks(
            waiting_events=remaining_events,
            current_time=self.engine.simulation_time,
            predictions=predictions,
        )
        for resource_id in temporarily_reserved_resources:
            self.engine.resourceEngine.busy.discard(resource_id)

        if self.reservation_manager is not None:
            for decision in decisions:
                if decision.decision_type == "reservation":
                    self.reservation_decisions += 1
                    self.reservation_manager.create_from_decision(
                        decision=decision,
                        current_time=current_time_value,
                    )

        decisions = [
            decision
            for _event, decision in reserved_assignments
        ] + decisions
        assigned_task_ids = {
            decision.task_id
            for decision in decisions
            if decision.decision_type == "assignment" and decision.task_id is not None
        }

        assigned_events = []
        for event in events:
            task_id = self.engine.resourceEngine._task_id_for_event(event)
            if task_id not in assigned_task_ids:
                continue

            event.update(
                {
                    "EventID": self.engine.event_counter,
                    "lifecycle:transition": (
                        EventType.ACTIVITY_RESUME
                        if resume
                        else EventType.ACTIVITY_START
                    ),
                    "time:timestamp": self.engine.simulation_time,
                }
            )
            self.engine.event_counter += 1

            processing_time = self.engine.processTimeEngine.getProcessingTime(event)
            if processing_time < timedelta(seconds=self.min_processing_seconds):
                processing_time = timedelta(seconds=self.min_processing_seconds)
            end_time_activity = event.time + processing_time
            self.engine.push_event(
                end_time_activity,
                EventType.ACTIVITY_END,
                event.activity,
                event.getAttribs(),
                event.eventCase,
            )
            self.engine.logger.log_event(event)
            assigned_events.append(event)

        return assigned_events

    def _assign_matching_reservations(self, events, current_time_value: float):
        if self.reservation_manager is None:
            return []

        assignments = []
        for event in events:
            reservation = self.reservation_manager.pop_matching(
                event=event,
                current_time=current_time_value,
            )
            if reservation is None:
                continue
            if not self._reservation_resource_can_execute(reservation, event):
                continue

            self.engine.resourceEngine.busy.add(reservation.resource_id)
            self.engine.resourceEngine.load[reservation.resource_id] = (
                self.engine.resourceEngine.load.get(reservation.resource_id, 0) + 1
            )
            event.resource = reservation.resource_id
            task_id = self.engine.resourceEngine._task_id_for_event(event)
            assignments.append(
                (
                    event,
                    AllocationDecision(
                        resource_id=reservation.resource_id,
                        task_id=task_id,
                        activity=event.activity,
                        case_id=self.engine.resourceEngine._case_id_for_event(event),
                        decision_type="assignment",
                        reason="Used ParkSongML resource reservation.",
                    ),
                )
            )

        return assignments

    def _hold_unmatched_reservations(self, current_time_value: float) -> set[str]:
        if self.reservation_manager is None:
            return set()

        resource_engine = self.engine.resourceEngine
        held_resources = (
            self.reservation_manager.reserved_resource_ids(current_time_value)
            - resource_engine.busy
        )
        resource_engine.busy.update(held_resources)
        return held_resources

    def _reservation_resource_can_execute(
        self,
        reservation: ResourceReservation,
        event,
    ) -> bool:
        resource_engine = self.engine.resourceEngine
        current_time = self.engine.simulation_time
        if reservation.resource_id not in resource_engine.availability.who_is_available(current_time):
            return False
        if reservation.resource_id in resource_engine.busy:
            return False
        if reservation.resource_id not in resource_engine.permissions.who_can(event.activity):
            return False
        return True


def attach_full_global_diagnostics(engine) -> dict[str, int]:
    diagnostics = {
        "global_strategy_calls": 0,
        "waiting_events_seen": 0,
        "global_assignments": 0,
        "old_path_assignments": 0,
        "max_waiting_queue_length": 0,
    }

    resource_engine = engine.resourceEngine
    original_allocate_resource = resource_engine.allocateResource
    original_allocate_waiting_tasks = resource_engine.allocate_waiting_tasks
    strategy = resource_engine.global_allocation_strategy
    original_strategy_allocate = getattr(strategy, "allocate", None)

    def counted_allocate_resource(self, event):
        allocated = original_allocate_resource(event)
        if allocated:
            diagnostics["old_path_assignments"] += 1
        return allocated

    def counted_allocate_waiting_tasks(self, waiting_events, current_time, predictions=None):
        waiting_count = len(waiting_events)
        diagnostics["waiting_events_seen"] += waiting_count
        diagnostics["max_waiting_queue_length"] = max(
            diagnostics["max_waiting_queue_length"],
            waiting_count,
        )

        decisions = original_allocate_waiting_tasks(
            waiting_events=waiting_events,
            current_time=current_time,
            predictions=predictions,
        )
        diagnostics["global_assignments"] += sum(
            1
            for decision in decisions
            if getattr(decision, "decision_type", None) == "assignment"
        )
        return decisions

    def counted_strategy_allocate(self, *args, **kwargs):
        diagnostics["global_strategy_calls"] += 1
        return original_strategy_allocate(*args, **kwargs)

    resource_engine.allocateResource = MethodType(
        counted_allocate_resource,
        resource_engine,
    )
    resource_engine.allocate_waiting_tasks = MethodType(
        counted_allocate_waiting_tasks,
        resource_engine,
    )
    if original_strategy_allocate is not None:
        strategy.allocate = MethodType(
            counted_strategy_allocate,
            strategy,
        )

    return diagnostics


def compute_metrics(log: pd.DataFrame) -> dict[str, float | int]:
    if log.empty:
        return {
            "n_events": 0,
            "n_cases": 0,
            "throughput": 0,
            "assigned_events": 0,
            "average_cycle_time": math.nan,
            "average_waiting_time": math.nan,
            "average_resource_occupation": math.nan,
            "resource_fairness": math.nan,
            "weighted_resource_fairness": math.nan,
        }

    prepared = log.copy()
    prepared[TIMESTAMP_COL] = pd.to_datetime(
        prepared[TIMESTAMP_COL],
        errors="coerce",
    )
    prepared = prepared.dropna(subset=[TIMESTAMP_COL])

    assigned = prepared[RESOURCE_COL].fillna("").astype(str).str.len() > 0
    complete = prepared["lifecycle:transition"].astype(str) == EventType.ACTIVITY_END.value
    completed_cases = prepared.loc[complete, CASE_COL].nunique() if CASE_COL in prepared else 0
    resource_counts = prepared.loc[assigned, RESOURCE_COL].astype(str).value_counts()

    return {
        "n_events": int(len(prepared)),
        "n_cases": int(prepared[CASE_COL].nunique()) if CASE_COL in prepared else 0,
        "throughput": int(completed_cases),
        "assigned_events": int(assigned.sum()),
        "average_cycle_time": average_cycle_time(prepared),
        "average_waiting_time": average_waiting_time(prepared),
        "average_resource_occupation": average_resource_occupation(prepared),
        "resource_fairness": gini(resource_counts.tolist()),
        "weighted_resource_fairness": math.nan,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full-simulation resource allocation comparison through global strategies."
    )
    parser.add_argument("--data-path", default=str(REPO_ROOT / "data" / "logData.xes"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--start", default="2000-01-03T09:00:00")
    parser.add_argument("--end", default="2000-01-03T12:00:00")
    parser.add_argument(
        "--min-processing-seconds",
        type=float,
        default=1.0,
        help="Experiment-only lower bound for sampled processing times.",
    )
    parser.add_argument(
        "--output",
        default=str(
            JOAO_ROOT
            / "results"
            / "full_simulation_resource_allocation_comparison.csv"
        ),
    )
    parser.add_argument(
        "--group-methods",
        action="store_true",
        help="Include executable group-level methods in addition to Joao methods.",
    )
    parser.add_argument(
        "--include-parksong-ml",
        action="store_true",
        help="Include online prediction-aware ParkSongML with reservation diagnostics.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds for a multi-seed run. Omit for a single --seed run.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Summary CSV path for multi-seed runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = datetime.fromisoformat(args.start)
    end_time = datetime.fromisoformat(args.end)

    if args.seeds:
        seeds = [
            int(seed.strip())
            for seed in args.seeds.split(",")
            if seed.strip()
        ]
        summary_output = (
            Path(args.summary_output)
            if args.summary_output is not None
            else Path(args.output).with_name(f"{Path(args.output).stem}_summary.csv")
        )
        result, summary = run_multi_seed_full_simulation_comparison(
            data_path=args.data_path,
            start_time=start_time,
            end_time=end_time,
            seeds=seeds,
            output_path=Path(args.output),
            summary_output_path=summary_output,
            min_processing_seconds=args.min_processing_seconds,
            group_methods=args.group_methods,
            include_parksong_ml=args.include_parksong_ml,
        )
        print(result.to_string(index=False))
        print("\nSummary:")
        print(summary.to_string(index=False))
        print(f"\nSaved full-simulation comparison to: {args.output}")
        print(f"Saved full-simulation summary to: {summary_output}")
        return

    result = run_full_simulation_comparison(
        data_path=args.data_path,
        start_time=start_time,
        end_time=end_time,
        seed=args.seed,
        output_path=Path(args.output),
        min_processing_seconds=args.min_processing_seconds,
        strategies=(
            build_group_strategies(args.seed)
            if args.group_methods
            else build_strategies(args.seed)
        ),
        include_parksong_ml=args.include_parksong_ml,
    )
    print(result.to_string(index=False))
    print(f"\nSaved full-simulation comparison to: {args.output}")


if __name__ == "__main__":
    main()
