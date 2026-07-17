from datetime import datetime, timedelta

import pandas as pd

from Helper import Case, Event, EventType
import SimulationEngineCore as simulation_core
from src.resource_allocation.AllocationStrategy import Prediction
from src.resource_allocation.ShortestQueueAllocation import ShortestQueueAllocation
from src.resource_allocation.ParkSongAllocation import ParkSongAllocation
from src.resource_allocation.integration.BranchPredictionContext import BranchPrediction
from src.resource_allocation.integration.IntegratedAllocationEngine import (
    IntegratedAllocationEngine,
    MIN_VISIBLE_PROCESSING_DURATION,
)
from src.resource_allocation.integration.TaskLifecycleContext import ResourceReservation


class FakeArrivalEngine:
    def __init__(self, log, seed=1):
        pass

    def nextArrivalTime(self, current_time):
        return timedelta(days=1)


class FakeProcessTimeEngine:
    def __init__(self, log, seed=1):
        pass

    def getProcessingTime(self, event, activity=None):
        return timedelta(seconds=1)

    def getWaitingTime(self, event, activity=None):
        return timedelta(seconds=0)

    def sampleTime_basic(self, activity, resource="", kind="processing"):
        return timedelta(seconds=1)


class ConfigurableProcessTimeEngine:
    def __init__(
        self,
        processing_duration,
        fallback_duration=timedelta(seconds=7),
        models_basic=None,
        fallback_models_basic=None,
    ):
        self.processing_duration = processing_duration
        self.fallback_duration = fallback_duration
        self.processing_calls = 0
        self.fallback_calls = 0
        self.models_basic = models_basic or {}
        self.fallback_models_basic = fallback_models_basic or {}

    def getProcessingTime(self, event, activity=None):
        self.processing_calls += 1
        activity = getattr(event, "activity", "")
        resource = getattr(event, "resource", "")
        if f"{activity}_{resource}_processing" in self.models_basic:
            source = "exact_model"
        elif f"{activity}_processing" in self.fallback_models_basic:
            source = "learned_activity_fallback"
        else:
            source = "missing_model"
        self.last_sample_diagnostics = {
            "kind": "processing",
            "activity": activity,
            "resource": resource,
            "source": source,
            "category": "",
            "resampling_attempts": 1 if source != "missing_model" else 0,
        }
        return self.processing_duration

    def getWaitingTime(self, event, activity=None):
        return timedelta(0)

    def sampleTime_basic(self, activity, resource="", kind="processing"):
        self.fallback_calls += 1
        return self.fallback_duration


class FakeBPMNEngine:
    def __init__(self):
        self.case_markings = {}
        self.final_marking = "final"
        self.net = None

    def initialize_case(self, case):
        self.case_markings[str(case)] = "start"

    def getStartActivity(self, data=None):
        return "A"

    def fire_activity(self, activity, case_id):
        if activity in {"B", "C"}:
            self.case_markings[str(case_id)] = self.final_marking
        else:
            self.case_markings[str(case_id)] = str(activity)
        return True

    def getPossibleNextActivities(self, current_activity, case_id=None):
        if current_activity == "A":
            return ["B", "C"]
        return []


class HourlyAvailability:
    def __init__(self, available_from_hour=10):
        self._all_resources = {"R1"}
        self.available_from_hour = available_from_hour
        self.calendars = None

    def who_is_available(self, time):
        if time.hour >= self.available_from_hour:
            return {"R1"}
        return set()


class FakeCompositeBranching:
    def __init__(self):
        self.engines = []
        self.calls = 0

    def get_statistics(self):
        return {
            "engine_success_counts": (
                {"AttributeBasedBranchingEngine": self.calls}
                if self.calls
                else {}
            ),
            "random_fallback_count": 0,
        }

    def getNextActivities(self, event, possible):
        self.calls += 1
        return ["B"]


class RecordingCompositeBranching(FakeCompositeBranching):
    def __init__(self, selected="B"):
        super().__init__()
        self.selected = selected
        self.possible_seen = []

    def getNextActivities(self, event, possible):
        self.calls += 1
        self.possible_seen.append(list(possible))
        return [self.selected] if self.selected in possible else [possible[0]]


def build_log():
    return pd.DataFrame(
        [
            {
                "case:concept:name": f"C{i}",
                "concept:name": "A",
                "time:timestamp": pd.Timestamp(f"2026-01-0{i + 1} 09:00:00"),
                "case:ApplicationType": "New credit",
                "case:LoanGoal": "Car",
                "case:RequestedAmount": 10000 + i,
                "org:resource": "R1",
                "lifecycle:transition": "complete",
            }
            for i in range(6)
        ]
    )


def test_integrated_engine_uses_composite_prediction_for_executed_branch(monkeypatch):
    monkeypatch.setattr(simulation_core.pm4py, "read_xes", lambda *args, **kwargs: build_log())
    monkeypatch.setattr(simulation_core, "ArrivalEngine", FakeArrivalEngine)
    monkeypatch.setattr(simulation_core, "ProcessTimeEngine", FakeProcessTimeEngine)

    strategy = ParkSongAllocation(allow_strategic_idling=False)
    engine = IntegratedAllocationEngine(
        dataPath="data/logData.xes",
        seed=1,
        allocation_strategy=strategy,
        branching_engine=FakeCompositeBranching(),
    )
    engine.bpmnEngine = FakeBPMNEngine()
    engine.resourceEngine.availability.calendars = None
    engine.resourceEngine.availability._all_resources = {"R1"}
    engine.resourceEngine.permissions._activity_to_resources = {
        "A": {"R1"},
        "B": {"R1"},
        "C": {"R1"},
    }

    engine.run(
        datetime(2026, 1, 5, 9, 0),
        datetime(2026, 1, 5, 9, 0, 5),
        format_type=[],
    )

    diagnostics = engine.get_integration_diagnostics()
    log = engine.logger.get_log()

    assert diagnostics["branch_predictions"] >= 1
    assert diagnostics["prediction_execution_mismatches"] == 0
    assert diagnostics["prediction_execution_matches"] >= 1
    assert diagnostics["park_song_predictions_consumed"] >= 1
    assert "B" in set(log["concept:name"])
    assert "C" not in set(log["concept:name"])
    assert "0" in engine.admitted_case_ids
    assert "0" in engine.completed_case_ids


def test_integrated_engine_drain_stops_arrivals_but_processes_existing_case(monkeypatch):
    monkeypatch.setattr(simulation_core.pm4py, "read_xes", lambda *args, **kwargs: build_log())
    monkeypatch.setattr(simulation_core, "ArrivalEngine", FakeArrivalEngine)
    monkeypatch.setattr(simulation_core, "ProcessTimeEngine", FakeProcessTimeEngine)

    engine = IntegratedAllocationEngine(
        dataPath="data/logData.xes",
        seed=1,
        allocation_strategy=ParkSongAllocation(allow_strategic_idling=False),
        branching_engine=FakeCompositeBranching(),
    )
    engine.bpmnEngine = FakeBPMNEngine()
    engine.resourceEngine.availability.calendars = None
    engine.resourceEngine.availability._all_resources = {"R1"}
    engine.resourceEngine.permissions._activity_to_resources = {
        "A": {"R1"},
        "B": {"R1"},
        "C": {"R1"},
    }

    start = datetime(2026, 1, 5, 9, 0)
    engine.run(
        start,
        start,
        format_type=[],
        drain_until=start + timedelta(seconds=5),
    )

    assert engine.admitted_case_ids == {"0"}
    assert engine.completed_case_ids == {"0"}


def test_deadlock_is_not_counted_as_completion(monkeypatch):
    engine = build_engine(monkeypatch)
    event = make_event(task_id="1", case_id="C1", activity="B")
    engine.bpmnEngine.case_markings["C1"] = "not-final"
    engine.bpmnEngine.final_marking = "final"

    engine._classify_no_next_case(event)

    assert engine.completed_case_ids == set()
    assert engine.deadlocked_case_ids == {"C1"}


def test_cycle_guard_filters_activity_after_empirical_visit_limit(monkeypatch):
    engine = build_engine(monkeypatch)
    event = make_event(task_id="1", case_id="C1", activity="A")
    engine.activity_visit_limits = {"A": 1, "B": 5}
    engine._completed_activity_counts_by_case["C1"] = {"A": 1}

    filtered = engine._filter_cycle_candidates(event, ["A", "B"])

    assert filtered == ["B"]
    assert engine.diagnostics["cycle_candidate_filtered"] == 1


def test_empirical_successor_filter_keeps_supported_bpmn_candidates(monkeypatch):
    engine = build_engine(monkeypatch)
    event = make_event(task_id="1", case_id="C1", activity="A")
    engine.empirical_successors = {"A": {"B"}}

    filtered = engine._filter_empirical_successors(event, ["B", "C"])

    assert filtered == ["B"]
    assert engine.diagnostics["empirical_successor_candidates_filtered"] == 1


def build_engine(monkeypatch, strategy=None, branching_engine=None, **kwargs):
    monkeypatch.setattr(simulation_core.pm4py, "read_xes", lambda *args, **kwargs: build_log())
    monkeypatch.setattr(simulation_core, "ArrivalEngine", FakeArrivalEngine)
    monkeypatch.setattr(simulation_core, "ProcessTimeEngine", FakeProcessTimeEngine)
    engine = IntegratedAllocationEngine(
        dataPath="data/logData.xes",
        seed=1,
        allocation_strategy=strategy or ParkSongAllocation(),
        branching_engine=branching_engine or FakeCompositeBranching(),
        **kwargs,
    )
    engine.bpmnEngine = FakeBPMNEngine()
    engine.simulation_time = datetime(2026, 1, 5, 9, 0)
    engine.resourceEngine.availability.calendars = None
    engine.resourceEngine.availability._all_resources = {"R1", "R2"}
    engine.resourceEngine.permissions._activity_to_resources = {
        "A": {"R1", "R2"},
        "B": {"R1", "R2"},
        "C": {"R1", "R2"},
    }
    return engine


def test_default_final_mode_does_not_learn_full_log_empirical_runtime_filters(monkeypatch):
    engine = build_engine(monkeypatch)

    assert engine.diagnostic_cycle_guard is False
    assert engine.empirical_successors == {}
    assert engine.activity_visit_limits == {}
    diagnostics = engine.get_integration_diagnostics()
    assert diagnostics["diagnostic_cycle_guard_enabled"] == 0


def test_diagnostic_cycle_guard_is_explicitly_labeled(monkeypatch):
    engine = build_engine(monkeypatch, diagnostic_cycle_guard=True)

    assert engine.diagnostic_cycle_guard is True
    assert engine.activity_visit_limits
    diagnostics = engine.get_integration_diagnostics()
    assert diagnostics["diagnostic_cycle_guard_enabled"] == 1


def test_cycle_repetition_detector_classifies_without_mutating_candidates(monkeypatch):
    engine = build_engine(monkeypatch, cycle_repetition_limit=2)
    event = make_event(task_id="1", case_id="C1", activity="A")
    engine._completed_activity_counts_by_case["C1"] = {"A": 2}
    candidates = ["A", "B"]

    assert engine._detect_repetition_cycle(event) is True
    assert candidates == ["A", "B"]
    assert engine.get_integration_diagnostics()["cycle_repetition_limit_hits"] == 1


def test_fixed_route_completes_and_cleans_case_state(monkeypatch):
    branching = RecordingCompositeBranching(selected="B")
    engine = build_engine(
        monkeypatch,
        branching_engine=branching,
        fixed_routes=[["A", "B"]],
        fixed_route_case_ids=["Application_X"],
    )
    start = datetime(2026, 1, 5, 9, 0)

    engine.run(start, start, format_type=[], drain_until=start + timedelta(seconds=10))

    assert engine.completed_case_ids == {"0"}
    assert not engine.waiting_processes
    assert not engine.reservations_by_resource_id
    diagnostics = engine.get_integration_diagnostics()
    assert diagnostics["fixed_route_cases_admitted"] == 1
    assert diagnostics["fixed_route_cases_completed"] == 1


def test_fixed_route_is_identical_across_allocation_strategies(monkeypatch):
    start = datetime(2026, 1, 5, 9, 0)
    routes = [["A", "B"]]
    sequences = []
    for strategy in [ParkSongAllocation(allow_strategic_idling=False), ShortestQueueAllocation()]:
        engine = build_engine(
            monkeypatch,
            strategy=strategy,
            branching_engine=RecordingCompositeBranching(selected="B"),
            fixed_routes=routes,
        )
        engine.run(start, start, format_type=[], drain_until=start + timedelta(seconds=10))
        log = engine.logger.get_log()
        sequences.append(log[log["lifecycle:transition"] == "complete"]["concept:name"].tolist())

    assert sequences[0] == sequences[1] == ["A", "B"]


def test_fixed_route_uses_historical_arrival_timestamps(monkeypatch):
    start = datetime(2026, 1, 5, 9, 0)
    arrivals = [start, start + timedelta(minutes=30)]
    engine = build_engine(
        monkeypatch,
        strategy=ParkSongAllocation(allow_strategic_idling=False),
        branching_engine=RecordingCompositeBranching(selected="B"),
        fixed_routes=[["A"], ["A"]],
        fixed_route_arrival_times=arrivals,
    )

    engine.run(start, start + timedelta(hours=1), format_type=[], drain_until=start + timedelta(hours=2))

    log = engine.logger.get_log()
    starts = log[log["lifecycle:transition"] == "start"]["time:timestamp"].tolist()
    assert starts[:2] == arrivals
    assert engine.admitted_case_ids == {"0", "1"}
    assert engine.get_integration_diagnostics()["fixed_route_historical_arrivals_scheduled"] == 2


def test_fixed_route_prediction_candidates_do_not_expose_actual_next_only(monkeypatch):
    branching = RecordingCompositeBranching(selected="B")
    engine = build_engine(
        monkeypatch,
        strategy=ParkSongAllocation(allow_strategic_idling=True),
        branching_engine=branching,
        fixed_routes=[["A", "C"]],
    )
    start = datetime(2026, 1, 5, 9, 0)

    engine.run(start, start, format_type=[], drain_until=start + timedelta(seconds=10))

    assert branching.possible_seen
    assert set(branching.possible_seen[0]) == {"B", "C"}
    assert branching.possible_seen[0] != ["C"]


def test_waiting_task_retried_at_next_availability(monkeypatch):
    start = datetime(2026, 1, 5, 9, 30)
    engine = build_engine(
        monkeypatch,
        strategy=ParkSongAllocation(allow_strategic_idling=False),
        fixed_routes=[["A"]],
    )
    engine.resourceEngine.availability = HourlyAvailability(available_from_hour=10)

    engine.run(start, start, format_type=[], drain_until=start + timedelta(hours=2))

    assert engine.completed_case_ids == {"0"}
    diagnostics = engine.get_integration_diagnostics()
    assert diagnostics["waiting_retry_events_scheduled"] >= 1
    assert diagnostics["waiting_retry_events_processed"] >= 1
    assert diagnostics["event_queue_empty_with_waiting_tasks"] == 0


def test_waiting_retry_duplicate_suppressed(monkeypatch):
    start = datetime(2026, 1, 5, 9, 30)
    engine = build_engine(monkeypatch)
    engine.simulation_time = start
    engine._event_horizon = start + timedelta(hours=2)
    engine.resourceEngine.availability = HourlyAvailability(available_from_hour=10)
    event = make_event(task_id="1", case_id="C1", activity="A")
    engine._suspend_event(event)

    engine._schedule_waiting_retry("test")
    engine._schedule_waiting_retry("test")

    diagnostics = engine.get_integration_diagnostics()
    assert diagnostics["waiting_retry_events_scheduled"] == 1
    assert diagnostics["waiting_retry_duplicate_suppressed"] == 1


def test_event_queue_empty_with_feasible_waiting_work_is_reported(monkeypatch):
    start = datetime(2026, 1, 5, 10, 0)
    engine = build_engine(monkeypatch)
    engine.simulation_time = start
    engine.resourceEngine.availability = HourlyAvailability(available_from_hour=10)
    engine._suspend_event(make_event(task_id="1", case_id="C1", activity="A"))

    engine._record_waiting_drain_feasibility()

    diagnostics = engine.get_integration_diagnostics()
    assert diagnostics["event_queue_empty_with_waiting_tasks"] == 1
    assert diagnostics["feasible_waiting_tasks_at_drain_end"] == 1


def make_event(task_id="1", case_id="C1", activity="B", resource=""):
    case = Case(case_id)
    event = Event(
        EventType.ACTIVITY_START,
        activity,
        datetime(2026, 1, 5, 9, 0),
        int(task_id),
        case,
        {
            "EventID": int(task_id),
            "case:concept:name": case_id,
            "concept:name": activity,
            "org:resource": resource,
        },
    )
    return event


def make_prediction(task_id="1", case_id="C1", activity="B", delay=10.0):
    return BranchPrediction(
        prediction_id=f"BP{task_id}",
        case_id=case_id,
        current_activity="A",
        decision_point="A",
        candidate_activities=(activity,),
        selected_activity=activity,
        probabilities={activity: 1.0},
        prediction_time=datetime(2026, 1, 5, 9, 0),
        target_task_id=task_id,
        expected_delay=delay,
    )


def add_reservation(engine, resource_id="R1", task_id="1", case_id="C1", activity="B", delay=10.0):
    prediction = make_prediction(task_id=task_id, case_id=case_id, activity=activity, delay=delay)
    engine.branch_prediction_by_task_id[task_id] = prediction
    engine.prediction_id_by_task_id[task_id] = prediction.prediction_id
    engine.future_predictions_by_task_id[task_id] = Prediction(
        case_id=case_id,
        activity=activity,
        probability=1.0,
        expected_delay=delay,
        source="test",
        confidence=1.0,
    )
    reservation = ResourceReservation(
        reservation_id=f"RES{task_id}",
        resource_id=resource_id,
        case_id=case_id,
        target_activity=activity,
        target_task_id=task_id,
        source_prediction_id=prediction.prediction_id,
        creation_time=engine.simulation_time,
        valid_from=engine.simulation_time,
        expiration_time=engine.simulation_time + timedelta(seconds=delay) if delay > 0 else None,
    )
    engine.reservations_by_resource_id[resource_id] = reservation
    return reservation


def test_shortest_queue_integration_skips_busy_lower_load_resource(monkeypatch):
    engine = build_engine(monkeypatch, strategy=ShortestQueueAllocation())
    engine.resourceEngine.busy.add("R1")
    engine.resourceEngine.load = {"R1": 0, "R2": 5}
    event = make_event(task_id="1", case_id="C1", activity="A")

    assigned = engine._allocate_enabled_events([event], include_waiting=False)

    assert assigned == [event]
    assert event.resource == "R2"
    diagnostics = engine.get_integration_diagnostics()
    assert diagnostics["last_selected_resource_load"] == 5.0


def test_shortest_queue_integration_passes_resource_engine_loads(monkeypatch):
    strategy = ShortestQueueAllocation()
    engine = build_engine(monkeypatch, strategy=strategy)
    engine.resourceEngine.load = {"R1": 5, "R2": 1}
    event = make_event(task_id="1", case_id="C1", activity="A")

    assigned = engine._allocate_enabled_events([event], include_waiting=False)

    assert assigned == [event]
    assert event.resource == "R2"
    assert strategy.last_resource_loads == {"R1": 5.0, "R2": 1.0}
    assert engine.resourceEngine.load["R2"] == 2


def test_resource_engine_load_is_cumulative_and_not_decremented_on_release(monkeypatch):
    engine = build_engine(monkeypatch, strategy=ShortestQueueAllocation())
    event = make_event(task_id="1", case_id="C1", activity="A")

    assigned = engine._allocate_enabled_events([event], include_waiting=False)
    assert assigned == [event]
    assigned_resource = event.resource
    assert engine.resourceEngine.load[assigned_resource] == 1

    engine.resourceEngine.releaseResource(event)

    assert assigned_resource not in engine.resourceEngine.busy
    assert engine.resourceEngine.load[assigned_resource] == 1


def test_shortest_queue_integration_respects_permission_and_availability_filters(monkeypatch):
    engine = build_engine(monkeypatch, strategy=ShortestQueueAllocation())
    engine.resourceEngine.availability._all_resources = {"R1", "R2"}
    engine.resourceEngine.permissions._activity_to_resources = {"A": {"R2"}}
    engine.resourceEngine.load = {"R1": 0, "R2": 10}
    event = make_event(task_id="1", case_id="C1", activity="A")

    assigned = engine._allocate_enabled_events([event], include_waiting=False)

    assert assigned == [event]
    assert event.resource == "R2"


def test_reservation_consumed_by_matching_case_activity_task(monkeypatch):
    engine = build_engine(monkeypatch)
    reservation = add_reservation(engine)

    resource = engine._pop_matching_reservation(make_event("1", "C1", "B"))

    assert resource == "R1"
    assert reservation.status == "consumed"
    assert engine.get_integration_diagnostics()["reservations_used"] == 1


def test_unrelated_task_does_not_consume_reservation(monkeypatch):
    engine = build_engine(monkeypatch)
    add_reservation(engine, task_id="1", case_id="C1", activity="B")

    assert engine._pop_matching_reservation(make_event("2", "C2", "B")) is None
    assert len(engine.reservations_by_resource_id) == 1


def test_stale_prediction_cancels_reservation(monkeypatch):
    engine = build_engine(monkeypatch)
    add_reservation(engine, task_id="1")
    engine.branch_prediction_by_task_id.pop("1")

    assert engine._active_future_predictions() == []

    diagnostics = engine.get_integration_diagnostics()
    assert diagnostics["reservations_cancelled"] == 1
    assert diagnostics["reservations_active_after_cleanup"] == 0


def test_stale_prediction_does_not_block_resource_build(monkeypatch):
    engine = build_engine(monkeypatch, strategy=ShortestQueueAllocation())
    add_reservation(engine, resource_id="R1", task_id="99", case_id="C9")
    engine.branch_prediction_by_task_id.pop("99")
    engine.resourceEngine.load = {"R1": 0, "R2": 10}
    event = make_event("1", "C1", "A")

    assigned = engine._allocate_enabled_events([event], include_waiting=False)

    assert assigned == [event]
    assert event.resource == "R1"
    diagnostics = engine.get_integration_diagnostics()
    assert diagnostics["reservations_cancelled"] == 1


def test_completed_case_cancels_reservation(monkeypatch):
    engine = build_engine(monkeypatch)
    add_reservation(engine, case_id="C1")

    engine._cancel_reservations_for_case("C1")

    assert engine.get_integration_diagnostics()["reservations_cancelled"] == 1
    assert not engine.reservations_by_resource_id


def test_resource_unavailable_expires_reservation(monkeypatch):
    engine = build_engine(monkeypatch)
    add_reservation(engine)
    engine.resourceEngine.availability._all_resources = {"R2"}

    assert engine._pop_matching_reservation(make_event("1", "C1", "B")) is None
    assert engine.get_integration_diagnostics()["reservations_expired"] == 1


def test_expired_reservation_does_not_block_resource_build(monkeypatch):
    engine = build_engine(monkeypatch, strategy=ShortestQueueAllocation())
    reservation = add_reservation(engine, resource_id="R1", task_id="99", case_id="C9")
    reservation.expiration_time = engine.simulation_time - timedelta(seconds=1)
    engine.resourceEngine.load = {"R1": 0, "R2": 10}
    event = make_event("1", "C1", "A")

    assigned = engine._allocate_enabled_events([event], include_waiting=False)

    assert assigned == [event]
    assert event.resource == "R1"
    diagnostics = engine.get_integration_diagnostics()
    assert diagnostics["reservations_expired"] == 1


def test_task_cache_reuse_preserves_shortest_queue_assignment(monkeypatch):
    cold_engine = build_engine(monkeypatch, strategy=ShortestQueueAllocation())
    cold_engine.resourceEngine.load = {"R1": 5, "R2": 1}
    cold_event = make_event("1", "C1", "A")

    warm_engine = build_engine(monkeypatch, strategy=ShortestQueueAllocation())
    warm_engine.resourceEngine.load = {"R1": 5, "R2": 1}
    warm_event = make_event("1", "C1", "A")
    warm_engine.resourceEngine.availability._all_resources = set()
    assert warm_engine._allocate_enabled_events([warm_event], include_waiting=False) == []
    warm_engine.resourceEngine.availability._all_resources = {"R1", "R2"}

    cold_assigned = cold_engine._allocate_enabled_events(
        [cold_event],
        include_waiting=False,
    )
    warm_assigned = warm_engine._allocate_enabled_events(
        [warm_event],
        include_waiting=False,
    )

    assert cold_assigned == [cold_event]
    assert warm_assigned == [warm_event]
    assert cold_event.resource == warm_event.resource == "R2"
    assert warm_engine.get_integration_diagnostics()["task_cache_hits"] == 1


def test_permission_loss_cancels_reservation(monkeypatch):
    engine = build_engine(monkeypatch)
    add_reservation(engine)
    engine.resourceEngine.permissions._activity_to_resources = {"B": {"R2"}}

    assert engine._pop_matching_reservation(make_event("1", "C1", "B")) is None
    assert engine.get_integration_diagnostics()["reservations_cancelled"] == 1


def test_target_task_starting_elsewhere_cancels_reservation(monkeypatch):
    engine = build_engine(monkeypatch)
    add_reservation(engine, resource_id="R1", task_id="1")
    event = make_event("1", "C1", "B", resource="R2")

    engine._cancel_reservation_for_started_task(event)

    assert engine.get_integration_diagnostics()["reservations_cancelled"] == 1
    assert not engine.reservations_by_resource_id


def test_reservation_overwrite_policy_keeps_existing_without_preference(monkeypatch):
    engine = build_engine(monkeypatch)
    existing = add_reservation(engine, resource_id="R1", task_id="1", delay=0.0)
    new_prediction = make_prediction(task_id="2", case_id="C2", activity="B", delay=5.0)
    new_reservation = engine._create_reservation(
        type("Decision", (), {"resource_id": "R1", "case_id": "C2", "activity": "B"})(),
        new_prediction,
    )

    assert engine._reservation_is_preferred(new_reservation, existing) is False


def test_reservation_expiration_multiplier_extends_expected_delay(monkeypatch):
    engine = build_engine(monkeypatch, reservation_expiration_multiplier=2.0)
    prediction = make_prediction(task_id="1", case_id="C1", activity="B", delay=10.0)

    reservation = engine._create_reservation(
        type("Decision", (), {"resource_id": "R1", "case_id": "C1", "activity": "B"})(),
        prediction,
    )

    assert reservation.expiration_time == engine.simulation_time + timedelta(seconds=20)


def test_horizon_cleanup_marks_unresolved_and_removes_active_reservation(monkeypatch):
    engine = build_engine(monkeypatch)
    reservation = add_reservation(engine)

    engine._cleanup_reservations_at_horizon()

    diagnostics = engine.get_integration_diagnostics()
    assert reservation.status == "unresolved_at_horizon"
    assert diagnostics["reservations_unresolved_at_horizon"] == 1
    assert diagnostics["reservations_active_after_cleanup"] == 0
    assert not engine.reservations_by_resource_id


def scheduled_end_events(engine):
    return [
        event
        for event in engine.event_queue
        if event.eventType == EventType.ACTIVITY_END
    ]


def test_zero_visible_processing_duration_advances_by_minimum_visible_duration(monkeypatch):
    engine = build_engine(monkeypatch)
    engine.processTimeEngine = ConfigurableProcessTimeEngine(
        timedelta(0),
        models_basic={"A_R1_processing": object()},
    )
    event = make_event("1", "C1", "A", resource="R1")

    engine._schedule_activity_end(event)

    ends = scheduled_end_events(engine)
    assert len(ends) == 1
    assert ends[0].time == engine.simulation_time + MIN_VISIBLE_PROCESSING_DURATION
    lifecycle = engine.task_lifecycle[engine.task_id_for_event(event)]
    assert lifecycle.sampled_processing_duration == MIN_VISIBLE_PROCESSING_DURATION
    diagnostics = engine.get_integration_diagnostics()
    assert diagnostics["minimum_visible_duration_applications"] == 1
    assert diagnostics["minimum_visible_duration_application_rate"] == 1.0
    assert diagnostics["minimum_visible_duration_activity_A"] == 1


def test_positive_processing_duration_is_preserved(monkeypatch):
    engine = build_engine(monkeypatch)
    engine.processTimeEngine = ConfigurableProcessTimeEngine(
        timedelta(seconds=3),
        models_basic={"A_R1_processing": object()},
    )
    event = make_event("1", "C1", "A", resource="R1")

    engine._schedule_activity_end(event)

    ends = scheduled_end_events(engine)
    assert len(ends) == 1
    assert ends[0].time == engine.simulation_time + timedelta(seconds=3)
    assert engine.task_lifecycle[engine.task_id_for_event(event)].sampled_processing_duration == timedelta(seconds=3)
    diagnostics = engine.get_integration_diagnostics()
    assert diagnostics["processing_time_model_hits"] == 1
    assert diagnostics["processing_time_activity_fallback_hits"] == 0
    assert diagnostics["processing_time_missing_model_count"] == 0
    assert diagnostics["final_processing_duration_min"] == 3.0
    assert diagnostics["final_processing_duration_median"] == 3.0
    assert diagnostics["final_processing_duration_mean"] == 3.0
    assert diagnostics["final_processing_duration_max"] == 3.0


def test_activity_processing_time_fallback_increments_only_fallback_counter(monkeypatch):
    engine = build_engine(monkeypatch)
    engine.processTimeEngine = ConfigurableProcessTimeEngine(
        timedelta(seconds=4),
        fallback_models_basic={"A_processing": object()},
    )
    event = make_event("1", "C1", "A", resource="R1")

    engine._schedule_activity_end(event)

    diagnostics = engine.get_integration_diagnostics()
    assert diagnostics["processing_time_model_hits"] == 0
    assert diagnostics["processing_time_activity_fallback_hits"] == 1
    assert diagnostics["processing_time_missing_model_count"] == 0


def test_missing_processing_time_model_increments_missing_counter(monkeypatch):
    engine = build_engine(monkeypatch)
    engine.processTimeEngine = ConfigurableProcessTimeEngine(timedelta(seconds=2))
    event = make_event("1", "C1", "A", resource="R1")

    engine._schedule_activity_end(event)

    diagnostics = engine.get_integration_diagnostics()
    assert diagnostics["processing_time_model_hits"] == 0
    assert diagnostics["processing_time_activity_fallback_hits"] == 0
    assert diagnostics["processing_time_missing_model_count"] == 1
    assert diagnostics["processing_time_missing_model_activity_A"] == 1


def test_invalid_processing_duration_uses_existing_basic_fallback(monkeypatch):
    engine = build_engine(monkeypatch)
    engine.processTimeEngine = ConfigurableProcessTimeEngine(
        timedelta(seconds=-5),
        fallback_duration=timedelta(seconds=7),
        models_basic={"A_R1_processing": object()},
    )
    event = make_event("1", "C1", "A", resource="R1")

    engine._schedule_activity_end(event)

    ends = scheduled_end_events(engine)
    assert len(ends) == 1
    assert ends[0].time == engine.simulation_time + timedelta(seconds=7)
    assert engine.processTimeEngine.fallback_calls == 1
    assert engine.get_integration_diagnostics()["processing_time_invalid_value_count"] == 1


def test_silent_processing_duration_may_remain_instantaneous(monkeypatch):
    engine = build_engine(monkeypatch)
    engine.processTimeEngine = ConfigurableProcessTimeEngine(timedelta(0))
    event = make_event("1", "C1", "", resource="R1")

    engine._schedule_activity_end(event)

    ends = scheduled_end_events(engine)
    assert len(ends) == 1
    assert ends[0].time == engine.simulation_time
    assert engine.task_lifecycle[engine.task_id_for_event(event)].sampled_processing_duration == timedelta(0)
    diagnostics = engine.get_integration_diagnostics()
    assert diagnostics["minimum_visible_duration_applications"] == 0
    assert diagnostics["visible_activity_processing_starts"] == 0
    assert diagnostics["zero_duration_silent_transitions"] == 1


def test_duplicate_activity_end_event_is_not_scheduled(monkeypatch):
    engine = build_engine(monkeypatch)
    engine.processTimeEngine = ConfigurableProcessTimeEngine(
        timedelta(seconds=3),
        models_basic={"A_R1_processing": object()},
    )
    event = make_event("1", "C1", "A", resource="R1")

    engine._schedule_activity_end(event)
    engine._schedule_activity_end(event)

    assert len(scheduled_end_events(engine)) == 1
    assert engine.processTimeEngine.processing_calls == 1
    diagnostics = engine.get_integration_diagnostics()
    assert diagnostics["processing_time_model_hits"] == 1
    assert diagnostics["visible_activity_processing_starts"] == 1


def test_processing_end_clears_duplicate_schedule_guard(monkeypatch):
    engine = build_engine(monkeypatch)
    engine.processTimeEngine = ConfigurableProcessTimeEngine(timedelta(seconds=3))
    event = make_event("1", "C1", "A", resource="R1")
    engine._schedule_activity_end(event)
    end_event = scheduled_end_events(engine)[0]
    engine.simulation_time = end_event.time

    engine._mark_processing_end(end_event)

    lifecycle = engine.task_lifecycle[engine.task_id_for_event(event)]
    assert lifecycle.processing_end_time == end_event.time
    assert lifecycle.processing_end_event_scheduled is False


def test_branch_prediction_selection_is_unchanged_by_duration_guard(monkeypatch):
    engine = build_engine(monkeypatch)
    event = make_event("1", "C1", "A", resource="R1")

    prediction = engine._predict_branch(event, ["B", "C"])

    assert prediction.selected_activity == "B"
    assert engine.branchingEngine.calls == 1
