from datetime import datetime, timedelta

import pandas as pd

from resources import ResourceEngine
from joao.src.resource_allocation.AllocationStrategy import Prediction
from joao.src.resource_allocation.ParkSongAllocation import ParkSongAllocation
from joao.src.resource_allocation.RandomResourceAllocation import RandomResourceAllocation
from joao.src.resource_allocation.ShortestQueueAllocation import ShortestQueueAllocation


class Case:
    def __init__(self, case_id):
        self.caseId = case_id


class Event:
    def __init__(self, event_id, activity, case_id="C1"):
        self.eventId = event_id
        self.activity = activity
        self.eventCase = Case(case_id)
        self.time = datetime(2026, 1, 5, 9, 0)
        self.resource = ""

    def getAttribs(self):
        return {
            "EventID": self.eventId,
            "concept:name": self.activity,
            "case:concept:name": self.eventCase.caseId,
            "time:timestamp": self.time,
        }


def build_engine():
    log = pd.DataFrame(
        [
            {
                "concept:name": "A",
                "org:resource": "R1",
                "time:timestamp": datetime(2026, 1, 5, 9, 0),
            },
            {
                "concept:name": "B",
                "org:resource": "R1",
                "time:timestamp": datetime(2026, 1, 5, 9, 0),
            },
        ]
    )
    engine = ResourceEngine(log, seed=1)
    engine.availability.calendars = None
    engine.availability._all_resources = {"R1"}
    engine.permissions._activity_to_resources = {
        "A": {"R1"},
        "B": {"R1"},
    }
    return engine


def test_allocate_waiting_tasks_supports_random_allocation():
    engine = build_engine()
    engine.global_allocation_strategy = RandomResourceAllocation(seed=1)
    event = Event(1, "A")

    decisions = engine.allocate_waiting_tasks(
        waiting_events=[event],
        current_time=datetime(2026, 1, 5, 9, 1),
    )

    assert decisions[0].decision_type == "assignment"
    assert event.resource == "R1"
    assert "R1" in engine.busy


def test_allocate_waiting_tasks_supports_shortest_queue_allocation():
    engine = build_engine()
    engine.global_allocation_strategy = ShortestQueueAllocation()
    events = [
        Event(1, "A", "C1"),
        Event(2, "A", "C2"),
        Event(3, "B", "C3"),
    ]

    decisions = engine.allocate_waiting_tasks(
        waiting_events=events,
        current_time=datetime(2026, 1, 5, 9, 1),
    )

    assignment = next(
        decision for decision in decisions
        if decision.decision_type == "assignment"
    )
    assert assignment.activity == "A"
    assert events[0].resource == "R1"


def test_allocate_waiting_tasks_passes_cumulative_resource_loads():
    engine = build_engine()
    engine.availability._all_resources = {"R1", "R2"}
    engine.permissions._activity_to_resources = {"A": {"R1", "R2"}, "B": {"R1", "R2"}}
    strategy = ShortestQueueAllocation()
    engine.global_allocation_strategy = strategy
    engine.load = {"R1": 5, "R2": 1}
    event = Event(1, "A")

    decisions = engine.allocate_waiting_tasks(
        waiting_events=[event],
        current_time=datetime(2026, 1, 5, 9, 1),
    )

    assignment = next(decision for decision in decisions if decision.decision_type == "assignment")
    assert assignment.resource_id == "R2"
    assert event.resource == "R2"
    assert strategy.last_resource_loads == {"R1": 5.0, "R2": 1.0}
    assert engine.load["R2"] == 2


def test_allocate_waiting_tasks_supports_parksong_without_predictions():
    engine = build_engine()
    engine.global_allocation_strategy = ParkSongAllocation(
        allow_strategic_idling=False,
    )
    event = Event(1, "A")

    decisions = engine.allocate_waiting_tasks(
        waiting_events=[event],
        current_time=datetime(2026, 1, 5, 9, 1),
    )

    assert decisions[0].decision_type == "assignment"
    assert event.resource == "R1"


def test_allocate_waiting_tasks_accepts_mock_predictions_for_parksong_ml_shape():
    engine = build_engine()
    engine.global_allocation_strategy = ParkSongAllocation(
        processing_time_estimates={
            ("R1", "A"): 10.0,
            ("R1", "B"): 1.0,
        },
        prediction_probability_threshold=0.5,
        uncertainty_weight=0.1,
        idling_weight=0.1,
        waiting_weight=0.0,
        allow_strategic_idling=True,
    )

    decisions = engine.allocate_waiting_tasks(
        waiting_events=[Event(1, "A")],
        current_time=datetime(2026, 1, 5, 9, 1),
        predictions=[
            Prediction(
                case_id="C_FUTURE",
                activity="B",
                probability=0.95,
                expected_delay=0.1,
                source="mock",
            )
        ],
    )

    assert decisions[0].decision_type == "reservation"
    assert decisions[0].activity == "B"
