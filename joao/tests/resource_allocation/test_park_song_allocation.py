# tests/resource_allocation/test_park_song_allocation.py

from src.resource_allocation.AllocationStrategy import Prediction, Resource, Task
from src.resource_allocation.ParkSongAllocation import ParkSongAllocation


def test_park_song_assigns_current_task_when_available():
    resources = [
        Resource(resource_id="R1", available=True, skills=["A"]),
    ]

    tasks = [
        Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0),
    ]

    strategy = ParkSongAllocation(
        processing_time_estimates={
            ("R1", "A"): 1.0,
        },
    )

    decisions = strategy.allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=1.0,
    )

    assert len(decisions) == 1
    assert decisions[0].decision_type == "assignment"
    assert decisions[0].resource_id == "R1"
    assert decisions[0].task_id == "T1"
    assert decisions[0].activity == "A"
    assert tasks[0].assigned is True


def test_park_song_can_reserve_resource_for_predicted_task():
    resources = [
        Resource(resource_id="R1", available=True, skills=["A", "B"]),
    ]

    tasks = [
        Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0),
    ]

    predictions = [
        Prediction(
            case_id="C2",
            activity="B",
            probability=0.95,
            expected_delay=0.1,
            source="test_prediction",
        ),
    ]

    strategy = ParkSongAllocation(
        processing_time_estimates={
            ("R1", "A"): 10.0,
            ("R1", "B"): 1.0,
        },
        prediction_probability_threshold=0.5,
        uncertainty_weight=1.0,
        idling_weight=0.1,
        waiting_weight=0.0,
        allow_strategic_idling=True,
    )

    decisions = strategy.allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=1.0,
        predictions=predictions,
    )

    assert len(decisions) == 1
    assert decisions[0].decision_type == "reservation"
    assert decisions[0].resource_id == "R1"
    assert decisions[0].task_id is None
    assert decisions[0].activity == "B"
    assert decisions[0].case_id == "C2"
    assert tasks[0].assigned is False


def test_park_song_falls_back_to_current_tasks_without_predictions():
    resources = [
        Resource(resource_id="R1", available=True, skills=["A"]),
    ]

    tasks = [
        Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0),
    ]

    strategy = ParkSongAllocation()

    decisions = strategy.allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=1.0,
    )

    assert len(decisions) == 1
    assert decisions[0].decision_type == "assignment"
    assert decisions[0].task_id == "T1"


def test_park_song_ignores_low_probability_predictions():
    resources = [
        Resource(resource_id="R1", available=True, skills=["A", "B"]),
    ]

    tasks = [
        Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0),
    ]

    predictions = [
        Prediction(
            case_id="C2",
            activity="B",
            probability=0.2,
            expected_delay=0.1,
            source="test_prediction",
        ),
    ]

    strategy = ParkSongAllocation(
        processing_time_estimates={
            ("R1", "A"): 1.0,
            ("R1", "B"): 0.1,
        },
        prediction_probability_threshold=0.5,
    )

    decisions = strategy.allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=1.0,
        predictions=predictions,
    )

    assert len(decisions) == 1
    assert decisions[0].decision_type == "assignment"
    assert decisions[0].task_id == "T1"


def test_park_song_can_disable_strategic_idling():
    resources = [
        Resource(resource_id="R1", available=True, skills=["A", "B"]),
    ]

    tasks = [
        Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0),
    ]

    predictions = [
        Prediction(
            case_id="C2",
            activity="B",
            probability=0.99,
            expected_delay=0.1,
            source="test_prediction",
        ),
    ]

    strategy = ParkSongAllocation(
        processing_time_estimates={
            ("R1", "A"): 10.0,
            ("R1", "B"): 1.0,
        },
        allow_strategic_idling=False,
    )

    decisions = strategy.allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=1.0,
        predictions=predictions,
    )

    assert len(decisions) == 1
    assert decisions[0].decision_type == "assignment"
    assert decisions[0].task_id == "T1"


def test_park_song_respects_resource_skills_for_predictions():
    resources = [
        Resource(resource_id="R1", available=True, skills=["A"]),
    ]

    tasks = [
        Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0),
    ]

    predictions = [
        Prediction(
            case_id="C2",
            activity="B",
            probability=0.99,
            expected_delay=0.1,
            source="test_prediction",
        ),
    ]

    strategy = ParkSongAllocation(
        processing_time_estimates={
            ("R1", "A"): 1.0,
            ("R1", "B"): 0.1,
        },
        prediction_probability_threshold=0.5,
    )

    decisions = strategy.allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=1.0,
        predictions=predictions,
    )

    assert len(decisions) == 1
    assert decisions[0].decision_type == "assignment"
    assert decisions[0].task_id == "T1"


def test_park_song_ignores_unavailable_resources():
    resources = [
        Resource(resource_id="R1", available=False, skills=["A"]),
    ]

    tasks = [
        Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0),
    ]

    strategy = ParkSongAllocation()

    decisions = strategy.allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=1.0,
    )

    assert len(decisions) == 0
    assert tasks[0].assigned is False


def test_park_song_does_not_assign_same_current_task_twice():
    resources = [
        Resource(resource_id="R1", available=True, skills=["A"]),
        Resource(resource_id="R2", available=True, skills=["A"]),
    ]

    tasks = [
        Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0),
    ]

    strategy = ParkSongAllocation()

    decisions = strategy.allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=1.0,
    )

    assignments = [
        decision
        for decision in decisions
        if decision.decision_type == "assignment"
    ]

    idle_decisions = [
        decision
        for decision in decisions
        if decision.decision_type == "idle"
    ]

    assert len(assignments) == 1
    assert len(idle_decisions) == 1
    assert assignments[0].task_id == "T1"
    assert tasks[0].assigned is True


def test_park_song_waiting_time_can_favor_old_waiting_current_task():
    resources = [
        Resource(resource_id="R1", available=True, skills=["A", "B"]),
    ]

    tasks = [
        Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0),
    ]

    predictions = [
        Prediction(
            case_id="C2",
            activity="B",
            probability=0.99,
            expected_delay=0.1,
            source="test_prediction",
        ),
    ]

    strategy = ParkSongAllocation(
        processing_time_estimates={
            ("R1", "A"): 5.0,
            ("R1", "B"): 1.0,
        },
        uncertainty_weight=1.0,
        idling_weight=1.0,
        waiting_weight=1.0,
        prediction_probability_threshold=0.5,
        allow_strategic_idling=True,
    )

    decisions = strategy.allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=10.0,
        predictions=predictions,
    )

    assert len(decisions) == 1
    assert decisions[0].decision_type == "assignment"
    assert decisions[0].task_id == "T1"