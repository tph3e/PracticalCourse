# tests/resource_allocation/test_park_song_allocation.py

from src.resource_allocation.AllocationStrategy import Prediction, Resource, Task
from src.resource_allocation.ParkSongAllocation import CandidateTask, ParkSongAllocation


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


def test_park_song_processing_time_weight_scales_processing_cost():
    strategy = ParkSongAllocation(
        processing_time_estimates={("R1", "A"): 100.0},
        processing_time_weight=0.25,
        waiting_weight=0.0,
        priority_weight=0.0,
    )
    candidate = CandidateTask(
        candidate_id="current::T1",
        case_id="C1",
        activity="A",
        candidate_type="current",
        task_id="T1",
    )

    cost = strategy._compute_cost(
        resource=Resource(resource_id="R1", available=True, skills=["A"]),
        candidate=candidate,
        current_time=0.0,
    )

    assert cost == 25.0


def test_park_song_cost_time_scale_normalizes_time_terms():
    strategy = ParkSongAllocation(
        processing_time_estimates={("R1", "A"): 3600.0},
        cost_time_scale=3600.0,
        processing_time_weight=2.0,
        waiting_weight=1.0,
        priority_weight=0.0,
    )
    candidate = CandidateTask(
        candidate_id="current::T1",
        case_id="C1",
        activity="A",
        candidate_type="current",
        task_id="T1",
        enabled_time=0.0,
    )

    cost = strategy._compute_cost(
        resource=Resource(resource_id="R1", available=True, skills=["A"]),
        candidate=candidate,
        current_time=1800.0,
    )

    assert cost == 1.5


def test_park_song_no_show_penalty_adds_expected_reservation_risk():
    strategy = ParkSongAllocation(
        processing_time_estimates={("R1", "B"): 0.0},
        cost_time_scale=3600.0,
        processing_time_weight=0.0,
        uncertainty_weight=0.0,
        idling_weight=0.0,
        no_show_penalty_weight=2.0,
    )
    candidate = CandidateTask(
        candidate_id="predicted::C2::B::0",
        case_id="C2",
        activity="B",
        candidate_type="predicted",
        probability=0.75,
        expected_delay=3600.0,
    )

    cost = strategy._compute_cost(
        resource=Resource(resource_id="R1", available=True, skills=["B"]),
        candidate=candidate,
        current_time=0.0,
    )

    assert cost == 0.5


def test_park_song_reservation_margin_filters_marginal_prediction():
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
            expected_delay=0.0,
            source="test_prediction",
        ),
    ]

    strategy = ParkSongAllocation(
        processing_time_estimates={
            ("R1", "A"): 1.0,
            ("R1", "B"): 0.9,
        },
        uncertainty_weight=0.0,
        idling_weight=0.0,
        waiting_weight=0.0,
        reservation_margin=0.2,
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
    assert strategy.get_diagnostics()["reservation_margin_filtered"] == 1


def test_park_song_reservation_margin_allows_clear_prediction_advantage():
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
            expected_delay=0.0,
            source="test_prediction",
        ),
    ]

    strategy = ParkSongAllocation(
        processing_time_estimates={
            ("R1", "A"): 2.0,
            ("R1", "B"): 1.0,
        },
        uncertainty_weight=0.0,
        idling_weight=0.0,
        waiting_weight=0.0,
        reservation_margin=0.2,
    )

    decisions = strategy.allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=1.0,
        predictions=predictions,
    )

    assert len(decisions) == 1
    assert decisions[0].decision_type == "reservation"
    assert decisions[0].activity == "B"


def test_park_song_solves_epoch_as_global_assignment_not_local_greedy():
    resources = [
        Resource(resource_id="R1", available=True, skills=["A", "B"]),
        Resource(resource_id="R2", available=True, skills=["A"]),
    ]

    tasks = [
        Task(task_id="T_A", case_id="C1", activity="A", enabled_time=0.0),
        Task(task_id="T_B", case_id="C2", activity="B", enabled_time=0.0),
    ]

    strategy = ParkSongAllocation(
        processing_time_estimates={
            ("R1", "A"): 1.0,
            ("R1", "B"): 2.0,
            ("R2", "A"): 2.0,
        },
    )

    decisions = strategy.allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=1.0,
    )

    assignments = {
        decision.resource_id: decision.task_id
        for decision in decisions
        if decision.decision_type == "assignment"
    }

    assert assignments == {"R1": "T_B", "R2": "T_A"}
    assert all(task.assigned for task in tasks)


def test_park_song_temporal_lookahead_can_protect_near_future_prediction():
    resources = [
        Resource(resource_id="R1", available=True, skills=["A", "B"]),
    ]

    tasks = [
        Task(task_id="T_A", case_id="C1", activity="A", enabled_time=0.0),
    ]

    predictions = [
        Prediction(
            case_id="C2",
            activity="B",
            probability=0.99,
            expected_delay=1.0,
            source="test_prediction",
        ),
    ]

    strategy = ParkSongAllocation(
        processing_time_estimates={
            ("R1", "A"): 2.0,
            ("R1", "B"): 1.0,
        },
        prediction_probability_threshold=0.5,
        uncertainty_weight=0.0,
        idling_weight=1.0,
        waiting_weight=0.0,
        future_delay_weight=5.0,
        planning_horizon=10.0,
    )

    decisions = strategy.allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=0.0,
        predictions=predictions,
    )

    assert len(decisions) == 1
    assert decisions[0].decision_type == "reservation"
    assert decisions[0].activity == "B"
    assert tasks[0].assigned is False
    assert strategy.get_diagnostics()["temporal_lookahead_penalties"] == 1


def test_park_song_temporal_lookahead_ignores_predictions_outside_horizon():
    resources = [
        Resource(resource_id="R1", available=True, skills=["A", "B"]),
    ]

    tasks = [
        Task(task_id="T_A", case_id="C1", activity="A", enabled_time=0.0),
    ]

    predictions = [
        Prediction(
            case_id="C2",
            activity="B",
            probability=0.99,
            expected_delay=20.0,
            source="test_prediction",
        ),
    ]

    strategy = ParkSongAllocation(
        processing_time_estimates={
            ("R1", "A"): 2.0,
            ("R1", "B"): 1.0,
        },
        prediction_probability_threshold=0.5,
        uncertainty_weight=0.0,
        idling_weight=1.0,
        waiting_weight=0.0,
        future_delay_weight=5.0,
        planning_horizon=10.0,
    )

    decisions = strategy.allocate(
        resources=resources,
        waiting_tasks=tasks,
        current_time=0.0,
        predictions=predictions,
    )

    assert len(decisions) == 1
    assert decisions[0].decision_type == "assignment"
    assert decisions[0].task_id == "T_A"
    assert tasks[0].assigned is True
    assert "temporal_lookahead_penalties" not in strategy.get_diagnostics()
