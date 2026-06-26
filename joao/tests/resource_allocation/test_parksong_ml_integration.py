from src.resource_allocation.AllocationStrategy import Resource, Prediction
from src.resource_allocation.ParkSongAllocation import ParkSongAllocation


def test_parksong_uses_ml_prediction_as_reservation():
    resources = [
        Resource(
            resource_id="R1",
            available=True,
            skills=["A_APPROVED", "A_REJECTED"],
        )
    ]

    waiting_tasks = []

    predictions = [
        Prediction(
            case_id="C1",
            activity="A_APPROVED",
            probability=0.9,
            expected_delay=1.0,
            source="PredictiveBranchingEngine",
            confidence=0.9,
        )
    ]

    allocator = ParkSongAllocation(
        prediction_probability_threshold=0.5,
        allow_strategic_idling=True,
    )

    decisions = allocator.allocate(
        resources=resources,
        waiting_tasks=waiting_tasks,
        current_time=0.0,
        predictions=predictions,
    )

    assert len(decisions) == 1
    assert decisions[0].resource_id == "R1"
    assert decisions[0].decision_type == "reservation"
    assert decisions[0].task_id is None
    assert decisions[0].activity == "A_APPROVED"
    assert decisions[0].case_id == "C1"


def test_parksong_ignores_low_probability_prediction():
    resources = [
        Resource(
            resource_id="R1",
            available=True,
            skills=["A_APPROVED"],
        )
    ]

    waiting_tasks = []

    predictions = [
        Prediction(
            case_id="C1",
            activity="A_APPROVED",
            probability=0.2,
            expected_delay=1.0,
            source="PredictiveBranchingEngine",
            confidence=0.2,
        )
    ]

    allocator = ParkSongAllocation(
        prediction_probability_threshold=0.5,
        allow_strategic_idling=True,
    )

    decisions = allocator.allocate(
        resources=resources,
        waiting_tasks=waiting_tasks,
        current_time=0.0,
        predictions=predictions,
    )

    assert len(decisions) == 1
    assert decisions[0].decision_type == "idle"
    assert decisions[0].activity is None
    assert decisions[0].case_id is None


def test_parksong_prefers_current_task_when_prediction_is_too_uncertain():
    from src.resource_allocation.AllocationStrategy import Task

    resources = [
        Resource(
            resource_id="R1",
            available=True,
            skills=["A_CURRENT", "A_FUTURE"],
        )
    ]

    waiting_tasks = [
        Task(
            task_id="T1",
            case_id="C1",
            activity="A_CURRENT",
            enabled_time=0.0,
            priority=0.0,
        )
    ]

    predictions = [
        Prediction(
            case_id="C2",
            activity="A_FUTURE",
            probability=0.55,
            expected_delay=5.0,
            source="PredictiveBranchingEngine",
            confidence=0.55,
        )
    ]

    allocator = ParkSongAllocation(
        prediction_probability_threshold=0.5,
        uncertainty_weight=5.0,
        idling_weight=1.0,
        allow_strategic_idling=True,
    )

    decisions = allocator.allocate(
        resources=resources,
        waiting_tasks=waiting_tasks,
        current_time=0.0,
        predictions=predictions,
    )

    assert len(decisions) == 1
    assert decisions[0].decision_type == "assignment"
    assert decisions[0].task_id == "T1"
    assert decisions[0].activity == "A_CURRENT"
