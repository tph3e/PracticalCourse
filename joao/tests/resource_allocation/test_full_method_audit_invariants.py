import copy

from src.resource_allocation.AllocationStrategy import Prediction, Resource, Task
from src.resource_allocation.ParkSongAllocation import ParkSongAllocation
from src.resource_allocation.RandomResourceAllocation import RandomResourceAllocation
from src.resource_allocation.RoundRobinResourceAllocation import RoundRobinResourceAllocation
from src.resource_allocation.ShortestQueueAllocation import ShortestQueueAllocation


def clone_snapshot(resources, tasks):
    return copy.deepcopy(resources), copy.deepcopy(tasks)


def assignments(decisions):
    return [decision for decision in decisions if decision.decision_type == "assignment"]


def test_all_strategies_respect_permissions_and_availability_on_same_snapshot():
    strategies = [
        RandomResourceAllocation(seed=4),
        RoundRobinResourceAllocation(),
        ShortestQueueAllocation(),
        ParkSongAllocation(allow_strategic_idling=False),
    ]
    resources = [
        Resource("R1", available=False, skills=["A"]),
        Resource("R2", available=True, skills=["B"]),
        Resource("R3", available=True, skills=["A"]),
    ]
    tasks = [Task("T1", "C1", "A", 0.0)]

    for strategy in strategies:
        snapshot_resources, snapshot_tasks = clone_snapshot(resources, tasks)
        decisions = strategy.allocate(
            snapshot_resources,
            snapshot_tasks,
            current_time=1.0,
            resource_loads={"R1": 0, "R2": 0, "R3": 0},
        )

        assigned = assignments(decisions)
        assert len(assigned) == 1
        assert assigned[0].resource_id == "R3"
        assert assigned[0].task_id == "T1"


def test_random_seed_reproducibility_and_input_resource_stability():
    resources = [Resource("R1", skills=["A"]), Resource("R2", skills=["A"])]
    tasks = [Task(f"T{i}", f"C{i}", "A", 0.0) for i in range(4)]
    original_resources = copy.deepcopy(resources)

    first_resources, first_tasks = clone_snapshot(resources, tasks)
    second_resources, second_tasks = clone_snapshot(resources, tasks)

    first = RandomResourceAllocation(seed=17).allocate(first_resources, first_tasks, 0.0)
    second = RandomResourceAllocation(seed=17).allocate(second_resources, second_tasks, 0.0)

    assert [(d.resource_id, d.task_id, d.decision_type) for d in first] == [
        (d.resource_id, d.task_id, d.decision_type) for d in second
    ]
    assert resources == original_resources


def test_shortest_queue_never_selects_strictly_more_loaded_candidate():
    strategy = ShortestQueueAllocation()
    resources = [
        Resource("R_high", skills=["A"]),
        Resource("R_low", skills=["A"]),
        Resource("R_mid", skills=["A"]),
    ]
    tasks = [Task("T1", "C1", "A", 0.0)]

    decisions = strategy.allocate(
        resources,
        tasks,
        current_time=0.0,
        resource_loads={"R_high": 10, "R_mid": 3, "R_low": 1},
    )

    assert assignments(decisions)[0].resource_id == "R_low"
    assert strategy.last_selected_resource_load == strategy.last_min_candidate_resource_load


def test_round_robin_separate_instances_have_independent_state():
    resources = [Resource("R1", skills=["A"]), Resource("R2", skills=["A"])]
    first = RoundRobinResourceAllocation()
    second = RoundRobinResourceAllocation()

    first.allocate(resources, [Task("T1", "C1", "A", 0.0)], 0.0)

    second_decisions = second.allocate(
        resources,
        [Task("T2", "C2", "A", 0.0)],
        0.0,
    )

    assert assignments(second_decisions)[0].resource_id == "R1"


def test_parksong_prediction_same_as_explicit_parksong_candidate():
    resources = [Resource("R1", skills=["A", "B"])]
    tasks = [Task("T1", "C1", "A", 0.0)]
    predictions = [Prediction("C2", "B", probability=0.99, expected_delay=0.1)]
    strategy = ParkSongAllocation(
        processing_time_estimates={("R1", "A"): 10.0, ("R1", "B"): 1.0},
        uncertainty_weight=1.0,
        idling_weight=0.1,
        waiting_weight=0.0,
    )

    decisions = strategy.allocate(resources, tasks, current_time=1.0, predictions=predictions)

    assert decisions[0].decision_type == "reservation"
    assert decisions[0].activity == "B"
    assert tasks[0].assigned is False
