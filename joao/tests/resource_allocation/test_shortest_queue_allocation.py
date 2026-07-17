from src.resource_allocation.AllocationStrategy import Resource, Task
from src.resource_allocation.ShortestQueueAllocation import ShortestQueueAllocation


def assignment(decisions):
    return next(decision for decision in decisions if decision.decision_type == "assignment")


def test_shortest_queue_selects_lowest_cumulative_resource_load():
    strategy = ShortestQueueAllocation()

    decisions = strategy.allocate(
        resources=[
            Resource(resource_id="R1", available=True, skills=["A"]),
            Resource(resource_id="R2", available=True, skills=["A"]),
        ],
        waiting_tasks=[Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0)],
        current_time=1.0,
        resource_loads={"R1": 5, "R2": 2},
    )

    assert assignment(decisions).resource_id == "R2"
    assert strategy.last_resource_loads == {"R1": 5.0, "R2": 2.0}
    assert strategy.last_selected_resource_load == 2.0
    assert strategy.get_diagnostics()["unequal_resource_load_comparisons"] == 1


def test_shortest_queue_missing_loads_start_at_zero():
    strategy = ShortestQueueAllocation()

    decisions = strategy.allocate(
        resources=[
            Resource(resource_id="R1", available=True, skills=["A"]),
            Resource(resource_id="R2", available=True, skills=["A"]),
        ],
        waiting_tasks=[Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0)],
        current_time=1.0,
        resource_loads={"R1": 5},
    )

    assert assignment(decisions).resource_id == "R2"
    assert strategy.last_resource_loads == {"R1": 5.0, "R2": 0.0}


def test_shortest_queue_uses_resource_id_for_equal_load_tie_break():
    strategy = ShortestQueueAllocation()

    decisions = strategy.allocate(
        resources=[
            Resource(resource_id="R2", available=True, skills=["A"]),
            Resource(resource_id="R1", available=True, skills=["A"]),
        ],
        waiting_tasks=[Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0)],
        current_time=1.0,
        resource_loads={"R1": 3, "R2": 3},
    )

    assert assignment(decisions).resource_id == "R1"
    diagnostics = strategy.get_diagnostics()
    assert diagnostics["equal_resource_load_ties"] == 1
    assert diagnostics["resource_load_tie_break_decisions"] == 1


def test_shortest_queue_skips_unavailable_lower_load_resource():
    strategy = ShortestQueueAllocation()

    decisions = strategy.allocate(
        resources=[
            Resource(resource_id="R1", available=False, skills=["A"]),
            Resource(resource_id="R2", available=True, skills=["A"]),
        ],
        waiting_tasks=[Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0)],
        current_time=1.0,
        resource_loads={"R1": 0, "R2": 10},
    )

    assert assignment(decisions).resource_id == "R2"


def test_shortest_queue_excludes_busy_resources_passed_as_unavailable():
    strategy = ShortestQueueAllocation()

    decisions = strategy.allocate(
        resources=[
            Resource(resource_id="R1", available=False, skills=["A"]),
            Resource(resource_id="R2", available=True, skills=["A"]),
        ],
        waiting_tasks=[Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0)],
        current_time=1.0,
        resource_loads={"R1": 0, "R2": 7},
    )

    assert assignment(decisions).resource_id == "R2"
    assert strategy.last_resource_loads == {"R2": 7.0}


def test_shortest_queue_respects_permission_mismatch():
    strategy = ShortestQueueAllocation()

    decisions = strategy.allocate(
        resources=[Resource(resource_id="R1", available=True, skills=["B"])],
        waiting_tasks=[Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0)],
        current_time=1.0,
        resource_loads={"R1": 0},
    )

    assert all(decision.decision_type == "idle" for decision in decisions)


def test_shortest_queue_respects_availability_and_permissions_together():
    strategy = ShortestQueueAllocation()

    decisions = strategy.allocate(
        resources=[
            Resource(resource_id="R1", available=False, skills=["A"]),
            Resource(resource_id="R2", available=True, skills=["B"]),
            Resource(resource_id="R3", available=True, skills=["A"]),
        ],
        waiting_tasks=[Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0)],
        current_time=1.0,
        resource_loads={"R1": 0, "R2": 0, "R3": 4},
    )

    assert assignment(decisions).resource_id == "R3"


def test_shortest_queue_changing_group_load_changes_decision():
    resources = [
        Resource(resource_id="R1", available=True, skills=["A"]),
        Resource(resource_id="R2", available=True, skills=["A"]),
    ]

    first = ShortestQueueAllocation().allocate(
        resources=resources,
        waiting_tasks=[Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0)],
        current_time=1.0,
        resource_loads={"R1": 0, "R2": 5},
    )
    second = ShortestQueueAllocation().allocate(
        resources=resources,
        waiting_tasks=[Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0)],
        current_time=1.0,
        resource_loads={"R1": 5, "R2": 0},
    )

    assert assignment(first).resource_id == "R1"
    assert assignment(second).resource_id == "R2"


def test_shortest_queue_has_no_internal_load_leak_across_calls():
    strategy = ShortestQueueAllocation()
    resources = [
        Resource(resource_id="R1", available=True, skills=["A"]),
        Resource(resource_id="R2", available=True, skills=["A"]),
    ]

    first = strategy.allocate(
        resources=resources,
        waiting_tasks=[Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0)],
        current_time=1.0,
        resource_loads={"R1": 0, "R2": 5},
    )
    second = strategy.allocate(
        resources=resources,
        waiting_tasks=[Task(task_id="T2", case_id="C2", activity="A", enabled_time=0.0)],
        current_time=2.0,
        resource_loads={"R1": 0, "R2": 5},
    )

    assert assignment(first).resource_id == "R1"
    assert assignment(second).resource_id == "R1"
    assert not hasattr(strategy, "resource_queue_lengths")
    assert not hasattr(strategy, "task_to_resource")


def test_shortest_queue_reproducible_without_random_state():
    resources = [
        Resource(resource_id="R2", available=True, skills=["A"]),
        Resource(resource_id="R1", available=True, skills=["A"]),
    ]
    task = Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0)

    first = ShortestQueueAllocation().allocate(
        resources,
        [task],
        current_time=1.0,
        resource_loads={"R1": 1, "R2": 1},
    )
    second = ShortestQueueAllocation().allocate(
        resources,
        [Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0)],
        current_time=1.0,
        resource_loads={"R1": 1, "R2": 1},
    )

    assert assignment(first).resource_id == assignment(second).resource_id == "R1"


def test_shortest_queue_empty_candidate_behavior():
    strategy = ShortestQueueAllocation()

    assert strategy.allocate([], [], current_time=1.0, resource_loads={}) == []

    decisions = strategy.allocate(
        [Resource(resource_id="R1", available=True, skills=["A"])],
        [],
        current_time=1.0,
        resource_loads={"R1": 0},
    )
    assert len(decisions) == 1
    assert decisions[0].decision_type == "idle"


def test_shortest_queue_resets_last_snapshot_diagnostics_without_candidates():
    strategy = ShortestQueueAllocation()
    strategy.allocate(
        [Resource(resource_id="R1", available=True, skills=["A"])],
        [Task(task_id="T1", case_id="C1", activity="A", enabled_time=0.0)],
        current_time=1.0,
        resource_loads={"R1": 3},
    )
    assert strategy.last_resource_loads == {"R1": 3.0}

    decisions = strategy.allocate(
        [Resource(resource_id="R1", available=True, skills=["A"])],
        [],
        current_time=2.0,
        resource_loads={"R1": 3},
    )

    assert decisions[0].decision_type == "idle"
    assert strategy.last_resource_loads == {}
    assert strategy.last_min_candidate_resource_load is None
    assert strategy.last_selected_resource_load is None
