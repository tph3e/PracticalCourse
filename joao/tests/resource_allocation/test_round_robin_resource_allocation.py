from src.resource_allocation.AllocationStrategy import Resource, Task
from src.resource_allocation.RoundRobinResourceAllocation import RoundRobinResourceAllocation


def task(task_id="T1", activity="A", enabled_time=0.0):
    return Task(
        task_id=task_id,
        case_id=f"C_{task_id}",
        activity=activity,
        enabled_time=enabled_time,
    )


def assignment_resource(decisions):
    return [
        decision.resource_id
        for decision in decisions
        if decision.decision_type == "assignment"
    ]


def test_round_robin_rotates_deterministically_and_wraps():
    strategy = RoundRobinResourceAllocation()
    resources = [
        Resource("R2", skills=["A"]),
        Resource("R1", skills=["A"]),
        Resource("R3", skills=["A"]),
    ]

    assert assignment_resource(strategy.allocate(resources, [task("T1")], 0.0)) == ["R1"]
    assert assignment_resource(strategy.allocate(resources, [task("T2")], 1.0)) == ["R2"]
    assert assignment_resource(strategy.allocate(resources, [task("T3")], 2.0)) == ["R3"]
    assert assignment_resource(strategy.allocate(resources, [task("T4")], 3.0)) == ["R1"]


def test_round_robin_state_persists_across_calls():
    strategy = RoundRobinResourceAllocation()
    resources = [Resource("R1", skills=["A"]), Resource("R2", skills=["A"])]

    strategy.allocate(resources, [task("T1")], 0.0)

    assert assignment_resource(strategy.allocate(resources, [task("T2")], 1.0)) == ["R2"]


def test_round_robin_continues_after_multi_assignment_epoch():
    strategy = RoundRobinResourceAllocation()
    resources = [
        Resource("R1", skills=["A"]),
        Resource("R2", skills=["A"]),
        Resource("R3", skills=["A"]),
    ]

    assert assignment_resource(strategy.allocate(resources, [task("T1")], 0.0)) == ["R1"]
    assert assignment_resource(
        strategy.allocate(resources, [task("T2"), task("T3"), task("T4")], 1.0)
    ) == ["R2", "R3", "R1"]
    assert assignment_resource(strategy.allocate(resources, [task("T5")], 2.0)) == ["R2"]


def test_round_robin_multi_assignment_with_idle_resource_continues_correctly():
    strategy = RoundRobinResourceAllocation()
    resources = [
        Resource("R1", skills=["A"]),
        Resource("R2", skills=["A"]),
        Resource("R3", skills=["B"]),
    ]

    assert assignment_resource(strategy.allocate(resources, [task("T1")], 0.0)) == ["R1"]
    decisions = strategy.allocate(
        resources,
        [task("T2", activity="A"), task("T3", activity="A")],
        1.0,
    )

    assert assignment_resource(decisions) == ["R2", "R1"]
    assert any(
        decision.resource_id == "R3" and decision.decision_type == "idle"
        for decision in decisions
    )
    assert assignment_resource(strategy.allocate(resources, [task("T4")], 2.0)) == ["R2"]


def test_round_robin_candidate_change_after_multi_assignment():
    strategy = RoundRobinResourceAllocation()
    all_resources = [
        Resource("R1", skills=["A"]),
        Resource("R2", skills=["A"]),
        Resource("R3", skills=["A"]),
    ]

    assert assignment_resource(strategy.allocate(all_resources, [task("T1")], 0.0)) == ["R1"]
    assert assignment_resource(
        strategy.allocate(all_resources, [task("T2"), task("T3"), task("T4")], 1.0)
    ) == ["R2", "R3", "R1"]

    changed_resources = [Resource("R1", skills=["A"]), Resource("R3", skills=["A"])]

    assert assignment_resource(strategy.allocate(changed_resources, [task("T5")], 2.0)) == ["R3"]


def test_round_robin_separate_instances_after_multi_assignment_are_independent():
    resources = [
        Resource("R1", skills=["A"]),
        Resource("R2", skills=["A"]),
        Resource("R3", skills=["A"]),
    ]
    first = RoundRobinResourceAllocation()
    second = RoundRobinResourceAllocation()

    first.allocate(resources, [task("T1")], 0.0)
    first.allocate(resources, [task("T2"), task("T3"), task("T4")], 1.0)

    assert assignment_resource(second.allocate(resources, [task("T5")], 2.0)) == ["R1"]


def test_round_robin_does_not_advance_pointer_when_no_assignment_occurs():
    strategy = RoundRobinResourceAllocation()
    resources = [
        Resource("R1", skills=["A"]),
        Resource("R2", skills=["A"]),
    ]

    assert assignment_resource(strategy.allocate(resources, [task("T1")], 0.0)) == ["R1"]
    decisions = strategy.allocate(resources, [task("T2", activity="B")], 1.0)

    assert assignment_resource(decisions) == []
    assert assignment_resource(strategy.allocate(resources, [task("T3")], 2.0)) == ["R2"]


def test_round_robin_multi_assignment_is_deterministic():
    resources = [
        Resource("R1", skills=["A"]),
        Resource("R2", skills=["A"]),
        Resource("R3", skills=["A"]),
    ]
    first = RoundRobinResourceAllocation()
    second = RoundRobinResourceAllocation()

    first_sequence = [
        assignment_resource(first.allocate(resources, [task("T1")], 0.0)),
        assignment_resource(first.allocate(resources, [task("T2"), task("T3"), task("T4")], 1.0)),
        assignment_resource(first.allocate(resources, [task("T5")], 2.0)),
    ]
    second_sequence = [
        assignment_resource(second.allocate(resources, [task("T1")], 0.0)),
        assignment_resource(second.allocate(resources, [task("T2"), task("T3"), task("T4")], 1.0)),
        assignment_resource(second.allocate(resources, [task("T5")], 2.0)),
    ]

    assert first_sequence == second_sequence == [["R1"], ["R2", "R3", "R1"], ["R2"]]


def test_round_robin_skips_unavailable_resources():
    strategy = RoundRobinResourceAllocation()
    resources = [
        Resource("R1", available=False, skills=["A"]),
        Resource("R2", available=True, skills=["A"]),
    ]

    assert assignment_resource(strategy.allocate(resources, [task("T1")], 0.0)) == ["R2"]


def test_round_robin_skips_resources_without_permission():
    strategy = RoundRobinResourceAllocation()
    resources = [
        Resource("R1", skills=["B"]),
        Resource("R2", skills=["A"]),
    ]

    decisions = strategy.allocate(resources, [task("T1", activity="A")], 0.0)

    assert assignment_resource(decisions) == ["R2"]
    assert any(decision.resource_id == "R1" and decision.decision_type == "idle" for decision in decisions)


def test_round_robin_preserves_rotation_when_candidate_set_changes():
    strategy = RoundRobinResourceAllocation()
    all_resources = [
        Resource("R1", skills=["A"]),
        Resource("R2", skills=["A"]),
        Resource("R3", skills=["A"]),
    ]
    strategy.allocate(all_resources, [task("T1")], 0.0)

    changed_resources = [Resource("R1", skills=["A"]), Resource("R3", skills=["A"])]

    assert assignment_resource(strategy.allocate(changed_resources, [task("T2")], 1.0)) == ["R3"]


def test_round_robin_returns_no_assignment_without_candidates():
    strategy = RoundRobinResourceAllocation()

    assert strategy.allocate([], [task("T1")], 0.0) == []


def test_round_robin_is_reproducible_without_seed():
    resources = [Resource("R1", skills=["A"]), Resource("R2", skills=["A"])]
    first = RoundRobinResourceAllocation()
    second = RoundRobinResourceAllocation()

    first_sequence = [
        assignment_resource(first.allocate(resources, [task(f"T{i}")], float(i)))[0]
        for i in range(4)
    ]
    second_sequence = [
        assignment_resource(second.allocate(resources, [task(f"T{i}")], float(i)))[0]
        for i in range(4)
    ]

    assert first_sequence == second_sequence == ["R1", "R2", "R1", "R2"]


def test_round_robin_avoids_starvation_under_stable_candidates():
    strategy = RoundRobinResourceAllocation()
    resources = [Resource("R1", skills=["A"]), Resource("R2", skills=["A"])]
    assigned = []

    for index in range(10):
        decisions = strategy.allocate(resources, [task(f"T{index}")], float(index))
        assigned.extend(assignment_resource(decisions))

    assert assigned.count("R1") == 5
    assert assigned.count("R2") == 5
