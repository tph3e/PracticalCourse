from src.branching.CompositeBranchingEngine import CompositeBranchingEngine


class InvalidEngine:
    def getNextActivities(self, event, possibleActivities):
        return ["INVALID_ACTIVITY"]


class ErrorEngine:
    def getNextActivities(self, event, possibleActivities):
        raise RuntimeError("Simulated engine failure")


class ValidEngine:
    def getNextActivities(self, event, possibleActivities):
        return [possibleActivities[-1]]


def test_composite_uses_first_valid_engine():
    engine = CompositeBranchingEngine(
        engines=[
            InvalidEngine(),
            ErrorEngine(),
            ValidEngine(),
        ],
        seed=1,
        use_default_hierarchy=False,
    )

    result = engine.getNextActivities(
        event=object(),
        possibleActivities=["A", "B"],
    )

    assert result == ["B"]

    stats = engine.get_statistics()
    assert stats["total_decisions"] == 1
    assert stats["engine_success_counts"]["ValidEngine"] == 1
    assert stats["engine_failure_counts"]["ErrorEngine"] == 1
    assert stats["invalid_result_counts"]["InvalidEngine"] == 1
    assert stats["random_fallback_count"] == 0


def test_composite_random_fallback_when_no_engine_is_configured():
    engine = CompositeBranchingEngine(
        seed=1,
        use_default_hierarchy=False,
    )

    result = engine.getNextActivities(
        event=object(),
        possibleActivities=["A", "B"],
    )

    assert result in [["A"], ["B"]]

    stats = engine.get_statistics()
    assert stats["total_decisions"] == 1
    assert stats["random_fallback_count"] == 1
    assert stats["configured_engines"] == []


def test_composite_returns_single_activity_directly():
    engine = CompositeBranchingEngine(
        seed=1,
        use_default_hierarchy=False,
    )

    result = engine.getNextActivities(
        event=object(),
        possibleActivities=["OnlyOne"],
    )

    assert result == ["OnlyOne"]

    stats = engine.get_statistics()
    assert stats["total_decisions"] == 0
    assert stats["random_fallback_count"] == 0


def test_composite_returns_empty_when_no_possible_activity():
    engine = CompositeBranchingEngine(
        seed=1,
        use_default_hierarchy=False,
    )

    result = engine.getNextActivities(
        event=object(),
        possibleActivities=[],
    )

    assert result == []

    stats = engine.get_statistics()
    assert stats["total_decisions"] == 0
    assert stats["random_fallback_count"] == 0


def test_composite_builds_default_hierarchy_without_crashing():
    engine = CompositeBranchingEngine(
        log=None,
        seed=1,
    )

    stats = engine.get_statistics()

    assert "configured_engines" in stats
    assert "default_build_errors" in stats
    assert isinstance(stats["configured_engines"], list)
    assert isinstance(stats["default_build_errors"], dict)


def test_composite_default_hierarchy_still_has_random_fallback():
    engine = CompositeBranchingEngine(
        log=None,
        seed=1,
    )

    result = engine.getNextActivities(
        event=object(),
        possibleActivities=["A", "B"],
    )

    assert result in [["A"], ["B"]]

    stats = engine.get_statistics()
    assert stats["total_decisions"] == 1

    total_successes = sum(stats["engine_success_counts"].values())
    total_failures = sum(stats["engine_failure_counts"].values())
    total_invalid = sum(stats["invalid_result_counts"].values())
    random_fallbacks = stats["random_fallback_count"]

    assert (
        total_successes
        + total_failures
        + total_invalid
        + random_fallbacks
        >= 1
    )


def test_composite_manual_engines_override_default_hierarchy():
    engine = CompositeBranchingEngine(
        engines=[ValidEngine()],
        log=None,
        seed=1,
    )

    stats = engine.get_statistics()

    assert stats["configured_engines"] == ["ValidEngine"]
    assert stats["default_build_errors"] == {}

    result = engine.getNextActivities(
        event=object(),
        possibleActivities=["A", "B"],
    )

    assert result == ["B"]