from datetime import datetime

import pandas as pd

from resources import ResourceEngine
from src.resource_allocation.GlobalAllocationAdapter import GlobalAllocationAdapter


class Event:
    def __init__(self, activity: str):
        self.activity = activity
        self.time = datetime(2026, 1, 5, 9, 0)
        self.resource = None


def test_global_adapter_random_pick_returns_candidate():
    adapter = GlobalAllocationAdapter(mode="random", seed=1)

    selected = adapter.pick({"R2", "R1"})

    assert selected in {"R1", "R2"}


def test_global_adapter_round_robin_pick_is_deterministic():
    adapter = GlobalAllocationAdapter(mode="round_robin")

    assert adapter.pick({"R2", "R1"}) == "R1"
    assert adapter.pick({"R2", "R1"}) == "R2"
    assert adapter.pick({"R2", "R1"}) == "R1"


def test_global_adapter_returns_none_without_candidates():
    adapter = GlobalAllocationAdapter(mode="random", seed=1)

    assert adapter.pick(set()) is None


def test_global_adapter_accepts_resource_engine_context_argument():
    adapter = GlobalAllocationAdapter(mode="round_robin")

    selected = adapter.pick({"R1"}, context=object())

    assert selected == "R1"


def test_global_adapter_can_be_plugged_into_resource_engine():
    log = pd.DataFrame(
        [
            {
                "concept:name": "A",
                "org:resource": "R1",
                "time:timestamp": datetime(2026, 1, 5, 9, 0),
            },
            {
                "concept:name": "A",
                "org:resource": "R2",
                "time:timestamp": datetime(2026, 1, 5, 9, 0),
            },
        ]
    )

    engine = ResourceEngine(log, seed=1)
    engine.allocation = GlobalAllocationAdapter(mode="round_robin")

    # Keep this smoke test independent of repository-level advanced artifacts.
    engine.availability.calendars = None
    engine.availability._all_resources = {"R1", "R2"}
    engine.permissions._activity_to_resources = {"A": {"R1", "R2"}}

    event = Event("A")

    assert engine.allocateResource(event) is True
    assert event.resource == "R1"
    assert "R1" in engine.busy
