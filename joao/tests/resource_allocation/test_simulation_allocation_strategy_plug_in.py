from datetime import datetime, timedelta

import pandas as pd
import pytest

import SimulationEngineCore as simulation_core
from joao.src.resource_allocation.GlobalAllocationAdapter import GlobalAllocationAdapter
from joao.src.resource_allocation.ParkSongAllocation import ParkSongAllocation
from joao.src.resource_allocation.ShortestQueueAllocation import ShortestQueueAllocation


class FakeArrivalEngine:
    def __init__(self, log, seed=1):
        pass

    def nextArrivalTime(self, current_time):
        return timedelta(seconds=1)


class FakeProcessTimeEngine:
    def __init__(self, log, seed=1):
        pass

    def getProcessingTime(self, event, activity=None):
        return timedelta(seconds=10)

    def getWaitingTime(self, event, activity=None):
        return timedelta(seconds=1)


def build_smoke_log():
    rows = []
    for index in range(6):
        rows.append(
            {
                "case:concept:name": f"C{index}",
                "concept:name": "A_Create Application",
                "time:timestamp": pd.Timestamp(f"2026-01-0{index + 1} 09:00:00"),
                "case:ApplicationType": "New credit",
                "case:LoanGoal": "Car",
                "case:RequestedAmount": 10000 + index * 1000,
                "org:resource": "R_SMOKE_1",
                "lifecycle:transition": "complete",
            }
        )
    return pd.DataFrame(rows)


def configure_resource_fixture(engine):
    activities = {
        transition.label
        for transition in engine.bpmnEngine.net.transitions
        if transition.label is not None
    }
    resources = {"R_SMOKE_1"}

    # Keep ResourceEngine real, but make this smoke test independent of
    # repository-level availability/permission artifacts.
    engine.resourceEngine.availability.calendars = None
    engine.resourceEngine.availability._all_resources = resources
    engine.resourceEngine.permissions._activity_to_resources = {
        activity: set(resources)
        for activity in activities
    }


@pytest.mark.parametrize("mode", ["random", "round_robin"])
def test_main_simulation_engine_can_use_global_allocation_adapter(
    monkeypatch,
    mode,
):
    monkeypatch.setattr(
        simulation_core.pm4py,
        "read_xes",
        lambda *args, **kwargs: build_smoke_log(),
    )
    monkeypatch.setattr(simulation_core, "ArrivalEngine", FakeArrivalEngine)
    monkeypatch.setattr(simulation_core, "ProcessTimeEngine", FakeProcessTimeEngine)

    engine = simulation_core.Engine(dataPath="data/logData.xes", seed=1)
    configure_resource_fixture(engine)

    adapter = GlobalAllocationAdapter(mode=mode, seed=1)
    engine.resourceEngine.allocation = adapter

    engine.run(
        datetime(2026, 1, 5, 9, 0),
        datetime(2026, 1, 5, 9, 0, 30),
        format_type=[],
    )

    log = engine.logger.get_log()

    assert not log.empty
    assert adapter.pick_count > 0
    assert log["org:resource"].astype(bool).any()
    assert engine.resourceEngine.allocation is adapter


class CountingShortestQueueAllocation(ShortestQueueAllocation):
    def __init__(self):
        self.call_count = 0

    def allocate(self, *args, **kwargs):
        self.call_count += 1
        return super().allocate(*args, **kwargs)


class CountingParkSongAllocation(ParkSongAllocation):
    def __init__(self):
        super().__init__(allow_strategic_idling=False)
        self.call_count = 0

    def allocate(self, *args, **kwargs):
        self.call_count += 1
        return super().allocate(*args, **kwargs)


@pytest.mark.parametrize(
    "strategy",
    [
        CountingShortestQueueAllocation(),
        CountingParkSongAllocation(),
    ],
)
def test_main_simulation_engine_can_allocate_waiting_events_with_part_ii_strategy(
    monkeypatch,
    strategy,
):
    monkeypatch.setattr(
        simulation_core.pm4py,
        "read_xes",
        lambda *args, **kwargs: build_smoke_log(),
    )
    monkeypatch.setattr(simulation_core, "ArrivalEngine", FakeArrivalEngine)
    monkeypatch.setattr(simulation_core, "ProcessTimeEngine", FakeProcessTimeEngine)

    engine = simulation_core.Engine(dataPath="data/logData.xes", seed=1)
    configure_resource_fixture(engine)
    engine.resourceEngine.global_allocation_strategy = strategy

    engine.run(
        datetime(2026, 1, 5, 9, 0),
        datetime(2026, 1, 5, 9, 0, 30),
        format_type=[],
    )

    log = engine.logger.get_log()

    assert not log.empty
    assert engine.resourceEngine.global_allocation_strategy is strategy
    assert log["org:resource"].astype(bool).any()
