from datetime import datetime, timedelta

import pandas as pd

from joao.scripts.resource_allocation import run_integrated_allocation_comparison as script


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
    return pd.DataFrame(
        [
            {
                "case:concept:name": f"C{index}",
                "concept:name": "A_Create Application",
                "time:timestamp": pd.Timestamp(f"2026-01-0{index + 1} 09:00:00"),
                "case:ApplicationType": "New credit",
                "case:LoanGoal": "Car",
                "case:RequestedAmount": 10000 + index,
                "org:resource": "R_SMOKE",
                "lifecycle:transition": "complete",
            }
            for index in range(6)
        ]
    )


def test_integrated_allocation_comparison_script_creates_expected_csv(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        script,
        "Engine",
        build_fake_engine_class(script.Engine),
    )

    output = tmp_path / "integrated_resource_allocation_comparison.csv"

    result = script.run_integrated_comparison(
        data_path="data/logData.xes",
        start_time=datetime(2026, 1, 5, 9, 0),
        end_time=datetime(2026, 1, 5, 9, 0, 30),
        seed=1,
        output_path=output,
    )

    assert output.exists()
    saved = pd.read_csv(output)
    assert set(saved["strategy"]) == {"Random", "ShortestQueue", "ParkSong"}
    assert list(saved.columns) == [
        "strategy",
        "seed",
        "simulation_start",
        "simulation_end",
        "n_events",
        "n_cases",
        "assigned_events",
        "average_cycle_time",
        "average_waiting_time",
        "average_resource_occupation",
        "resource_fairness",
        "weighted_resource_fairness",
        "allocate_waiting_task_calls",
        "global_strategy_calls",
        "waiting_events_seen",
        "global_assignments",
        "old_path_assignments",
        "suspended_events",
        "resumed_events",
        "max_waiting_queue_length",
    ]
    assert set(result["strategy"]) == {"Random", "ShortestQueue", "ParkSong"}


def build_fake_engine_class(real_engine_class):
    class FakeEngine(real_engine_class):
        def __init__(self, dataPath="data/logData.xes", seed=1):
            import SimulationEngineCore as simulation_core

            original_read_xes = simulation_core.pm4py.read_xes
            original_arrival = simulation_core.ArrivalEngine
            original_process_time = simulation_core.ProcessTimeEngine

            simulation_core.pm4py.read_xes = lambda *args, **kwargs: build_smoke_log()
            simulation_core.ArrivalEngine = FakeArrivalEngine
            simulation_core.ProcessTimeEngine = FakeProcessTimeEngine

            try:
                super().__init__(dataPath=dataPath, seed=seed)
            finally:
                simulation_core.pm4py.read_xes = original_read_xes
                simulation_core.ArrivalEngine = original_arrival
                simulation_core.ProcessTimeEngine = original_process_time

            activities = {
                transition.label
                for transition in self.bpmnEngine.net.transitions
                if transition.label is not None
            }
            resources = {"R_SMOKE"}
            self.resourceEngine.availability.calendars = None
            self.resourceEngine.availability._all_resources = resources
            self.resourceEngine.permissions._activity_to_resources = {
                activity: set(resources)
                for activity in activities
            }

    return FakeEngine
