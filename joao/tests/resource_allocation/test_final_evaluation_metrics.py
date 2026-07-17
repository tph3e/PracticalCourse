from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from joao.src.resource_allocation.ParkSongAllocation import ParkSongAllocation
from joao.src.resource_allocation.RoundRobinResourceAllocation import (
    RoundRobinResourceAllocation,
)
from joao.src.resource_allocation.ShortestQueueAllocation import ShortestQueueAllocation
from scripts.resource_allocation.run_final_resource_allocation_evaluation import (
    aggregate,
    build_strategy,
    completed_case_cycle_durations,
    lifecycle_metrics,
    lifecycle_waiting_durations,
    paired_comparisons,
    parse_parksong_params,
    parksong_processing_time_estimates_from_log,
    processing_time_coverage,
    reservation_diagnostics,
)


def lifecycle(case_id, activity, enabled, assigned, completed):
    return SimpleNamespace(
        case_id=case_id,
        activity=activity,
        resource_id="R1",
        enabled_time=enabled,
        process_wait_start=enabled,
        resource_queue_entry_time=enabled,
        resource_assignment_time=assigned,
        process_wait_end=assigned,
        processing_start_time=assigned,
        processing_end_time=completed,
    )


class AlwaysAvailable:
    _all_resources = {"R1"}

    def who_is_available(self, _time):
        return {"R1"}


def test_cycle_time_uses_completed_cases_only():
    start = datetime(2026, 1, 1, 9, 0, 0)
    engine = SimpleNamespace(
        task_lifecycle={
            "C1:A": lifecycle(
                "C1",
                "A",
                start,
                start + timedelta(seconds=5),
                start + timedelta(seconds=10),
            ),
            "C2:A": lifecycle(
                "C2",
                "A",
                start,
                start + timedelta(seconds=1),
                start + timedelta(seconds=1000),
            ),
        },
    )

    durations = completed_case_cycle_durations(engine, {"C1"})

    assert durations.tolist() == [10.0]


def test_waiting_time_uses_queue_to_assignment_not_processing_duration():
    start = datetime(2026, 1, 1, 9, 0, 0)
    engine = SimpleNamespace(
        task_lifecycle={
            "C1:A": lifecycle(
                "C1",
                "A",
                start,
                start + timedelta(seconds=7),
                start + timedelta(seconds=60),
            ),
        },
    )

    waits = lifecycle_waiting_durations(engine)

    assert waits.tolist() == [7.0]


def test_lifecycle_metrics_reports_mutually_exclusive_terminal_counts():
    start = datetime(2026, 1, 1, 9, 0, 0)
    end = start + timedelta(hours=1)
    engine = SimpleNamespace(
        admitted_case_ids={"C1", "C2", "C3", "C4"},
        completed_case_ids={"C1"},
        deadlocked_case_ids={"C2"},
        cyclic_case_ids={"C3"},
        censored_case_ids=set(),
        waiting_processes=[],
        drain_stopped_by_limit=False,
        resourceEngine=SimpleNamespace(availability=AlwaysAvailable()),
        task_lifecycle={
            "C1:A": lifecycle(
                "C1",
                "A",
                start,
                start + timedelta(seconds=2),
                start + timedelta(seconds=10),
            ),
        },
    )
    log = pd.DataFrame(
        [
            {
                "case:concept:name": "C1",
                "concept:name": "A",
                "org:resource": "R1",
                "lifecycle:transition": "complete",
                "time:timestamp": start + timedelta(seconds=10),
            }
        ]
    )

    metrics = lifecycle_metrics(log, engine, start, end)

    assert metrics["cases_admitted"] == 4
    assert metrics["cases_completed"] == 1
    assert metrics["fixed_routes_completed"] == 1
    assert metrics["fixed_route_completion_rate"] == 0.25
    assert metrics["cases_deadlocked"] == 1
    assert metrics["cases_cyclic"] == 1
    assert metrics["cases_active"] == 1
    assert metrics["cycle_time_mean_s"] == 10.0
    assert metrics["waiting_time_mean_s"] == 2.0
    assert metrics["horizon_normalized_throughput_cases_per_hour"] == metrics["throughput_cases_per_hour"]
    assert pd.isna(metrics["horizon_normalized_resource_occupation_mean"])
    assert pd.isna(metrics["resource_occupation_mean"])


def test_parse_parksong_params_accepts_known_numeric_overrides():
    params = parse_parksong_params(
        "prediction_probability_threshold=0.8,idling_weight=2.5,"
        "future_delay_weight=0.001,reservation_margin=0.2,"
        "processing_time_weight=0.1,cost_time_scale=3600,"
        "no_show_penalty_weight=0.3"
    )

    assert params == {
        "prediction_probability_threshold": 0.8,
        "idling_weight": 2.5,
        "future_delay_weight": 0.001,
        "reservation_margin": 0.2,
        "processing_time_weight": 0.1,
        "cost_time_scale": 3600.0,
        "no_show_penalty_weight": 0.3,
    }


def test_parse_parksong_params_rejects_unknown_override():
    with pytest.raises(ValueError, match="Unknown ParkSong parameter"):
        parse_parksong_params("unknown_weight=1.0")


def test_build_strategy_applies_parksong_overrides_only_to_parksong():
    strategy = build_strategy(
        "ParkSong-Composite",
        seed=1,
        parksong_params={
            "prediction_probability_threshold": 0.8,
            "idling_weight": 2.5,
        },
    )

    assert isinstance(strategy, ParkSongAllocation)
    assert strategy.prediction_probability_threshold == 0.8
    assert strategy.idling_weight == 2.5
    assert strategy.processing_time_weight == 1.0
    assert strategy.waiting_weight == 0.5


def test_build_strategy_instantiates_base_heuristics_directly():
    assert isinstance(build_strategy("RoundRobin", seed=1), RoundRobinResourceAllocation)
    assert isinstance(build_strategy("ShortestQueue", seed=1), ShortestQueueAllocation)


def test_build_strategy_injects_parksong_processing_time_estimates():
    estimates = {("R1", "A"): 42.0}

    strategy = build_strategy(
        "ParkSong-Composite",
        seed=1,
        parksong_processing_time_estimates=estimates,
        parksong_default_processing_time=9.0,
    )

    assert isinstance(strategy, ParkSongAllocation)
    assert strategy.processing_time_estimates == estimates
    assert strategy.default_processing_time == 9.0


def test_parksong_processing_time_estimates_use_positive_resource_activity_medians():
    start = datetime(2026, 1, 1, 9, 0, 0)
    log = pd.DataFrame(
        [
            {
                "case:concept:name": "C1",
                "concept:name": "A",
                "org:resource": "R1",
                "lifecycle:transition": "start",
                "time:timestamp": start,
            },
            {
                "case:concept:name": "C1",
                "concept:name": "A",
                "org:resource": "R1",
                "lifecycle:transition": "complete",
                "time:timestamp": start + timedelta(seconds=10),
            },
            {
                "case:concept:name": "C2",
                "concept:name": "A",
                "org:resource": "R1",
                "lifecycle:transition": "start",
                "time:timestamp": start + timedelta(minutes=1),
            },
            {
                "case:concept:name": "C2",
                "concept:name": "A",
                "org:resource": "R1",
                "lifecycle:transition": "complete",
                "time:timestamp": start + timedelta(minutes=1, seconds=30),
            },
            {
                "case:concept:name": "C3",
                "concept:name": "A",
                "org:resource": "R2",
                "lifecycle:transition": "start",
                "time:timestamp": start + timedelta(minutes=2),
            },
            {
                "case:concept:name": "C3",
                "concept:name": "A",
                "org:resource": "R2",
                "lifecycle:transition": "complete",
                "time:timestamp": start + timedelta(minutes=2, seconds=20),
            },
        ]
    )

    estimates, default_processing_time, summary = parksong_processing_time_estimates_from_log(log)

    assert estimates[("R1", "A")] == 20.0
    assert estimates[("R2", "A")] == 20.0
    assert default_processing_time == 20.0
    assert summary["estimate_count"] == 2
    assert summary["positive_occurrence_count"] == 3


def test_build_strategy_uses_calibrated_parksong_composite_waiting_weight():
    strategy = build_strategy("ParkSong-Composite", seed=1)

    assert isinstance(strategy, ParkSongAllocation)
    assert strategy.waiting_weight == 0.5


def test_build_strategy_allows_parksong_composite_waiting_weight_override():
    strategy = build_strategy(
        "ParkSong-Composite",
        seed=1,
        parksong_params={"waiting_weight": 0.2},
    )

    assert isinstance(strategy, ParkSongAllocation)
    assert strategy.waiting_weight == 0.2


def test_aggregate_uses_student_t_ci_and_keeps_identifier_columns_out():
    raw = pd.DataFrame(
        [
            {
                "strategy": "A",
                "seed": 1,
                "cycle_time_mean_s": 10.0,
                "waiting_time_mean_s": 1.0,
                "horizon_normalized_throughput_cases_per_hour": 0.1,
                "horizon_normalized_resource_occupation_mean": 0.2,
                "resource_fairness_gini": 0.3,
                "weighted_resource_fairness": 0.4,
                "fixed_route_completion_rate": 1.0,
                "completed_case_ids": "C1;C2",
            },
            {
                "strategy": "A",
                "seed": 2,
                "cycle_time_mean_s": 14.0,
                "waiting_time_mean_s": 3.0,
                "horizon_normalized_throughput_cases_per_hour": 0.2,
                "horizon_normalized_resource_occupation_mean": 0.4,
                "resource_fairness_gini": 0.5,
                "weighted_resource_fairness": 0.6,
                "fixed_route_completion_rate": 0.5,
                "completed_case_ids": "C3",
            },
        ]
    )

    aggregated = aggregate(raw)

    assert aggregated.loc[0, "ci_method"] == "student_t"
    assert aggregated.loc[0, "ci_level"] == 0.95
    assert aggregated.loc[0, "n_runs"] == 2
    assert "completed_case_ids_mean" not in aggregated.columns
    assert aggregated.loc[0, "cycle_time_mean_s_ci95_half_width"] > 0


def test_paired_comparisons_reports_descriptive_wins_and_t_ci():
    raw = pd.DataFrame(
        [
            {"strategy": "ParkSong-Composite", "seed": 1, "cycle_time_mean_s": 9.0, "waiting_time_mean_s": 1.0, "horizon_normalized_throughput_cases_per_hour": 1.0, "horizon_normalized_resource_occupation_mean": 1.0, "resource_fairness_gini": 1.0, "weighted_resource_fairness": 1.0},
            {"strategy": "RoundRobin", "seed": 1, "cycle_time_mean_s": 10.0, "waiting_time_mean_s": 2.0, "horizon_normalized_throughput_cases_per_hour": 1.0, "horizon_normalized_resource_occupation_mean": 1.0, "resource_fairness_gini": 1.0, "weighted_resource_fairness": 1.0},
            {"strategy": "ParkSong-Composite", "seed": 2, "cycle_time_mean_s": 12.0, "waiting_time_mean_s": 4.0, "horizon_normalized_throughput_cases_per_hour": 1.0, "horizon_normalized_resource_occupation_mean": 1.0, "resource_fairness_gini": 1.0, "weighted_resource_fairness": 1.0},
            {"strategy": "RoundRobin", "seed": 2, "cycle_time_mean_s": 11.0, "waiting_time_mean_s": 3.0, "horizon_normalized_throughput_cases_per_hour": 1.0, "horizon_normalized_resource_occupation_mean": 1.0, "resource_fairness_gini": 1.0, "weighted_resource_fairness": 1.0},
        ]
    )

    comparisons = paired_comparisons(raw)
    row = comparisons[
        (comparisons["comparison"] == "ParkSong-Composite - RoundRobin")
        & (comparisons["metric"] == "cycle_time_mean_s")
    ].iloc[0]

    assert row["ci_method"] == "student_t"
    assert row["paired_seed_count"] == 2
    assert row["wins"] == 1
    assert row["losses"] == 1


def test_processing_time_and_reservation_diagnostics_are_derived_from_engine_counters():
    diagnostics = pd.DataFrame(
        [
            {
                "strategy": "ParkSong-Composite",
                "seed": 1,
                "processing_time_model_hits": 2,
                "processing_time_activity_fallback_hits": 3,
                "processing_time_empirical_activity_fallback_hits": 1,
                "processing_time_category_fallback_hits": 4,
                "processing_time_global_fallback_hits": 10,
                "processing_time_emergency_guard_hits": 0,
                "reservations_created": 5,
                "reservations_used": 2,
                "park_song_predictions_consumed": 7,
                "prediction_execution_matches": 3,
                "prediction_execution_mismatches": 1,
                "simulation_runtime_seconds": 0.5,
            }
        ]
    )

    coverage = processing_time_coverage(diagnostics)
    reservations = reservation_diagnostics(diagnostics)

    assert coverage.loc[0, "total_processing_time_sampling_calls"] == 20
    assert coverage.loc[0, "global_fallback_rate"] == 0.5
    assert reservations.loc[0, "reservation_utilization_rate"] == 0.4
    assert reservations.loc[0, "predictions_consumed"] == 7
