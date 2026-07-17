import json
from datetime import datetime

import pandas as pd
import pytest

from joao.scripts.resource_allocation import run_my_methods_integrated_comparison as script


def test_parse_strategy_filter_defaults_to_all_available_names():
    available = ["Random", "RoundRobin", "ShortestQueue", "ParkSong", "Batch"]

    assert script.parse_strategy_filter(None, available) == available
    assert script.parse_strategy_filter("", available) == available


def test_parse_strategy_filter_preserves_requested_order():
    available = ["Random", "RoundRobin", "ShortestQueue", "ParkSong", "Batch"]

    selected = script.parse_strategy_filter(
        "ShortestQueue,Random",
        available,
    )

    assert selected == ["ShortestQueue", "Random"]


def test_parse_strategy_filter_rejects_unknown_name():
    available = ["Random", "RoundRobin", "ShortestQueue", "ParkSong", "Batch"]

    with pytest.raises(ValueError, match="Unknown strategy name"):
        script.parse_strategy_filter("Random,Unknown", available)


def test_write_configuration_records_selected_strategies(tmp_path):
    script.write_configuration(
        output_dir=tmp_path,
        data_path="data/logData.xes",
        start_time=datetime(2000, 1, 3, 9, 0),
        end_time=datetime(2000, 1, 3, 9, 15),
        seeds=[1],
        branching_artifact="joao/models/branching/composite_branching_evaluation_train70.pkl",
        processing_time_artifact="joao/models/process_time/final_process_time_coverage_v2.pkl",
        strategy_names=["Random", "ParkSong"],
    )

    config = json.loads((tmp_path / "my_methods_configuration.json").read_text())

    assert config["methods"] == ["Random", "ParkSong"]
    assert config["selected_strategies"] == ["Random", "ParkSong"]
    assert (
        config["processing_time_artifact"]
        == "joao/models/process_time/final_process_time_coverage_v2.pkl"
    )


def test_processing_time_diagnostics_are_exported_to_summary():
    raw = pd.DataFrame(
        [
            {
                "strategy": "Random",
                "seed": 1,
                "n_events": 1,
                "n_cases": 1,
                "assigned_events": 1,
            }
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {
                "strategy": "Random",
                "processing_time_global_fallback_hits": 2,
                "processing_time_emergency_guard_hits": 0,
                "processing_time_any_non_emergency_coverage_rate": 1.0,
                "final_zero_visible_duration_count": 0,
                "processing_time_source_activity_A_Submitted_global_fallback": 1,
            }
        ]
    )

    summary = script.summarize(raw, diagnostics)

    assert "processing_time_global_fallback_hits" in summary.columns
    assert "processing_time_any_non_emergency_coverage_rate" in summary.columns
    assert "final_zero_visible_duration_count" in summary.columns
    assert (
        "processing_time_source_activity_A_Submitted_global_fallback"
        in summary.columns
    )
