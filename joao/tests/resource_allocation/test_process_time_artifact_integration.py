from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from processTimes import process_time_engine
from processTimes.process_time_engine import ProcessTimeEngine


def test_process_time_engine_respects_configured_model_path(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "configured_process_time.pkl"
    model_path.write_bytes(b"placeholder")
    loaded_paths: list[str] = []

    def fake_load(path):
        loaded_paths.append(str(path))
        return {
            "basic": {},
            "quantiles": {},
            "advanced": {},
            "fallback_basic": {},
        }

    monkeypatch.setattr(process_time_engine.joblib, "load", fake_load)

    engine = ProcessTimeEngine(model_path=str(model_path))

    assert loaded_paths == [str(model_path)]
    assert engine.model_path == str(model_path)


def test_waiting_time_accepts_next_activity_override() -> None:
    engine = ProcessTimeEngine(metricProcessing=True)
    engine.waiting_advanced = False
    engine.rng = SimpleNamespace(random=lambda: 1.0, poisson=lambda lam: lam)
    engine.models_basic = {
        "NextActivity_R1_waiting": {
            "0-proportion": 0.0,
            "distribution": "poisson",
            "parameters": {"lambda": 7},
        }
    }
    engine.fallback_models_basic = {}

    event = SimpleNamespace(activity="CurrentActivity", resource="R1")

    assert engine.getWaitingTime(event, "NextActivity").total_seconds() == 7
