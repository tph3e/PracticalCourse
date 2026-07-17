import pandas as pd
import pytest

from joao.src.resource_allocation.integration.IntegratedAllocationEngine import (
    IntegratedAllocationEngine,
)
from src.resource_allocation.integration import RequestedAmountCompatibility as compat


def log_frame():
    return pd.DataFrame(
        {
            "case:ApplicationType": ["New credit"] * 6,
            "case:LoanGoal": ["Car"] * 6,
            "case:RequestedAmount": [1000, 1001, 1002, 1003, 1004, 1005],
        }
    )


def test_requested_amount_fit_keeps_full_precision_without_rounding(monkeypatch):
    calls = []

    def fake_fit(amounts):
        calls.append(len(amounts))
        return (0.004, 0.0, 0.004)

    monkeypatch.setattr(compat.stats.lognorm, "fit", fake_fit)

    _freq, amount_dists, global_params = compat.fit_requested_amount_distributions(log_frame())

    assert global_params == (0.004, 0.0, 0.004)
    assert amount_dists[("New credit", "Car")] == (0.004, 0.0, 0.004)
    assert calls


def test_requested_amount_fit_uses_valid_global_fit_for_invalid_group(monkeypatch):
    returned = [(0.5, 10.0, 100.0), (0.0, 0.0, 0.0)]

    def fake_fit(amounts):
        return returned.pop(0)

    monkeypatch.setattr(compat.stats.lognorm, "fit", fake_fit)

    _freq, amount_dists, global_params = compat.fit_requested_amount_distributions(log_frame())

    assert global_params == (0.5, 10.0, 100.0)
    assert amount_dists[("New credit", "Car")] == global_params


def test_requested_amount_fit_fails_if_global_fit_is_invalid(monkeypatch):
    monkeypatch.setattr(compat.stats.lognorm, "fit", lambda amounts: (0.0, 0.0, 1.0))

    with pytest.raises(ValueError):
        compat.fit_requested_amount_distributions(log_frame())


def test_integrated_engine_clips_sampled_requested_amount_to_historical_bounds(monkeypatch):
    engine = IntegratedAllocationEngine.__new__(IntegratedAllocationEngine)
    engine.freq = pd.DataFrame(
        {
            "case:ApplicationType": ["New credit"],
            "case:LoanGoal": ["Car"],
            "prob": [1.0],
        }
    )
    engine.amount_dists = {("New credit", "Car"): (1.0, 0.0, 1.0)}
    engine.global_params = (1.0, 0.0, 1.0)
    engine.requested_amount_bounds = (1000.0, 1005.0)
    engine.requested_amount_rng = None

    monkeypatch.setattr(
        compat.stats.lognorm,
        "rvs",
        lambda *args, **kwargs: 10**12,
    )

    assert engine.sample_case_data()["case:RequestedAmount"] == 1005.0
