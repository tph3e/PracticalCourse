from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats


LogNormParams = Tuple[float, float, float]


def fit_requested_amount_distributions(
    log: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[tuple[str, str], LogNormParams], LogNormParams]:
    """
    Fit the simulator's requested-amount distributions without rounding.

    The root simulator rounds fitted parameters for presentation, which can turn
    small positive parameters into invalid zeros. This helper keeps full
    precision internally and falls back only to the valid global historical fit.
    """

    freq = (
        log.groupby(["case:ApplicationType", "case:LoanGoal"])
        .size()
        .rename("count")
        .reset_index()
    )
    freq["prob"] = freq["count"] / freq["count"].sum()

    global_amounts = _amounts(log)
    global_params = _fit_valid_lognorm(global_amounts)
    if global_params is None:
        raise ValueError(
            "Cannot fit requested-amount model: global historical lognormal "
            "fit is invalid."
        )

    amount_dists: Dict[tuple[str, str], LogNormParams] = {}
    for keys, group in log.groupby(["case:ApplicationType", "case:LoanGoal"]):
        amounts = _amounts(group)
        params = None
        if len(amounts) >= 5 and np.var(amounts) > 0:
            params = _fit_valid_lognorm(amounts)
        amount_dists[keys] = params or global_params

    return freq, amount_dists, global_params


def _amounts(log: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(
        log["case:RequestedAmount"],
        errors="coerce",
    ).dropna().to_numpy()


def _fit_valid_lognorm(amounts: np.ndarray) -> LogNormParams | None:
    if len(amounts) == 0:
        return None

    try:
        shape, loc, scale = stats.lognorm.fit(amounts)
    except Exception:
        return None

    params = (float(shape), float(loc), float(scale))
    if not all(np.isfinite(value) for value in params):
        return None
    if params[0] <= 0 or params[2] <= 0:
        return None
    return params
