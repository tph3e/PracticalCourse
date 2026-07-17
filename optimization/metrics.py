# 1.2 (basic): evaluation metrics for resource allocation methods.

# Three metrics from a standard XES event-log DataFrame (simulator output or BPIC-17 slim log): average cycle time, average resource occupation and weighted resource fairness.



from __future__ import annotations

import numpy as np
import pandas as pd

CASE = "case:concept:name"
ACT = "concept:name"
RES = "org:resource"
TS = "time:timestamp"
LC = "lifecycle:transition"

SECONDS_PER_WEEK = 7 * 24 * 3600


# helpers
def activity_durations(log_df: pd.DataFrame) -> pd.DataFrame:
    #  Pair start/complete events per (case, activity, occurrence) -> durations.

    df = log_df[[CASE, ACT, RES, TS, LC]].copy()
    df[TS] = pd.to_datetime(df[TS], utc=True)

    # Work begins at a "start" or "resume" event. In the simulator log a suspended
    # activity emits suspend -> resume -> complete with no "start", so pairing only
    # start<->complete would drop every resumed instance. The suspend->resume gap
    # is waiting, not busy.
    s = df[df[LC].isin(["start", "resume"])].sort_values(TS).copy()
    c = df[df[LC] == "complete"].sort_values(TS).copy()
    s["occ"] = s.groupby([CASE, ACT]).cumcount()
    c["occ"] = c.groupby([CASE, ACT]).cumcount()

    merged = s.merge(c, on=[CASE, ACT, "occ"], suffixes=("_s", "_c"))
    merged["duration_s"] = (merged[f"{TS}_c"] - merged[f"{TS}_s"]).dt.total_seconds()
    merged = merged[merged["duration_s"] >= 0]
    return merged.rename(
        columns={f"{RES}_s": "resource", f"{TS}_s": "start", f"{TS}_c": "complete"}
    )[[CASE, ACT, "resource", "start", "complete", "duration_s"]]


def busy_seconds(dur: pd.DataFrame) -> pd.Series:
    # Per-resource actual occupied (wall-clock) time = UNION of busy intervals.

    out: dict[str, float] = {}
    for resource, grp in dur.groupby("resource"):
        intervals = sorted(zip(grp["start"], grp["complete"]))
        total = 0.0
        cur_s = cur_e = None
        for s, e in intervals:
            if cur_e is None:
                cur_s, cur_e = s, e
            elif s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                total += (cur_e - cur_s).total_seconds()
                cur_s, cur_e = s, e
        if cur_e is not None:
            total += (cur_e - cur_s).total_seconds()
        out[str(resource)] = total
    return pd.Series(out, dtype=float)


def _window_seconds(log_df: pd.DataFrame, window_s: float | None = None) -> float:
    if window_s is not None:
        return window_s
    ts = pd.to_datetime(log_df[TS], utc=True)
    return float((ts.max() - ts.min()).total_seconds())


def _available_seconds(availability, window_s: float) -> dict[str, float] | None:
    # Per-resource available seconds from the 1.6 learned calendars.

    calendars = getattr(availability, "calendars", None)
    if not calendars:
        return None
    weeks = window_s / SECONDS_PER_WEEK
    return {r: len(buckets) * 3600.0 * weeks for r, buckets in calendars.items()}


def gini(values) -> float:
    #  Gini coefficient (0 = perfectly equal, ->1 = maximally unequal).

    x = np.sort(np.asarray(list(values), dtype=float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


# metrics
def average_cycle_time(log_df: pd.DataFrame) -> dict:
    #  Per-case span (last - first event timestamp), averaged over cases.

    ts = pd.to_datetime(log_df[TS], utc=True)
    g = pd.DataFrame({CASE: log_df[CASE].values, TS: ts.values}).groupby(CASE)[TS]
    span = (g.max() - g.min()).dt.total_seconds()
    return {
        "mean_s": float(span.mean()),
        "median_s": float(span.median()),
        "mean_days": float(span.mean() / 86400),
        "n_cases": int(span.size),
    }


def average_resource_occupation(
    log_df: pd.DataFrame, availability=None, window_s: float | None = None
) -> dict:
    # Mean per-resource occupation = busy time / available time.

    dur = activity_durations(log_df)
    busy = busy_seconds(dur)  # union of intervals -> true occupied time

    window_s = _window_seconds(log_df, window_s)
    avail = _available_seconds(availability, window_s) if availability is not None else None

    if avail is not None:
        occ = {r: busy.get(r, 0.0) / sec for r, sec in avail.items() if sec > 0}
        basis = "availability-calendar"
    else:
        occ = {r: b / window_s for r, b in busy.items()} if window_s > 0 else {}
        basis = "total-window"

    vals = list(occ.values())
    return {
        "mean": float(np.mean(vals)) if vals else 0.0,
        "median": float(np.median(vals)) if vals else 0.0,
        "max": float(np.max(vals)) if vals else 0.0,
        "basis": basis,
        "n_resources": len(occ),
        "per_resource": occ,
    }


def resource_fairness(
    log_df: pd.DataFrame,
    by: str = "busy",
    weighted: bool = False,
    availability=None,
    window_s: float | None = None,
) -> dict:
    # Gini coefficient over per-resource load (lower = fairer).
    # by="busy" uses busy seconds, by="count" uses activities

    dur = activity_durations(log_df)
    if by == "count":
        load = dur.groupby("resource").size().astype(float)
    else:
        load = busy_seconds(dur)  # union of intervals -> true occupied time

    if weighted and availability is not None:
        window_s = _window_seconds(log_df, window_s)
        avail = _available_seconds(availability, window_s)
        if avail:
            load = pd.Series(
                {r: load.get(r, 0.0) / avail[r] for r in avail if avail[r] > 0}
            )

    return {
        "gini": gini(load.values),
        "by": by,
        "weighted": bool(weighted and availability is not None),
        "n_resources": int(load.size),
    }


def compute_all(log_df: pd.DataFrame, availability=None, window_s: float | None = None) -> dict:
    # All three metrics in one dict (reused as the RL reward signal)
    return {
        "cycle_time": average_cycle_time(log_df),
        "occupation": average_resource_occupation(log_df, availability, window_s),
        "fairness": resource_fairness(log_df, availability=availability, window_s=window_s),
    }


def compare_on_sim(sim_logs: dict, availability=None) -> pd.DataFrame:
    # Cross-method comparison on the integrated simulator output (task F bridge).
    # sim_logs: {method_name: simulator_output_log_df}, one table row per method.
    rows = []
    for method, slog in sim_logs.items():
        m = compute_all(slog, availability=availability)
        rows.append([
            method,
            round(m["cycle_time"]["mean_days"], 1),
            round(m["occupation"]["mean"], 3),
            round(m["fairness"]["gini"], 3),
        ])
    return pd.DataFrame(
        rows, columns=["method", "cycle_time_days", "occupation", "fairness_gini"]
    )
