# 1.6 (Part 1) Resource availabilities

from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd

DEFAULT_ARTIFACT = "results/availability_calendars.json"


def learn_calendars(
    slim_df: pd.DataFrame,
    resource_col: str = "org:resource",
    time_col: str = "time:timestamp",
    threshold_frac: float = 0.1,
    min_events: int = 1,
) -> dict[str, list[list[int]]]:
    
    df = slim_df[[resource_col, time_col]].dropna()
    ts = pd.to_datetime(df[time_col], utc=True)
    df = df.assign(weekday=ts.dt.weekday, hour=ts.dt.hour)

    calendars: dict[str, list[list[int]]] = {}
    for resource, group in df.groupby(resource_col):
        counts = group.groupby(["weekday", "hour"]).size()
        if counts.empty:
            continue
        cutoff = max(min_events, threshold_frac * counts.max())
        buckets = [[int(wd), int(h)] for (wd, h), c in counts.items() if c >= cutoff]
        if buckets:
            calendars[str(resource)] = buckets
    return calendars


def save_calendars(calendars: dict, path: str = DEFAULT_ARTIFACT) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calendars, f)


class AvailabilityModel:
    def __init__(
        self,
        log,
        resource_col: str = "org:resource",
        start_hour: int = 8,
        end_hour: int = 18,
        working_weekdays: frozenset[int] = frozenset({0, 1, 2, 3, 4}),  # Mon to Fri
        artifact_path: str | None = DEFAULT_ARTIFACT,
    ):
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.working_weekdays = working_weekdays

        # advanced: load learned per-resource calendars if the artifact exists
        self.calendars: dict[str, set[tuple[int, int]]] | None = None
        if artifact_path and os.path.exists(artifact_path):
            with open(artifact_path, encoding="utf-8") as f:
                raw = json.load(f)
            self.calendars = {
                res: {(int(wd), int(h)) for wd, h in buckets}
                for res, buckets in raw.items()
            }

        # resource universe (basic mode). Advanced mode draws it from calendars
        if self.calendars is not None:
            self._all_resources: set[str] = set(self.calendars.keys())
        elif log is not None:
            self._all_resources = set(log[resource_col].dropna().astype(str).unique())
        else:
            self._all_resources = set()

    @property
    def mode(self) -> str:
        return "advanced" if self.calendars is not None else "basic"

    def _is_working_time(self, time: datetime) -> bool:
        return (
            time.weekday() in self.working_weekdays
            and self.start_hour <= time.hour < self.end_hour
        )

    def who_is_available(self, time: datetime) -> set[str]:
        if time is None:
            return self._all_resources  # no time context (dummy run) -> all

        if self.calendars is not None:  # advanced: per-resource learned calendar
            bucket = (time.weekday(), time.hour)
            return {res for res, cal in self.calendars.items() if bucket in cal}

        # basic: single interval applied to every resource
        return self._all_resources if self._is_working_time(time) else set()
