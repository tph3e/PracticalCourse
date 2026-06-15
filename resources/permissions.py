# 1.7 Resource permissions

from __future__ import annotations

import json
import os

import pandas as pd

DEFAULT_ARTIFACT = "results/permissions_roles.json"


class PermissionModel:
    # Maps an activity to the set of resources allowed to perform it.

    def __init__(
        self,
        log,
        activity_col: str = "concept:name",
        resource_col: str = "org:resource",
        artifact_path: str | None = DEFAULT_ARTIFACT,
    ):
        self._activity_to_resources: dict[str, set[str]] = {}

        if artifact_path and os.path.exists(artifact_path):
            self._load_artifact(artifact_path)  # advanced: role-based mapping
            self._mode = "advanced"
        else:
            if log is not None:
                self._build_matrix(log, activity_col, resource_col)  # basic
            self._mode = "basic"

    @property
    def mode(self) -> str:
        return self._mode

    def _build_matrix(self, log, activity_col: str, resource_col: str) -> None:
        df = log[[activity_col, resource_col]].dropna()
        for activity, group in df.groupby(activity_col):
            self._activity_to_resources[str(activity)] = set(
                group[resource_col].astype(str).unique()
            )

    def _load_artifact(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)  
        self._activity_to_resources = {a: set(rs) for a, rs in raw.items()}

    def who_can(self, activity) -> set[str]:
        # Return the set of resources permitted to perform "activity".
        return self._activity_to_resources.get(str(activity), set())
