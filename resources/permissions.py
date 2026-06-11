from __future__ import annotations


class PermissionModel:
    def __init__(self, log):
        self._activity_to_resources: dict[str, set[str]] = {}

    def who_can(self, activity) -> set[str]:
        return self._activity_to_resources.get(activity, set())
