from __future__ import annotations

from datetime import datetime

class AvailabilityModel:
    def __init__(self, log):
        self._all_resources: set[str] = set()

    def who_is_available(self, time: datetime) -> set[str]:
        return self._all_resources
