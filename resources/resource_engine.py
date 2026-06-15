# Facade the simulation core talks to. Orchestrates tasks 1.6–1.8.

from __future__ import annotations

from .availability import AvailabilityModel
from .permissions import PermissionModel
from .allocation import RandomAllocation


class ResourceEngine:
    def __init__(self, log, seed: int = 1):
        self.availability = AvailabilityModel(log)   # 1.6
        self.permissions = PermissionModel(log)      # 1.7
        self.allocation = RandomAllocation(seed)     # 1.8 (swappable for Part II)
        # resources currently executing an activity
        self.busy: set[str] = set()

    def allocateResource(self, event) -> bool:
        # Try to assign a resource to "event". Returns True on success.

        eligible = self.permissions.who_can(event.activity)         # 1.7
        available = self.availability.who_is_available(event.time)  # 1.6
        candidates = (eligible & available) - self.busy
        chosen = self.allocation.pick(candidates)                   # 1.8
        if chosen is None:
            return False
        self.busy.add(chosen)
        event.resource = chosen
        return True

    def releaseResource(self, event) -> None:
        self.busy.discard(getattr(event, "resource", None))
