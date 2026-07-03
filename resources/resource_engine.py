# Facade the simulation core talks to. Orchestrates tasks 1.6–1.8 (Part 1 + Part 2).

from __future__ import annotations

from .availability import AvailabilityModel
from .permissions import PermissionModel
from .allocation import RandomAllocation, AllocationContext


class ResourceEngine:
    def __init__(self, log, seed: int = 1):
        self.availability = AvailabilityModel(log)   # 1.6
        self.permissions = PermissionModel(log)      # 1.7
        self.allocation = RandomAllocation(seed)     # 1.8 (swappable for Part 2)
        # resources currently executing an activity
        self.busy: set[str] = set()
        # activities allocated per resource (load signal for Part 2)
        self.load: dict[str, float] = {}

    def allocateResource(self, event) -> bool:
        # Try to assign a resource to "event". Returns True on success.

        eligible = self.permissions.who_can(event.activity)         # 1.7
        available = self.availability.who_is_available(event.time)  # 1.6
        candidates = (eligible & available) - self.busy
        # context lets Part II strategies (push heuristics, RL) see runtime state
        context = AllocationContext(
            time=getattr(event, "time", None),
            event=event,
            busy=self.busy,
            load=self.load,
        )
        chosen = self.allocation.pick(candidates, context)          # 1.8
        if chosen is None:
            return False
        self.busy.add(chosen)
        self.load[chosen] = self.load.get(chosen, 0) + 1
        event.resource = chosen
        return True

    def releaseResource(self, event) -> None:
        self.busy.discard(getattr(event, "resource", None))
