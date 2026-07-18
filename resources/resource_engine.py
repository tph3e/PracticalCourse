# Facade the simulation core talks to. Orchestrates tasks 1.6 to 1.8 (Part 1 + Part 2).

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
        # activity each busy resource is currently executing (eta feature for the Task D RL state)
        self.busy_activity: dict[str, str] = {}

    def set_allocation(self, strategy) -> None:
        # Swap the 1.8 allocation strategy (Part 2: heuristics, batch, RL policy).
        # The core keeps calling allocateResource unchanged, only pick() differs.
        self.allocation = strategy

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
            busy_activity=self.busy_activity,
        )
        chosen = self.allocation.pick(candidates, context)          # 1.8
        if chosen is None:
            return False
        self.busy.add(chosen)
        self.load[chosen] = self.load.get(chosen, 0) + 1
        self.busy_activity[chosen] = getattr(event, "activity", None)  # eta
        event.resource = chosen
        return True

    def releaseResource(self, event) -> None:
        resource = getattr(event, "resource", None)
        self.busy.discard(resource)
        self.busy_activity.pop(resource, None)
