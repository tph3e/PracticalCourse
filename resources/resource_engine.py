# Facade the simulation core talks to. Orchestrates tasks 1.6 to 1.8 (Part 1 + Part 2).

from __future__ import annotations

from .availability import AvailabilityModel
from .permissions import PermissionModel
from .allocation import RandomAllocation, AllocationContext
from joao.src.resource_allocation.AllocationStrategy import AllocationDecision, Resource, Task


class ResourceEngine:
    def __init__(self, log, seed: int = 1):
        self.availability = AvailabilityModel(log)   # 1.6
        self.permissions = PermissionModel(log)      # 1.7
        self.allocation = RandomAllocation(seed)     # 1.8 (swappable for Part 2)
        self.global_allocation_strategy = None
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

    def allocate_waiting_tasks(
        self,
        waiting_events,
        current_time,
        predictions=None,
        **kwargs,
    ) -> list[AllocationDecision]:
        strategy = self.global_allocation_strategy
        if strategy is None:
            return []

        available_ids = self.availability.who_is_available(current_time) - self.busy
        predicted_activities = {
            str(getattr(prediction, "activity", ""))
            for prediction in (predictions or [])
            if getattr(prediction, "activity", "")
        }
        activities = {
            str(getattr(event, "activity", ""))
            for event in waiting_events
        } | predicted_activities
        activity_permissions = {
            activity: self.permissions.who_can(
                activity
            )
            for activity in activities
        }
        resource_ids = set(available_ids)
        for permitted in activity_permissions.values():
            resource_ids.update(permitted)

        resources = [
            Resource(
                resource_id=str(resource_id),
                available=str(resource_id) in available_ids,
                skills=[
                    activity
                    for activity, permitted in activity_permissions.items()
                    if str(resource_id) in permitted
                ],
            )
            for resource_id in sorted(resource_ids)
        ]
        tasks = [
            Task(
                task_id=self._task_id_for_event(event, index),
                case_id=str(getattr(getattr(event, "eventCase", None), "caseId", "")),
                activity=str(getattr(event, "activity", "")),
                enabled_time=self._seconds_since_epoch(getattr(event, "time", current_time)),
            )
            for index, event in enumerate(waiting_events)
        ]

        decisions = strategy.allocate(
            resources=resources,
            waiting_tasks=tasks,
            current_time=self._seconds_since_epoch(current_time),
            predictions=predictions or [],
            resource_loads={str(resource): float(load) for resource, load in self.load.items()},
            **kwargs,
        )
        events_by_task_id = {
            self._task_id_for_event(event, index): event
            for index, event in enumerate(waiting_events)
        }
        for decision in decisions:
            if decision.decision_type != "assignment" or decision.task_id is None:
                continue
            event = events_by_task_id.get(str(decision.task_id))
            if event is None:
                continue
            resource_id = str(decision.resource_id)
            event.resource = resource_id
            self.busy.add(resource_id)
            self.load[resource_id] = self.load.get(resource_id, 0) + 1
            self.busy_activity[resource_id] = getattr(event, "activity", None)
        return decisions

    @staticmethod
    def _seconds_since_epoch(value) -> float:
        timestamp = getattr(value, "timestamp", None)
        if callable(timestamp):
            return float(timestamp())
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _time_value(self, value) -> float:
        return self._seconds_since_epoch(value)

    @staticmethod
    def _task_id_for_event(event, fallback=None) -> str:
        return str(getattr(event, "eventId", fallback if fallback is not None else id(event)))
