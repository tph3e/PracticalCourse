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
        self.global_allocation_strategy = None
        # resources currently executing an activity
        self.busy: set[str] = set()
        # activities allocated per resource (load signal for Part 2)
        self.load: dict[str, float] = {}
        # activity each busy resource is currently executing (eta feature for the Part 2: Task 1.1 (advanced)  RL state)
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
    ):
        """
        Optional Part II batch allocation path for waiting simulator events.

        The existing allocateResource(event) path remains the default. This method
        is used only when a richer Joao allocation strategy is explicitly assigned
        to self.global_allocation_strategy.
        """

        if self.global_allocation_strategy is None or not waiting_events:
            return []

        from joao.src.resource_allocation.AllocationStrategy import Resource, Task

        current_time_value = self._time_value(current_time)
        prediction_activities = {
            prediction.activity
            for prediction in (predictions or [])
            if getattr(prediction, "activity", None) is not None
        }
        waiting_activities = {event.activity for event in waiting_events}
        candidate_activities = waiting_activities | prediction_activities
        available_resources = self.availability.who_is_available(current_time)
        resource_ids = sorted(available_resources - self.busy)

        resources = []
        for resource_id in resource_ids:
            skills = [
                activity
                for activity in candidate_activities
                if resource_id in self.permissions.who_can(activity)
            ]
            if skills:
                resources.append(
                    Resource(
                        resource_id=resource_id,
                        available=True,
                        skills=skills,
                    )
                )

        event_by_task_id = {
            self._task_id_for_event(event): event
            for event in waiting_events
        }

        tasks = [
            Task(
                task_id=task_id,
                case_id=self._case_id_for_event(event),
                activity=event.activity,
                enabled_time=self._time_value(getattr(event, "time", current_time)),
                priority=float(getattr(event, "priority", 0.0) or 0.0),
            )
            for task_id, event in event_by_task_id.items()
        ]

        decisions = self.global_allocation_strategy.allocate(
            resources=resources,
            waiting_tasks=tasks,
            current_time=current_time_value,
            predictions=predictions or [],
            resource_loads=dict(self.load),
        )

        for decision in decisions:
            if decision.decision_type != "assignment" or decision.task_id is None:
                continue

            event = event_by_task_id.get(decision.task_id)
            if event is None:
                continue

            self.busy.add(decision.resource_id)
            self.load[decision.resource_id] = self.load.get(decision.resource_id, 0) + 1
            self.busy_activity[decision.resource_id] = event.activity
            event.resource = decision.resource_id

        return decisions

    def _task_id_for_event(self, event) -> str:
        event_id = getattr(event, "eventId", None)
        if event_id is None:
            event_id = getattr(event, "event_id", None)
        if event_id is None and hasattr(event, "getAttribs"):
            event_id = event.getAttribs().get("EventID")
        return str(event_id if event_id is not None else id(event))

    def _case_id_for_event(self, event) -> str:
        event_case = getattr(event, "eventCase", None)
        case_id = getattr(event_case, "caseId", None)
        if case_id is None and hasattr(event, "getAttribs"):
            case_id = event.getAttribs().get("case:concept:name")
        return str(case_id if case_id is not None else "UNKNOWN_CASE")

    def _time_value(self, value) -> float:
        if hasattr(value, "timestamp"):
            return float(value.timestamp())
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
