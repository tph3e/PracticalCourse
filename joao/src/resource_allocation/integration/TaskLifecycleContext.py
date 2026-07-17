from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class TaskLifecycle:
    task_id: str
    case_id: str
    activity: str
    enabled_time: datetime
    process_wait_start: datetime | None = None
    process_wait_end: datetime | None = None
    resource_queue_entry_time: datetime | None = None
    resource_assignment_time: datetime | None = None
    resource_id: str | None = None
    processing_start_time: datetime | None = None
    processing_end_time: datetime | None = None
    sampled_processing_duration: timedelta | None = None
    processing_end_event_scheduled: bool = False
    processing_end_event_id: str | None = None


@dataclass
class ResourceReservation:
    reservation_id: str
    resource_id: str
    case_id: str
    target_activity: str
    target_task_id: str | None
    source_prediction_id: str
    creation_time: datetime
    valid_from: datetime
    expiration_time: datetime | None
    status: str = "created"
