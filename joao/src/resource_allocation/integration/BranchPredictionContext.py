from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class BranchPrediction:
    prediction_id: str
    case_id: str
    current_activity: str
    decision_point: str
    candidate_activities: tuple[str, ...]
    selected_activity: str | None
    probabilities: dict[str, float] = field(default_factory=dict)
    prediction_source: str = "unknown"
    prediction_time: datetime | None = None
    target_task_id: str | None = None
    expected_delay: float = 0.0
    status: str = "created"
    scheduled_activity: str | None = None
    started_activity: str | None = None
    completed_activity: str | None = None
    candidate_transition_ids: tuple[str, ...] = ()
    selected_transition_id: str | None = None
    marking_signature: str | None = None
    transition_ambiguity: bool = False
    fallback_source: str | None = None
    rejected_activity: str | None = None

    @property
    def confidence(self) -> float | None:
        if self.selected_activity is None:
            return None
        return self.probabilities.get(self.selected_activity)
