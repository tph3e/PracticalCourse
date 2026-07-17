from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BranchDecision:
    activities: list[str]
    decision_source: str
    probability_source: str | None = None
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    support: int | None = None
    used_fallback: bool = False
    decision_point_id: str | None = None
    candidate_activities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_valid_for(self, possible_activities: list[str]) -> bool:
        if not isinstance(self.activities, list):
            return False
        if not possible_activities:
            return self.activities == []
        if not self.activities:
            return False
        return all(activity in possible_activities for activity in self.activities)
