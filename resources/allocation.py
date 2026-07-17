# 1.8 (Part 1) Resource allocation

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AllocationContext:
    # Runtime state handed to a strategy at an allocation decision point.

    time: datetime | None = None                          # current simulation time
    event: object = None                                  # event being allocated (activity, case, ...)
    busy: set[str] = field(default_factory=set)           # resources currently busy
    load: dict[str, float] = field(default_factory=dict)  # cumulative work per resource so far
    busy_activity: dict[str, str] = field(default_factory=dict)  # resource -> activity it runs (eta, Part 2: Task 1.1 (advanced)  RL state)


class AllocationStrategy:
    # context is optional so simple strategies (e.g. random) can ignore it.
    def pick(self, candidates: set[str], context: AllocationContext | None = None) -> str | None:
        raise NotImplementedError


class RandomAllocation(AllocationStrategy):
    def __init__(self, seed: int = 1):
        self._rng = random.Random(seed)

    def pick(self, candidates: set[str], context: AllocationContext | None = None) -> str | None:
        if not candidates:
            return None
        # sort first so the seeded RNG is deterministic regardless of set order
        return self._rng.choice(sorted(candidates))
