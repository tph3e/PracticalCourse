
from __future__ import annotations

import random


class AllocationStrategy:
    def pick(self, candidates: set[str]) -> str | None:
        raise NotImplementedError


class RandomAllocation(AllocationStrategy):
    def __init__(self, seed: int = 1):
        self._rng = random.Random(seed)

    def pick(self, candidates: set[str]) -> str | None:
        if not candidates:
            return None
        return self._rng.choice(sorted(candidates))
