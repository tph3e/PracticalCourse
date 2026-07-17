from __future__ import annotations

import random
from typing import Any, Iterable


class GlobalAllocationAdapter:
    """
    Adapter for the global ResourceEngine allocation interface.

    The global simulator calls:

        pick(candidates, context=None) -> resource_id | None

    Joao's richer allocation strategies operate on scenario objects with
    resources, waiting_tasks, current_time, and optional predictions. This
    adapter intentionally exposes only the basic resource-picking behavior that
    the global ResourceEngine can support from its candidate set.

    ParkSong/ParkSongML are kept in the scenario-based evaluation pipeline
    because deploying them at runtime requires richer context: waiting_tasks,
    current_time, predictions, and possible future tasks.
    """

    SUPPORTED_MODES = {"random", "round_robin"}

    def __init__(self, mode: str = "random", seed: int | None = None):
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported global allocation mode: {mode}. "
                f"Supported modes: {sorted(self.SUPPORTED_MODES)}"
            )

        self.mode = mode
        self.random = random.Random(seed)
        self._round_robin_index = 0
        self.pick_count = 0
        self.last_candidates: list[str] = []

    def pick(self, candidates: Iterable[str], context: Any | None = None) -> str | None:
        """
        Select one resource id from the already feasible candidate set.

        The global ResourceEngine is responsible for applying permissions,
        availability, and busy-resource filtering before calling this method.
        """

        ordered_candidates = sorted(candidates)
        self.pick_count += 1
        self.last_candidates = ordered_candidates

        if not ordered_candidates:
            return None

        if self.mode == "random":
            return self.random.choice(ordered_candidates)

        selected = ordered_candidates[
            self._round_robin_index % len(ordered_candidates)
        ]
        self._round_robin_index += 1
        return selected
