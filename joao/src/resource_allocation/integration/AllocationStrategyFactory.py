from __future__ import annotations

from joao.src.resource_allocation.BatchAllocationAdapter import BatchAllocationAdapter
from joao.src.resource_allocation.KunklerAllocationAdapter import KunklerAllocationAdapter
from joao.src.resource_allocation.ParkSongAllocation import ParkSongAllocation
from joao.src.resource_allocation.RandomResourceAllocation import RandomResourceAllocation
from joao.src.resource_allocation.RoundRobinResourceAllocation import (
    RoundRobinResourceAllocation,
)
from joao.src.resource_allocation.ShortestQueueAllocation import ShortestQueueAllocation


def build_my_allocation_strategies(seed: int) -> dict[str, object]:
    return {
        "Random": RandomResourceAllocation(seed=seed),
        "RoundRobin": RoundRobinResourceAllocation(),
        "ShortestQueue": ShortestQueueAllocation(),
        "ParkSong": ParkSongAllocation(allow_strategic_idling=True),
        "Kunkler": KunklerAllocationAdapter(seed=seed),
        "Batch": BatchAllocationAdapter(k_limit=5),
    }
