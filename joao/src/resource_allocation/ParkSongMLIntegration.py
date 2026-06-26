from __future__ import annotations

from typing import Any, List

from .AllocationStrategy import AllocationDecision, Resource, Task
from .MLPredictionAdapter import MLPredictionAdapter
from .ParkSongAllocation import ParkSongAllocation


class ParkSongMLIntegration:
    """
    Integration layer between the predictive branching model and the
    Park & Song-inspired resource allocation strategy.

    Pipeline:
        simulation event
            -> ML next-activity prediction
            -> Prediction objects
            -> ParkSongAllocation
            -> allocation decisions
    """

    def __init__(
        self,
        prediction_adapter: MLPredictionAdapter,
        allocator: ParkSongAllocation,
    ):
        self.prediction_adapter = prediction_adapter
        self.allocator = allocator

    def allocate_with_ml_predictions(
        self,
        event: Any,
        possible_activities: List[str],
        resources: List[Resource],
        waiting_tasks: List[Task],
        current_time: float,
    ) -> List[AllocationDecision]:
        """
        Generate ML predictions for the current simulation event and pass them
        to the Park & Song-inspired allocation strategy.
        """

        predictions = self.prediction_adapter.predict_for_event(
            event=event,
            possible_activities=possible_activities,
        )

        return self.allocator.allocate(
            resources=resources,
            waiting_tasks=waiting_tasks,
            current_time=current_time,
            predictions=predictions,
        )
