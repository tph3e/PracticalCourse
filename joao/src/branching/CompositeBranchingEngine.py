from __future__ import annotations

from typing import Any
import random

from .BranchingUtils import choose_random_valid_activity

class CompositeBranchingEngine:
    """
    Composite branching engine

    This engine combines several branching strategies in a fallback hierarchy

    Example hierarchy:
        1. PredictiveBranchingEngine
        2. AttributeSamplingBranchingEngine
        3. AttributeBasedBranchingEngine
        4. ProbabilityBranchingEngine
        5. Random RPMN-valid fallback

    The simulation core still only calls:

        getNextActivities(event, possibleActivities)

    The composite engine then tries each internal engine until one returns a BPMN-valid result
    """

    def __init__(
            self,
            engines: list[Any] | None = None,
            seed: int = 1,
    ):
        self.engines = engines or []
        self.seed = seed
        self.random = random.Random(seed)

        self.total_decisions = 0
        self.engine_success_counts: dict[str, int] = {}
        self.random_fallback_count = 0

    
    def getNextActivities(
            self,
            event: Any,
            possibleActivities: list[str],
    ) -> list[str]:
        """
        Selects the next activity using the first engine that returns a BPMN-valid result
        """

        if not possibleActivities:
            return []
        
        if len(possibleActivities) == 1:
            return possibleActivities
        
        self.total_decisions += 1

        for engine in self.engines:
            result = engine.getNextActivities(event, possibleActivities)

            if self._is_valid_result(result, possibleActivities):
                engine_name = engine.__class__.__name__
                self.engine_success_counts[engine_name] = (
                    self.engine_success_counts.get(engine_name, 0) + 1
                )
                return result
            
        self.random_fallback_count += 1

        return choose_random_valid_activity(
            possible_activities=possibleActivities,
            random_generator=self.random,
        )
    

    def add_engine(self, engine: Any) -> None:
        """
        Adds a branching engine to the fallback hierarchy
        """

        self.engines.append(engine)


    def _is_valid_result(
            self, 
            result: list[str],
            possibleActivities: list[str],
    ) -> bool:
        """
        Checks whether a result is BPMN-valid
        """

        if not isinstance(result, list):
            return False
        
        if not possibleActivities:
            return result == []
        
        if len(result) == 0:
            return False
        
        return all(activity in possibleActivities for activity in result)
    

    def get_statistics(self) -> dict[str, Any]:
        """
        Returns simple statistics for reporting and debugging
        """

        return {
            "total_decisions" : self.total_decisions,
            "engine_success_counts" : self.engine_success_counts,
            "random_fallback_count" : self.random_fallback_count,
        }
    


