from __future__ import annotations

from typing import Any
import random

from .BranchingUtils import choose_random_valid_activity


class AttributeSamplingBranchingEngine:
    """
    Advanced branching engine for sampled and modified runtime attributes

    This engine enriches event/case attributes before delegating the actual branch decision to a base engine, usually AttributeBasedBranchingEngine

    It supports:
    - sampling missing attributes
    - deriving new attributes
    - modifying runtime attributes
    """

    def __init__(
            self,
            base_engine: Any,
            sampling_config: dict[str, dict[Any, float]] | None = None,
            fallback_engine: Any | None = None,
            seed: int = 1,
    ): 
        self.base_engine = base_engine
        self.sampling_config = sampling_config or {}
        self.fallback_engine = fallback_engine
        self.seed = seed
        self.random = random.Random(seed)

        self.sampled_attribute_count = 0
        self.derived_attribute_count = 0
        self.modified_attribute_count = 0
        self.fallback_count = 0
        self.total_decisions = 0

    
    def train(self, event_log=None) -> None:
        """
        Placeholder for future learning of attribute distributions from the event log

        For now, sampling distributions are passed through sampling_config
        """

        # Later this can learn distributions from the event log
        return None
    

    def getNextActivities(
            self,
            event: Any,
            possibleActivities: list[str],
    ) -> list[str]:
        """
        Enriches attributes and delegates branch selection to the base engine
        """

        if not possibleActivities:
            return []
        
        if len(possibleActivities) == 1:
            return possibleActivities
        
        self.total_decisions += 1

        original_attributes = self.extract_attributes(event)
        enriched_attributes = dict(original_attributes)

        enriched_attributes = self.sample_missing_attributes(enriched_attributes)
        enriched_attributes = self.derive_attributes(enriched_attributes, event)
        enriched_attributes = self.modify_runtime_attributes(enriched_attributes, event)

        enriched_event = self.EnrichedEventWrapper(
            original_event=event,
            enriched_attributes=enriched_attributes,
        )

        result = self.base_engine.getNextActivities(
            enriched_event,
            possibleActivities,
        )

        if self._is_valid_result(result, possibleActivities):
            return result
        
        return self.fallback(event, possibleActivities)
    

    def extract_attributes(self, event: Any) -> dict[str, Any]:
        """
        Extracts attributes from different possible event structures
        """

        attributes: dict[str, Any] = {}

        if event is None:
            return attributes
        
        if hasattr(event, "getAttribs"):
            event_attributes = event.getAttribs()
            if isinstance(event_attributes, dict):
                attributes.update(event_attributes)

        if hasattr(event, "attributes"):
            event_attributes = event.attributes
            if isinstance(event_attributes, dict):
                attributes.update(event_attributes)

        if hasattr(event, "data"):
            event_data = event.data
            if isinstance(event_data, dict):
                attributes.update(event_data)

        if isinstance(event, dict):
            attributes.update(event)

        if hasattr(event, "activity"):
            attributes.setdefault("concept:name", event.activity)
            attributes.setdefault("activity", event.activity)

        return attributes
    
    
    def sample_missing_attributes(
            self,
            attributes: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Samples missing attributes based on sampling_config

        sampling_config example:
        {
            "CustomerType" : {
            "standard" : 0.7,
            "premium" : 0.2,
            "business" : 0.1
            }
        }
        """

        for attribute_name, distribution in self.sampling_config.items():
            if attribute_name in attributes and attributes[attribute_name] is not None:
                continue

            sampled_value = self._sample_from_distribution(distribution)
            attributes[attribute_name] = sampled_value
            self.sampled_attribute_count += 1

        return attributes
    

    def _sample_from_distribution(
            self,
            distribution: dict[Any, float],
    ) -> Any:
        """
        Samples one value from a probability distribution
        """

        if not distribution:
            return None
        
        values = list(distribution.keys())
        weights = list(distribution.values())

        return self.random.choices(
            population=values,
            weights=weights,
            k=1,
        )[0]
    

    def derive_attributes(
            self,
            attributes: dict[str, Any],
            event: Any,
    ) -> dict[str, Any]:
        """
        Derives new attributes from existing attributes

        Current implemented examples:
        - RequestedAmount -> AmountCategory
        - CreditScore -> RiskCategory
        - AmoungCategory / RiskCategory -> CaseComplexity
        """

        before_count = len(attributes)

        requested_amount = self._get_first_available(
            attributes,
            [
                "RequestedAmount",
                "case:RequestedAmount",
                "requested_amount",
            ],
        )

        if requested_amount is not None and "AmountCategory" not in attributes:
            attributes["AmountCategory"] = self._derive_amount_category(requested_amount)

        credit_score = self._get_first_available(
            attributes,
            [
                "CreditScore",
                "case:CrediScore",
                "credit_score",
            ],
        )

        if credit_score is not None and "RiskCategory" not in attributes:
            attributes["RiskCategory"] = self._derive_risk_category(credit_score)

        if "CaseComplexity" not in attributes:
            amount_category = attributes.get("AmountCategory")
            risk_category = attributes.get("RiskCategory")

            if amount_category is not None or risk_category is not None:
                attributes["CaseComplexity"] = self._derive_case_complexity(
                    amount_category=amount_category,
                    risk_category=risk_category,
                )

        after_count = len(attributes)
        self.derived_attribute_count += max(0, after_count - before_count)
        return attributes 
    

    def modify_runtime_attributes(
            self,
            attributes: dict[str, Any],
            event: Any,
    ) -> dict[str, Any]:
        """
        Modifies runtime attributes based on process state

        Current simple implementation:
        - If elapsed_case_time is high, set Priority = high
        - If previous activities contain Review more than once, set ReworkIndicator = True
        """

        modified = 0

        elapsed_time = self._get_first_available(
            attributes,
            [
                "elapsed_case_time",
                "ElapsedCaseTime",
                "case_elapsed_time",
            ],
        )

        if elapsed_time is not None:
            try:
                if float(elapsed_time) > 10:
                    if attributes.get("Priority") != "high":
                        attributes["Priority"] = "high"
                        modified += 1
            except (TypeError, ValueError):
                pass

        history = self._extract_history(event)

        review_count = sum(
            1 for activity in history
            if isinstance(activity, str) and "review" in activity.lower()
        )

        if review_count > 1:
            if attributes.get("ReworkIndicator") is not True:
                attributes["ReworkIndicator"] = True
                modified += 1

        self.modified_attribute_count += modified

        return attributes
    

    def _derive_amount_category(self, requested_amount: Any) -> str:
        """
        Derives an amount category from a numeric requested amount
        """

        try:
            amount = float(requested_amount)
        except (TypeError, ValueError):
            return "unknown"
        
        if amount < 20000:
            return "low"
        
        if amount < 50000:
            return "medium"
        
        return "high"
    
    def _derive_risk_category(self, credit_score: Any) -> str:
        """
        Derives a risk category from a numeric credit score

        Lower scores are treated as higher risk
        """

        try:
            score = float(credit_score)
        except (TypeError, ValueError):
            return "unknown"
        
        if score < 500:
            return "high risk"
        
        if score < 700:
            return "medium risk"
        
        return "low risk"
    

    def _derive_case_complexity(
            self,
            amount_category: str | None,
            risk_category: str | None,
    ) -> str:
        """
        Derives case complexity from amount and risk categories
        """

        if amount_category == "high" or risk_category == "high risk":
            return "complex"
        
        if amount_category == "unknown" or risk_category == "unknown":
            return "unknown"
        
        return "standard"
    

    def _get_first_available(
            self,
            attributes: dict[str, Any],
            keys: list[str],
    ) -> Any:
        """
        Returns the first available value from a list of possible keys
        """

        for key in keys:
            if key in attributes and attributes[key] is not None:
                return attributes[key]
            
        return None
    

    def _extract_history(self, event: Any) -> list[str]:
        """
        Extracts previous activity history if available
        """

        if event is None:
            return []
        
        if hasattr(event, "history"):
            history = event.history
            if isinstance(history, list):
                return history
            
        if hasattr(event, "getAttribOfLastEvents"):
            try:
                history_attributes = event.getAttribOfLastEvents(-1)
                if isinstance(history_attributes, dict):
                    value = history_attributes.get("history")
                    if isinstance(value, list):
                        return value
            except TypeError:
                pass

        return []


    def fallback(
            self,
            event: Any,
            possibleActivities: list[str],
    ) -> list[str]:
        """
        Uses fallback engine if available

        Otherwise uses random valid choice
        """

        self.fallback_count += 1

        if self.fallback_engine is not None:
            result = self.fallback_engine.getNextActivities(
                event,
                possibleActivities,
            )

            if self._is_valid_result(result, possibleActivities):
                return result
        
        return choose_random_valid_activity(
            possible_activities=possibleActivities,
            random_generator=self.random,
        )
    

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
        
        return all(activity in possibleActivities for activity in result)
    

    def get_sampling_statistics(self) -> dict[str, float | int]:
        """
        Returns simple counters for reporting/evaluation
        """

        return {
            "total_decisions" : self.total_decisions,
            "sampled_attribute_count" : self.sampled_attribute_count,
            "derived_attribute_count" : self.derived_attribute_count,
            "modified_attribute_count" : self.modified_attribute_count,
            "fallback_count" : self.fallback_count,
        }
    

    class EnrichedEventWrapper:
        """
        Wraps an original event and exposes enriched attributes through getAttribs()

        This avoids modifying the original event object directly
        """

        def __init__(
                self,
                original_event: Any,
                enriched_attributes: dict[str, Any],
        ):
            self.original_event = original_event
            self.enriched_attributes = enriched_attributes

            if hasattr(original_event, "activity"):
                self.activity = original_event.activity
            else:
                self.activity = enriched_attributes.get("concept:name")

            if hasattr(original_event, "history"):
                self.history = original_event.history
            else:
                self.history = []

        
        def getAttribs(self) -> dict[str, Any]:
            return self.enriched_attributes
        
        def getAttribOfLastEvents(self, amount: int = -1) -> dict[str, Any]:
            if hasattr(self.original_event, "getAttribOfLastEvents"):
                try:
                    return self.original_event.getAttribOfLastEvents(amount)
                except TypeError:
                    pass

            return {"history" : self.history}
        

    
