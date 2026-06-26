from __future__ import annotations

from typing import Any
import random
import operator

from .BranchingUtils import (
    choose_random_valid_activity,
    extract_current_activity,
)


class AttributeBasedBranchingEngine:
    """
    Attribute-based branching engine

    This engine applies configurable rules based on runtime, event and case attribute

    It always respects BPMN constraints by only returning activities contained in possibleActivities
    """

    SUPPORTED_OPERATORS = {
        "==" : operator.eq,
        "!=" : operator.ne,
        ">" : operator.gt,
        ">=" : operator.ge,
        "<" : operator.lt,
        "<=" : operator.le,
        "in" : lambda actual, expected: actual in expected,
        "not in" : lambda actual, expected: actual not in expected,
    }

    
    def __init__(
            self,
            rules: list[dict[str, Any]] | None = None,
            fallback_engine: Any | None = None,
            seed: int = 1,
            activity_col: str = "concept:name"
    ):
        self.rules = rules or []
        self.fallback_engine = fallback_engine
        self.seed = seed
        self.random = random.Random(seed)
        self.activity_col = activity_col

        self.rule_matches = 0
        self.fallback_count = 0
        self.total_decisions = 0

    
    def getNextActivities(
            self,
            event: Any,
            possibleActivities: list[str],
    ) -> list[str]:
        """
        Selects the next activity using attribute-based rules

        The selected activity must always be BPMN-valid
        """

        if not possibleActivities:
            return []
        
        if len(possibleActivities) == 1:
            return possibleActivities
        
        self.total_decisions += 1

        current_activity = extract_current_activity(
            event,
            activity_col=self.activity_col,
        )

        attributes = self.extract_attributes(event)

        selected_activity = self.apply_rules(
            current_activity=current_activity,
            attributes=attributes,
            possibleActivities=possibleActivities,
        )

        if selected_activity is not None:
            self.rule_matches += 1
            return selected_activity
        
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
            attributes.setdefault(self.activity_col, event.activity)
            attributes.setdefault("activity", event.activity)

        return attributes
    

    def apply_rules(
            self,
            current_activity: str | None,
            attributes: dict[str, Any],
            possibleActivities: list[str],
    ) -> str | None:
        """
        Applies rules for the current activity

        Returns the first BPMN-valid preferred activity whose condition is true
        """

        if current_activity is None:
            return None
        
        for rule in self.rules:
            decision_point = rule.get("decision_point")

            if decision_point is not None and decision_point != current_activity:
                continue

            attribute_name = rule.get("attribute")
            operator_symbol = rule.get("operator")
            expected_value = rule.get("value")
            preferred_activities = rule.get("preferred_activities", [])

            if not isinstance(preferred_activities, list):
                preferred_activities = [preferred_activities]

            actual_value = attributes.get(attribute_name)

            if actual_value is None:
                continue

            condition_matches = self.evaluate_condition(
                actual_value=actual_value,
                operator_symbol=operator_symbol,
                expected_value=expected_value,
            )

            if not condition_matches:
                continue

            for preferred_activity in preferred_activities:
                if preferred_activity in possibleActivities:
                    return [preferred_activity]
            
        return None 
    

    def evaluate_condition(
            self,
            actual_value: Any,
            operator_symbol: str,
            expected_value: Any,
    ) -> bool:
        """
        Evaluates a rule condition
        """

        if operator_symbol not in self.SUPPORTED_OPERATORS:
            raise ValueError(f"Unsupported operator: {operator_symbol}")
        
        operation = self.SUPPORTED_OPERATORS[operator_symbol]

        try:
            return bool(operation(actual_value, expected_value))
        except TypeError:
            return False
        
    
    def fallback(
            self,
            event: Any,
            possibleActivities: list[str],
    ) -> list[str]:
        """
        Uses fallback engine if available

        Otherwise selects a random BPMN-valid activity
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
        Checks whether a fallback result is BPMN-valid
        """

        if not isinstance(result, list):
            return False
        
        if not possibleActivities:
            return result == []
        
        return all(activity in possibleActivities for activity in result)
    

    def get_rule_coverage(self) -> float:
        """
        Returns the share of decisions handled by attribute-based rules
        """

        if self.total_decisions == 0:
            return 0.0
        
        return self.rule_matches / self.total_decisions
    
    
    def get_fallback_rate(self) -> float:
        """
        Returns the share of decisions handled by fallback
        """

        if self.total_decisions == 0:
            return 0.0
        
        return self.fallback_count / self.total_decisions
    
