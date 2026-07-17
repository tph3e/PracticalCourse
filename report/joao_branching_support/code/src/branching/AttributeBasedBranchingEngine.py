from __future__ import annotations

from typing import Any
import random
import operator
import math

from .BranchDecision import BranchDecision
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
        self.rules = self._validate_rules(rules or [])
        self.fallback_engine = fallback_engine
        self.seed = seed
        self.random = random.Random(seed)
        self.activity_col = activity_col

        self.rule_matches = 0
        self.fallback_count = 0
        self.total_decisions = 0
        self.rule_matches_by_id: dict[str, int] = {}
        self.rule_missing_attribute_count = 0
        self.rule_invalid_type_count = 0
        self.rule_candidate_rejected_count = 0

    
    def getNextActivities(
            self,
            event: Any,
            possibleActivities: list[str],
    ) -> list[str]:
        """
        Selects the next activity using attribute-based rules

        The selected activity must always be BPMN-valid
        """

        decision = self.decide(event, possibleActivities)
        if decision is not None:
            return decision.activities
        return self.fallback(event, possibleActivities)

    def decide(
        self,
        event: Any,
        possibleActivities: list[str],
        context: dict[str, Any] | None = None,
    ) -> BranchDecision | None:
        if not possibleActivities:
            return BranchDecision(
                activities=[],
                decision_source="attribute_rule_empty_candidates",
                candidate_activities=[],
                used_fallback=True,
            )

        if len(possibleActivities) == 1:
            return BranchDecision(
                activities=possibleActivities,
                decision_source="single_bpmn_candidate",
                confidence=1.0,
                candidate_activities=list(possibleActivities),
            )

        self.total_decisions += 1

        current_activity = extract_current_activity(
            event,
            activity_col=self.activity_col,
        )

        attributes = self.extract_attributes(event)

        selected = self.apply_rules(
            current_activity=current_activity,
            attributes=attributes,
            possibleActivities=possibleActivities,
        )

        if selected is not None:
            self.rule_matches += 1
            self.rule_matches_by_id[selected["rule_id"]] = (
                self.rule_matches_by_id.get(selected["rule_id"], 0) + 1
            )
            return BranchDecision(
                activities=[selected["activity"]],
                decision_source="attribute_rule",
                probability_source=None,
                probabilities=None,
                confidence=None,
                support=None,
                used_fallback=False,
                candidate_activities=list(possibleActivities),
                metadata={"rule_id": selected["rule_id"]},
            )

        return None
    

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
    ) -> dict[str, str] | None:
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

            attribute_name = rule["attribute"]
            operator_symbol = rule["operator"]
            expected_value = rule["value"]
            preferred_activities = rule["preferred_activities"]

            actual_value = attributes.get(attribute_name)

            if self._is_missing(actual_value):
                self.rule_missing_attribute_count += 1
                continue

            try:
                condition_matches = self.evaluate_condition(
                    actual_value=actual_value,
                    operator_symbol=operator_symbol,
                    expected_value=expected_value,
                )
            except TypeError:
                self.rule_invalid_type_count += 1
                continue

            if not condition_matches:
                continue

            for preferred_activity in preferred_activities:
                if preferred_activity in possibleActivities:
                    return {"activity": preferred_activity, "rule_id": rule["rule_id"]}
                self.rule_candidate_rejected_count += 1
            
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

        coerced_actual, coerced_expected = self._coerce_values(actual_value, expected_value)
        return bool(operation(coerced_actual, coerced_expected))

    def _validate_rules(self, rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
        validated = []
        for index, rule in enumerate(rules):
            operator_symbol = rule.get("operator")
            if operator_symbol not in self.SUPPORTED_OPERATORS:
                raise ValueError(f"Unsupported operator in rule {index}: {operator_symbol}")
            preferred = rule.get("preferred_activities", [])
            if not isinstance(preferred, list):
                preferred = [preferred]
            preferred = [str(activity) for activity in preferred if activity]
            if not preferred:
                raise ValueError(f"Rule {index} has no preferred activities.")
            if not rule.get("attribute"):
                raise ValueError(f"Rule {index} has no attribute.")
            validated.append(
                {
                    **rule,
                    "rule_id": str(rule.get("rule_id") or f"rule_{index}_{rule.get('attribute')}_{operator_symbol}"),
                    "preferred_activities": preferred,
                }
            )
        return validated

    def _is_missing(self, value: Any) -> bool:
        if value is None:
            return True
        try:
            return bool(math.isnan(value))
        except (TypeError, ValueError):
            return False

    def _coerce_values(self, actual: Any, expected: Any) -> tuple[Any, Any]:
        if isinstance(expected, bool) and isinstance(actual, str):
            lowered = actual.strip().lower()
            if lowered in {"true", "1", "yes"}:
                return True, expected
            if lowered in {"false", "0", "no"}:
                return False, expected
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            try:
                return float(actual), float(expected)
            except (TypeError, ValueError):
                raise TypeError("Cannot coerce numeric rule value.")
        if isinstance(expected, list) and not isinstance(actual, list):
            return actual, expected
        return actual, expected
        
    
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

        if len(result) == 0:
            return False

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

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "total_decisions": self.total_decisions,
            "rule_matches": self.rule_matches,
            "rule_matches_by_id": dict(self.rule_matches_by_id),
            "rule_missing_attribute_count": self.rule_missing_attribute_count,
            "rule_invalid_type_count": self.rule_invalid_type_count,
            "rule_candidate_rejected_count": self.rule_candidate_rejected_count,
            "rule_coverage": self.get_rule_coverage(),
            "fallback_count": self.fallback_count,
        }
    
