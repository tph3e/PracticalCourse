from __future__ import annotations

from typing import Any
import random

import pandas as pd

from .BranchDecision import BranchDecision
from .BranchingUtils import choose_random_valid_activity


class CompositeBranchingEngine:
    """
    Composite branching engine.

    Default hierarchy:
        1. PredictiveBranchingEngine
        2. ProbabilityBranchingEngine
        3. Random BPMN-valid fallback

    Legacy branching engines remain importable for backwards compatibility, but
    they are not part of the default hierarchy or the final evaluated composite.

    The simulation core only calls:

        getNextActivities(event, possibleActivities)

    The composite engine tries each internal engine until one returns a BPMN-valid result.
    If no internal engine works, it falls back to a random BPMN-valid activity.
    """

    def __init__(
        self,
        engines: list[Any] | None = None,
        log: Any | None = None,
        seed: int = 1,
        use_default_hierarchy: bool = True,
        train_on_init: bool | None = None,
        predictive_use_bpmn_replay: bool = False,
        bpmn_model_path: str = "models/v4_replay.bpmn",
    ):
        self.seed = seed
        self.random = random.Random(seed)

        self.total_decisions = 0
        self.engine_success_counts: dict[str, int] = {}
        self.engine_failure_counts: dict[str, int] = {}
        self.invalid_result_counts: dict[str, int] = {}
        self.random_fallback_count = 0
        self.default_build_errors: dict[str, str] = {}
        self.engine_abstention_counts: dict[str, int] = {}
        self.decision_trace: list[dict[str, Any]] = []

        if engines is not None:
            self.engines = engines
        elif use_default_hierarchy:
            if train_on_init is None:
                train_on_init = log is not None

            self.engines = self._build_default_hierarchy(
                log=log,
                seed=seed,
                train_on_init=train_on_init,
                predictive_use_bpmn_replay=predictive_use_bpmn_replay,
                bpmn_model_path=bpmn_model_path,
            )
        else:
            self.engines = []

    def _build_default_hierarchy(
        self,
        log: Any | None,
        seed: int,
        train_on_init: bool,
        predictive_use_bpmn_replay: bool,
        bpmn_model_path: str,
    ) -> list[Any]:
        """
        Builds the default fallback hierarchy.

        The method is intentionally robust:
        - If one engine cannot be imported or initialized, it is skipped.
        - The simulation can still continue with the remaining engines.
        - If no engine can be built, the random BPMN-valid fallback is still available.
        """

        engines: list[Any] = []

        probability_engine = self._build_probability_engine(log, seed, train_on_init)
        attribute_engine = None
        sampling_engine = None
        predictive_engine = self._build_predictive_engine(
            log=log,
            seed=seed,
            train_on_init=train_on_init,
            fallback_engine=probability_engine,
            use_bpmn_replay=predictive_use_bpmn_replay,
            bpmn_model_path=bpmn_model_path,
        )

        for engine in [
            predictive_engine,
            probability_engine,
        ]:
            if engine is not None:
                engines.append(engine)

        return engines

    def _build_probability_engine(
        self,
        log: Any | None,
        seed: int,
        train_on_init: bool,
    ) -> Any | None:
        try:
            engine_class = self._import_engine_class(
                module_name=".ProbabilityBranchingEngine",
                class_name="ProbabilityBranchingEngine",
            )
            engine = engine_class(log=log, seed=seed)

            if train_on_init and log is not None:
                engine.train(log)

            return engine
        except Exception as exc:
            self.default_build_errors["ProbabilityBranchingEngine"] = str(exc)
            return None

    def _build_attribute_engine(
        self,
        seed: int,
        fallback_engine: Any | None,
    ) -> Any | None:
        try:
            engine_class = self._import_engine_class(
                module_name=".AttributeBasedBranchingEngine",
                class_name="AttributeBasedBranchingEngine",
            )
            return engine_class(
                rules=[],
                fallback_engine=fallback_engine,
                seed=seed,
            )
        except Exception as exc:
            self.default_build_errors["AttributeBasedBranchingEngine"] = str(exc)
            return None

    def _build_sampling_engine(
        self,
        seed: int,
        attribute_engine: Any | None,
        fallback_engine: Any | None,
    ) -> Any | None:
        if attribute_engine is None:
            self.default_build_errors["AttributeSamplingBranchingEngine"] = (
                "AttributeBasedBranchingEngine is unavailable."
            )
            return None

        try:
            engine_class = self._import_engine_class(
                module_name=".AttributeSamplingBranchingEngine",
                class_name="AttributeSamplingBranchingEngine",
            )
            return engine_class(
                base_engine=attribute_engine,
                sampling_config={},
                fallback_engine=fallback_engine,
                seed=seed,
            )
        except Exception as exc:
            self.default_build_errors["AttributeSamplingBranchingEngine"] = str(exc)
            return None

    def _build_predictive_engine(
        self,
        log: Any | None,
        seed: int,
        train_on_init: bool,
        fallback_engine: Any | None,
        use_bpmn_replay: bool,
        bpmn_model_path: str,
    ) -> Any | None:
        try:
            engine_class = self._import_engine_class(
                module_name=".PredictiveBranchingEngine",
                class_name="PredictiveBranchingEngine",
            )
            engine = engine_class(
                fallback_engine=fallback_engine,
                feature_columns=self._default_feature_columns(log),
                seed=seed,
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=2,
                use_bpmn_replay=use_bpmn_replay,
                bpmn_model_path=bpmn_model_path,
            )

            if train_on_init and log is not None:
                engine.train(log)

            return engine
        except Exception as exc:
            self.default_build_errors["PredictiveBranchingEngine"] = str(exc)
            return None

    def _default_feature_columns(self, log: Any | None) -> list[str]:
        candidate_features = [
            "case:ApplicationType",
            "case:LoanGoal",
            "case:RequestedAmount",
            "CreditScore",
            "EventOrigin",
            "org:resource",
        ]

        if isinstance(log, pd.DataFrame):
            return [
                column for column in candidate_features
                if column in log.columns
            ]

        return candidate_features

    def _import_engine_class(self, module_name: str, class_name: str) -> type:
        """
        Imports an engine class from the current branching package.
        """

        import importlib

        package = __package__
        module = importlib.import_module(module_name, package=package)
        return getattr(module, class_name)

    def _instantiate_engine(
        self,
        engine_class: type,
        log: Any | None,
        seed: int,
    ) -> Any:
        """
        Instantiates an engine while supporting different constructor styles.

        This is useful because the individual branching engines may not all use
        the exact same __init__ signature.
        """

        attempts = [
            lambda: engine_class(log=log, seed=seed),
            lambda: engine_class(log, seed),
            lambda: engine_class(seed=seed),
            lambda: engine_class(log=log),
            lambda: engine_class(log),
            lambda: engine_class(),
        ]

        last_error: Exception | None = None

        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_error = exc

        raise TypeError(
            f"Could not instantiate {engine_class.__name__}. "
            f"Last error: {last_error}"
        )

    def getNextActivities(
        self,
        event: Any,
        possibleActivities: list[str],
    ) -> list[str]:
        """
        Selects the next activity using the first engine that returns a BPMN-valid result.
        """

        decision = self.decide(event, possibleActivities)
        return decision.activities if decision is not None else []

    def decide(
        self,
        event: Any,
        possibleActivities: list[str],
        context: dict[str, Any] | None = None,
    ) -> BranchDecision:
        if not possibleActivities:
            return BranchDecision(
                activities=[],
                decision_source="composite_empty_candidates",
                candidate_activities=[],
                used_fallback=True,
            )

        if len(possibleActivities) == 1:
            return BranchDecision(
                activities=possibleActivities,
                decision_source="single_bpmn_candidate",
                probability_source="single_candidate",
                probabilities={possibleActivities[0]: 1.0},
                confidence=1.0,
                candidate_activities=list(possibleActivities),
            )

        self.total_decisions += 1
        attempts: list[dict[str, Any]] = []

        for engine in self.engines:
            engine_name = engine.__class__.__name__

            try:
                if hasattr(engine, "decide"):
                    decision = engine.decide(event, possibleActivities, context=context)
                    if decision is None:
                        self.engine_abstention_counts[engine_name] = (
                            self.engine_abstention_counts.get(engine_name, 0) + 1
                        )
                        attempts.append({"engine": engine_name, "status": "abstained"})
                        continue
                    result = decision.activities
                else:
                    result = engine.getNextActivities(event, possibleActivities)
                    decision = BranchDecision(
                        activities=result,
                        decision_source=engine_name,
                        used_fallback=False,
                        candidate_activities=list(possibleActivities),
                    )
            except Exception as exc:
                self.engine_failure_counts[engine_name] = (
                    self.engine_failure_counts.get(engine_name, 0) + 1
                )
                attempts.append({"engine": engine_name, "status": "failed", "error": str(exc)})
                continue

            if self._is_valid_result(result, possibleActivities):
                self.engine_success_counts[engine_name] = (
                    self.engine_success_counts.get(engine_name, 0) + 1
                )
                decision.metadata = {
                    **decision.metadata,
                    "composite_attempts": attempts,
                    "selected_engine": engine_name,
                }
                self.decision_trace.append(
                    {
                        "decision_source": decision.decision_source,
                        "probability_source": decision.probability_source,
                        "engine": engine_name,
                        "used_fallback": decision.used_fallback,
                    }
                )
                return decision

            self.invalid_result_counts[engine_name] = (
                self.invalid_result_counts.get(engine_name, 0) + 1
            )
            attempts.append({"engine": engine_name, "status": "invalid_result"})

        self.random_fallback_count += 1
        selected = choose_random_valid_activity(
            possible_activities=possibleActivities,
            random_generator=self.random,
        )
        return BranchDecision(
            activities=selected,
            decision_source="composite_random_fallback",
            probability_source="random_fallback",
            probabilities={activity: 1 / len(possibleActivities) for activity in possibleActivities},
            confidence=1 / len(possibleActivities),
            used_fallback=True,
            candidate_activities=list(possibleActivities),
            metadata={"composite_attempts": attempts},
        )

    def add_engine(self, engine: Any) -> None:
        """
        Adds a branching engine manually to the fallback hierarchy.
        """

        self.engines.append(engine)

    def _is_valid_result(
        self,
        result: list[str],
        possibleActivities: list[str],
    ) -> bool:
        """
        Checks whether a result is BPMN-valid.
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
        Returns statistics for reporting and debugging.
        """

        return {
            "total_decisions": self.total_decisions,
            "engine_success_counts": self.engine_success_counts,
            "engine_failure_counts": self.engine_failure_counts,
            "invalid_result_counts": self.invalid_result_counts,
            "engine_abstention_counts": self.engine_abstention_counts,
            "random_fallback_count": self.random_fallback_count,
            "configured_engines": [
                engine.__class__.__name__ for engine in self.engines
            ],
            "default_build_errors": self.default_build_errors,
        }
