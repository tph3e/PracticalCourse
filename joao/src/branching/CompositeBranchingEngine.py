from __future__ import annotations

from typing import Any
import random

from .BranchingUtils import choose_random_valid_activity


class CompositeBranchingEngine:
    """
    Composite branching engine.

    Default hierarchy:
        1. PredictiveBranchingEngine
        2. AttributeSamplingBranchingEngine
        3. AttributeBasedBranchingEngine
        4. ProbabilityBranchingEngine
        5. Random BPMN-valid fallback

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
        train_on_init: bool = False,
    ):
        self.seed = seed
        self.random = random.Random(seed)

        self.total_decisions = 0
        self.engine_success_counts: dict[str, int] = {}
        self.engine_failure_counts: dict[str, int] = {}
        self.invalid_result_counts: dict[str, int] = {}
        self.random_fallback_count = 0
        self.default_build_errors: dict[str, str] = {}

        if engines is not None:
            self.engines = engines
        elif use_default_hierarchy:
            self.engines = self._build_default_hierarchy(
                log=log,
                seed=seed,
                train_on_init=train_on_init,
            )
        else:
            self.engines = []

    def _build_default_hierarchy(
        self,
        log: Any | None,
        seed: int,
        train_on_init: bool,
    ) -> list[Any]:
        """
        Builds the default fallback hierarchy.

        The method is intentionally robust:
        - If one engine cannot be imported or initialized, it is skipped.
        - The simulation can still continue with the remaining engines.
        - If no engine can be built, the random BPMN-valid fallback is still available.
        """

        hierarchy: list[tuple[str, str]] = [
            ("PredictiveBranchingEngine", ".PredictiveBranchingEngine"),
            ("AttributeSamplingBranchingEngine", ".AttributeSamplingBranchingEngine"),
            ("AttributeBasedBranchingEngine", ".AttributeBasedBranchingEngine"),
            ("ProbabilityBranchingEngine", ".ProbabilityBranchingEngine"),
        ]

        engines: list[Any] = []

        for class_name, module_name in hierarchy:
            try:
                engine_class = self._import_engine_class(
                    module_name=module_name,
                    class_name=class_name,
                )

                engine = self._instantiate_engine(
                    engine_class=engine_class,
                    log=log,
                    seed=seed,
                )

                if train_on_init and log is not None and hasattr(engine, "train"):
                    engine.train(log)

                engines.append(engine)

            except Exception as exc:
                self.default_build_errors[class_name] = str(exc)

        return engines

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

        if not possibleActivities:
            return []

        if len(possibleActivities) == 1:
            return possibleActivities

        self.total_decisions += 1

        for engine in self.engines:
            engine_name = engine.__class__.__name__

            try:
                result = engine.getNextActivities(event, possibleActivities)
            except Exception:
                self.engine_failure_counts[engine_name] = (
                    self.engine_failure_counts.get(engine_name, 0) + 1
                )
                continue

            if self._is_valid_result(result, possibleActivities):
                self.engine_success_counts[engine_name] = (
                    self.engine_success_counts.get(engine_name, 0) + 1
                )
                return result

            self.invalid_result_counts[engine_name] = (
                self.invalid_result_counts.get(engine_name, 0) + 1
            )

        self.random_fallback_count += 1

        return choose_random_valid_activity(
            possible_activities=possibleActivities,
            random_generator=self.random,
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
            "random_fallback_count": self.random_fallback_count,
            "configured_engines": [
                engine.__class__.__name__ for engine in self.engines
            ],
            "default_build_errors": self.default_build_errors,
        }