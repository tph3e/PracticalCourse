# src/resource_allocation/ResourceAllocationEvaluator.py

from typing import Dict, List, Optional

from .ResourceAllocationMetrics import (
    average_cycle_time,
    average_resource_occupation,
    average_waiting_time,
    resource_fairness,
    weighted_resource_fairness,
)


class ResourceAllocationEvaluator:
    """
    Evaluates resource allocation strategies based on simulator output.

    This class does not run the simulator itself.
    It receives the output logs produced by a simulator run and computes metrics.
    """

    def evaluate(
        self,
        strategy_name: str,
        case_times: List[Dict],
        task_times: List[Dict],
        resource_intervals: List[Dict],
        availability_times: Dict[str, float],
    ) -> Dict[str, Optional[float] | str]:
        """
        Compute all Basic 1.2 metrics for one allocation strategy.

        Args:
            strategy_name: Name of the evaluated allocation strategy.
            case_times: Case-level arrival and completion timestamps.
            task_times: Task-level enabled and start timestamps.
            resource_intervals: Resource busy intervals.
            availability_times: Available time per resource.

        Returns:
            Dictionary containing all evaluation metrics.
        """

        return {
            "strategy": strategy_name,
            "average_cycle_time": average_cycle_time(case_times),
            "average_waiting_time": average_waiting_time(task_times),
            "average_resource_occupation": average_resource_occupation(
                resource_intervals=resource_intervals,
                availability_times=availability_times,
            ),
            "resource_fairness": resource_fairness(
                resource_intervals=resource_intervals,
                availability_times=availability_times,
            ),
            "weighted_resource_fairness": weighted_resource_fairness(
                resource_intervals=resource_intervals,
                availability_times=availability_times,
            ),
        }

    def evaluate_multiple(self, simulation_results: Dict[str, Dict]) -> List[Dict]:
        """
        Evaluate several allocation strategies.

        Expected input format:

        {
            "R-RRA": {
                "case_times": [...],
                "task_times": [...],
                "resource_intervals": [...],
                "availability_times": {...}
            },
            "R-SHQ": {
                ...
            }
        }

        Returns:
            List of dictionaries, one per strategy.
        """

        results = []

        for strategy_name, output in simulation_results.items():
            result = self.evaluate(
                strategy_name=strategy_name,
                case_times=output["case_times"],
                task_times=output["task_times"],
                resource_intervals=output["resource_intervals"],
                availability_times=output["availability_times"],
            )

            results.append(result)

        return results