import random
from typing import Any, List, Optional

from .AllocationStrategy import AllocationDecision, AllocationStrategy, Resource, Task
from .AllocationUtils import (
    get_available_resources,
    get_eligible_tasks,
    mark_task_assigned,
)

class RandomResourceAllocation(AllocationStrategy):
    """
    R-RRA: Random Resource Allocation

    This is a simple pull-based allocation heuristic
    Each available resource randomly selects one eligible waiting task

    Design decision under uncertain resource availability:
    unavailable resources are filtered out before the allocation decision
    """

    def __init__(self, seed: Optional[int] = None):
        self.random = random.Random(seed)

    
    def allocate(
            self,
            resources: List[Resource],
            waiting_tasks: List[Task],
            current_time: float,
            **kwargs: Any
    ) -> List[AllocationDecision]:
        
        decisions: List[AllocationDecision] = []
        available_resources = get_available_resources(resources)

        for resource in available_resources:
            eligible_tasks = get_eligible_tasks(resource, waiting_tasks)

            if not eligible_tasks:
                decisions.append(
                    AllocationDecision(
                        resource_id=resource.resource_id,
                        task_id=None,
                        activity=None,
                        case_id=None,
                        decision_type="idle",
                        reason="No eligible waiting task available."
                    )
                )
                continue

            selected_task = self.random.choice(eligible_tasks)
            mark_task_assigned(waiting_tasks, selected_task.task_id)

            decisions.append(
                AllocationDecision(
                    resource_id=resource.resource_id,
                    task_id=selected_task.task_id,
                    activity=selected_task.activity,
                    case_id=selected_task.case_id,
                    decision_type="assignment",
                    reason="Randomly selected among eligible waiting tasks."
                )
            )

        return decisions
    

    