from typing import Any, List

from .AllocationStrategy import AllocationDecision, AllocationStrategy, Resource, Task
from .AllocationUtils import (
    compute_activity_queue_lengths,
    get_available_resources,
    get_eligible_tasks,
    mark_task_assigned,
)


class ShortestQueueAllocation(AllocationStrategy):
    """
    R-SHQ: Shortest Queue Heurisstic

    This is a simple pull-based allocation heuristic
    Each available resource selects an eligible task from the activity queue with the smallest number of currently waiting tasks

    Tie-breaking:
    1. shortest activity queue
    2. oldest enabled task
    3. highest task priority
    """

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
                        reason="No eligible waiting task available"
                    )
                )
                continue

            queue_lengths = compute_activity_queue_lengths(waiting_tasks)

            selected_task = min(
                eligible_tasks,
                key=lambda task: (
                    queue_lengths.get(task.activity, 0),
                    task.enabled_time,
                    -task.priority
                )
            )

            mark_task_assigned(waiting_tasks, selected_task.task_id)

            decisions.append(
                AllocationDecision(
                    resource_id=resource.resource_id,
                    task_id=selected_task.task_id,
                    activity=selected_task.activity,
                    case_id=selected_task.case_id,
                    decision_type="assignment",
                    reason="Selected task from shortest eligible activity queue"
                )
            )

        return decisions
    

    