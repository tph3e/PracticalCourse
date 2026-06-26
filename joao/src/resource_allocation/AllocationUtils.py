from collections import Counter
from typing import Dict, List

from .AllocationStrategy import Resource, Task


def get_available_resources(resources: List[Resource]) -> List[Resource]:
    """
    Return only resources that are currently available

    This is important because the assignment requires design decisions for
    uncertain resource availability. Basic heuristics should not assign tasks to unavailable resources
    """
    return [resource for resource in resources if resource.available]


def is_resource_eligible(resource: Resource, task: Task) -> bool:
    """
    Check whether a resource-task pair is feasible

    A pair is feasible if:
    - the resource is currently available
    - the task is not already assigned 
    - the task/case is not blocked
    - the resource is allowed to execute the task activity

    If resource.skills is None, I assume that the reouce can execute all activities
    """

    if not resource.available:
        return False
    
    if task.assigned:
        return False
    
    if task.blocked:
        return False
    
    if resource.skills is None:
        return True
    
    return task.activity in resource.skills


def get_eligible_tasks(resource: Resource, waiting_tasks: List[Task]) -> List[Task]:
    """
    Return all waiting tasks that can be executed by the given resource
    """

    return [
        task for task in waiting_tasks
        if is_resource_eligible(resource, task)
    ]


def compute_activity_queue_lengths(waiting_tasks: List[Task]) -> Dict[str, int]:
    """
    Compute the number of waiting tasks per activity

    Example:
        A, A, B -> {"A": 2, "B": 1}

    Assigned and blocked tasks are ignored
    """

    valid_tasks = [
        task for task in waiting_tasks
        if not task.assigned and not task.blocked
    ]

    return dict(Counter(task.activity for task in valid_tasks))


def mark_task_assigned(waiting_tasks: List[Task], task_id: str) -> None:
    """
    Mark a task as assigned

    This prevents assigning the same task to multiple resources during the same allocation step
    """

    for task in waiting_tasks:
        if task.task_id == task_id:
            task.assigned = True
            return