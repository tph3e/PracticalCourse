from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

@dataclass
class Resource:
    """
    Representation of a resource in the allocation module

    Attributes:
        resource_id: Unique identifier of the resource
        available: Whether the resource is currently available
        skills: List of activities the resource is allowed to execute
                If skills is NOne, the resource is assumed to be able to execute all activities
    """

    resource_id: str
    available: bool = True
    skills: Optional[List[str]] = None


@dataclass
class Task:
    """
    Representation of a currently enabled task

    Attributes:
        task_id: Unique identifier of the task
        case_id: Identifier of the process instance
        activity: Activity name of the task
        enabled_time: Simulation time at which the task became enabled
        assigned: Whether the task has already been assigned to a resource
        blocked: Whether the case/task is currently blocked
        priority: Optional priority value. Higher means more important
    """

    task_id: str
    case_id: str
    activity: str
    enabled_time: float
    assigned: bool = False
    blocked: bool = False
    priority: float = 0.0


@dataclass
class Prediction:
    """
    Representation of a predicted future task

    Used by the Park & Song inspired allocation strategy

    Attributes:
        case_id: Identifier of the process instance
        activity: Predicted next activity
        probability: Probability/confidence of the prediction
        expected_delay: Expected time until the predicted task becomes enabled
        source: Source of the prediction model
        confidence: Optional additional confidence score
    """

    case_id: str
    activity: str
    probability: float
    expected_delay: float
    source: str = "unknown"
    confidence: Optional[float] = None


@dataclass
class AllocationDecision:
    """
    Output of an allocation strategy

    decision_type can be:
        - "assignment": resource is assigned to a current taks
        - "reservation": resource is kept idle for a predicted task
        - "idle": resource remains idle because no feasible assignment exists
    """

    resource_id: str
    task_id: Optional[str]
    activity: Optional[str]
    case_id: Optional[str]
    decision_type: str
    reason: str = ""


class AllocationStrategy(ABC):
    """
    Common interface for all resource allocation strategies
    """

    @abstractmethod
    def allocate(
        self,
        resource: List[Resource],
        waiting_tasks: List[Task],
        current_time: float,
        **kwargs: Any
    ) -> List[AllocationDecision]:
        """
        Allocate available resources to waiting tasks

        Args:
            resources: List of resources in the simulator state
            waiting_tasks: List of currently enabled waiting tasks
            current_time: Current simulation time
            **kwargs: Optional strategy-specific inputs, e.g. predictions

        Returns:
            A list of allocation decisions
        """

        pass