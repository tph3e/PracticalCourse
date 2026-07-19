import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Any, List

from joao.src.resource_allocation.AllocationStrategy import AllocationDecision, AllocationStrategy, Resource, Task
from joao.src.resource_allocation.AllocationUtils import (
    compute_activity_queue_lengths,
    get_available_resources,
    get_eligible_tasks,
    mark_task_assigned,
)

class AnticipatoryAssignmentAllocator(AllocationStrategy):
    def __init__(self, processing_time_model, task_prediction_model, resource_model, wait_penalty_weight=1.0, delta=1.0):
        self.processTimeEngine = processing_time_model
        self.task_prediction_model = task_prediction_model
        self.resourceModel = resource_model
        self.w = wait_penalty_weight
        self.delta=delta
        self.first_phase = True
        self.usePhases = False

    def allocate(self, resource: List[Resource], waiting_tasks: List[Task], current_time: float, **kwargs: Any):
        count_allocation=0.0
        count_no_allocation=0.0
        
        available_resources = get_available_resources(resource)
        if not available_resources or not waiting_tasks:
            return []
        cost_matrix = np.zeros((len(available_resources), len(waiting_tasks)))

        all_resources =  []
        dummy_resources = []
        
        for i, spec_resource in enumerate(all_resources+dummy_resources):
            for j, task in enumerate(waiting_tasks):
                # Base performance cost
                p_rt = self.processTimeEngine.getMedian(task.activity, spec_resource.resource_id)*self.delta
                
                if task in waiting_tasks:
                    cost_matrix[i, j] = p_rt
                else:
                    # Apply the weight 'w' to the expected waiting time
                    wait_time = max(0, task['enabled_time'] - current_time)
                    cost_matrix[i, j] = p_rt + (self.w * wait_time)

        #solve the assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        #allocate the processes according to the solved assignment problem
        allocations: List[AllocationDecision] = []
        for i, j in zip(row_ind, col_ind):
            spec_resource = available_resources[i]
            assigned_task = waiting_tasks[j]
            
            if assigned_task in waiting_tasks:
                allocations.append(
                AllocationDecision(
                    resource_id=spec_resource.resource_id,
                    task_id=assigned_task.task_id,
                    activity=assigned_task.activity,
                    case_id=assigned_task.case_id,
                    decision_type="assignment",
                    reason="Selected task from Kunkler Rinderle-Ma"
                ))
            else:
                allocations.append(
                AllocationDecision(
                    resource_id=spec_resource.resource_id,
                    task_id=assigned_task.task_id,
                    activity=assigned_task.activity,
                    case_id=assigned_task.case_id,
                    decision_type="idle",
                    reason="Selected task from Kunkler Rinderle-Ma"
                ))
        if self.usePhases:
            if self.first_phase:
                if count_allocation/(count_allocation+count_no_allocation)>0.8:
                    self.delta=1.1*self.delta
                else:
                    self.first_phase=False
            elif count_allocation/(count_allocation+count_no_allocation)>0.95:
                self.delta=self.delta+0.005
            elif count_allocation/(count_allocation+count_no_allocation)<0.5:
                self.delta=self.delta*0.9

                
        return allocations