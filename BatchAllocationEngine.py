import copy

class BatchAllocationEngine:
    def __init__(self, k_limit = 5):
        self.k_limit = k_limit
        self.batch_buffer = []

    def add_to_buffer(self, task, available_resources, current_time):
        self.batch_buffer.append(task)
        print(f"Task '{task.task_type}' added. Buffer: {len(self.batch_buffer)}/{self.k_limit}")

        if len(self.batch_buffer) >= self.k_limit:
            resources_copy = copy.deepcopy(available_resources)
            return self.fire_batch(resources_copy, current_time)
        return []
    
    def fire_batch(self, available_resources, current_time):
        assignments = []
        used_resources = set()
        remaining_tasks = []

        for task in self.batch_buffer:
            assigned = False
            for resource in available_resources:
                if resource not in used_resources and self.is_authorized(resource, task.task_type):
                    assignments.append({
                        "task": task,
                        "resource": resource,
                        "start_time": current_time
                    })
                    used_resources.add(resource)
                    assigned = True
                    break
            if not assigned:
                remaining_tasks.append(task)

        self.batch_buffer = remaining_tasks
        print(f"[Batch] fire complete. Assigned: {len(assignments)}, Left in buffer: {len(self.batch_buffer)}")

        return assignments
    
    def is_authorized(self, resource, task_type) -> bool:
        if hasattr(resource, 'authorized_tasks'):
            return task_type in resource.authorized_tasks
        return True
    
    def flush(self, available_resources, current_time):
        if self.batch_buffer:
            print(f"[Flush] Force-firing {len(self.batch_buffer)} remaining tasks at simulation end.")
            resources_copy = copy.deepcopy(available_resources)
            return self.fire_batch(resources_copy, current_time)
        return []
    