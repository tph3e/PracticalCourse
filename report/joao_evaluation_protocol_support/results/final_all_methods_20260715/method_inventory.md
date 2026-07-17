# Method Inventory

All methods allocate resources/tasks only. None selects BPMN routes; branching remains separate and is enforced by the BPMN/transition-aware path.

## RoundRobin
- Source: `joao/src/resource_allocation/RoundRobinResourceAllocation.py`
- Objective/semantics: pull/global deterministic rotation
- Decision inputs: eligible resources,current waiting tasks
- Simulator path: IntegratedAllocationEngine
- Existing tests: joao/tests/resource_allocation/test_round_robin_resource_allocation.py
- Status: included
- Notes: Deterministic resource-id rotation over currently available eligible resources.

## ShortestQueue
- Source: `joao/src/resource_allocation/ShortestQueueAllocation.py`
- Objective/semantics: pull/global load-balancing
- Decision inputs: eligible resources,current waiting tasks,resource loads
- Simulator path: IntegratedAllocationEngine
- Existing tests: joao/tests/resource_allocation/test_shortest_queue_allocation.py
- Status: included
- Notes: Uses ResourceEngine cumulative load as the current queue/load proxy.

## ParkSong-Composite
- Source: `joao/src/resource_allocation/ParkSongAllocation.py;joao/src/resource_allocation/integration/CompositeBranchingAdapter.py`
- Objective/semantics: prediction-aware,reservation-based
- Decision inputs: eligible resources,current waiting tasks,Composite predictions
- Simulator path: IntegratedAllocationEngine
- Existing tests: joao/tests/resource_allocation/test_integrated_allocation_engine.py
- Status: included
- Notes: Consumes current CompositeBranchingAdapter predictions and creates reservations.

## Batch
- Source: `BatchAllocationEngine.py;joao/src/resource_allocation/BatchAllocationAdapter.py`
- Objective/semantics: batch/current-queue snapshot
- Decision inputs: eligible resources,current waiting tasks,k_limit
- Simulator path: IntegratedAllocationEngine via BatchAllocationAdapter
- Existing tests: joao/tests/resource_allocation/test_batch_allocation_adapter.py
- Status: included
- Notes: Adapter preserves group engine assignment rule but fires per integrated decision epoch.

## Kunkler-Rinderle-Ma
- Source: `resourceAllocation_KunklerRinderleMa.py;joao/src/resource_allocation/KunklerAllocationAdapter.py;notebooks/2.3.1_formalization_kunkler.ipynb`
- Objective/semantics: anticipatory assignment/cost matrix
- Decision inputs: eligible resources,current waiting tasks,processing-time quantile shim
- Simulator path: IntegratedAllocationEngine via KunklerAllocationAdapter
- Existing tests: joao/tests/resource_allocation/test_kunkler_allocation_adapter.py
- Status: included
- Notes: Adapter invokes the original allocator and enforces integrated eligibility/assignment semantics; root class cost-matrix limitations are diagnosed.
