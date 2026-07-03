# João - Branching Decisions and Resource Allocation Optimization

This folder contains João's individual contribution for the Business Process Prediction, Simulation, and Optimization project.

## Covered tasks

### Part I - Task 1.5 Branching Decisions

Implemented approaches:

1. ProbabilityBranchingEngine
2. AttributeBasedBranchingEngine
3. AttributeSamplingBranchingEngine
4. PredictiveBranchingEngine

The final selected validation model is the PredictiveBranchingEngine with a temporal case-based 70/30 split.  
After model selection, a full deployment model was trained on the complete event log and used in the simulator.

### Part II - 1.1 Basic Resource Allocation

Implemented basic resource allocation heuristics:

1. R-RRA - Random Resource Allocation
2. R-SHQ - Shortest Queue Allocation

Design decision for uncertain resource availability:
Unavailable resources are filtered before allocation. Tasks that are already assigned, blocked, or incompatible with resource skills are excluded.

### Part II - Advanced Resource Allocation

Implemented a Park & Song-inspired prediction-based allocation approach:

- ParkSongAllocation
- MLPredictionAdapter
- ParkSongMLIntegration

The ML model provides predicted future task candidates. The allocation strategy can assign current tasks, reserve resources for predicted tasks, or keep resources idle.

### Part II - 1.2 Evaluation

Implemented evaluation metrics:

- average_cycle_time
- average_waiting_time
- average_resource_occupation
- resource_fairness
- weighted_resource_fairness

## Folder structure

- `src/branching/`: branching decision engines
- `src/resource_allocation/`: allocation strategies, ParkSong integration, and evaluation metrics
- `scripts/branching/`: training and report export scripts
- `scripts/resource_allocation/`: allocation evaluation scripts
- `scripts/simulation/`: simulator integration script
- `tests/`: unit tests
- `results/`: generated CSV results

## Notes

Large `.pkl` model files are not committed to Git. They can be regenerated using the training scripts.
