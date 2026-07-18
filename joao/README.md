# João - Branching Decisions and Resource Allocation Optimization

This folder contains João's individual contribution for the Business Process Prediction, Simulation, and Optimization project.

## Covered tasks

### Part I - Task 1.5 Branching Decisions

Implemented approaches:

1. ProbabilityBranchingEngine
2. PredictiveBranchingEngine
3. CompositeBranchingEngine

The final branching scope is ProbabilityBranching as the Basic method,
PredictiveBranching with a Random Forest as the Advanced-II method, and a
CompositeBranching runtime chain:

`Predictive Random Forest -> ProbabilityBranching fallback -> random BPMN-valid fallback`

The corrected final evaluation uses a leakage-free temporal case-based 70/30
split and the optimized evaluation artifact
`models/branching/composite_branching_evaluation_train70_rfopt_v1.pkl`. A
separate deployment artifact,
`models/branching/composite_branching_deployment_full_rfopt_v1.pkl`, is trained
on the full log but is not used for held-out metrics.

### Part II - 1.1 Basic Resource Allocation

Implemented basic resource allocation heuristics:

1. R-RRA - Resource-aware Round Robin Allocation
2. R-SHQ - Shortest Queue Allocation

Design decision for uncertain resource availability:
Unavailable resources are filtered before allocation. Tasks that are already assigned, blocked, or incompatible with resource skills are excluded.

### Part II - Advanced Resource Allocation

Implemented a Park & Song-inspired prediction-based allocation approach:

- ParkSongAllocation
- MLPredictionAdapter
- ParkSongMLIntegration

The ML model provides predicted future task candidates. The allocation strategy can assign current tasks, reserve resources for predicted tasks, or keep resources idle.
In the canonical fixed replay this is reported as a Park & Song-inspired
rolling-epoch approximation with reservations, not as the original temporal
optimization formulation.

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

The corrected canonical packages are:

- `results/branching_corrected_20260717/`
- `results/final_canonical_branching_corrected_20260717/`

The older `results/final_canonical_20260716/` package and
`models/branching/final_composite_branching.pkl` artifact are historical
pre-correction material only. They should not be used to reconstruct the final
tables.

## Full Method Audit

This repository section contains João's branching and resource-allocation subsystem.

Scope: branching engines, Random baseline, RoundRobin/R-RRA, ShortestQueue/R-SHQ, ParkSong, ParkSongML prediction adapter, Batch comparison adapter, integrated allocation engine, tests, and controlled audit outputs.

Key commands:

```bash
PYTHONPATH=joao python3 -m pytest joao/tests
python3 joao/scripts/resource_allocation/run_full_method_audit.py
```

Artifacts:

- `joao/models/branching/composite_branching_evaluation_train70_rfopt_v1.pkl`
- `joao/models/branching/composite_branching_deployment_full_rfopt_v1.pkl`
- `joao/models/branching/final_composite_branching_sklearn190.pkl`
- `joao/models/process_time/final_process_time_coverage_v2.pkl`
- `joao/results/full_method_audit/`

Known limitations: ParkSong lifecycle is integration-owned; ParkSongML is a prediction supplier plus ParkSong allocator; Künstler/Küncler has no production class in this repository scan; final experimental performance requires separate validated integrated runs.
