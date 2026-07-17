# Corrected Branching Reproducibility

## Environment

- Python: 3.14.3
- Platform: macOS-26.4-arm64-arm-64bit-Mach-O
- pandas: 3.0.2
- numpy: 2.4.4
- scipy: 1.17.1
- scikit-learn: 1.8.0
- pm4py: 2.7.22.2
- pyarrow: 24.0.0
- joblib: 1.5.3

Pinned package versions for this package are recorded in
`requirements-reproducibility.txt`. The corrected branching artifacts were
trained and exported with scikit-learn 1.8.0. The resource-allocation fixed
replay also loads `joao/models/process_time/final_process_time_coverage_v2.pkl`,
which emits scikit-learn compatibility warnings in the current environment
because that processing-time artifact was trained with a different scikit-learn
version. The warning is documented and the artifact was not retrained in this
branching-only correction.

## Git State

- Branch: `joao/integrate-methods-2026-07-15`
- HEAD: `24a5545ac7e8d170c3ce1a8e2ad6f7d9204dce99`
- Dirty worktree during generation: true

## Required Data and Artifacts

- Event log: `data/logData.xes`
- Event log SHA-256: `d653a36d36fac668638d65c90b803670bab6e599aa23e3f7dd4f0d5d0b216b1c`
- BPMN: `models/v4_replay.bpmn`
- BPMN SHA-256: `329c6d18b42a680cf45ebc3f22e0d1683318f1c638ffdcbd79064385f5c82586`
- Evaluation branching artifact: `joao/models/branching/composite_branching_evaluation_train70.pkl`
- Evaluation artifact SHA-256: `ebb16554ecf0e42faa87c7334faef3ab81b742b8e29820df1ce032bd4109f1c4`
- Deployment branching artifact: `joao/models/branching/composite_branching_deployment_full.pkl`
- Deployment artifact SHA-256: `6dcd01744f635d0fdd24008c3e7c4ae28bedb340c4f37aabbc1bc53fd7e7ab3e`
- Transition-aware artifact: `joao/models/branching/transition_aware_branching_v1_20260715.pkl`
- Transition-aware artifact SHA-256: `79127e6b1cea6cb58fc1f1f19b1ce96564837a24aa5adf80efcbcc71fc183e54`
- Processing-time artifact: `joao/models/process_time/final_process_time_coverage_v2.pkl`
- Processing-time artifact SHA-256: `c540304cdbb6f60159ad1023112e6d5e71aeecae68b885ee2e2b1ac3c826a886`

The BPIC17 event log is external data and is not versioned in this repository.

## Corrected Branching Command

Run from the repository root:

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=joao:. python3 joao/scripts/branching/run_corrected_branching_evaluation.py \
  --log data/logData.xes \
  --bpmn models/v4_replay.bpmn \
  --output-dir joao/results/branching_corrected_20260717 \
  --seed 1
```

This command creates the common BPMN-replay decision dataset, the temporal split
manifest, the corrected method metrics, the evaluation artifact and the
deployment artifact. It does not overwrite the historical
`final_composite_branching.pkl` artifact.

## Corrected Fixed Replay Command

Run from the repository root:

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=joao:. python3 joao/scripts/resource_allocation/run_final_resource_allocation_evaluation.py \
  --data-path data/logData.xes \
  --branching-artifact joao/models/branching/composite_branching_evaluation_train70.pkl \
  --processing-time-artifact joao/models/process_time/final_process_time_coverage_v2.pkl \
  --start 2016-09-16T15:07:00+00:00 \
  --end 2016-09-17T15:07:00+00:00 \
  --drain-until 2016-11-16T15:07:00+00:00 \
  --seeds 1,2,3,4,5 \
  --strategies RoundRobin,ShortestQueue,ParkSong-Composite,Kunkler-Rinderle-Ma,Batch \
  --route-mode fixed-replay \
  --split-ratio 0.7 \
  --parksong-processing-times train-median \
  --reservation-expiration-multiplier 1.0 \
  --output-dir joao/results/final_canonical_branching_corrected_20260717/fixed_replay
```

This rerun was required because the historical fixed replay used a full-log
branching artifact. The corrected rerun keeps the same workload, methods, seeds,
processing-time artifact and ParkSong parameters, but uses the leakage-free
evaluation branching artifact.

## RF Composite Generative Smoke Command

Run from the repository root:

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=joao:. python3 joao/scripts/resource_allocation/run_final_resource_allocation_evaluation.py \
  --data-path data/logData.xes \
  --branching-artifact joao/models/branching/composite_branching_evaluation_train70.pkl \
  --processing-time-artifact joao/models/process_time/final_process_time_coverage_v2.pkl \
  --start 2016-09-16T15:07:00+00:00 \
  --end 2016-09-16T18:07:00+00:00 \
  --drain-until 2016-09-17T02:07:00+00:00 \
  --seeds 1,2,3 \
  --strategies RoundRobin,ShortestQueue,ParkSong-Composite,Kunkler-Rinderle-Ma,Batch \
  --route-mode generative \
  --split-ratio 0.7 \
  --parksong-processing-times train-median \
  --reservation-expiration-multiplier 1.0 \
  --output-dir joao/results/branching_corrected_20260717/generative_rf_composite_smoke
```

This is an integration smoke for RF-composite participation in simulation. It
must not be interpreted as a ranking benchmark.

## Result Reconstruction

- Branching metrics: `branching_method_metrics_aggregated.csv`
- Branching per-seed metrics: `branching_method_metrics_by_seed.csv`
- Decision-point metrics: `branching_per_decision_point_metrics.csv`
- Coverage: `branching_coverage.csv`
- Feature parity diagnostics: `branching_feature_parity_diagnostics.csv`
- Fixed replay rerun: `../final_canonical_branching_corrected_20260717/fixed_replay/`
- Old-vs-corrected comparison: `../final_canonical_branching_corrected_20260717/fixed_replay_old_vs_corrected.csv`

Report tables in `report/sections/branching_eval_joao.tex`,
`report/sections/evaluation_protocol_joao.tex`,
`report/sections/cross_method_joao.tex` and
`report/sections/parksong_eval_joao.tex` are based on these corrected files.
