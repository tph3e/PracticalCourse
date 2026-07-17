# Branching-Corrected Fixed Replay

This directory contains the fixed-replay rerun produced after replacing the
historical full-log branching artifact with the leakage-free evaluation artifact.
It is the corrected resource-allocation comparison for report claims that depend
on branching.

## Scope

- Corrected fixed replay: `joao/results/final_canonical_branching_corrected_20260717/fixed_replay/`
- Historical pre-correction package: `joao/results/final_canonical_20260716/`
- Old-vs-corrected comparison: `fixed_replay_old_vs_corrected.csv`
- Old-vs-corrected summary: `fixed_replay_old_vs_corrected.md`

The historical package is preserved, but the corrected package should be used
for final claims after the branching leakage correction.

## Key Artifacts

- Branching artifact: `joao/models/branching/composite_branching_evaluation_train70.pkl`
- Branching artifact SHA-256: `ebb16554ecf0e42faa87c7334faef3ab81b742b8e29820df1ce032bd4109f1c4`
- Processing-time artifact: `joao/models/process_time/final_process_time_coverage_v2.pkl`
- Processing-time artifact SHA-256: `c540304cdbb6f60159ad1023112e6d5e71aeecae68b885ee2e2b1ac3c826a886`
- Event log: `data/logData.xes`
- Event log SHA-256: `d653a36d36fac668638d65c90b803670bab6e599aa23e3f7dd4f0d5d0b216b1c`

The 76 fixed-replay cases are fully contained in the outer held-out split, with
zero overlap against the branching evaluation artifact's training cases.

## Command

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

## Interpretation

The corrected rerun did not change the descriptive mean-cycle-time ranking:

1. ShortestQueue
2. Batch
3. RoundRobin
4. ParkSong-Composite
5. Kunkler-Rinderle-Ma

Completion, censorship and conditional cycle-time metrics must be interpreted
together. Cycle-time averages are conditional on completed fixed routes and can
show survivorship bias when completion is below 100%.
