# Reproducibility manifest

Generated: 2026-07-17T00:29:46.710541+00:00
Branch: `joao/integrate-methods-2026-07-15`
HEAD: `24a5545ac7e8d170c3ce1a8e2ad6f7d9204dce99`
Git dirty: `True`

Interpretation: this manifest captures the canonical pre-commit artifact state.
The canonical code, results and support package must be committed or otherwise
distributed together with the hashes below before this is reproducible from a
fresh checkout. After the final commit, refresh only HEAD/dirty metadata with
the command at the end of this file; the experimental results do not need to be
regenerated for that metadata update.

## Hashes
- `data/logData.xes`: `d653a36d36fac668638d65c90b803670bab6e599aa23e3f7dd4f0d5d0b216b1c`
- `models/v4_replay.bpmn`: `329c6d18b42a680cf45ebc3f22e0d1683318f1c638ffdcbd79064385f5c82586`
- `branching_artifact`: `e364dce5e00b05d2f6afa1b0eee835cfd224c3884d82c7b2522a47778c6ae9af`
- `transition_aware_artifact`: `79127e6b1cea6cb58fc1f1f19b1ce96564837a24aa5adf80efcbcc71fc183e54`
- `processing_time_artifact`: `c540304cdbb6f60159ad1023112e6d5e71aeecae68b885ee2e2b1ac3c826a886`
- `fixed_replay_config`: `37513cab56bbe6913adbc54d32fea783e11983cc0e520a8186a75d1377e7a26c`
- `generative_config`: `b698bae5086ad59eaa4e779d45a10580ce7191bc45e1c36dea8ab679ee5cad7b`

## Commands

Fixed replay:
```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=joao python3 joao/scripts/resource_allocation/run_final_resource_allocation_evaluation.py --data-path data/logData.xes --branching-artifact joao/models/branching/final_composite_branching.pkl --processing-time-artifact joao/models/process_time/final_process_time_coverage_v2.pkl --start 2016-09-16T15:07:00+00:00 --end 2016-09-17T15:07:00+00:00 --drain-until 2016-11-16T15:07:00+00:00 --seeds 1,2,3,4,5 --strategies RoundRobin,ShortestQueue,ParkSong-Composite,Kunkler-Rinderle-Ma,Batch --route-mode fixed-replay --split-ratio 0.7 --parksong-processing-times train-median --reservation-expiration-multiplier 1.0 --output-dir joao/results/final_canonical_20260716/fixed_replay
```

Generative smoke:
```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=joao python3 joao/scripts/branching/run_transition_aware_generative_smoke.py --log data/logData.xes --transition-artifact joao/models/branching/transition_aware_branching_v1_20260715.pkl --processing-time-artifact joao/models/process_time/final_process_time_coverage_v2.pkl --output-dir joao/results/final_canonical_20260716/generative --hours 3 --drain-hours 8 --event-cap 100 --seeds 1,2,3 --strategies round_robin,shortest_queue,parksong_composite,kunkler,batch --split-ratio 0.7
```

The generative smoke uses the transition-aware BPMN path with an empty composite
branching hierarchy. It validates transition-aware BPMN/resource-allocation
integration, but it is not a ranking benchmark and not an evaluation of the
Random-Forest composite branching classifier.

After final commit, update only HEAD/dirty metadata with:
```bash
PYTHONDONTWRITEBYTECODE=1 python3 joao/scripts/resource_allocation/write_reproducibility_manifest.py --results-dir joao/results/final_canonical_20260716
```
