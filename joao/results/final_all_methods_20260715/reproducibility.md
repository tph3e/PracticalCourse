# Reproducibility

```json
{
  "branch": "joao/integrate-methods-2026-07-15",
  "commit": "24a5545ac7e8d170c3ce1a8e2ad6f7d9204dce99",
  "python": "3.14.3 (v3.14.3:323c59a5e34, Feb  3 2026, 11:41:37) [Clang 16.0.0 (clang-1600.0.26.6)]",
  "platform": "macOS-26.4-arm64-arm-64bit-Mach-O",
  "selected_bpmn": "models/v4_replay.bpmn",
  "fixed_replay": {
    "start": "2016-09-16T15:07:00+00:00",
    "end": "2016-09-17T15:07:00+00:00",
    "drain_until": "2016-11-16T15:07:00+00:00",
    "seeds": [
      1,
      2,
      3,
      4,
      5
    ],
    "route_mode": "held-out fixed replay",
    "routes": 76
  },
  "generative": {
    "start": "2016-01-04T09:00:00",
    "hours": 3,
    "drain_hours": 8,
    "seeds": [
      1,
      2,
      3
    ],
    "route_mode": "BPMN-enforced transition-aware generative"
  },
  "artifacts": {
    "branching": "joao/results/joao_calibration_20260715_roundrobin_split_v4/models/composite_branching_temporal_split.pkl",
    "transition_aware": "joao/models/branching/transition_aware_branching_v1_20260715.pkl",
    "processing_time": "joao/models/process_time/final_process_time_coverage_v2.pkl"
  },
  "commands": [
    "pytest -q joao/tests/resource_allocation/test_kunkler_allocation_adapter.py joao/tests/resource_allocation/test_batch_allocation_adapter.py joao/tests/resource_allocation/test_integrated_allocation_engine.py",
    "python3 joao/scripts/resource_allocation/run_final_resource_allocation_evaluation.py --data-path data/logData.xes --branching-artifact joao/results/joao_calibration_20260715_roundrobin_split_v4/models/composite_branching_temporal_split.pkl --processing-time-artifact joao/models/process_time/final_process_time_coverage_v2.pkl --start 2016-09-16T15:07:00+00:00 --end 2016-09-17T15:07:00+00:00 --drain-until 2016-11-16T15:07:00+00:00 --seeds 1,2,3,4,5 --strategies RoundRobin,ShortestQueue,ParkSong-Composite,Kunkler-Rinderle-Ma,Batch --route-mode fixed-replay --split-ratio 0.7 --output-dir joao/results/final_all_methods_20260715/fixed_replay --resume",
    "python3 joao/scripts/branching/run_transition_aware_generative_smoke.py --hours 3 --drain-hours 8 --seeds 1,2,3 --strategies round_robin,shortest_queue,parksong_composite,kunkler,batch --transition-artifact joao/models/branching/transition_aware_branching_v1_20260715.pkl --processing-time-artifact joao/models/process_time/final_process_time_coverage_v2.pkl --output-dir joao/results/final_all_methods_20260715/generative"
  ]
}
```

Protected hash checks are in `protected_hashes_before.txt`, `protected_hashes_after.txt`, and `protected_hash_diff.txt`.
