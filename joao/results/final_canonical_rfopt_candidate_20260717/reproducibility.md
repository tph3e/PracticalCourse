# Part II A Fixed-Replay Reproducibility

Generated: 2026-07-17T19:33:59.757355+00:00
Branch: `joao/integrate-methods-2026-07-15`
HEAD: `24a5545ac7e8d170c3ce1a8e2ad6f7d9204dce99`
Git dirty: `True`

This package was generated before the final commit. After committing, run:

```bash
python3 joao/scripts/resource_allocation/write_reproducibility_manifest.py --results-dir joao/results/final_canonical_rfopt_candidate_20260717
```

That refresh updates only git metadata unless result files changed.

## Command

```bash
MPLCONFIGDIR=/tmp/matplotlib \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=joao:. \
  python3 joao/scripts/resource_allocation/run_final_resource_allocation_evaluation.py \
  --data-path data/logData.xes \
  --branching-artifact joao/models/branching/composite_branching_evaluation_train70_rfopt_v1.pkl \
  --processing-time-artifact joao/models/process_time/final_process_time_coverage_v2.pkl \
  --start 2016-09-16T15:07:00+00:00 \
  --end 2016-09-17T15:07:00+00:00 \
  --drain-until 2016-11-16T15:07:00+00:00 \
  --seeds 1,2,3,4,5 \
  --strategies RoundRobin,ShortestQueue,ParkSong-Composite,Kunkler-Rinderle-Ma,Batch \
  --route-mode fixed-replay \
  --split-ratio 0.7 \
  --parksong-processing-times train-median \
  --parksong-params cost_time_scale=3600.0,future_delay_weight=0.0,no_show_penalty_weight=1.0,reservation_margin=0.0 \
  --reservation-expiration-multiplier 1.0 \
  --output-dir joao/results/final_canonical_rfopt_candidate_20260717/fixed_replay
```

## Artifact Hashes

- `data_sha256`: `d653a36d36fac668638d65c90b803670bab6e599aa23e3f7dd4f0d5d0b216b1c`
- `branching_sha256`: `490da841440cd019a0819892ab659e5b61f5ee55182f6db12a536a9c4d23ff82`
- `processing_time_sha256`: `c540304cdbb6f60159ad1023112e6d5e71aeecae68b885ee2e2b1ac3c826a886`

## File Hashes

- `fixed_replay/final_run_config.json`: `3e9d22797ecf762b18dab49295a39357b6bd1ffbfb5083a7e33ddff07557c2a0`
- `fixed_replay/final_artifact_hashes.json`: `54226fd5ea53643b4a9837c5c90524a893f786923fdf17d29233b818af4b10f2`
- `fixed_replay/final_route_ids.csv`: `bae638664681a94ce7ce4f84ed38c11f9b02cb288911ff5a4ea340aac1356deb`
- `fixed_replay/final_raw_metrics.csv`: `6149379789bfad38af13214891703917c192556fba631ff1a6f92afd84ff6ff6`
- `fixed_replay/final_aggregated_metrics.csv`: `ebc7d6723dc17f67aa63a33ce0cedc86b513801cdb9cf5504c208b61b515b8bb`
- `fixed_replay/final_report_table.csv`: `e3691a6acbc9e6572d0524b7f3f3877ce84a3576e2e8a49cf8440d9896a3fdf5`
- `fixed_replay/final_paired_comparisons.csv`: `d5ba5fd5e1f53aba286820ce13b2a489324f70ca3fe010beaac150c1dea91019`
