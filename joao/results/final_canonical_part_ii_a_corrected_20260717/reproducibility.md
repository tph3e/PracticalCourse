# Part II A Fixed-Replay Reproducibility

Generated: 2026-07-17T18:53:15.433733+00:00
Branch: `joao/integrate-methods-2026-07-15`
HEAD: `24a5545ac7e8d170c3ce1a8e2ad6f7d9204dce99`
Git dirty: `True`

This package was generated before the final commit. After committing, run:

```bash
python3 joao/scripts/resource_allocation/write_reproducibility_manifest.py --results-dir joao/results/final_canonical_part_ii_a_corrected_20260717
```

That refresh updates only git metadata unless result files changed.

## Command

```bash
MPLCONFIGDIR=/tmp/matplotlib \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=joao:. \
  python3 joao/scripts/resource_allocation/run_final_resource_allocation_evaluation.py \
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
  --parksong-params cost_time_scale=3600.0,future_delay_weight=0.0,no_show_penalty_weight=1.0,reservation_margin=0.0 \
  --reservation-expiration-multiplier 1.0 \
  --output-dir joao/results/final_canonical_part_ii_a_corrected_20260717/fixed_replay
```

## Artifact Hashes

- `data_sha256`: `d653a36d36fac668638d65c90b803670bab6e599aa23e3f7dd4f0d5d0b216b1c`
- `branching_sha256`: `ebb16554ecf0e42faa87c7334faef3ab81b742b8e29820df1ce032bd4109f1c4`
- `processing_time_sha256`: `c540304cdbb6f60159ad1023112e6d5e71aeecae68b885ee2e2b1ac3c826a886`

## File Hashes

- `fixed_replay/final_run_config.json`: `2472cda6d19968905d25f04d10c54c275628f8e980eb9ba7296c6a367a245bb9`
- `fixed_replay/final_artifact_hashes.json`: `f88ed4690f1576d0ba58d4c9843a9f8213490e192c386415ed53f13850f24389`
- `fixed_replay/final_route_ids.csv`: `bae638664681a94ce7ce4f84ed38c11f9b02cb288911ff5a4ea340aac1356deb`
- `fixed_replay/final_raw_metrics.csv`: `c66c19362f0bf4e723618edacaee82ac432afb1bc0aec12c3394e613f39ff8e8`
- `fixed_replay/final_aggregated_metrics.csv`: `afbfbcd164a06d219b74d0c59d9a6c22966565395475920f3ec639d2f2a14d9e`
- `fixed_replay/final_report_table.csv`: `e54290d5dfda949115bd63d3e5b654117abd467fe434c9c329c1c8df3a07f49d`
- `fixed_replay/final_paired_comparisons.csv`: `40c92686e357764db4d192fb498b25ba16a2528fcad9d62e845234b1735a049b`
