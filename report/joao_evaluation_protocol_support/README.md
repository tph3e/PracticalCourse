# Joao Evaluation Protocol Support Package

This package mirrors the corrected Resource Allocation and ParkSong material used
by the report after the 2026-07-17 branching leakage correction.
`support_manifest.csv` records the canonical path, support copy path and SHA-256
for every synchronized file.

Corrected canonical results are under:

- `results/final_canonical_branching_corrected_20260717/fixed_replay/`
- `results/final_canonical_branching_corrected_20260717/fixed_replay_old_vs_corrected.csv`
- `results/final_canonical_branching_corrected_20260717/fixed_replay_old_vs_corrected.md`

Historical folders such as `final_canonical_20260716`,
`final_all_methods_20260715` and `parksong_calibration_20260716` are retained
only as audit history and should not be used for the corrected final table.

Primary fixed-replay inputs:

- Branching artifact: `joao/models/branching/composite_branching_evaluation_train70.pkl`
- Branching SHA-256: `ebb16554ecf0e42faa87c7334faef3ab81b742b8e29820df1ce032bd4109f1c4`
- Processing-time artifact: `joao/models/process_time/final_process_time_coverage_v2.pkl`
- Route mode: fixed replay, 76 held-out routes, seeds 1-5
- ParkSong: train-median estimates, `cost_time_scale=3600`,
  `no_show_penalty_weight=1.0`, `future_delay_weight=0`,
  `reservation_margin=0`

Primary fixed-replay table source:

- `results/final_canonical_branching_corrected_20260717/fixed_replay/fixed_replay_aggregated_metrics.csv`
- `results/final_canonical_branching_corrected_20260717/fixed_replay/fixed_replay_raw_metrics.csv`
- `results/final_canonical_branching_corrected_20260717/fixed_replay/parksong_reservation_diagnostics.csv`
- `results/final_canonical_branching_corrected_20260717/fixed_replay/processing_time_coverage.csv`
- `results/final_canonical_branching_corrected_20260717/fixed_replay/resource_pressure_diagnostics.csv`

Generative smoke source:

- `generative/generative_runs.csv`
- `generative/generative_summary.csv`
- `generative/generative_integration_smoke.json`

The generative smoke is transition-aware BPMN/allocation integration evidence,
not the main ranking benchmark and not an evaluation of the Random-Forest
composite classifier. The corrected branching package separately includes
`joao/results/branching_corrected_20260717/generative_rf_composite_smoke/`,
which checks RF-composite runtime integration without changing the benchmark
ranking.

Metric interpretation:

- Fixed-replay completion is fixed-route completion, not BPMN final-marking
  completion.
- Cycle time is computed only for completed route instances.
- Waiting time is computed for tasks with observed resource assignment.
- The current canonical manifest was generated before the final repository
  commit and therefore records a dirty worktree; rerun only the manifest writer
  after committing to refresh HEAD/dirty metadata.
