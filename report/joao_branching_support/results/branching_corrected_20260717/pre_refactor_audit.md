# Branching Pre-Refactor Audit
Generated before branching correction edits.
## Git State
- Branch: `joao/integrate-methods-2026-07-15`
- HEAD: `24a5545ac7e8d170c3ce1a8e2ad6f7d9204dce99`
```text
## joao/integrate-methods-2026-07-15
 M BPMN_engine.py
 M SimulationEngineCore.py
 M joao/.gitignore
 M joao/README.md
 M joao/results/final_predictive_model_full_summary.csv
 M joao/results/final_predictive_model_metrics_0_6.csv
 M joao/results/final_predictive_model_metrics_0_7.csv
 M joao/results/final_predictive_model_metrics_0_8.csv
 M joao/results/final_predictive_model_split_comparison.csv
 M joao/scripts/branching/compare_branching_approaches_and_splits.py
 M joao/scripts/branching/export_branching_report.py
 M joao/scripts/branching/train_final_predictive_model.py
 M joao/scripts/branching/train_full_predictive_model.py
 M joao/scripts/resource_allocation/run_resource_allocation_scenarios.py
 M joao/src/branching/AttributeBasedBranchingEngine.py
 M joao/src/branching/AttributeSamplingBranchingEngine.py
 M joao/src/branching/BranchingUtils.py
 M joao/src/branching/CompositeBranchingEngine.py
 M joao/src/branching/PredictiveBranchingEngine.py
 M joao/src/branching/ProbabilityBranchingEngine.py
 M joao/src/resource_allocation/MLPredictionAdapter.py
 M joao/src/resource_allocation/ParkSongAllocation.py
 M joao/src/resource_allocation/ShortestQueueAllocation.py
 M joao/tests/resource_allocation/test_ml_prediction_adapter.py
 M joao/tests/resource_allocation/test_park_song_allocation.py
 M joao/tests/resource_allocation/test_parksong_ml_integration_layer.py
 M joao/tests/resource_allocation/test_shortest_queue_allocation.py
 M joao/tests/test_composite_branching_engine.py
 M processTimes/process_time_engine.py
 M report/main.tex
 M report/references.bib
 M report/sections/formalization_park.tex
 M report/sections/heuristics.tex
 M resources/resource_engine.py
?? joao/models/
?? joao/results/branching_split_approach_comparison.csv
?? joao/results/final_all_methods_20260715/
?? joao/results/final_canonical_20260716/
?? joao/results/final_predictive_model_bpmn_replay_diagnostics.csv
?? joao/results/final_validation_20260715/
?? joao/results/parksong_calibration_20260716/
?? joao/results/parksong_global_assignment_audit_20260716/
?? joao/results/parksong_temporal_lookahead_audit_20260716/
?? joao/results/transition_aware_branching_20260715/
?? joao/scripts/branching/export_final_composite_branching_artifact.py
?? joao/scripts/branching/final_validation_analysis.py
?? joao/scripts/branching/run_transition_aware_generative_smoke.py
?? joao/scripts/branching/train_transition_aware_branching.py
?? joao/scripts/process_model/
?? joao/scripts/resource_allocation/build_processing_time_coverage_v2.py
?? joao/scripts/resource_allocation/run_final_resource_allocation_evaluation.py
?? joao/scripts/resource_allocation/run_full_method_audit.py
?? joao/scripts/resource_allocation/run_full_simulation_allocation_comparison.py
?? joao/scripts/resource_allocation/run_integrated_allocation_comparison.py
?? joao/scripts/resource_allocation/run_my_methods_integrated_comparison.py
?? joao/scripts/resource_allocation/write_reproducibility_manifest.py
?? joao/src/branching/CompositeBranchingArtifact.py
?? joao/src/resource_allocation/BatchAllocationAdapter.py
?? joao/src/resource_allocation/GlobalAllocationAdapter.py
?? joao/src/resource_allocation/KunklerAllocationAdapter.py
?? joao/src/resource_allocation/PickInterfaceAllocationAdapter.py
?? joao/src/resource_allocation/RoundRobinResourceAllocation.py
?? joao/src/resource_allocation/integration/
?? joao/tests/conftest.py
?? joao/tests/resource_allocation/test_batch_allocation_adapter.py
?? joao/tests/resource_allocation/test_composite_branching_adapter.py
?? joao/tests/resource_allocation/test_final_evaluation_metrics.py
?? joao/tests/resource_allocation/test_full_method_audit_invariants.py
?? joao/tests/resource_allocation/test_full_simulation_allocation_comparison_script.py
?? joao/tests/resource_allocation/test_global_allocation_adapter.py
?? joao/tests/resource_allocation/test_integrated_allocation_comparison_script.py
?? joao/tests/resource_allocation/test_integrated_allocation_engine.py
?? joao/tests/resource_allocation/test_kunkler_allocation_adapter.py
?? joao/tests/resource_allocation/test_my_methods_integrated_comparison_script.py
?? joao/tests/resource_allocation/test_pick_interface_allocation_adapter.py
?? joao/tests/resource_allocation/test_requested_amount_compatibility.py
?? joao/tests/resource_allocation/test_resource_engine_waiting_task_allocation.py
?? joao/tests/resource_allocation/test_round_robin_resource_allocation.py
?? joao/tests/resource_allocation/test_simulation_allocation_strategy_plug_in.py
?? joao/tests/resource_allocation/test_transition_aware_branching_adapter.py
?? joao/tests/resource_allocation/test_weighted_fairness_adapter.py
?? joao/tests/test_bpmn_engine_transition_api.py
?? joao/tests/test_composite_branching_artifact.py
?? joao/tests/test_final_canonical_reproducibility.py
?? joao/tests/test_full_method_audit_branching.py
?? joao/tests/test_project_integration_smoke.py
?? models/v5_simulation.bpmn
?? models/v5_simulation.pnml
?? pytest.ini
?? report/joao_branching_support/
?? report/joao_evaluation_protocol_support/
?? report/sections/base_heuristics_eval_joao.tex
?? report/sections/branching_eval_joao.tex
?? report/sections/branching_joao.tex
?? report/sections/cross_method_joao.tex
?? report/sections/evaluation_protocol_joao.tex
?? report/sections/joao_appendix.tex
?? report/sections/parksong_eval_joao.tex
```
## Existing Branching Artifacts
- `joao/models/branching/final_composite_branching.pkl`: size=306566, sha256=`e364dce5e00b05d2f6afa1b0eee835cfd224c3884d82c7b2522a47778c6ae9af`
- `joao/models/branching/final_composite_branching_sklearn190.pkl`: size=13629022, sha256=`9cc504fc84603a8db5a2a00507b329c0fbcf6303aa948990dcf14f10289b48ab`
- `joao/models/branching/transition_aware_branching_v1_20260715.pkl`: size=1552, sha256=`79127e6b1cea6cb58fc1f1f19b1ce96564837a24aa5adf80efcbcc71fc183e54`
- `joao/results/final_canonical_20260716/fixed_replay/fixed_replay_config.json`: size=1138, sha256=`37513cab56bbe6913adbc54d32fea783e11983cc0e520a8186a75d1377e7a26c`
- `joao/results/final_canonical_20260716/fixed_replay/final_artifact_hashes.json`: size=1365, sha256=`2eaafea51f39694d39ec39ff4894dc34212bc0b9807b885c2c01e25c19f3fb8f`
- `joao/results/final_canonical_20260716/fixed_replay/final_route_ids.csv`: size=4481, sha256=`bae638664681a94ce7ce4f84ed38c11f9b02cb288911ff5a4ea340aac1356deb`
- `joao/results/final_canonical_20260716/fixed_replay/fixed_route_workload_summary.json`: size=7882, sha256=`c5e3188e655e09fda94ab8219cb1bebb8cbb299d527463ad25c6da79c79190b4`
- `joao/results/final_canonical_20260716/reproducibility_manifest.json`: size=9925, sha256=`e4a3f9b418120d98b890e8d49d1f96ad1c7252d0574d1b44bef3e31c25617d25`

## Current Composite Artifact Metadata
```json
{
  "training_log": "data/logData.xes",
  "training_events": 1202267,
  "training_cases": 31509,
  "seed": 1,
  "selected_validation_train_ratio": 0.7,
  "selected_validation_evidence": "joao/results/final_predictive_model_split_comparison.csv",
  "deployment_training_mode": "full_log_after_model_selection",
  "predictive_training_dataset_mode": "bpmn_replay",
  "bpmn_model": "models/v4_replay.bpmn",
  "feature_columns": [
    "case:ApplicationType",
    "case:LoanGoal",
    "case:RequestedAmount",
    "CreditScore",
    "EventOrigin",
    "org:resource"
  ],
  "composite_hierarchy": [
    "PredictiveBranchingEngine",
    "AttributeSamplingBranchingEngine",
    "AttributeBasedBranchingEngine",
    "ProbabilityBranchingEngine"
  ],
  "sklearn_version": "1.8.0",
  "runtime_state_persisted": false
}
```

## Leakage Assessment
- Artifact inspected: `joao/models/branching/final_composite_branching.pkl`
- Declared training mode: `full_log_after_model_selection`
- Declared training cases: `31509`
- Fixed replay route IDs: `76`
- Fixed replay case-id SHA-256: `54c159c6efddef14e20d6cd190946634dc0a3c8a16e278a01b7ed2d374987ae6`
- Artifact training case-id SHA-256: not available in v1 artifact metadata
- Confirmed case overlap: `76` because the artifact declares full-log training and fixed replay routes come from the same log.

## Results Depending On Current Artifact
- `joao/results/final_canonical_20260716/fixed_replay/` uses `joao/models/branching/final_composite_branching.pkl`.
- `report/joao_branching_support/` mirrors the previous artifact and branching report sections.
- `report/sections/branching_joao.tex`, `branching_eval_joao.tex`, and `joao_appendix.tex` depend on the previous metrics.

## Historical / Pre-Correction Material
- `joao/models/branching/final_composite_branching.pkl` is treated as historical/deployment-style full-log artifact.
- `joao/results/final_canonical_20260716/` remains preserved as pre-correction canonical results.
- `joao/results/final_predictive_model_*` and `joao/results/branching_split_approach_comparison.csv` remain historical unless regenerated in this correction directory.

## Mandatory Reruns
- Build a leakage-free evaluation artifact trained only on the outer development split.
- Recompute common BPMN-replay branching metrics on the outer held-out test.
- Rerun fixed replay with the leakage-free evaluation artifact if the same held-out route set is valid.
- Generate a separate RF-composite generative smoke without overwriting the existing transition-aware smoke.
