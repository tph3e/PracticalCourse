# Corrected Branching Results

This package contains the leakage-free branching evaluation generated on
2026-07-17. It replaces the old branching evaluation package for claims about
held-out predictive quality. Historical results remain preserved, but they
should be read as pre-correction material.

## Canonical Scope

- Corrected result directory: `joao/results/branching_corrected_20260717/`
- Pre-optimization evaluation artifact: `joao/models/branching/composite_branching_evaluation_train70.pkl`
- Pre-optimization evaluation artifact SHA-256: `ebb16554ecf0e42faa87c7334faef3ab81b742b8e29820df1ce032bd4109f1c4`
- Final optimized evaluation candidate: `joao/models/branching/composite_branching_evaluation_train70_rfopt_v1.pkl`
- Final optimized evaluation candidate SHA-256: `490da841440cd019a0819892ab659e5b61f5ee55182f6db12a536a9c4d23ff82`
- Deployment artifact: `joao/models/branching/composite_branching_deployment_full_rfopt_v1.pkl`
- Deployment artifact SHA-256: `14b61cd4e13762e584fc35dfbd657142d1bfb4868c0cff3daf2b9b83d75e47e3`
- Historical full-log artifact: `joao/models/branching/final_composite_branching.pkl`
- Historical artifact SHA-256: `e364dce5e00b05d2f6afa1b0eee835cfd224c3884d82c7b2522a47778c6ae9af`

Both evaluation artifacts are trained only on the outer development split. The
`rfopt_v1` artifact is the final recommended branching candidate after
inner-validation optimization. The deployment artifact is trained on the full
log and is not valid for held-out metrics. The ambiguous
`final_composite_branching.pkl` name is retained only as historical/deployment-
style material.

## Data and Split

- Event log: `data/logData.xes`
- Event log SHA-256: `d653a36d36fac668638d65c90b803670bab6e599aa23e3f7dd4f0d5d0b216b1c`
- BPMN model: `models/v4_replay.bpmn`
- BPMN SHA-256: `329c6d18b42a680cf45ebc3f22e0d1683318f1c638ffdcbd79064385f5c82586`
- Outer development cases: 22,056
- Outer held-out cases: 9,453
- Inner train cases: 18,747
- Inner validation cases: 3,309
- Fixed-replay cases: 76
- Fixed-replay cases in outer held-out: true
- Outer train/test overlap: 0
- Training/fixed-replay overlap: 0

The full split metadata and hashes are in `branching_split_manifest.json`.

## Main Outputs

- `datasets/branching_common_dataset_filtered.csv`
- `branching_coverage.csv`
- `branching_method_metrics_by_seed.csv`
- `branching_method_metrics_aggregated.csv`
- `branching_per_decision_point_metrics.csv`
- `branching_confusion_matrix.csv`
- `branching_feature_parity_diagnostics.csv`
- `branching_artifacts.csv`
- `branching_artifacts_metadata.json`
- `generative_transition_aware_smoke/`
- `generative_rf_composite_smoke/`

The common evaluation uses synchronized BPMN-replay decision observations with
more than one BPMN-valid candidate and the true next label enabled. All methods
are evaluated on the same final rows.

## Interpretation Notes

- The corrected pre-optimization RF held-out accuracy is `0.814225`, not a
  perfect score.
- The RF training optimization package
  `joao/results/rf_training_optimization_20260717/` contains the final
  optimized core-method table with ProbabilityBranching, Predictive RF and
  CompositeRuntime only.
- The optimized RF candidate reaches held-out accuracy `0.935341`, macro-F1
  `0.707269` and weighted-F1 `0.935807` on the same `29,385` BPMN-replay
  observations.
- The majority baseline is strong (`0.838455`) on this filtered dataset and is
  retained as a baseline, not as the runtime policy.
- The RF is not calibrated; probabilities should not be interpreted as calibrated
  outcome likelihoods.
- `generative_transition_aware_smoke/` is an identical copy of the historical
  transition-aware / empty-composite smoke, included here so final documentation
  does not need to point to the pre-correction package. Its `copy_manifest.json`
  records source and copied SHA-256 values.
- `generative_rf_composite_smoke/` is an integration smoke for the RF composite
  artifact. It is not a statistical ranking benchmark.

## Reproduction

Use `reproducibility.md` for commands and environment details. The BPIC17 log is
external and must be available at `data/logData.xes`; it should not be committed
to Git.
