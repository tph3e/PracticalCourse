# Joao Branching Support Package

This package mirrors the corrected Branching Decisions / Predictive ML material
used by the report after the 2026-07-17 leakage correction. `support_manifest.csv`
records SHA-256 equality between each canonical file and its support copy.

Canonical evaluation branching artifact:

- `models/composite_branching_evaluation_train70_rfopt_v1.pkl`
- SHA-256: `490da841440cd019a0819892ab659e5b61f5ee55182f6db12a536a9c4d23ff82`
- Scope: evaluation
- Training dataset mode: common BPMN replay
- Outer development/test split: temporal case split 70/30
- Training/fixed-replay case overlap: 0
- Runtime hierarchy: PredictiveBranchingEngine, ProbabilityBranchingEngine

Deployment-only branching artifact:

- `models/composite_branching_deployment_full_rfopt_v1.pkl`
- SHA-256: `14b61cd4e13762e584fc35dfbd657142d1bfb4868c0cff3daf2b9b83d75e47e3`
- Scope: deployment
- Not valid for held-out evaluation metrics

Historical full-log artifact:

- `models/final_composite_branching.pkl`
- SHA-256: `e364dce5e00b05d2f6afa1b0eee835cfd224c3884d82c7b2522a47778c6ae9af`
- Retained as pre-correction historical material only

Canonical transition-aware integration artifact:

- `models/transition_aware_branching_v1_20260715.pkl`
- SHA-256: `79127e6b1cea6cb58fc1f1f19b1ce96564837a24aa5adf80efcbcc71fc183e54`

The RF-optimized candidate fixed replay uses
`composite_branching_evaluation_train70_rfopt_v1.pkl`. The older
`final_composite_branching.pkl`, `composite_branching_temporal_split.pkl` and
`final_all_methods_20260715` results are historical audit material only.

Important report/support files:

- `report_sections/branching_joao.tex`
- `report_sections/branching_eval_joao.tex`
- `report_sections/joao_appendix.tex`
- `code/src/branching/PredictiveBranchingEngine.py`
- `code/src/branching/CompositeBranchingArtifact.py`
- `code/scripts/run_corrected_branching_evaluation.py`
- `results/branching_corrected_20260717/`
- `code/scripts/final_validation_analysis.py`

The corrected BPMN-replay offline scores are computed only on synchronized
multi-candidate decision observations with the true next label enabled. They
should not be read as full event-log next-activity accuracy. The original
generative smoke in the evaluation package uses the transition-aware BPMN path
with an empty composite branching hierarchy; it is integration evidence and does
not evaluate the Random-Forest composite artifact. The corrected package also
contains a separate RF-composite generative smoke, which remains an integration
smoke rather than a ranking benchmark.
