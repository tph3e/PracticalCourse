# RF Training Optimization 20260717

Controlled inner-validation optimization for the final branching scope:
ProbabilityBranching, Predictive Random Forest, and CompositeRuntime. Existing
artifacts were not overwritten.

## Selection

The predeclared selection rule chose `rf_27` using only inner temporal
validation. The selected configuration is:

- `n_estimators=200`
- `max_depth=16`
- `min_samples_leaf=5`
- `min_samples_split=5`
- `max_features=0.5`
- `class_weight=balanced_subsample`
- `criterion=gini`

Inner-validation macro-F1 improved from `0.777951` to `0.841642` averaged over
seeds 1-5. The final single outer-held-out evaluation improved RF macro-F1 from
`0.662517` to `0.707269`.

## Artifacts

- Evaluation candidate: `joao/models/branching/composite_branching_evaluation_train70_rfopt_v1.pkl`
- Evaluation SHA-256: `490da841440cd019a0819892ab659e5b61f5ee55182f6db12a536a9c4d23ff82`
- Deployment candidate: `joao/models/branching/composite_branching_deployment_full_rfopt_v1.pkl`
- Deployment SHA-256: `14b61cd4e13762e584fc35dfbd657142d1bfb4868c0cff3daf2b9b83d75e47e3`
- Composite hierarchy: `PredictiveBranchingEngine -> ProbabilityBranchingEngine -> random BPMN-valid fallback`

## Fixed Replay Candidate

Because a new evaluation artifact was recommended, all five fixed-replay methods
were rerun on the same 76 held-out routes and seeds 1-5:

`joao/results/final_canonical_rfopt_candidate_20260717/fixed_replay/`

No run failures were recorded. The ranking by conditional mean cycle time changed
because the optimized artifact completed all ParkSong-Composite and RoundRobin
fixed routes, whereas the previous corrected run censored one case in each of
those strategies. This makes cycle-time averages not directly comparable without
also reading completion and censorship counts.

## Main Files

- `rf_selection_decision.json`
- `rf_candidate_hyperparameters.csv`
- `rf_seed_stability_aggregated.csv`
- `rf_outer_heldout_metrics.csv`
- `branching_final_core_metrics.csv`
- `fixed_replay_current_vs_rfopt.csv`
- `fixed_replay_current_vs_rfopt.md`
- `generative_rf_composite_smoke/`
