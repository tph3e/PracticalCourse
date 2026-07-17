# Historical Pre-Correction Results - 2026-07-16

This directory is preserved as historical pre-correction evidence. It is not the
current canonical result package and must not be used to reconstruct the final
report tables after the branching leakage correction.

The corrected final sources are:

- Branching evaluation: `joao/results/branching_corrected_20260717/`
- Corrected fixed replay:
  `joao/results/final_canonical_branching_corrected_20260717/`
- Corrected evaluation artifact:
  `joao/models/branching/composite_branching_evaluation_train70.pkl`
- Corrected evaluation artifact SHA-256:
  `ebb16554ecf0e42faa87c7334faef3ab81b742b8e29820df1ce032bd4109f1c4`

## Why This Package Is Historical

This package was produced before the explicit separation between:

- an evaluation artifact trained only on the outer development split; and
- a deployment artifact trained on the full log.

The fixed replay in this directory used:

- `joao/models/branching/final_composite_branching.pkl`
- SHA-256:
  `e364dce5e00b05d2f6afa1b0eee835cfd224c3884d82c7b2522a47778c6ae9af`

That artifact is a full-log/deployment-style artifact and is not valid as
held-out evaluation evidence. It is retained only for audit and old-vs-corrected
comparison.

## Preserved Inputs

- Event log: `data/logData.xes`
- Event-log SHA-256:
  `d653a36d36fac668638d65c90b803670bab6e599aa23e3f7dd4f0d5d0b216b1c`
- BPMN model: `models/v4_replay.bpmn`
- BPMN SHA-256:
  `329c6d18b42a680cf45ebc3f22e0d1683318f1c638ffdcbd79064385f5c82586`
- Historical composite branching artifact:
  `joao/models/branching/final_composite_branching.pkl`
- Transition-aware artifact:
  `joao/models/branching/transition_aware_branching_v1_20260715.pkl`
- Transition-aware SHA-256:
  `79127e6b1cea6cb58fc1f1f19b1ce96564837a24aa5adf80efcbcc71fc183e54`
- Processing-time artifact:
  `joao/models/process_time/final_process_time_coverage_v2.pkl`
- Processing-time SHA-256:
  `c540304cdbb6f60159ad1023112e6d5e71aeecae68b885ee2e2b1ac3c826a886`

## Historical Fixed Replay

The historical fixed replay used 76 held-out historical routes, seeds 1-5 and
the five strategies RoundRobin, ShortestQueue, ParkSong-Composite,
Kunkler-Rinderle-Ma and Batch. These files are preserved to compare the
pre-correction and corrected runs, but they are not the current final benchmark.

The corrected fixed replay is under:

```text
joao/results/final_canonical_branching_corrected_20260717/fixed_replay/
```

The corrected old-vs-corrected comparison is under:

```text
joao/results/final_canonical_branching_corrected_20260717/fixed_replay_old_vs_corrected.csv
```

## Historical Transition-Aware Smoke

The generative smoke in this directory uses the transition-aware BPMN path with
an empty composite branching hierarchy. It validates BPMN-constrained transition
execution and allocation integration. It does not evaluate the Random-Forest
composite branching artifact and is not a ranking benchmark.

For final documentation, an identical copy is available under:

```text
joao/results/branching_corrected_20260717/generative_transition_aware_smoke/
```

That copy has a `copy_manifest.json` recording source and copied SHA-256 values.

## Historical Commands

Commands in this directory are retained to reproduce the historical package
only. Do not use them as final reconstruction commands. In particular, the
historical fixed-replay command uses:

```text
--branching-artifact joao/models/branching/final_composite_branching.pkl
```

Final fixed-replay reconstruction must instead use:

```text
--branching-artifact joao/models/branching/composite_branching_evaluation_train70.pkl
```

## Result Reconstruction

Do not use this package to reconstruct final report tables. Use:

- `joao/results/branching_corrected_20260717/`
- `joao/results/final_canonical_branching_corrected_20260717/`
- `report/joao_branching_support/`
- `report/joao_evaluation_protocol_support/`
