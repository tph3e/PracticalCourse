# Transition-Aware Empty-Composite Smoke Copy

This directory is an identical copy of the transition-aware / empty-composite
generative smoke originally produced under:

```text
joao/results/final_canonical_20260716/generative/
```

It is copied here so the corrected branching package has both smoke-test
families in one place:

- `generative_transition_aware_smoke/`: transition-aware BPMN execution with an
  empty CompositeBranchingEngine hierarchy; not a Random-Forest evaluation.
- `generative_rf_composite_smoke/`: RF-composite runtime integration smoke; not
  a ranking benchmark.

The experiment was not regenerated. `copy_manifest.json` records source paths,
source SHA-256 values, copied SHA-256 values and `identical=true` for every
copied file.

Some copied JSON fields, such as `event_log_path`, still contain the original
2026-07-16 directory because they are generated run metadata. They are retained
unchanged to preserve hash identity with the source files.
