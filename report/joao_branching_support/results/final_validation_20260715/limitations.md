# Limitations

- `v4_replay.bpmn` is enforceable but not fully representative of BPIC17; conformance gaps remain.
- Full alignment over every train/test case was not run in this final pass; bounded replay reports coverage explicitly.
- Macro-F1, weighted-F1, log-loss, and top-k transition metrics require a larger labeled transition decision dataset; unavailable values are reported as `null`, not inferred.
- Generative validation is larger than the 12-case smoke when rerun with multiple seeds, but still bounded to avoid a large final evaluation.
- Existing processing-time pickle emits sklearn version warnings under the current environment.
- Untracked `models/v5_simulation.*` files exist from an interrupted earlier exploration and are not selected or used for this validation.