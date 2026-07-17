# Limitations

- `v4_replay.bpmn` is enforceable but not fully representative of BPIC17; conformance gaps remain.
- The BPMN-replay classifier is evaluated only on synchronized decision observations reached by replay on `v4_replay.bpmn`; it must not be interpreted as global next-activity accuracy over the full log.
- The bounded transition audit reports replay observations attempted, multi-candidate observations, synchronized observations and skipped observations separately; the attempted replay count is not equivalent to classifier decision rows.
- The preserved fixed-replay allocation ranking uses the earlier `composite_branching_temporal_split.pkl` artifact, not the newly exported BPMN-replay classifier artifact.
- Macro-F1, weighted-F1, log-loss, and top-k transition metrics for the transition-alignment audit require a larger labeled transition decision dataset; unavailable values are reported as `null`, not inferred.
- Generative validation is larger than the 12-case smoke when rerun with multiple seeds, but still bounded to avoid a large final evaluation.
- Existing processing-time pickle emits sklearn version warnings under the current environment.
- Untracked `models/v5_simulation.*` files exist from an interrupted earlier exploration and are not selected or used for this validation.