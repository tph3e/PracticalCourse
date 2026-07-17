# Final Validation Report

Outcome: B. Functionally validated, with quantified modeling limitations.

Implementation correctness: transition-aware BPMN enforcement is implemented and regression-tested.

BPMN/log conformance: `v4_replay.bpmn` does not cover all BPIC17 behavior; skipped/nonconformant observations are counted separately.

BPMN-replay classifier: the final Random-Forest classifier is trained and evaluated on synchronized BPMN-replay decision rows. Its perfect offline score applies only to that subset and not to the full event log.

Generative simulation validity: bounded generative runs complete with exact transition fires and final markings in the reported configuration.

Fixed-replay validity: protected fixed-replay outputs are preserved and remain separate evidence for allocation under controlled historical routes. They use the earlier temporal-split composite branching artifact, not the new BPMN-replay artifact.

Resource-allocation evaluation validity: RoundRobin, ShortestQueue, and ParkSong-Composite are exercised through the integrated simulator; fixed-replay final allocation comparison is unchanged.

Quantitative tables: `conformance_v4.csv`, `branching_evaluation.csv`, `generative_runs.csv`, and `generative_summary.csv`.