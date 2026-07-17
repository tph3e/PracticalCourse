# Final Validation Report

Outcome: B. Functionally validated, with quantified modeling limitations.

Implementation correctness: transition-aware BPMN enforcement is implemented and regression-tested.

BPMN/log conformance: `v4_replay.bpmn` does not cover all BPIC17 behavior; skipped/nonconformant observations are counted separately.

Generative simulation validity: bounded generative runs complete with exact transition fires and final markings in the reported configuration.

Fixed-replay validity: protected fixed-replay outputs are preserved and remain separate evidence for allocation under controlled historical routes.

Resource-allocation evaluation validity: RoundRobin, ShortestQueue, and ParkSong-Composite are exercised through the integrated simulator; fixed-replay final allocation comparison is unchanged.

Quantitative tables: `conformance_v4.csv`, `branching_evaluation.csv`, `generative_runs.csv`, and `generative_summary.csv`.