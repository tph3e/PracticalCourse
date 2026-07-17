# Requirements Traceability

Validation date: 2026-07-15

Sources:
- `/Users/joaofmzuffo/Downloads/Practical_course__Group_assignment-2.pdf`
- `/Users/joaofmzuffo/Downloads/Practical_course__Group_assignment_II-4.pdf`

Repository state at start:
- Branch: `joao/integrate-methods-2026-07-15`
- Commit: `24a5545ac7e8d170c3ce1a8e2ad6f7d9204dce99`
- Worktree: dirty before this validation pass; existing João/resource-allocation work preserved.

## Matrix

| Assignment requirement | Responsible component | Current status | Validation evidence | Remaining limitation |
|---|---|---:|---|---|
| Part I 1.1 discrete-event core with one global event queue | `SimulationEngineCore.Engine`, `IntegratedAllocationEngine` | Implemented | Event queue uses `heapq`; generated logs exported by `EventLogger`; full tests pass. | Further long-horizon stress validation still bounded in this report. |
| Part I 1.2 case arrivals from parametric distribution | `arrival_engine.py` | Implemented | Generative smoke initializes `ArrivalEngine`; logs admitted cases. | Advanced dynamic spawn rates not claimed. |
| Part I 1.3 processing times fit from historical data and ML option | `processTimes/process_time_engine.py`, `joao/models/process_time/final_process_time_coverage_v2.pkl` | Implemented | Smoke uses processing-time engine and reports positive resource releases/completions. | Existing sklearn pickle version warnings observed; no dependency upgrade performed. |
| Part I 1.3 Advanced II processing vs waiting times | `ProcessTimeEngine.getProcessingTime`, `getWaitingTime`, queues in `IntegratedAllocationEngine` | Partially implemented | Waiting retry/queue diagnostics exist; resource contention tests cover waiting paths. | Need larger validation to quantify waiting distributions under real contention. |
| Part I 1.4 selected process model must be enforced | `BPMN_engine.py`, `models/v4_replay.bpmn` | Implemented for selected BPMN | Transition-aware engine enumerates enabled/reachable candidates and fires by transition ID; smoke has exact transition fires only. | `v4_replay.bpmn` does not cover all BPIC17 traces; conformance limitations are separate from enforcement correctness. |
| Part I 1.4 Advanced BPMN loaded and converted to Petri net | `BPMN_engine.py`, PM4Py | Implemented | `v4_replay.bpmn` loads; inventory: 24 places, 32 transitions, 21 visible, 11 silent. | Woflan full soundness can be expensive; bounded reachability used where necessary. |
| Part I 1.5 Basic branch probabilities/runtime decisions | `ProbabilityBranchingEngine`, `CompositeBranchingEngine` | Implemented | Existing tests and transition-aware fallback over current candidates. | Branch probabilities must be interpreted through BPMN candidates, not as free successors. |
| Part I 1.5 Advanced II next-activity/branch prediction using case history, data identified by replaying log on process model | `PredictiveBranchingEngine`, transition-aware replay script | Implemented with bounded replay evidence | `train_transition_aware_branching.py` uses train split only and synchronized observations; artifact is versioned. | Full alignment over all cases not run; nonconformant observations skipped and counted. |
| Part I 1.6 resource availabilities | `resources/availability.py`, `ResourceEngine` | Implemented | Allocation tests cover availability; smoke uses complete resource engine. | Quantitative availability-calendar fidelity not fully audited in this pass. |
| Part I 1.7 resource permissions | `resources/permissions.py`, `ResourceEngine` | Implemented | Permission tests and integration allocation tests pass. | Large-run violation counts still need explicit reporting. |
| Part I 1.8 random resource allocation | Existing resource allocation baseline | Implemented by group | Existing tests pass. | João validation focuses on RoundRobin, ShortestQueue, ParkSong-Composite. |
| Part II 1.1 base allocation heuristics | `RoundRobinResourceAllocation`, `ShortestQueueAllocation` | Implemented | Smoke runs both strategies end-to-end with resources allocated/released. | Design adaptation to uncertain availability should be documented in final report. |
| Part II 1.1 batch allocation | `BatchAllocationAdapter` | Implemented in existing João work | Existing full tests include batch allocation tests. | Not exercised in transition-aware generative smoke. |
| Part II 1.1 advanced allocation approach | `ParkSongAllocation`, integration reservations | Implemented | ParkSong-Composite smoke created 214 reservations and completed 12/12 cases. | Uses approximation, not full min-cost max-flow LSTM implementation. |
| Part II 1.2 at least three simple metrics | `ResourceAllocationMetrics`, result CSVs | Implemented | Existing fixed-replay final results include resource/cycle/fairness metrics. | Generative metric summary still to be consolidated under final validation. |
| Part II 1.2 evaluate allocation methods on simulator and describe settings | Fixed-replay final evaluation plus transition-aware generative smoke | Partially validated | Fixed-replay outputs preserved; smoke validates generative integration. | Fixed-replay evaluates allocation under controlled routes, not branching quality. |
| Reporting: design decisions, challenges, individual contribution, AI tools | `report/sections`, final validation reports | Partially implemented | Existing final reports present; this directory adds validation evidence. | Final course report still must integrate these results concisely. |

## Validation Plan

1. Preserve protected baselines by hashing fixed-replay result files and model artifacts before and after validation.
2. Audit the full generative path: arrival, case creation, BPMN marking, transition candidates, branching, queues, resources, processing time, release, next marking, final/deadlock.
3. Audit branching leakage and coverage: train-only observations, held-out isolation, candidate restriction, invalid prediction rejection, fallback diagnostics.
4. Quantify `v4_replay.bpmn` conformance separately from simulator correctness.
5. Run bounded but meaningful generative validation for RoundRobin, ShortestQueue, and ParkSong-Composite using deterministic seeds.
6. Keep fixed-replay evidence separate and do not rerun or modify protected final fixed-replay results.
7. Add or improve regression tests only for genuine gaps found during the audit.
8. Run the full test suite and verify protected hashes did not change.
