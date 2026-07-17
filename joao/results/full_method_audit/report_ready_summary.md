# Report Ready Summary

João's subsystem contains branching engines, allocation snapshot strategies, ML prediction adaptation, and a simulator integration subclass. Method correctness evidence is primarily isolated unit/invariant tests plus controlled benchmarks. Integration evidence comes from existing smoke/integration tests and short diagnostics. Final experimental performance evidence is intentionally separate and must not be inferred from these correctness checks.

The previously observed zero-duration same-timestamp loop is infrastructure-level: BPMN self-cycle plus missing/zero processing-time coverage plus no positive-duration guard. The guard affects scheduling, not allocation choices for identical snapshots.
