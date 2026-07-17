# Limitations

- Results are conditional on `models/v4_replay.bpmn`; this BPMN does not cover every BPIC17 trace variant.
- Fixed replay evaluates resource allocation under historical held-out routes. It does not evaluate branching quality.
- Generative evaluation is bounded to a three-hour arrival window and eight-hour drain, not a full-log simulation.
- Some generative cases are censored by the drain horizon even though no deadlocks occurred.
- Kunkler required an adapter repair because the original repository class leaves its cost-matrix resource list empty and can emit infeasible pairs. The adapter still invokes the real implementation and records the repair count.
- Batch Allocation is integrated as a current waiting-queue snapshot because the integrated simulator owns queue persistence, retries, resource release, and event lifecycle.
- Predictive label metrics and transition-alignment metrics are reported separately.
