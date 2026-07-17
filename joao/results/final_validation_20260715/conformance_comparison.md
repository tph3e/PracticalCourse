# BPMN/Log Conformance

Model under validation: `models/v4_replay.bpmn`.

The conformance analysis is intentionally separated from simulator enforcement. A trace can be nonconformant to `v4_replay.bpmn` while the simulator still correctly enforces that BPMN for generated cases.

Summary rows are available in `conformance_v4.csv`; detailed metrics are in `conformance_v4.json`.

Known limitation: `v4_replay.bpmn` does not represent all BPIC17 observed behavior. Nonconformant observations are skipped for transition accuracy rather than treated as identifiable transition targets.