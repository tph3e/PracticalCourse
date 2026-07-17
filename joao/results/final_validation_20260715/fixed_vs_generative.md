# Fixed-Replay vs Generative Evaluation

Fixed-replay preserves historical routing and is a controlled resource-allocation experiment. It evaluates resource dynamics and allocation choices under known routes; it does not validate branching quality or BPMN generative conformance.

Generative runs use BPMN transition candidates and branching decisions to create routes. These runs validate process-model enforcement, final markings, resource integration, and termination behavior.

Protected fixed-replay outputs were not rerun or modified. Hash comparison is recorded in `protected_hash_diff.txt`.

The BPMN-replay predictive classifier generated after the final audit is an offline branching artifact. The preserved fixed-replay allocation ranking uses the earlier temporal-split composite branching artifact and should not be presented as having used the new BPMN-replay artifact.