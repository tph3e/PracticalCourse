# Pipeline Audit

Audited path: arrival -> case creation -> BPMN marking -> exact transition candidates -> branching decision -> task/queue -> resource eligibility -> allocation -> processing duration -> release -> next BPMN state -> final/deadlock.

Evidence:
- `BPMN_engine.py` normalizes case IDs and stores independent markings per case.
- Transition candidates include transition ID, label, source marking, pre-visible marking, silent path, and resulting marking.
- `IntegratedAllocationEngine` stores selected transition IDs per task and fires exact transition IDs on activity completion.
- Legacy label firing is retained only as compatibility fallback and ambiguous labels are rejected.
- Generative runs report exact transition fires, legacy fires, deadlocks, censoring, resource allocation, and reservations.

Current limitation: queue-length/utilization/fairness are inherited from existing metric outputs for fixed-replay; this final pass did not recompute full rich metrics for large generative runs.