# Final Report Material

## Methods

R-RRA / RoundRobin assigns each available eligible resource to the oldest eligible waiting task while rotating deterministically through resource identifiers. It is a resource-allocation method only and does not choose process routes.

R-SHQ / ShortestQueue assigns tasks to the eligible available resource with the smallest current load proxy, using deterministic resource-id tie-breaking.

ParkSong-Composite combines the Park and Song inspired reservation allocator with Composite branching predictions. Reservations are created for predicted future tasks, while actual execution still requires the task to become enabled and eligible.

Kunkler uses the repository `AnticipatoryAssignmentAllocator` through `KunklerAllocationAdapter`. The adapter supplies the expected processing-time quantile API, filters eligible resources/tasks, invokes the real allocator, validates assignments, and records sequential eligibility repairs required by limitations in the root implementation.

Batch Allocation uses the repository `BatchAllocationEngine` through `BatchAllocationAdapter`. The adapter evaluates the current waiting queue as a batch snapshot because the integrated simulator owns queue persistence, retries, resource release, and event lifecycle.

## Experimental Setup

All strategies were evaluated under `models/v4_replay.bpmn`; therefore absolute results are conditional on this BPMN. Fixed replay used 76 held-out temporal-test routes from 2016-09-16T15:07:00+00:00 to 2016-09-17T15:07:00+00:00, seeds 1-5, and a drain horizon of 2016-11-16T15:07:00+00:00. Generative evaluation used transition-aware BPMN enforcement, seeds 1-3, a three-hour arrival window, and an eight-hour drain.

## Branching Evaluation

Label-level predictive quality on the 70/30 temporal split: accuracy 0.5632, macro-F1 0.5190, weighted-F1 0.5957. BPMN transition alignment is separate: held-out sample coverage was 0.7080, exact transition accuracy over identifiable synchronized observations was 1.0, and coverage-adjusted accuracy was 0.7080.

## Resource-Allocation Results

Fixed replay provides the controlled allocation comparison and generative mode demonstrates end-to-end BPMN integration. Aggregate tables are in `method_summary.csv` and `comparison_table.md`. No strategy selects BPMN routes.

## Comparison

With few seeds, results should be treated descriptively. The comparison emphasizes completion reliability, cycle time, waiting time, throughput, utilization/fairness, and method-specific overhead.

## Discussion

Relative comparisons are valid for the selected BPMN and shared configuration. Absolute generalization to the complete BPIC17 language is limited by known BPMN/log conformance gaps.

## Threats To Validity And Limitations

The bounded generative evaluation is not a full production-scale simulation. Fixed replay does not validate branching. Kunkler required adapter-level repair around an incomplete root implementation. Batch is evaluated as a current-queue snapshot rather than a separate queue-owning engine.
