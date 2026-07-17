# Integration Data Flow

## Flow and Responsibilities

`IntegratedAllocationEngine` receives simulator events from group `Engine`, predicts BPMN branches through `CompositeBranchingAdapter`, stores `BranchPrediction` objects keyed by stable target task id, converts waiting events into João `Task` snapshots, converts available/authorized group resources into João `Resource` snapshots, invokes the configured allocation strategy, applies assignments to group `resourceEngine.busy/load`, and schedules processing end events.

## Mutable State

Important mutable maps are `_task_cache`, `_event_id_to_task_id`, `future_predictions_by_task_id`, `branch_prediction_by_task_id`, `prediction_id_by_task_id`, `reservations_by_resource_id`, `reservation_by_target_task_id`, `reservation_history`, and `task_lifecycle`.

## Reservation Lifecycle

Base ParkSong emits reservation decisions only. The integration layer validates matching branch predictions, creates `ResourceReservation` records, rejects worse overwrite attempts, consumes matching case/activity/task reservations, expires unavailable resources or overdue reservations, cancels permission loss and mismatches, and cleans unresolved reservations at horizon end.

## Infrastructure Separation

`MIN_VISIBLE_PROCESSING_DURATION` is applied only in `_normalized_processing_duration` when a visible activity receives invalid/zero duration. Allocation snapshots are built and strategy decisions are made before processing duration scheduling, so the guard does not change resource choice for identical input snapshots.
