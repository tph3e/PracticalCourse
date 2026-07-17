# Test Strength Report

- ShortestQueue selects the lowest cumulative load: status=pass; guard=test_shortest_queue_selects_lowest_cumulative_resource_load; invariant strictly lower-load test
- RoundRobin advances after multi-assignment epoch: status=pass; guard=test_round_robin_continues_after_multi_assignment_epoch
- ParkSong respects permissions: status=pass; guard=ParkSong skills/permissions tests and same-snapshot invariant
- ML adapter filters invalid activities: status=pass; guard=MLPredictionAdapter impossible-activity filtering
- Duplicate task/resource assignments are rejected: status=pass; guard=duplicate task/resource invariant tests
