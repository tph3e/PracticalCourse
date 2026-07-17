# Code Quality Findings

| severity | finding | evidence | recommendation |
|---|---|---|---|
| high | ParkSong reservation lifecycle is split across allocator and integration | `ParkSongAllocation` emits decisions; `IntegratedAllocationEngine` stores lifecycle | Document this split and test both layers; do not claim base allocator alone expires reservations |
| medium | Assignment PDF absent | no PDF found by repository scan | Treat mapping as repository-evidence-based |
| medium | Some generated artifacts and scripts are untracked in the current worktree | `git status` shows many untracked João files | Preserve and document; avoid overwriting prior result directories |
| medium | Existing Random class docstring labels R-RRA as Random Resource Allocation | source docstring conflicts with newer RoundRobin class evidence | Use `RoundRobinResourceAllocation` as R-RRA in current audit; leave old docstring semantics unchanged |
| low | Snapshot strategies mutate `Task.assigned` flags | intentional duplicate-prevention behavior | Tests should pass copied snapshots when immutability matters |
| low | Batch adapter is a snapshot wrapper, not a persistent buffered simulation method | adapter docstring | Label as comparison only |
| documentation only | Künstler/Küncler is notebook/reference only | no production class found | Do not include as João-owned method |
