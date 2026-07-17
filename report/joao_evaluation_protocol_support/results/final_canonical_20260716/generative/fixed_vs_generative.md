# Fixed replay vs generative smoke

Fixed replay is the ranking benchmark for resource allocation on held-out historical routes.
The generative run is a transition-aware BPMN and allocation integration smoke, not a ranking benchmark.
It uses the transition-aware BPMN path with an empty CompositeBranchingEngine hierarchy, so it does not evaluate the Random Forest composite classifier.

Completion is reported both as pooled completion and as mean seed-level completion.