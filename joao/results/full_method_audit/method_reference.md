# Method Reference

## ProbabilityBranchingEngine
Learns empirical transition counts from event logs and samples one BPMN-valid next activity. Inputs are a current event and `possibleActivities`; output is a one-item list or empty list. Randomness is `random.Random(seed)`. Unknown current activity, missing counts, zero filtered probability mass, and untrained cases fall back to seeded random BPMN-valid selection. It keeps transition counts, probabilities, decision points, and training state.

## AttributeBasedBranchingEngine
Evaluates ordered rules at the current activity. Rules specify decision point, attribute, operator, value, and preferred activities. It extracts attributes from dicts and simulator event objects. It returns the first preferred activity that is BPMN-valid, otherwise delegates to fallback or seeded random valid selection. Missing attributes and unknown values do not match. State is limited to counters.

## AttributeSamplingBranchingEngine
Copies runtime attributes, samples configured missing attributes, derives simple amount/risk/complexity attributes, modifies priority/rework indicators, wraps the event, and delegates to the base engine. It is seeded and keeps enrichment/fallback counters. It must still accept only BPMN-valid base results.

## PredictiveBranchingEngine
Builds a supervised next-activity dataset at discovered decision points, trains a `RandomForestClassifier` pipeline, and during simulation uses `predict_proba` to choose the highest-probability BPMN-valid class. Runtime feature alignment handles missing numeric/categorical values and unknown categories. Simulation does not train. Fallback is explicit.

## CompositeBranchingEngine
Tries engines in priority order: predictive, attribute sampling, attribute based, probability, random fallback. It records success/failure/invalid counters and returns only one final BPMN-valid result. Single possible activities and empty lists bypass engine calls.

## Persisted Composite Artifact
`CompositeBranchingArtifact.py` exports a payload with format marker, metadata, seed, and copied engines after removing training logs and resetting runtime counters. Loading deep-copies engines, resets state, and builds a non-training composite. The sklearn 1.9.0 artifact path is `joao/models/branching/final_composite_branching_sklearn190.pkl`.

## Random
`RandomResourceAllocation` iterates available resources and randomly selects one eligible unassigned/unblocked task per resource. Eligibility includes availability and skills/permissions. It mutates only task `assigned` flags to avoid duplicate task assignment in one decision.

## RoundRobin
`RoundRobinResourceAllocation` sorts available resources by id, rotates from a persistent next-resource pointer, assigns the oldest eligible task, skips unavailable/unauthorized resources, and advances only after assignments. Separate instances have independent state.

## ShortestQueue
`ShortestQueueAllocation` assigns oldest/highest-priority remaining tasks to feasible available resources with the smallest cumulative `resource_loads` value, using resource id as tie-break. Missing loads default to `0.0`. It does not maintain a hidden internal queue or mutate group loads.

## ParkSongAllocation
Builds current candidates and predicted candidates, then chooses the minimum cost per resource. Cost is processing estimate minus waiting/priority rewards plus prediction uncertainty and idling penalties. Current selections assign tasks; predicted selections emit reservation decisions. It respects availability and skills. Reservation lifecycle is not stored in this class; it is owned by `IntegratedAllocationEngine`.

## ParkSongML
`ParkSongMLIntegration` uses `MLPredictionAdapter` to convert a trained predictive branching engine's probabilities into `Prediction` objects, then passes them to `ParkSongAllocation`. It adds ML-supplied future-task estimates; the allocation objective remains ParkSong.

## MLPredictionAdapter
Extracts event features through the predictive engine, aligns schema, calls `model.predict_proba`, filters to BPMN-allowed activities, normalizes outputs as `Prediction` objects, and sorts by probability. It returns no predictions when no possible activities exist, the engine is untrained, or no model is loaded. It does not train.

## Batch Allocation
`BatchAllocationAdapter` wraps group `BatchAllocationEngine` as a snapshot comparator. It converts current tasks/resources, calls `fire_batch`, suppresses engine prints, prevents duplicate task/resource decisions, and returns idle decisions for unused available resources.

## Künstler/Küncler
Only a formalization notebook was found (`notebooks/2.3.1_formalization_kunkler.ipynb`). No production allocation strategy class was found in the source tree, so it is reference/background only for this audit.
