# Assignment Method Mapping

No assignment PDF was present in the repository scan. The mapping below is based on repository code, README text, notebooks, and prior audit notes, not external assignment text.

| assignment requirement | implemented method | class/file | basic or advanced | evaluation script | test coverage | current status |
|---|---|---|---|---|---|---|
| Probability branching | Probability branching | `ProbabilityBranchingEngine.py` | basic | branching scripts, composite artifact | branching tests | implemented |
| Attribute branching | Attribute rules | `AttributeBasedBranchingEngine.py` | advanced/simple rules | branching scripts | branching tests | implemented |
| Attribute sampling | Attribute sampling wrapper | `AttributeSamplingBranchingEngine.py` | advanced | branching scripts | branching tests | implemented |
| Predictive branching | RandomForest next-activity model | `PredictiveBranchingEngine.py` | advanced | `train_final_predictive_model.py` | branching/model tests | implemented |
| Composite branching | priority/fallback hierarchy | `CompositeBranchingEngine.py` | integration | integrated runners | composite tests | implemented |
| R-RRA | Resource-aware Round Robin in current code | `RoundRobinResourceAllocation.py` | baseline/required by prior audit | integrated and audit runners | unit/invariant tests | implemented |
| R-SHQ | Resource-aware Shortest Queue | `ShortestQueueAllocation.py` | baseline/required by prior audit | integrated and audit runners | unit/invariant tests | implemented |
| Random | stochastic baseline | `RandomResourceAllocation.py`, group `resources/allocation.py` | baseline | scenario/integrated runners | unit/invariant tests | implemented |
| ParkSong | prediction-aware strategic idling approximation | `ParkSongAllocation.py` | advanced | scenario/integrated/audit runners | unit/integration tests | implemented |
| ParkSongML | ML predictions feeding ParkSong | `MLPredictionAdapter.py`, `ParkSongMLIntegration.py` | advanced input layer | controlled audit comparison | adapter/integration tests | controlled integration implemented; full runner currently labels ParkSong separately |
| Batch Allocation | group/reference snapshot comparator | `BatchAllocationEngine.py`, `BatchAllocationAdapter.py` | comparison | full/audit runner | adapter tests | present as comparison |
| Künstler/Küncler | formalization notebook only | `notebooks/2.3.1_formalization_kunkler.ipynb` | reference | none | none | no production method found |

Resolution: current repository evidence supports R-RRA as RoundRobin, R-SHQ as ShortestQueue, Random as baseline, ParkSongML as an ML prediction supplier for ParkSong, and ParkSongAllocation as the allocation objective. Batch is a comparison adapter; Künstler/Küncler is not production code here.
