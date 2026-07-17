# Method Complexity Analysis

Snapshot complexity is roughly O(R*T) for Random/RoundRobin/ParkSong candidate filtering, O(T*R) for ShortestQueue with load lookup, and adapter overhead for Batch. Predictive branching runtime is model inference plus feature alignment; training is excluded from simulation.
