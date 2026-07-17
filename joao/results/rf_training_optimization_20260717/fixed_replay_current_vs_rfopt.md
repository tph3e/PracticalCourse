# Fixed Replay Current vs RF-Optimized Candidate

Current package: `joao/results/final_canonical_branching_corrected_20260717/fixed_replay`
Candidate package: `joao/results/final_canonical_rfopt_candidate_20260717/fixed_replay`
Ranking metric: `cycle_time_mean_s_mean` lower is better.
Ranking changed: `True`.
Current failures: `0`; candidate failures: `0`.

## Ranking

Current:
1. ShortestQueue - cycle mean 65184.340s, completion 0.997368
2. Batch - cycle mean 71654.255s, completion 0.997368
3. RoundRobin - cycle mean 77197.108s, completion 0.997368
4. ParkSong-Composite - cycle mean 93346.778s, completion 0.997368
5. Kunkler-Rinderle-Ma - cycle mean 102065.265s, completion 0.997368

RF optimized candidate:
1. ShortestQueue - cycle mean 65184.340s, completion 0.997368
2. Batch - cycle mean 71654.255s, completion 0.997368
3. Kunkler-Rinderle-Ma - cycle mean 103557.963s, completion 0.997368
4. RoundRobin - cycle mean 105187.338s, completion 1.000000
5. ParkSong-Composite - cycle mean 116029.344s, completion 1.000000

## Key Deltas

### fixed_route_completion_rate_mean
- Batch: 0.997368 -> 0.997368 (delta 0.000000, 0.000%)
- Kunkler-Rinderle-Ma: 0.997368 -> 0.997368 (delta 0.000000, 0.000%)
- ParkSong-Composite: 0.997368 -> 1.000000 (delta 0.002632, 0.264%)
- RoundRobin: 0.997368 -> 1.000000 (delta 0.002632, 0.264%)
- ShortestQueue: 0.997368 -> 0.997368 (delta 0.000000, 0.000%)

### cases_completed_mean
- Batch: 75.800000 -> 75.800000 (delta 0.000000, 0.000%)
- Kunkler-Rinderle-Ma: 75.800000 -> 75.800000 (delta 0.000000, 0.000%)
- ParkSong-Composite: 75.800000 -> 76.000000 (delta 0.200000, 0.264%)
- RoundRobin: 75.800000 -> 76.000000 (delta 0.200000, 0.264%)
- ShortestQueue: 75.800000 -> 75.800000 (delta 0.000000, 0.000%)

### cases_censored_mean
- Batch: 0.200000 -> 0.200000 (delta 0.000000, 0.000%)
- Kunkler-Rinderle-Ma: 0.200000 -> 0.200000 (delta 0.000000, 0.000%)
- ParkSong-Composite: 0.200000 -> 0.000000 (delta -0.200000, -100.000%)
- RoundRobin: 0.200000 -> 0.000000 (delta -0.200000, -100.000%)
- ShortestQueue: 0.200000 -> 0.200000 (delta 0.000000, 0.000%)

### cycle_time_mean_s_mean
- Batch: 71654.254544 -> 71654.254544 (delta 0.000000, 0.000%)
- Kunkler-Rinderle-Ma: 102065.265440 -> 103557.963083 (delta 1492.697642, 1.462%)
- ParkSong-Composite: 93346.778082 -> 116029.344411 (delta 22682.566328, 24.299%)
- RoundRobin: 77197.108168 -> 105187.337745 (delta 27990.229577, 36.258%)
- ShortestQueue: 65184.340368 -> 65184.340368 (delta 0.000000, 0.000%)

### waiting_time_mean_s_mean
- Batch: 1638.605784 -> 1638.605784 (delta 0.000000, 0.000%)
- Kunkler-Rinderle-Ma: 1642.174206 -> 1635.290685 (delta -6.883521, -0.419%)
- ParkSong-Composite: 2875.380778 -> 4401.933001 (delta 1526.552222, 53.090%)
- RoundRobin: 1559.589418 -> 1637.309121 (delta 77.719704, 4.983%)
- ShortestQueue: 1677.330286 -> 1677.330286 (delta 0.000000, 0.000%)

### resource_fairness_gini_mean
- Batch: 0.491152 -> 0.491152 (delta 0.000000, 0.000%)
- Kunkler-Rinderle-Ma: 0.490708 -> 0.489621 (delta -0.001088, -0.222%)
- ParkSong-Composite: 0.516557 -> 0.528371 (delta 0.011814, 2.287%)
- RoundRobin: 0.571047 -> 0.532718 (delta -0.038329, -6.712%)
- ShortestQueue: 0.473771 -> 0.473771 (delta 0.000000, 0.000%)

### weighted_resource_fairness_mean
- Batch: 0.005439 -> 0.005439 (delta 0.000000, 0.000%)
- Kunkler-Rinderle-Ma: 0.005397 -> 0.005413 (delta 0.000015, 0.280%)
- ParkSong-Composite: 0.004358 -> 0.004440 (delta 0.000082, 1.880%)
- RoundRobin: 0.006466 -> 0.005267 (delta -0.001200, -18.551%)
- ShortestQueue: 0.003236 -> 0.003236 (delta 0.000000, 0.000%)

### horizon_normalized_throughput_cases_per_hour_mean
- Batch: 0.051776 -> 0.051776 (delta 0.000000, 0.000%)
- Kunkler-Rinderle-Ma: 0.051776 -> 0.051776 (delta 0.000000, 0.000%)
- ParkSong-Composite: 0.051776 -> 0.051913 (delta 0.000137, 0.264%)
- RoundRobin: 0.051776 -> 0.051913 (delta 0.000137, 0.264%)
- ShortestQueue: 0.051776 -> 0.051776 (delta 0.000000, 0.000%)

### horizon_normalized_resource_occupation_mean_mean
- Batch: 0.002827 -> 0.002827 (delta 0.000000, 0.000%)
- Kunkler-Rinderle-Ma: 0.002789 -> 0.002796 (delta 0.000007, 0.241%)
- ParkSong-Composite: 0.001801 -> 0.001796 (delta -0.000005, -0.254%)
- RoundRobin: 0.002481 -> 0.002122 (delta -0.000359, -14.461%)
- ShortestQueue: 0.001416 -> 0.001416 (delta 0.000000, 0.000%)

### tasks_assigned_mean
- Batch: 2318.000000 -> 2318.000000 (delta 0.000000, 0.000%)
- Kunkler-Rinderle-Ma: 2326.400000 -> 2331.800000 (delta 5.400000, 0.232%)
- ParkSong-Composite: 2802.400000 -> 2786.600000 (delta -15.800000, -0.564%)
- RoundRobin: 2317.000000 -> 2325.000000 (delta 8.000000, 0.345%)
- ShortestQueue: 2311.200000 -> 2311.200000 (delta 0.000000, 0.000%)
