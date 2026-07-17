# Five-Method Comparison Table

| mode | method | runs | cases_admitted_mean | cases_completed_mean | completion_rate_mean | final_marking_rate_mean | cycle_time_mean_s_mean | waiting_time_mean_s_mean | throughput_cases_per_hour_mean | resource_fairness_gini_mean | simulation_runtime_seconds_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_replay | R-RRA / RoundRobin | 5 | 76 | 76 | 1 | 1 | 9.229e+04 | 1725 | 0.05191 | 0.5455 | 8.623 |
| fixed_replay | R-SHQ / ShortestQueue | 5 | 76 | 75.8 | 0.9974 | 0.9974 | 6.518e+04 | 1677 | 0.05178 | 0.4738 | 8.5 |
| fixed_replay | ParkSong-Composite | 5 | 76 | 76 | 1 | 1 | 1.046e+05 | 4063 | 0.05191 | 0.5319 | 18.02 |
| fixed_replay | Kunkler | 5 | 76 | 75.8 | 0.9974 | 0.9974 | 9.896e+04 | 1633 | 0.05178 | 0.4892 | 7.91 |
| fixed_replay | Batch Allocation | 5 | 76 | 75.8 | 0.9974 | 0.9974 | 7.165e+04 | 1639 | 0.05178 | 0.4912 | 8.082 |
| generative | R-RRA / RoundRobin | 3 | 37.33 | 29.33 | 0.7793 | 0.7793 | 1.346e+04 | 2656 | nan | nan | 0.272 |
| generative | R-SHQ / ShortestQueue | 3 | 37.33 | 30.33 | 0.8142 | 0.8142 | 1.474e+04 | 2540 | nan | nan | 0.332 |
| generative | ParkSong-Composite | 3 | 37.33 | 30 | 0.7973 | 0.7973 | 1.482e+04 | 317.3 | nan | nan | 0.5463 |
| generative | Kunkler | 3 | 37.33 | 30.67 | 0.8175 | 0.8175 | 1.417e+04 | 2957 | nan | nan | 0.3117 |
| generative | Batch Allocation | 3 | 37.33 | 30.33 | 0.8125 | 0.8125 | 1.376e+04 | 1843 | nan | nan | 0.3063 |
