# PracticalCourse
This git repository contains a student team's work for the **practical course** in **Business Process Prediction, Simulation, and Optimization (IN0012)** at Technical University Munich (TUM) in the summer term 2026.

### Data
It works with the data of the **Business Process Intelligence Challenge (BPIC) 2017**. 

Download BPI Challenge 2017.xes from the official 4TU.ResearchData record and place it in **data/**:
- van Dongen, B. (2017). BPI Challenge 2017. 4TU.ResearchData.
- dataset page: https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884

### Simulation architecture
For each of the following tasks an own file is created.
- Task 1.1 (Simulation Engine Core)
- Task 1.2 (Case Arrivals)
- Task 1.3 (Processing Times)
- Task 1.4 (Process Model)
- Task 1.5 (Branching Decisions)
- Task 1.6 to 1.8 (Resource Availabilities, Resource Permissions, Resource Allocation)

### Project layout

```
.
├── SimulationEngineCore.py   # core (Task 1.1)
├── DummyEngines.py           # placeholder engines, replaced per task
├── Helper.py                 # Event / Case / EventType
├── resources/                # tasks 1.6–1.8 
├── notebooks/                # offline fitting
├── data/                     # BPIC-17 log (not in repo)
├── results/                  # generated tables/figures for the report
├── EngineCoreChanges.md             # engine core changes requested
└── requirements.txt
```

Details: TBD

This project is not finished yet and subject to ongoing changes.

The _requirements.txt_ contains all dependencies necessary to execute the project.