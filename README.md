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
├── data/                     # BPIC-17 log (not in repo)
├── models/                   # BPMN models (with corresponding PDF view)
├── notebooks/                # offline fitting
├── optimization/             # Part II: task 1.1–1.2 
├── processTimes/             # Part I: task 1.3 
├── report/                   # Report basic framework (please continue writing and share it here)
├── resources/                # Part I: task 1.6–1.8 
├── results/                  # generated tables/figures for the report
├── scripts/                  # offline/helper scripts
├── arrival_engine.py         # Part I: task 1.2
├── BPMN_engine.py            # Part I: task 1.4 
├── DummyEngines.py           # placeholder engines, replaced per task
├── EngineCoreChanges.md      # engine core changes requested
├── Helper.py                 # Event / Case / EventType
├── requirements.txt
└── SimulationEngineCore.py   # Part I: task 1.1 (Core)
```

Details: TBD

This project is not finished yet and subject to ongoing changes.

### Setup
The _requirements.txt_ contains all dependencies necessary to execute the project.

**bash:** 

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt