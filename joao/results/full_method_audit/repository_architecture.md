# Repository Architecture

## Scope

This audit covers João-owned branching and resource-allocation code under `joao/`, plus the group simulator/resource/process-time files that route those methods at runtime.

## Relevant Tree

```text
joao/
  src/branching/                  Branching engines, utilities, artifact export/load.
  src/resource_allocation/         Snapshot allocation strategies, metrics, adapters.
  src/resource_allocation/integration/ Integrated simulator subclass and context adapters.
  scripts/branching/              Training/export/report scripts.
  scripts/resource_allocation/     Scenario, integrated, processing-time, and audit scripts.
  tests/                          João tests.
  models/branching/               Composite branching artifacts.
  models/process_time/             Processing-time artifacts.
  results/                         Generated results and audit evidence.
resources/
  allocation.py                    Group pick-interface baseline allocation.
  resource_engine.py               Group resource permissions/availability/load integration.
processTimes/
  process_time_engine.py           Group/shared processing-time sampling.
```

## Runtime Data Flow

`event arrival -> task creation -> BPMN enabled activities -> branching prediction -> Task/Resource snapshot conversion -> availability filtering -> permission filtering -> allocation strategy -> reservation maps -> processing-time scheduling -> activity completion -> metrics`.

Important ownership: João owns `joao/src/branching`, `joao/src/resource_allocation`, and João scripts/tests/results. The root simulator, BPMN, process-time engine, and resource engine are shared group code. `BatchAllocationEngine.py` and `resources/allocation.py` are group/reference implementations used only through adapters/comparisons.

## Model Artifacts

- `joao/models/branching/final_composite_branching_sklearn190.pkl`: present, sha256=9cc504fc84603a8db5a2a00507b329c0fbcf6303aa948990dcf14f10289b48ab
- `joao/models/process_time/final_process_time_coverage_v2.pkl`: present, sha256=c540304cdbb6f60159ad1023112e6d5e71aeecae68b885ee2e2b1ac3c826a886

See `file_inventory.csv` for every reviewed file, class/function summary, imports, tests, ownership, and category.
