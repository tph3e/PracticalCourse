# Results Interpretation

The five strategies were evaluated in the same selected process environment, `models/v4_replay.bpmn`. Absolute performance is conditional on that BPMN model and its known incomplete coverage of BPIC17 behavior.

Fixed replay is the controlled allocation comparison: all methods receive the same 76 held-out historical routes, seeds 1-5, one-day arrival interval, and 60-day drain. Generative mode is a bounded BPMN-enforced integration check: routes are selected by the transition-aware branching path and resource allocation only assigns tasks/resources.

Best fixed-replay mean waiting time: Kunkler (1633.08s). Best generative completion rate: Kunkler (0.817). These are descriptive comparisons only; no statistical significance is claimed from five or three seeds.
