# Canonical Consistency Audit - 2026-07-17

This audit records the consistency cleanup performed after the branching leakage
correction. No models, experimental CSV values, BPMN files or fixed-replay
results were regenerated in this cleanup step.

| arquivo | trecho | classificacao | correcao necessaria |
|---|---|---|---|
| `joao/results/final_canonical_20260716/README.md` | `# Canonical Final Results` and "canonical result package" | contraditoria | Mark as historical pre-correction; point to corrected 2026-07-17 packages. |
| `joao/results/final_canonical_20260716/README.md` | fixed replay command using `final_composite_branching.pkl` | comando obsoleto within historical package | Keep as historical command, explicitly not final reconstruction command. |
| `report/sections/evaluation_protocol_joao.tex` | canonical run uses `final_composite_branching.pkl` | relatorio obsoleto | Replace with `composite_branching_evaluation_train70.pkl` and corrected SHA. |
| `joao/README.md` | final package is `results/final_canonical_20260716/` | contraditoria | Point to corrected branching and fixed-replay packages; mark 2026-07-16 historical. |
| `joao/tests/test_final_canonical_reproducibility.py` | protects `final_canonical_20260716` and `final_composite_branching.pkl` | teste obsoleto | Replace with corrected canonical reproducibility test. |
| `joao/tests/resource_allocation/test_my_methods_integrated_comparison_script.py` | example config writes `final_composite_branching.pkl` | teste obsoleto | Use `composite_branching_evaluation_train70.pkl` in the example configuration. |
| `joao/.gitignore` | exception for `final_composite_branching.pkl` | support-package obsoleto | Remove exception so the historical full-log artifact is not accidentally treated as final. |
| `report/joao_evaluation_protocol_support/results/final_canonical_20260716/` | copied historical package paths | historica explicitamente identificada after README cleanup | Preserve; do not use for final claims. |
| `joao/results/branching_corrected_20260717/pre_refactor_audit.md` | old artifact and old canonical paths | historica explicitamente identificada | Keep unchanged as audit evidence. |
| `joao/results/branching_corrected_20260717/branching_artifacts_metadata.json` | absolute local paths from generated artifact metadata | correta but nonportable generated metadata | Leave experimental metadata unchanged; report path caveat. |
| `report/sections/branching_joao.tex` | evaluation/deployment distinction | correta e atual | No correction. |
| `report/sections/branching_eval_joao.tex` | corrected held-out metrics and historical old artifact note | correta e atual | No correction. |
| `report/joao_branching_support/README.md` | evaluation/deployment/historical distinction | correta e atual | No correction except manifest refresh. |
| `report/joao_evaluation_protocol_support/README.md` | corrected package paths | correta e atual | No correction except manifest refresh. |

Canonical matrix after cleanup:

| finalidade | path | artifact/hash | status |
|---|---|---|---|
| Branching evaluation | `joao/results/branching_corrected_20260717/` | evaluation artifact `ebb16554ecf0e42faa87c7334faef3ab81b742b8e29820df1ce032bd4109f1c4` | canonical |
| Branching evaluation artifact | `joao/models/branching/composite_branching_evaluation_train70.pkl` | `ebb16554ecf0e42faa87c7334faef3ab81b742b8e29820df1ce032bd4109f1c4` | canonical evaluation |
| Branching deployment artifact | `joao/models/branching/composite_branching_deployment_full.pkl` | `6dcd01744f635d0fdd24008c3e7c4ae28bedb340c4f37aabbc1bc53fd7e7ab3e` | deployment only |
| Corrected fixed replay | `joao/results/final_canonical_branching_corrected_20260717/fixed_replay/` | uses evaluation artifact `ebb16554ecf0e42faa87c7334faef3ab81b742b8e29820df1ce032bd4109f1c4` | canonical |
| Transition-aware artifact | `joao/models/branching/transition_aware_branching_v1_20260715.pkl` | `79127e6b1cea6cb58fc1f1f19b1ce96564837a24aa5adf80efcbcc71fc183e54` | canonical support |
| Processing-time artifact | `joao/models/process_time/final_process_time_coverage_v2.pkl` | `c540304cdbb6f60159ad1023112e6d5e71aeecae68b885ee2e2b1ac3c826a886` | canonical support |
| Historical full-log artifact | `joao/models/branching/final_composite_branching.pkl` | `e364dce5e00b05d2f6afa1b0eee835cfd224c3884d82c7b2522a47778c6ae9af` | historical/pre-correction |
| Historical fixed replay | `joao/results/final_canonical_20260716/` | uses full-log artifact | historical/pre-correction |
