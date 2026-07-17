# Reproducibility

Branch: `joao/integrate-methods-2026-07-15`
Commit: `24a5545ac7e8d170c3ce1a8e2ad6f7d9204dce99`
Python: `3.14.3`
PM4Py: `2.7.22.2`

Commands:
```bash
python3 joao/scripts/branching/train_transition_aware_branching.py --train-case-limit 500 --test-case-limit 200
python3 joao/scripts/branching/train_final_predictive_model.py --log data/logData.xes --train-ratio 0.7 --seed 1
python3 joao/scripts/branching/train_full_predictive_model.py --log data/logData.xes --seed 1
python3 joao/scripts/branching/export_final_composite_branching_artifact.py --log data/logData.xes --seed 1
python3 joao/scripts/branching/run_transition_aware_generative_smoke.py --hours 3 --drain-hours 8 --seeds 1,2,3 --output-dir joao/results/final_validation_20260715/generative
python3 joao/scripts/branching/final_validation_analysis.py
pytest -q
```

Temporal split: train=22056, held-out test=9453, overlap=0.

Important artifact distinction:
- Offline BPMN-replay composite artifact: `joao/models/branching/final_composite_branching.pkl`.
- Preserved fixed-replay ranking artifact: `joao/results/joao_calibration_20260715_roundrobin_split_v4/models/composite_branching_temporal_split.pkl`.