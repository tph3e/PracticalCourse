# Reproducibility

Branch: `joao/integrate-methods-2026-07-15`
Commit: `24a5545ac7e8d170c3ce1a8e2ad6f7d9204dce99`
Python: `3.14.3`
PM4Py: `2.7.22.2`

Commands:
```bash
python3 joao/scripts/branching/train_transition_aware_branching.py --train-case-limit 500 --test-case-limit 200
python3 joao/scripts/branching/run_transition_aware_generative_smoke.py --hours 3 --drain-hours 8 --seeds 1,2,3 --output-dir joao/results/final_validation_20260715/generative
python3 joao/scripts/branching/final_validation_analysis.py
pytest -q
```

Temporal split: train=22056, held-out test=9453, overlap=0.