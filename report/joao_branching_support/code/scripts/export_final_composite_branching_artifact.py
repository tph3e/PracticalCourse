from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pm4py

JOAO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = JOAO_ROOT.parent
sys.path.insert(0, str(JOAO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from joao.src.branching.CompositeBranchingArtifact import (
    export_composite_branching_artifact,
)
from joao.src.branching.CompositeBranchingEngine import CompositeBranchingEngine


def load_log(path: str) -> pd.DataFrame:
    if path.endswith(".xes"):
        return pm4py.read_xes(path, variant="r4pm")
    return pd.read_csv(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and export a deployment-only full-log Composite branching "
            "artifact. Use the leakage-free evaluation artifacts for metrics."
        )
    )
    parser.add_argument("--log", default=str(REPO_ROOT / "data" / "logData.xes"))
    parser.add_argument(
        "--output",
        default=str(JOAO_ROOT / "models" / "branching" / "composite_branching_deployment_full.pkl"),
    )
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log = load_log(args.log)
    composite = CompositeBranchingEngine(
        log=log,
        seed=args.seed,
        use_default_hierarchy=True,
        train_on_init=True,
        predictive_use_bpmn_replay=True,
        bpmn_model_path="models/v4_replay.bpmn",
    )
    metadata = export_composite_branching_artifact(
        composite=composite,
        path=args.output,
        metadata={
            "training_log": args.log,
            "training_events": int(len(log)),
            "training_cases": int(log["case:concept:name"].nunique()),
            "seed": args.seed,
            "selected_validation_train_ratio": 0.7,
            "selected_validation_evidence": (
                "joao/results/final_predictive_model_split_comparison.csv"
            ),
            "deployment_training_mode": "full_log_after_model_selection",
            "artifact_scope": "deployment",
            "deployment_only": True,
            "not_for_held_out_evaluation": True,
            "predictive_training_dataset_mode": "bpmn_replay",
            "bpmn_model": "models/v4_replay.bpmn",
            "feature_columns": [
                "case:ApplicationType",
                "case:LoanGoal",
                "case:RequestedAmount",
                "CreditScore",
                "EventOrigin",
                "org:resource",
            ],
            "composite_hierarchy": [
                engine.__class__.__name__
                for engine in composite.engines
            ],
        },
    )
    print(f"Saved Composite branching artifact to: {args.output}")
    print(f"Artifact hash: {metadata['artifact_sha256']}")


if __name__ == "__main__":
    main()
