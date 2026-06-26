import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import pickle
import xml.etree.ElementTree as ET

import pandas as pd
import pm4py

from src.branching.ProbabilityBranchingEngine import ProbabilityBranchingEngine
from src.branching.PredictiveBranchingEngine import PredictiveBranchingEngine


CASE_COL = "case:concept:name"
ACTIVITY_COL = "concept:name"
TIMESTAMP_COL = "time:timestamp"


def load_log(path: str) -> pd.DataFrame:
    log_path = Path(path)

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    if log_path.suffix.lower() == ".csv":
        log = pd.read_csv(log_path)

    elif log_path.suffix.lower() == ".xes":
        try:
            raw_log = pm4py.read_xes(str(log_path))
            log = pm4py.convert_to_dataframe(raw_log)
        except Exception as error:
            print(f"[load_log] PM4Py conversion failed: {error}")
            print("[load_log] Falling back to manual XES parsing.")

            tree = ET.parse(log_path)
            root = tree.getroot()

            namespace = ""
            if root.tag.startswith("{"):
                namespace = root.tag.split("}")[0] + "}"

            rows = []

            for trace in root.findall(f"{namespace}trace"):
                case_id = None

                for child in trace:
                    key = child.attrib.get("key")
                    value = child.attrib.get("value")

                    if key == "concept:name":
                        case_id = value

                for event in trace.findall(f"{namespace}event"):
                    row = {}

                    for attr in event:
                        key = attr.attrib.get("key")
                        value = attr.attrib.get("value")

                        if key is not None:
                            row[key] = value

                    if CASE_COL not in row:
                        row[CASE_COL] = case_id

                    rows.append(row)

            log = pd.DataFrame(rows)

    else:
        raise ValueError("Only CSV and XES logs are supported.")

    log[TIMESTAMP_COL] = pd.to_datetime(log[TIMESTAMP_COL], errors="coerce")

    log = log.dropna(
        subset=[
            CASE_COL,
            ACTIVITY_COL,
            TIMESTAMP_COL,
        ]
    ).reset_index(drop=True)

    print(
        f"[load_log] Loaded {len(log)} events and "
        f"{log[CASE_COL].nunique()} cases."
    )

    return log


def main():
    parser = argparse.ArgumentParser(
        description="Train final FULL predictive branching model."
    )

    parser.add_argument(
        "--log",
        required=True,
        help="Path to CSV or XES event log.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    log = load_log(args.log)

    print("\n=== Training FULL model ===")
    print(f"Training events: {len(log)}")
    print(f"Training cases: {log[CASE_COL].nunique()}")

    probability_engine = ProbabilityBranchingEngine(
        log=log,
        seed=args.seed,
    )
    probability_engine.train()

    candidate_features = [
        "case:ApplicationType",
        "case:LoanGoal",
        "case:RequestedAmount",
        "CreditScore",
        "EventOrigin",
        "org:resource",
    ]

    feature_columns = [
        column for column in candidate_features
        if column in log.columns
    ]

    predictive_engine = PredictiveBranchingEngine(
        fallback_engine=probability_engine,
        feature_columns=feature_columns,
        seed=args.seed,
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=2,
    )

    predictive_engine.train(log)

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "final_predictive_model_full.pkl"

    with open(model_path, "wb") as file:
        pickle.dump(
            {
                "predictive_engine": predictive_engine,
                "probability_engine": probability_engine,
                "feature_columns": feature_columns,
                "training_mode": "full_log",
                "train_events": len(log),
                "train_cases": log[CASE_COL].nunique(),
                "reference_validation_model": "final_predictive_model_0_7.pkl",
            },
            file,
        )

    summary = pd.DataFrame(
        [
            {
                "model": "PredictiveML_with_ProbabilityFallback_FULL",
                "training_mode": "full_log",
                "train_events": len(log),
                "train_cases": log[CASE_COL].nunique(),
                "feature_columns": ", ".join(feature_columns),
                "reference_validation_model": "final_predictive_model_0_7.pkl",
            }
        ]
    )

    summary_path = output_dir / "final_predictive_model_full_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n=== FULL model trained ===")
    print(summary.to_string(index=False))
    print(f"\nSaved model to: {model_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
