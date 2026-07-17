import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import pickle
import xml.etree.ElementTree as ET

import pandas as pd
import pm4py

from src.branching.BranchingUtils import temporal_train_test_split_by_case
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
        description="Train final predictive branching model."
    )

    parser.add_argument(
        "--log",
        required=True,
        help="Path to CSV or XES event log.",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    log = load_log(args.log)

    train_log, test_log = temporal_train_test_split_by_case(
        log=log,
        case_col=CASE_COL,
        timestamp_col=TIMESTAMP_COL,
        train_ratio=args.train_ratio,
    )

    print("\n=== Split ===")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Train events: {len(train_log)}")
    print(f"Test events: {len(test_log)}")
    print(f"Train cases: {train_log[CASE_COL].nunique()}")
    print(f"Test cases: {test_log[CASE_COL].nunique()}")

    probability_engine = ProbabilityBranchingEngine(
        log=train_log,
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
        if column in train_log.columns
    ]

    predictive_engine = PredictiveBranchingEngine(
        fallback_engine=probability_engine,
        feature_columns=feature_columns,
        seed=args.seed,
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=2,
        use_bpmn_replay=True,
        bpmn_model_path="models/v4_replay.bpmn",
    )

    print("\n=== Training PredictiveBranchingEngine ===")
    predictive_engine.train(train_log)

    print("\n=== Evaluating Final Model ===")
    metrics = predictive_engine.evaluate(test_log)

    results = {
        "model": "PredictiveML_BPMNReplay_with_ProbabilityFallback",
        "training_dataset_mode": predictive_engine.dataset_mode,
        "bpmn_model": predictive_engine.bpmn_model_path,
        "train_ratio": args.train_ratio,
        "train_events": len(train_log),
        "test_events": len(test_log),
        "train_cases": train_log[CASE_COL].nunique(),
        "test_cases": test_log[CASE_COL].nunique(),
        "feature_columns": ", ".join(feature_columns),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "n_samples": metrics["n_samples"],
    }

    results_df = pd.DataFrame([results])

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    ratio_label = str(args.train_ratio).replace(".", "_")
    metrics_path = output_dir / f"final_predictive_model_metrics_{ratio_label}.csv"
    model_path = output_dir / f"final_predictive_model_{ratio_label}.pkl"

    results_df.to_csv(metrics_path, index=False)

    with open(model_path, "wb") as file:
        pickle.dump(
            {
                "predictive_engine": predictive_engine,
                "probability_engine": probability_engine,
                "feature_columns": feature_columns,
                "train_ratio": args.train_ratio,
                "metrics": results,
                "training_dataset_mode": predictive_engine.dataset_mode,
                "training_bpmn_replay_diagnostics": (
                    predictive_engine.training_bpmn_replay_diagnostics
                ),
                "evaluation_bpmn_replay_diagnostics": (
                    predictive_engine.evaluation_bpmn_replay_diagnostics
                ),
            },
            file,
        )

    print("\n=== Final Model Metrics ===")
    print(results_df.to_string(index=False))

    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
