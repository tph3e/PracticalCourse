import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import pandas as pd
import pm4py

from sklearn.metrics import accuracy_score, f1_score

from src.branching.BranchingUtils import temporal_train_test_split_by_case
from src.branching.ProbabilityBranchingEngine import ProbabilityBranchingEngine
from src.branching.AttributeBasedBranchingEngine import AttributeBasedBranchingEngine
from src.branching.AttributeSamplingBranchingEngine import AttributeSamplingBranchingEngine
from src.branching.PredictiveBranchingEngine import PredictiveBranchingEngine


CASE_COL = "case:concept:name"
ACTIVITY_COL = "concept:name"
TIMESTAMP_COL = "time:timestamp"


def load_log(path: str) -> pd.DataFrame:
    """
    Load an event log from CSV or XES.

    For XES, this function first tries PM4Py. If PM4Py cannot convert the log
    because timestamps were exported as strings, it falls back to a lightweight
    XML parser and manually converts time:timestamp to datetime.
    """

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

            import xml.etree.ElementTree as ET

            tree = ET.parse(log_path)
            root = tree.getroot()

            rows = []

            # Support XES namespace and non-namespace XML.
            namespace = ""
            if root.tag.startswith("{"):
                namespace = root.tag.split("}")[0] + "}"

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

                    if "case:concept:name" not in row:
                        row["case:concept:name"] = case_id

                    rows.append(row)

            log = pd.DataFrame(rows)

    else:
        raise ValueError("Only .csv and .xes logs are supported.")

    if TIMESTAMP_COL not in log.columns:
        raise ValueError(
            f"Missing timestamp column: {TIMESTAMP_COL}. "
            f"Available columns: {list(log.columns)}"
        )

    log[TIMESTAMP_COL] = pd.to_datetime(
        log[TIMESTAMP_COL],
        errors="coerce",
    )

    log = log.dropna(
        subset=[
            CASE_COL,
            ACTIVITY_COL,
            TIMESTAMP_COL,
        ]
    ).reset_index(drop=True)

    print(
        f"[load_log] Loaded log with {len(log)} events and "
        f"{log[CASE_COL].nunique()} cases."
    )

    return log

def build_test_instances(test_log: pd.DataFrame):
    prepared = test_log.copy()
    prepared[TIMESTAMP_COL] = pd.to_datetime(prepared[TIMESTAMP_COL], errors="coerce")
    prepared = prepared.dropna(subset=[CASE_COL, ACTIVITY_COL, TIMESTAMP_COL])
    prepared = prepared.sort_values([CASE_COL, TIMESTAMP_COL])

    instances = []

    for _, case_events in prepared.groupby(CASE_COL):
        case_events = case_events.sort_values(TIMESTAMP_COL).reset_index(drop=True)

        for index in range(len(case_events) - 1):
            current_row = case_events.iloc[index]
            next_row = case_events.iloc[index + 1]

            event = current_row.to_dict()
            event["event_index"] = index

            # Evaluation uses successors observed for this current activity
            # inside the test log as possible activities.
            possible_activities = (
                prepared[prepared[ACTIVITY_COL] == current_row[ACTIVITY_COL]]
                .groupby(ACTIVITY_COL)[ACTIVITY_COL]
            )

            instances.append(
                {
                    "event": event,
                    "current_activity": current_row[ACTIVITY_COL],
                    "true_next_activity": next_row[ACTIVITY_COL],
                }
            )

    return instances


def build_transition_possible_activities(train_log: pd.DataFrame, test_log: pd.DataFrame):
    combined = pd.concat([train_log, test_log], ignore_index=True)
    combined = combined.sort_values([CASE_COL, TIMESTAMP_COL])

    transitions = {}

    for _, case_events in combined.groupby(CASE_COL):
        case_events = case_events.sort_values(TIMESTAMP_COL).reset_index(drop=True)

        for index in range(len(case_events) - 1):
            current_activity = case_events.iloc[index][ACTIVITY_COL]
            next_activity = case_events.iloc[index + 1][ACTIVITY_COL]
            transitions.setdefault(current_activity, set()).add(next_activity)

    return {
        current_activity: sorted(successors)
        for current_activity, successors in transitions.items()
    }


def build_attribute_rules(train_log: pd.DataFrame):
    """
    Simple example rules for BPIC-like data.

    These are intentionally conservative. If attributes are missing or rules
    do not match, the engine falls back to probabilistic branching.
    """

    rules = []

    if "CreditScore" in train_log.columns:
        rules.extend(
            [
                {
                    "decision_point": None,
                    "attribute": "CreditScore",
                    "operator": ">=",
                    "value": 700,
                    "preferred_activities": ["A_APPROVED", "O_Create Offer"],
                },
                {
                    "decision_point": None,
                    "attribute": "CreditScore",
                    "operator": "<",
                    "value": 500,
                    "preferred_activities": ["A_REJECTED", "A_Denied"],
                },
            ]
        )

    if "case:RequestedAmount" in train_log.columns:
        rules.append(
            {
                "decision_point": None,
                "attribute": "case:RequestedAmount",
                "operator": ">",
                "value": 50000,
                "preferred_activities": ["A_Denied", "A_REJECTED"],
            }
        )

    return rules


def evaluate_engine(engine, test_instances, possible_map):
    y_true = []
    y_pred = []

    for instance in test_instances:
        current_activity = instance["current_activity"]
        true_next = instance["true_next_activity"]
        event = instance["event"]

        possible_activities = possible_map.get(current_activity, [])

        if not possible_activities:
            continue

        prediction = engine.getNextActivities(event, possible_activities)

        if not prediction:
            continue

        y_true.append(true_next)
        y_pred.append(prediction[0])

    if not y_true:
        return {
            "accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
            "n_samples": 0,
        }

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "n_samples": len(y_true),
    }


def evaluate_split(log: pd.DataFrame, train_ratio: float, seed: int):
    train_log, test_log = temporal_train_test_split_by_case(
        log=log,
        case_col=CASE_COL,
        timestamp_col=TIMESTAMP_COL,
        train_ratio=train_ratio,
    )

    possible_map = build_transition_possible_activities(train_log, test_log)
    test_instances = build_test_instances(test_log)

    probability_engine = ProbabilityBranchingEngine(
        log=train_log,
        seed=seed,
    )
    probability_engine.train()

    attribute_rules = build_attribute_rules(train_log)

    attribute_engine = AttributeBasedBranchingEngine(
        rules=attribute_rules,
        fallback_engine=probability_engine,
        seed=seed,
    )

    sampling_engine = AttributeSamplingBranchingEngine(
        base_engine=attribute_engine,
        sampling_config={
            "CreditScore": {
                450: 0.3,
                650: 0.4,
                750: 0.3,
            }
        },
        fallback_engine=probability_engine,
        seed=seed,
    )

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
        seed=seed,
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=2,
    )

    predictive_engine.train(train_log)

    engines = {
        "Probability": probability_engine,
        "AttributeBased": attribute_engine,
        "AttributeSampling": sampling_engine,
        "PredictiveML": predictive_engine,
    }

    rows = []

    for approach_name, engine in engines.items():
        metrics = evaluate_engine(
            engine=engine,
            test_instances=test_instances,
            possible_map=possible_map,
        )

        rows.append(
            {
                "train_ratio": train_ratio,
                "approach": approach_name,
                "train_events": len(train_log),
                "test_events": len(test_log),
                "train_cases": train_log[CASE_COL].nunique(),
                "test_cases": test_log[CASE_COL].nunique(),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "n_samples": metrics["n_samples"],
                "feature_columns": ", ".join(feature_columns),
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Compare branching approaches across train/test splits."
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

    ratios = [0.6, 0.7, 0.8]
    all_rows = []

    for ratio in ratios:
        print(f"\nEvaluating split train_ratio={ratio}...")
        rows = evaluate_split(
            log=log,
            train_ratio=ratio,
            seed=args.seed,
        )
        all_rows.extend(rows)

    results_df = pd.DataFrame(all_rows)

    output_path = PROJECT_ROOT / "results" / "branching_split_approach_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print("\n=== Branching Split and Approach Comparison ===")
    print(
        results_df[
            [
                "train_ratio",
                "approach",
                "train_cases",
                "test_cases",
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "n_samples",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
