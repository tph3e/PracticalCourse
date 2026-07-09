import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pickle
import xml.etree.ElementTree as ET

import pandas as pd
import pm4py
from sklearn.metrics import accuracy_score, f1_score

from src.branching.ProbabilityBranchingEngine import ProbabilityBranchingEngine
from src.branching.AttributeBasedBranchingEngine import AttributeBasedBranchingEngine
from src.branching.AttributeSamplingBranchingEngine import AttributeSamplingBranchingEngine
from src.branching.PredictiveBranchingEngine import PredictiveBranchingEngine
from src.branching.BranchingUtils import temporal_train_test_split_by_case


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


def build_test_instances(test_log: pd.DataFrame, max_instances: int = 50000):
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

            instances.append(
                {
                    "event": event,
                    "current_activity": current_row[ACTIVITY_COL],
                    "true_next_activity": next_row[ACTIVITY_COL],
                }
            )

            if len(instances) >= max_instances:
                return instances

    return instances


def build_possible_activity_map(train_log: pd.DataFrame, test_log: pd.DataFrame):
    combined = pd.concat([train_log, test_log], ignore_index=True)
    combined[TIMESTAMP_COL] = pd.to_datetime(combined[TIMESTAMP_COL], errors="coerce")
    combined = combined.dropna(subset=[CASE_COL, ACTIVITY_COL, TIMESTAMP_COL])
    combined = combined.sort_values([CASE_COL, TIMESTAMP_COL])

    possible_map = {}

    for _, case_events in combined.groupby(CASE_COL):
        case_events = case_events.sort_values(TIMESTAMP_COL).reset_index(drop=True)

        for index in range(len(case_events) - 1):
            current_activity = case_events.iloc[index][ACTIVITY_COL]
            next_activity = case_events.iloc[index + 1][ACTIVITY_COL]

            possible_map.setdefault(current_activity, set()).add(next_activity)

    return {
        current_activity: sorted(successors)
        for current_activity, successors in possible_map.items()
    }


def build_attribute_rules(train_log: pd.DataFrame):
    """
    Conservative example rules.

    These rules are intentionally generic. If they do not match, the engine
    falls back to probabilistic branching.
    """

    rules = []

    activities = set(train_log[ACTIVITY_COL].dropna().unique())

    # Use real activities from the log if they exist.
    possible_positive_targets = [
        activity for activity in activities
        if "offer" in str(activity).lower()
        or "approve" in str(activity).lower()
        or "accept" in str(activity).lower()
    ]

    possible_negative_targets = [
        activity for activity in activities
        if "reject" in str(activity).lower()
        or "deny" in str(activity).lower()
        or "cancel" in str(activity).lower()
    ]

    positive_target = possible_positive_targets[0] if possible_positive_targets else None
    negative_target = possible_negative_targets[0] if possible_negative_targets else None

    if "CreditScore" in train_log.columns and positive_target is not None:
        rules.append(
            {
                "decision_point": None,
                "attribute": "CreditScore",
                "operator": ">=",
                "value": 700,
                "preferred_activities": [positive_target],
            }
        )

    if "CreditScore" in train_log.columns and negative_target is not None:
        rules.append(
            {
                "decision_point": None,
                "attribute": "CreditScore",
                "operator": "<",
                "value": 500,
                "preferred_activities": [negative_target],
            }
        )

    if "case:RequestedAmount" in train_log.columns and negative_target is not None:
        rules.append(
            {
                "decision_point": None,
                "attribute": "case:RequestedAmount",
                "operator": ">",
                "value": 50000,
                "preferred_activities": [negative_target],
            }
        )

    return rules


def evaluate_engine(engine, test_instances, possible_map):
    y_true = []
    y_pred = []

    for instance in test_instances:
        event = instance["event"]
        current_activity = instance["current_activity"]
        true_next = instance["true_next_activity"]

        possible_activities = possible_map.get(current_activity, [])

        if not possible_activities:
            continue

        result = engine.getNextActivities(event, possible_activities)

        if not result:
            continue

        prediction = result[0]

        if prediction not in possible_activities:
            continue

        y_true.append(true_next)
        y_pred.append(prediction)

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


def main():
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    log_path = PROJECT_ROOT / "../PracticalCourse/data/BPI Challenge 2017.xes"
    log = load_log(str(log_path))

    train_log, test_log = temporal_train_test_split_by_case(
        log=log,
        case_col=CASE_COL,
        timestamp_col=TIMESTAMP_COL,
        train_ratio=0.7,
    )

    print("\n=== Train/Test Split for Branching Approach Report ===")
    print(f"Train events: {len(train_log)}")
    print(f"Test events: {len(test_log)}")
    print(f"Train cases: {train_log[CASE_COL].nunique()}")
    print(f"Test cases: {test_log[CASE_COL].nunique()}")

    probability_engine = ProbabilityBranchingEngine(
        log=train_log,
        seed=1,
    )
    probability_engine.train()

    attribute_rules = build_attribute_rules(train_log)

    attribute_engine = AttributeBasedBranchingEngine(
        rules=attribute_rules,
        fallback_engine=probability_engine,
        seed=1,
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
        seed=1,
    )

    final_model_path = output_dir / "final_predictive_model_0_7.pkl"

    if final_model_path.exists():
        with open(final_model_path, "rb") as file:
            bundle = pickle.load(file)

        predictive_engine = bundle["predictive_engine"]
    else:
        print("[warning] final_predictive_model_0_7.pkl not found. Training predictive model now.")

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
            seed=1,
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=2,
        )
        predictive_engine.train(train_log)

    possible_map = build_possible_activity_map(train_log, test_log)
    test_instances = build_test_instances(test_log, max_instances=50000)

    engines = {
        "ProbabilityBranching": probability_engine,
        "AttributeBasedBranching": attribute_engine,
        "AttributeSamplingBranching": sampling_engine,
        "PredictiveML": predictive_engine,
    }

    rows = []

    for approach_name, engine in engines.items():
        print(f"\nEvaluating {approach_name}...")
        metrics = evaluate_engine(
            engine=engine,
            test_instances=test_instances,
            possible_map=possible_map,
        )

        rows.append(
            {
                "approach": approach_name,
                "train_ratio": 0.7,
                "train_events": len(train_log),
                "test_events": len(test_log),
                "train_cases": train_log[CASE_COL].nunique(),
                "test_cases": test_log[CASE_COL].nunique(),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "n_samples": metrics["n_samples"],
            }
        )

    approach_df = pd.DataFrame(rows)

    transition_table = probability_engine.export_transition_table()
    decision_points_table = probability_engine.export_decision_points_table()

    summary_df = pd.DataFrame(
        [
            {
                "component": "ProbabilityBranching",
                "description": "Learns empirical transition probabilities from the event log.",
                "output": "branching_transition_probabilities.csv, branching_decision_points.csv",
            },
            {
                "component": "AttributeBasedBranching",
                "description": "Applies configurable attribute-based rules and falls back to probabilistic branching.",
                "output": "branching_approach_comparison.csv",
            },
            {
                "component": "AttributeSamplingBranching",
                "description": "Samples or derives missing runtime attributes before delegating to attribute-based branching.",
                "output": "branching_approach_comparison.csv",
            },
            {
                "component": "PredictiveML",
                "description": "Uses a RandomForest-based next-activity classifier with probabilistic fallback.",
                "output": "branching_approach_comparison.csv, final_predictive_model_metrics_0_7.csv",
            },
        ]
    )

    approach_path = output_dir / "branching_approach_comparison.csv"
    transition_path = output_dir / "branching_transition_probabilities.csv"
    decision_points_path = output_dir / "branching_decision_points.csv"
    summary_path = output_dir / "branching_approach_summary.csv"

    approach_df.to_csv(approach_path, index=False)
    transition_table.to_csv(transition_path, index=False)
    decision_points_table.to_csv(decision_points_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n=== Branching Approach Comparison ===")
    print(
        approach_df[
            [
                "approach",
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "n_samples",
            ]
        ].to_string(index=False)
    )

    print("\n=== Branching Report Files ===")
    print(f"Saved approach comparison to: {approach_path}")
    print(f"Saved transition probabilities to: {transition_path}")
    print(f"Saved decision points to: {decision_points_path}")
    print(f"Saved approach summary to: {summary_path}")


if __name__ == "__main__":
    main()
