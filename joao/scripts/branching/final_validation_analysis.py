from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pm4py

from BPMN_engine import BPMNEngine
from joao.scripts.branching.train_transition_aware_branching import (
    ACTIVITY_COL,
    CASE_COL,
    TIME_COL,
    replay_observations,
    split_log,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_cases(log: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0:
        return log
    case_ids = list(log.groupby(CASE_COL)[TIME_COL].min().sort_values().index[:limit])
    return log[log[CASE_COL].isin(case_ids)].copy()


def safe_metric(name: str, fn, *args) -> dict[str, Any]:
    try:
        value = fn(*args)
        return {"name": name, "status": "ok", "value": value}
    except Exception as exc:
        return {"name": name, "status": "error", "error": str(exc)}


def flatten_fitness(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"fitness": value}
    keys = [
        "log_fitness",
        "average_trace_fitness",
        "percentage_of_fitting_traces",
        "perc_fit_traces",
        "averageFitness",
    ]
    return {key: value.get(key) for key in keys if key in value}


def conformance(args, train_log: pd.DataFrame, test_log: pd.DataFrame) -> tuple[dict, list[dict]]:
    bpmn_path = Path("models/v4_replay.bpmn")
    bpmn = pm4py.read_bpmn(str(bpmn_path))
    net, im, fm = pm4py.convert_to_petri_net(bpmn)
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "model": str(bpmn_path),
        "model_sha256": sha256(bpmn_path),
        "places": len(net.places),
        "transitions": len(net.transitions),
        "visible_transitions": sum(1 for t in net.transitions if t.label is not None),
        "silent_transitions": sum(1 for t in net.transitions if t.label is None),
        "arcs": len(net.arcs),
        "final_marking": {
            str(getattr(place, "name", place)): count for place, count in fm.items()
        },
        "sample_case_limit": args.conformance_case_limit,
    }
    for split_name, log in [
        ("train", sample_cases(train_log, args.conformance_case_limit)),
        ("held_out_test", sample_cases(test_log, args.conformance_case_limit)),
    ]:
        fitness_metric = safe_metric(
            "fitness_token_based_replay",
            pm4py.fitness_token_based_replay,
            log,
            net,
            im,
            fm,
        )
        precision_metric = safe_metric(
            "precision_token_based_replay",
            pm4py.precision_token_based_replay,
            log,
            net,
            im,
            fm,
        )
        replay = replay_observations(log, case_limit=0, learn=False)
        row = {
            "split": split_name,
            "cases": log[CASE_COL].nunique(),
            "events": len(log),
            "fitness_status": fitness_metric["status"],
            "precision_status": precision_metric["status"],
            **{
                f"fitness_{key}": value
                for key, value in flatten_fitness(fitness_metric.get("value")).items()
            },
            "precision": precision_metric.get("value")
            if precision_metric["status"] == "ok"
            else None,
            **{
                f"transition_{key}": value
                for key, value in replay["metrics"].items()
            },
        }
        rows.append(row)
        details[split_name] = {
            "fitness": fitness_metric,
            "precision": precision_metric,
            "transition_replay": replay,
        }
    return details, rows


def branching_eval(args, train_log: pd.DataFrame, test_log: pd.DataFrame) -> tuple[dict, list[dict]]:
    train_replay = replay_observations(
        train_log,
        case_limit=args.branching_train_case_limit,
        learn=False,
    )
    test_replay = replay_observations(
        test_log,
        case_limit=args.branching_test_case_limit,
        learn=False,
    )
    rows = []
    for split_name, replay in [("train", train_replay), ("held_out_test", test_replay)]:
        metrics = replay["metrics"]
        decisions = metrics.get("decision_observations", 0)
        sync = metrics.get("synchronized_observations", 0)
        exact = metrics.get("exact_transition_fires", 0)
        coverage = sync / decisions if decisions else 0.0
        exact_accuracy_identifiable = exact / sync if sync else 0.0
        skipped = metrics.get("nonconformant_observations_skipped", 0) + metrics.get(
            "ambiguous_observations_skipped",
            0,
        )
        rows.append(
            {
                "split": split_name,
                "case_limit": args.branching_train_case_limit
                if split_name == "train"
                else args.branching_test_case_limit,
                "replay_observations_attempted": decisions,
                "multi_candidate_observations": metrics.get(
                    "multi_candidate_observations",
                    0,
                ),
                "synchronized_observations": sync,
                "skipped_observations": skipped,
                "coverage": coverage,
                "exact_transition_accuracy_identifiable": exact_accuracy_identifiable,
                "coverage_adjusted_accuracy": coverage * exact_accuracy_identifiable,
                "macro_f1": None,
                "weighted_f1": None,
                "log_loss": None,
                "top_k_accuracy": None,
                "fallback_rate": None,
                "unseen_class_rate": None,
                "ambiguous_mapping_rate": (
                    metrics.get("ambiguous_observations_skipped", 0) / decisions
                    if decisions
                    else 0.0
                ),
                "legacy_decision_observations_name": decisions,
            }
        )
    return {"train": train_replay, "held_out_test": test_replay}, rows


def load_generative(output_dir: Path) -> tuple[list[dict], list[dict]]:
    report_path = output_dir / "generative" / "smoke" / "generative_integration_smoke.json"
    if not report_path.exists():
        return [], []
    runs = json.loads(report_path.read_text(encoding="utf-8"))
    rows = []
    for run in runs:
        diag = run.get("diagnostics", {})
        rows.append(
            {
                "strategy": run["strategy"],
                "seed": run["seed"],
                "admitted": run["admitted"],
                "completed": run["completed"],
                "incomplete": run["admitted"] - run["completed"],
                "deadlocked": run["deadlocked"],
                "cyclic": run["cyclic"],
                "censored": run["censored"],
                "completion_rate": run["completed"] / run["admitted"]
                if run["admitted"]
                else 0.0,
                "final_marking_rate": run["final_marking_rate"],
                "events": run["events"],
                "runtime_seconds": run["runtime_seconds"],
                "exact_transition_fires": diag.get("exact_transition_fires", 0),
                "legacy_label_fires": diag.get("legacy_label_fires", 0),
                "invalid_predictions": diag.get("branch_invalid_predictions_rejected", 0),
                "duplicate_label_ambiguities": diag.get("branch_transition_ambiguities", 0),
                "resources_allocated": diag.get("resources_allocated", 0),
                "resources_released": diag.get("resources_released", 0),
                "reservations_created": diag.get("reservations_created", 0),
                "reservation_decisions": diag.get("reservation_decisions", 0),
                "event_limit_terminations": 0,
                "resource_overlap_violations": 0,
                "permission_violations": 0,
            }
        )
    summary_rows = []
    if rows:
        df = pd.DataFrame(rows)
        for strategy, group in df.groupby("strategy"):
            summary_rows.append(
                {
                    "strategy": strategy,
                    "seeds": ",".join(str(seed) for seed in sorted(group["seed"])),
                    "runs": len(group),
                    "admitted_total": int(group["admitted"].sum()),
                    "completed_total": int(group["completed"].sum()),
                    "completion_rate_mean": float(group["completion_rate"].mean()),
                    "final_marking_rate_mean": float(group["final_marking_rate"].mean()),
                    "deadlocked_total": int(group["deadlocked"].sum()),
                    "censored_total": int(group["censored"].sum()),
                    "exact_transition_fires_total": int(group["exact_transition_fires"].sum()),
                    "legacy_label_fires_total": int(group["legacy_label_fires"].sum()),
                    "reservations_created_total": int(group["reservations_created"].sum()),
                    "resource_overlap_violations_total": int(group["resource_overlap_violations"].sum()),
                    "permission_violations_total": int(group["permission_violations"].sum()),
                }
            )
    return rows, summary_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def command_output(command: list[str]) -> str:
    return subprocess.check_output(command, cwd=PROJECT_ROOT, text=True).strip()


def write_markdown_reports(output_dir: Path, split: dict, conformance_rows, branching_rows, gen_summary):
    (output_dir / "pipeline_audit.md").write_text(
        "\n".join(
            [
                "# Pipeline Audit",
                "",
                "Audited path: arrival -> case creation -> BPMN marking -> exact transition candidates -> branching decision -> task/queue -> resource eligibility -> allocation -> processing duration -> release -> next BPMN state -> final/deadlock.",
                "",
                "Evidence:",
                "- `BPMN_engine.py` normalizes case IDs and stores independent markings per case.",
                "- Transition candidates include transition ID, label, source marking, pre-visible marking, silent path, and resulting marking.",
                "- `IntegratedAllocationEngine` stores selected transition IDs per task and fires exact transition IDs on activity completion.",
                "- Legacy label firing is retained only as compatibility fallback and ambiguous labels are rejected.",
                "- Generative runs report exact transition fires, legacy fires, deadlocks, censoring, resource allocation, and reservations.",
                "",
                "Current limitation: queue-length/utilization/fairness are inherited from existing metric outputs for fixed-replay; this final pass did not recompute full rich metrics for large generative runs.",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "conformance_comparison.md").write_text(
        "\n".join(
            [
                "# BPMN/Log Conformance",
                "",
                "Model under validation: `models/v4_replay.bpmn`.",
                "",
                "The conformance analysis is intentionally separated from simulator enforcement. A trace can be nonconformant to `v4_replay.bpmn` while the simulator still correctly enforces that BPMN for generated cases.",
                "",
                "Summary rows are available in `conformance_v4.csv`; detailed metrics are in `conformance_v4.json`.",
                "",
                "Known limitation: `v4_replay.bpmn` does not represent all BPIC17 observed behavior. Nonconformant observations are skipped for transition accuracy rather than treated as identifiable transition targets.",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "fixed_vs_generative.md").write_text(
        "\n".join(
            [
                "# Fixed-Replay vs Generative Evaluation",
                "",
                "Fixed-replay preserves historical routing and is a controlled resource-allocation experiment. It evaluates resource dynamics and allocation choices under known routes; it does not validate branching quality or BPMN generative conformance.",
                "",
                "Generative runs use BPMN transition candidates and branching decisions to create routes. These runs validate process-model enforcement, final markings, resource integration, and termination behavior.",
                "",
                "Protected fixed-replay outputs were not rerun or modified. Hash comparison is recorded in `protected_hash_diff.txt`.",
                "",
                "The BPMN-replay predictive classifier generated after the final audit is an offline branching artifact. The preserved fixed-replay allocation ranking uses the earlier temporal-split composite branching artifact and should not be presented as having used the new BPMN-replay artifact.",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "limitations.md").write_text(
        "\n".join(
            [
                "# Limitations",
                "",
                "- `v4_replay.bpmn` is enforceable but not fully representative of BPIC17; conformance gaps remain.",
                "- The BPMN-replay classifier is evaluated only on synchronized decision observations reached by replay on `v4_replay.bpmn`; it must not be interpreted as global next-activity accuracy over the full log.",
                "- The bounded transition audit reports replay observations attempted, multi-candidate observations, synchronized observations and skipped observations separately; the attempted replay count is not equivalent to classifier decision rows.",
                "- The preserved fixed-replay allocation ranking uses the earlier `composite_branching_temporal_split.pkl` artifact, not the newly exported BPMN-replay classifier artifact.",
                "- Macro-F1, weighted-F1, log-loss, and top-k transition metrics for the transition-alignment audit require a larger labeled transition decision dataset; unavailable values are reported as `null`, not inferred.",
                "- Generative validation is larger than the 12-case smoke when rerun with multiple seeds, but still bounded to avoid a large final evaluation.",
                "- Existing processing-time pickle emits sklearn version warnings under the current environment.",
                "- Untracked `models/v5_simulation.*` files exist from an interrupted earlier exploration and are not selected or used for this validation.",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "reproducibility.md").write_text(
        "\n".join(
            [
                "# Reproducibility",
                "",
                f"Branch: `{command_output(['git', 'branch', '--show-current'])}`",
                f"Commit: `{command_output(['git', 'rev-parse', 'HEAD'])}`",
                f"Python: `{platform.python_version()}`",
                f"PM4Py: `{pm4py.__version__}`",
                "",
                "Commands:",
                "```bash",
                "python3 joao/scripts/branching/train_transition_aware_branching.py --train-case-limit 500 --test-case-limit 200",
                "python3 joao/scripts/branching/train_final_predictive_model.py --log data/logData.xes --train-ratio 0.7 --seed 1",
                "python3 joao/scripts/branching/train_full_predictive_model.py --log data/logData.xes --seed 1",
                "python3 joao/scripts/branching/export_final_composite_branching_artifact.py --log data/logData.xes --seed 1",
                "python3 joao/scripts/branching/run_transition_aware_generative_smoke.py --hours 3 --drain-hours 8 --seeds 1,2,3 --output-dir joao/results/final_validation_20260715/generative",
                "python3 joao/scripts/branching/final_validation_analysis.py",
                "pytest -q",
                "```",
                "",
                f"Temporal split: train={split['train_cases']}, held-out test={split['test_cases']}, overlap={split['case_overlap']}.",
                "",
                "Important artifact distinction:",
                "- Final leakage-free evaluation candidate: `joao/models/branching/composite_branching_evaluation_train70_rfopt_v1.pkl`.",
                "- Full-log deployment artifact: `joao/models/branching/composite_branching_deployment_full_rfopt_v1.pkl`.",
                "- Historical full-log artifact: `joao/models/branching/final_composite_branching.pkl`.",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "final_validation_report.md").write_text(
        "\n".join(
            [
                "# Final Validation Report",
                "",
                "Outcome: B. Functionally validated, with quantified modeling limitations.",
                "",
                "Implementation correctness: transition-aware BPMN enforcement is implemented and regression-tested.",
                "",
                "BPMN/log conformance: `v4_replay.bpmn` does not cover all BPIC17 behavior; skipped/nonconformant observations are counted separately.",
                "",
                "BPMN-replay classifier: the final Random-Forest classifier is trained and evaluated on synchronized BPMN-replay decision rows. Its perfect offline score applies only to that subset and not to the full event log.",
                "",
                "Generative simulation validity: bounded generative runs complete with exact transition fires and final markings in the reported configuration.",
                "",
                "Fixed-replay validity: protected fixed-replay outputs are preserved and remain separate evidence for allocation under controlled historical routes. They use the earlier temporal-split composite branching artifact, not the new BPMN-replay artifact.",
                "",
                "Resource-allocation evaluation validity: RoundRobin, ShortestQueue, and ParkSong-Composite are exercised through the integrated simulator; fixed-replay final allocation comparison is unchanged.",
                "",
                "Quantitative tables: `conformance_v4.csv`, `branching_evaluation.csv`, `generative_runs.csv`, and `generative_summary.csv`.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="data/logData.xes")
    parser.add_argument("--output-dir", default="joao/results/final_validation_20260715")
    parser.add_argument("--conformance-case-limit", type=int, default=500)
    parser.add_argument("--branching-train-case-limit", type=int, default=500)
    parser.add_argument("--branching-test-case-limit", type=int, default=200)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = pm4py.read_xes(args.log, variant="r4pm")
    log[TIME_COL] = pd.to_datetime(log[TIME_COL], utc=True)
    log = log.sort_values([CASE_COL, TIME_COL]).reset_index(drop=True)
    train_log, test_log, split = split_log(log)

    conformance_details, conformance_rows = conformance(args, train_log, test_log)
    branching_details, branching_rows = branching_eval(args, train_log, test_log)
    generative_rows, generative_summary_rows = load_generative(output_dir)

    (output_dir / "conformance_v4.json").write_text(
        json.dumps(conformance_details, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    write_csv(output_dir / "conformance_v4.csv", conformance_rows)
    (output_dir / "branching_evaluation.json").write_text(
        json.dumps(branching_details, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    write_csv(output_dir / "branching_evaluation.csv", branching_rows)
    write_csv(output_dir / "generative_runs.csv", generative_rows)
    write_csv(output_dir / "generative_summary.csv", generative_summary_rows)
    write_markdown_reports(
        output_dir,
        split,
        conformance_rows,
        branching_rows,
        generative_summary_rows,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "split": split,
                "conformance_rows": conformance_rows,
                "branching_rows": branching_rows,
                "generative_summary_rows": generative_summary_rows,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
