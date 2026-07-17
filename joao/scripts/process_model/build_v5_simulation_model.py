from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict, deque
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import pandas as pd
import pm4py
from pm4py.objects.petri_net.semantics import enabled_transitions, execute


CASE_COL = "case:concept:name"
ACTIVITY_COL = "concept:name"
TIME_COL = "time:timestamp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="data/logData.xes")
    parser.add_argument("--output-dir", default="joao/results/process_model_v5")
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--thresholds", default="0.0,0.2,0.4")
    parser.add_argument("--max-replay-cases", type=int, default=1500)
    parser.add_argument("--discovery-case-limit", type=int, default=0)
    parser.add_argument("--max-reachability-markings", type=int, default=8000)
    parser.add_argument("--run-woflan", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def temporal_split(log: pd.DataFrame, case_limit: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    case_ranges = (
        log.groupby(CASE_COL)[TIME_COL]
        .agg(["min", "max"])
        .sort_values(["min", "max"])
    )
    if case_limit > 0:
        case_ranges = case_ranges.iloc[:case_limit]
    split_index = int(len(case_ranges) * 0.7)
    train_cases = set(case_ranges.index[:split_index])
    test_cases = set(case_ranges.index[split_index:])
    overlap = train_cases & test_cases
    assert not overlap

    train_log = log[log[CASE_COL].isin(train_cases)].copy()
    test_log = log[log[CASE_COL].isin(test_cases)].copy()
    metadata = {
        "train_cases": len(train_cases),
        "test_cases": len(test_cases),
        "case_overlap": len(overlap),
        "train_case_range": [str(case_ranges.index[0]), str(case_ranges.index[split_index - 1])],
        "test_case_range": [str(case_ranges.index[split_index]), str(case_ranges.index[-1])],
        "train_time_range": [str(train_log[TIME_COL].min()), str(train_log[TIME_COL].max())],
        "test_time_range": [str(test_log[TIME_COL].min()), str(test_log[TIME_COL].max())],
    }
    return train_log, test_log, metadata


def sample_cases(log: pd.DataFrame, max_cases: int) -> pd.DataFrame:
    if max_cases <= 0:
        return log
    case_ids = list(log.groupby(CASE_COL)[TIME_COL].min().sort_values().index[:max_cases])
    return log[log[CASE_COL].isin(case_ids)].copy()


def safe_metric(fn, *args):
    start = perf_counter()
    try:
        value = fn(*args)
        status = "ok"
    except Exception as exc:
        value = {"error": str(exc)}
        status = "error"
    return value, status, round(perf_counter() - start, 3)


def visible_labels(net) -> set[str]:
    return {str(t.label) for t in net.transitions if t.label is not None}


def marking_signature(marking) -> str:
    return "|".join(
        sorted(f"{getattr(place, 'name', place)}:{count}" for place, count in marking.items())
    )


def transition_id(transition) -> str:
    return str(getattr(transition, "name", None) or id(transition))


def reachable_model_relations(net, im, fm, max_markings: int) -> dict[str, Any]:
    queue = deque([(im, None)])
    visited: set[str] = set()
    labels = set()
    dfr = set()
    terminal = set()
    dead_transitions = {transition_id(t) for t in net.transitions}
    scc_edges: dict[str, set[str]] = defaultdict(set)

    while queue and len(visited) < max_markings:
        marking, previous_label = queue.popleft()
        signature = marking_signature(marking)
        if signature in visited:
            continue
        visited.add(signature)
        if marking == fm and previous_label:
            terminal.add(previous_label)
        for transition in sorted(enabled_transitions(net, marking), key=transition_id):
            tid = transition_id(transition)
            dead_transitions.discard(tid)
            next_marking = execute(transition, net, marking)
            label = str(transition.label) if transition.label is not None else None
            if label:
                labels.add(label)
                if previous_label:
                    dfr.add((previous_label, label))
                    scc_edges[previous_label].add(label)
                next_previous = label
            else:
                next_previous = previous_label
            next_signature = marking_signature(next_marking)
            if next_signature not in visited:
                queue.append((next_marking, next_previous))

    loop_nodes = strongly_connected_nodes(scc_edges)
    return {
        "reachable_markings_examined": len(visited),
        "reachability_truncated": bool(queue),
        "final_marking_reachable": marking_signature(fm) in visited,
        "visible_labels": sorted(labels),
        "directly_follows": sorted([list(item) for item in dfr]),
        "dead_transition_count": len(dead_transitions),
        "dead_transitions": sorted(dead_transitions)[:50],
        "terminal_labels": sorted(terminal),
        "loop_scc_activity_count": len(loop_nodes),
        "loop_scc_activities": sorted(loop_nodes),
    }


def strongly_connected_nodes(edges: dict[str, set[str]]) -> set[str]:
    nodes = set(edges) | {target for targets in edges.values() for target in targets}
    result = set()
    for node in nodes:
        stack = list(edges.get(node, set()))
        seen = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            if current == node:
                result.update(seen)
                break
            stack.extend(edges.get(current, set()) - seen)
    return result


def log_dfr(log: pd.DataFrame) -> set[tuple[str, str]]:
    relations = set()
    for _, group in log.sort_values([CASE_COL, TIME_COL]).groupby(CASE_COL):
        activities = [str(item) for item in group[ACTIVITY_COL]]
        relations.update(zip(activities, activities[1:]))
    return relations


def first_divergences(log: pd.DataFrame, net, im, max_cases: int) -> Counter[str]:
    divergences: Counter[str] = Counter()
    for case_id, group in sample_cases(log, max_cases).sort_values([CASE_COL, TIME_COL]).groupby(CASE_COL):
        marking = im
        for activity in [str(item) for item in group[ACTIVITY_COL]]:
            queue = deque([marking])
            visited = set()
            fired = False
            while queue and not fired:
                current = queue.popleft()
                signature = marking_signature(current)
                if signature in visited:
                    continue
                visited.add(signature)
                for transition in sorted(enabled_transitions(net, current), key=transition_id):
                    if transition.label == activity:
                        marking = execute(transition, net, current)
                        fired = True
                        break
                    if transition.label is None:
                        queue.append(execute(transition, net, current))
            if not fired:
                divergences[activity] += 1
                break
    return divergences


def audit_model(path: Path, train_log: pd.DataFrame, test_log: pd.DataFrame, args) -> dict[str, Any]:
    bpmn = pm4py.read_bpmn(str(path))
    net, im, fm = pm4py.convert_to_petri_net(bpmn)
    train_sample = sample_cases(train_log, args.max_replay_cases)
    test_sample = sample_cases(test_log, args.max_replay_cases)
    represented = reachable_model_relations(net, im, fm, args.max_reachability_markings)
    train_relations = log_dfr(train_sample)
    test_relations = log_dfr(test_sample)
    represented_relations = {tuple(item) for item in represented["directly_follows"]}

    train_fitness, train_fitness_status, train_fitness_seconds = safe_metric(
        pm4py.fitness_token_based_replay, train_sample, net, im, fm
    )
    test_fitness, test_fitness_status, test_fitness_seconds = safe_metric(
        pm4py.fitness_token_based_replay, test_sample, net, im, fm
    )
    precision, precision_status, precision_seconds = safe_metric(
        pm4py.precision_token_based_replay, train_sample, net, im, fm
    )
    generalization, generalization_status, generalization_seconds = safe_metric(
        pm4py.generalization_tbr, train_sample, net, im, fm
    )
    if args.run_woflan:
        soundness, soundness_status, _ = safe_metric(pm4py.check_soundness, net, im, fm)
    else:
        soundness = {
            "method": "bounded_reachability_proxy",
            "final_marking_reachable": represented["final_marking_reachable"],
            "dead_transition_count": represented["dead_transition_count"],
            "reachability_truncated": represented["reachability_truncated"],
        }
        soundness_status = "proxy"

    train_activities = set(train_log[ACTIVITY_COL].astype(str))
    test_activities = set(test_log[ACTIVITY_COL].astype(str))
    return {
        "model": str(path),
        "sha256": sha256(path),
        "places": len(net.places),
        "transitions": len(net.transitions),
        "visible_transitions": len(visible_labels(net)),
        "arcs": len(net.arcs),
        "soundness": soundness,
        "soundness_status": soundness_status,
        "train_token_fitness": train_fitness,
        "train_fitness_status": train_fitness_status,
        "train_fitness_seconds": train_fitness_seconds,
        "test_token_fitness": test_fitness,
        "test_fitness_status": test_fitness_status,
        "test_fitness_seconds": test_fitness_seconds,
        "precision": precision,
        "precision_status": precision_status,
        "precision_seconds": precision_seconds,
        "generalization": generalization,
        "generalization_status": generalization_status,
        "generalization_seconds": generalization_seconds,
        "uncovered_train_activities": sorted(train_activities - set(represented["visible_labels"])),
        "uncovered_test_activities": sorted(test_activities - set(represented["visible_labels"])),
        "train_dfr_not_represented_count": len(train_relations - represented_relations),
        "test_dfr_not_represented_count": len(test_relations - represented_relations),
        "train_dfr_not_represented_sample": sorted([list(item) for item in train_relations - represented_relations])[:50],
        "test_dfr_not_represented_sample": sorted([list(item) for item in test_relations - represented_relations])[:50],
        "first_divergence_train": dict(first_divergences(train_sample, net, im, args.max_replay_cases)),
        "first_divergence_test": dict(first_divergences(test_sample, net, im, args.max_replay_cases)),
        "reachability": represented,
        "explicit_activity_coverage": {
            activity: activity in represented["visible_labels"]
            for activity in [
                "A_Denied",
                "A_Cancelled",
                "O_Cancelled",
                "A_Pending",
                "A_Validating",
                "O_Sent",
                "O_Create Offer",
                "W_Complete application",
            ]
        },
    }


def discover_candidate(train_log: pd.DataFrame, threshold: float) -> tuple[Any, Any, Any, Any]:
    tree = pm4py.discover_process_tree_inductive(train_log, noise_threshold=threshold)
    bpmn = pm4py.convert_to_bpmn(tree)
    net, im, fm = pm4py.convert_to_petri_net(bpmn)
    return tree, bpmn, net, im, fm


def scalar_fitness(metric: Any) -> float:
    if isinstance(metric, dict):
        for key in ["log_fitness", "average_trace_fitness", "percentage_of_fitting_traces"]:
            if key in metric and isinstance(metric[key], (int, float)):
                return float(metric[key])
    if isinstance(metric, (int, float)):
        return float(metric)
    return 0.0


def soundness_score(row: dict[str, Any]) -> int:
    soundness = row.get("soundness")
    if row.get("soundness_status") == "ok":
        return 1 if bool(soundness) else 0
    if isinstance(soundness, dict):
        return 1 if soundness.get("final_marking_reachable") else 0
    return 0


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log)

    log = pm4py.read_xes(str(log_path), variant="r4pm")
    log[TIME_COL] = pd.to_datetime(log[TIME_COL], utc=True)
    log = log.sort_values([CASE_COL, TIME_COL]).reset_index(drop=True)
    train_log, test_log, split_metadata = temporal_split(log, args.case_limit)

    model_paths = sorted(Path(".").glob("**/*.bpmn"))
    model_paths = [
        path for path in model_paths
        if "models" in path.parts and path.name != "v5_simulation.bpmn"
    ]

    audits = []
    for path in model_paths:
        audits.append(audit_model(path, train_log, test_log, args))

    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    discovery_rows = []
    discovered = []
    train_discovery_log = sample_cases(train_log, args.discovery_case_limit)
    train_discovery_metric_log = sample_cases(train_log, args.max_replay_cases)
    for threshold in thresholds:
        tree, bpmn, net, im, fm = discover_candidate(train_discovery_log, threshold)
        fitness, fitness_status, fitness_seconds = safe_metric(
            pm4py.fitness_token_based_replay, train_discovery_metric_log, net, im, fm
        )
        precision, precision_status, precision_seconds = safe_metric(
            pm4py.precision_token_based_replay, train_discovery_metric_log, net, im, fm
        )
        represented = reachable_model_relations(net, im, fm, args.max_reachability_markings)
        if args.run_woflan:
            soundness, soundness_status, _ = safe_metric(pm4py.check_soundness, net, im, fm)
        else:
            soundness = {
                "method": "bounded_reachability_proxy",
                "final_marking_reachable": represented["final_marking_reachable"],
                "dead_transition_count": represented["dead_transition_count"],
                "reachability_truncated": represented["reachability_truncated"],
            }
            soundness_status = "proxy"
        complexity = len(net.places) + len(net.transitions) + len(net.arcs)
        row = {
            "threshold": threshold,
            "fitness": fitness,
            "fitness_status": fitness_status,
            "fitness_seconds": fitness_seconds,
            "precision": precision,
            "precision_status": precision_status,
            "precision_seconds": precision_seconds,
            "soundness": soundness,
            "soundness_status": soundness_status,
            "places": len(net.places),
            "transitions": len(net.transitions),
            "arcs": len(net.arcs),
            "complexity": complexity,
        }
        discovery_rows.append(row)
        discovered.append((row, bpmn, net, im, fm))

    best_row, best_bpmn, best_net, best_im, best_fm = max(
        discovered,
        key=lambda item: (
            soundness_score(item[0]),
            scalar_fitness(item[0]["fitness"]),
            float(item[0]["precision"]) if isinstance(item[0]["precision"], (int, float)) else 0.0,
            -item[0]["complexity"],
        ),
    )
    v5_path = Path("models/v5_simulation.bpmn")
    v5_pnml_path = Path("models/v5_simulation.pnml")
    pm4py.write_bpmn(best_bpmn, str(v5_path))
    pm4py.write_pnml(best_net, best_im, best_fm, str(v5_pnml_path))
    v5_audit = audit_model(v5_path, train_log, test_log, args)

    summary = {
        "log": {
            "path": str(log_path),
            "sha256": sha256(log_path),
            "events": len(log),
            "cases": log[CASE_COL].nunique(),
        },
        "split": split_metadata,
        "bpmn_hashes": {str(path): sha256(path) for path in model_paths + [v5_path]},
        "audit_case_limit": args.max_replay_cases,
        "discovery_case_limit": args.discovery_case_limit,
        "existing_model_audits": audits,
        "discovery_candidates": discovery_rows,
        "selected_model": {
            "path": str(v5_path),
            "pnml_path": str(v5_pnml_path),
            "metadata_path": "models/v5_simulation_metadata.json",
            "selection_basis": "training split only; best sound candidate by train fitness, train precision, and lower complexity",
            "selected_threshold": best_row["threshold"],
            "audit": v5_audit,
        },
    }
    metadata = {
        "source_log_sha256": sha256(log_path),
        "train_cases": split_metadata["train_cases"],
        "test_cases": split_metadata["test_cases"],
        "case_overlap": split_metadata["case_overlap"],
        "selected_threshold": best_row["threshold"],
        "selection_uses_test_log": False,
        "pnml_path": str(v5_pnml_path),
        "bpmn_sha256": sha256(v5_path),
        "pnml_sha256": sha256(v5_pnml_path),
        "complexity": {
            "places": len(best_net.places),
            "transitions": len(best_net.transitions),
            "arcs": len(best_net.arcs),
        },
    }
    Path("models/v5_simulation_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "process_model_v5_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    pd.DataFrame(discovery_rows).to_csv(output_dir / "discovery_candidates.csv", index=False)
    pd.DataFrame(audits + [v5_audit]).to_csv(output_dir / "model_comparison.csv", index=False)
    print(json.dumps({
        "split": split_metadata,
        "selected_model": str(v5_path),
        "selected_threshold": best_row["threshold"],
        "summary": str(output_dir / "process_model_v5_summary.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
