from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from scipy.stats import t as student_t


METHOD_ORDER = [
    "RoundRobin",
    "ShortestQueue",
    "ParkSong-Composite",
    "Kunkler-Rinderle-Ma",
    "Batch",
]


def tex_escape(value: object) -> str:
    return str(value).replace("_", r"\_")


def fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def json_clean(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    if value is pd.NA:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def completion(raw: pd.DataFrame, strategy: str) -> str:
    rows = raw[raw["strategy"] == strategy]
    completed = int(rows["fixed_routes_completed"].sum())
    admitted = int(rows["cases_admitted"].sum())
    return f"{completed}/{admitted}"


def fixed_replay_table(raw: pd.DataFrame, aggregated: pd.DataFrame) -> str:
    rows = []
    by_strategy = {row["strategy"]: row for _, row in aggregated.iterrows()}
    rows.append(r"\begin{table*}[t]")
    rows.append(
        r"    \caption{Canonical fixed-replay allocation results over five seeds. "
        r"Cycle time is in hours for completed fixed routes; waiting time is in minutes. "
        r"Values are mean $\pm$ sample standard deviation where shown.}"
    )
    rows.append(r"    \centering")
    rows.append(r"    \renewcommand{\arraystretch}{1.15}")
    rows.append(r"    \scriptsize")
    rows.append(r"    \resizebox{\textwidth}{!}{")
    rows.append(r"    \begin{tabular}{lrrrrrr}")
    rows.append(r"        \toprule")
    rows.append(
        r"        \textbf{Method} & \textbf{Fixed-route completion} & "
        r"\textbf{Cycle h} & \textbf{Wait min} & \textbf{Gini} & "
        r"\textbf{Weighted fairness} & \textbf{Runtime s} \\"
    )
    rows.append(r"        \midrule")
    for method in METHOD_ORDER:
        row = by_strategy[method]
        cycle_h = row["cycle_time_mean_s_mean"] / 3600.0
        cycle_h_std = row["cycle_time_mean_s_std"] / 3600.0
        wait_min = row["waiting_time_mean_s_mean"] / 60.0
        wait_min_std = row["waiting_time_mean_s_std"] / 60.0
        gini = row["resource_fairness_gini_mean"]
        gini_std = row["resource_fairness_gini_std"]
        weighted = row["weighted_resource_fairness_mean"]
        runtime = row["simulation_runtime_seconds_mean"]
        rows.append(
            "        "
            f"{tex_escape(method)} & {completion(raw, method)} & "
            f"${fmt(cycle_h)} \\pm {fmt(cycle_h_std)}$ & "
            f"${fmt(wait_min)} \\pm {fmt(wait_min_std)}$ & "
            f"${fmt(gini)} \\pm {fmt(gini_std)}$ & "
            f"${fmt(weighted)}$ & ${fmt(runtime, 3)}$ \\\\"
        )
    rows.append(r"        \bottomrule")
    rows.append(r"    \end{tabular}}")
    rows.append(r"    \label{tab:joao_fixed_replay}")
    rows.append(r"\end{table*}")
    return "\n".join(rows) + "\n"


def per_seed_table(raw: pd.DataFrame) -> str:
    rows = []
    ordered = raw.copy()
    ordered["strategy"] = pd.Categorical(ordered["strategy"], METHOD_ORDER, ordered=True)
    ordered = ordered.sort_values(["strategy", "seed"])
    rows.append(r"\begin{table*}[p]")
    rows.append(
        r"    \caption{Per-seed canonical fixed-replay metrics. Cycle time is in hours; "
        r"waiting time is in minutes; occupation is normalized by the full arrival-plus-drain horizon.}"
    )
    rows.append(r"    \centering")
    rows.append(r"    \tiny")
    rows.append(r"    \resizebox{\textwidth}{!}{")
    rows.append(r"    \begin{tabular}{llrrrrrr}")
    rows.append(r"        \toprule")
    rows.append(
        r"        \textbf{Method} & \textbf{Seed} & \textbf{Completed/admitted} & "
        r"\textbf{Cycle h} & \textbf{Wait min} & \textbf{Occ. horizon \%} & "
        r"\textbf{Gini} & \textbf{Runtime s} \\"
    )
    rows.append(r"        \midrule")
    for _, row in ordered.iterrows():
        rows.append(
            "        "
            f"{tex_escape(row['strategy'])} & {int(row['seed'])} & "
            f"{int(row['fixed_routes_completed'])}/{int(row['cases_admitted'])} & "
            f"{fmt(row['cycle_time_mean_s'] / 3600.0)} & "
            f"{fmt(row['waiting_time_mean_s'] / 60.0)} & "
            f"{fmt(row['horizon_normalized_resource_occupation_mean'] * 100.0)} & "
            f"{fmt(row['resource_fairness_gini'])} & "
            f"{fmt(row['simulation_runtime_seconds'], 3)} \\\\"
        )
    rows.append(r"        \bottomrule")
    rows.append(r"    \end{tabular}}")
    rows.append(r"\end{table*}")
    return "\n".join(rows) + "\n"


def direct_shortest_queue_round_robin(raw: pd.DataFrame) -> list[dict[str, object]]:
    metric_direction = {
        "fixed_route_completion_rate": "higher_is_better",
        "cycle_time_mean_s": "lower_is_better",
        "waiting_time_mean_s": "lower_is_better",
        "resource_fairness_gini": "lower_is_better",
        "weighted_resource_fairness": "lower_is_better",
        "horizon_normalized_resource_occupation_mean": "descriptive",
        "horizon_normalized_throughput_cases_per_hour": "higher_is_better",
    }
    rows: list[dict[str, object]] = []
    for metric, direction in metric_direction.items():
        if metric not in raw.columns:
            continue
        pivot = raw.pivot_table(index="seed", columns="strategy", values=metric, aggfunc="first")
        if "ShortestQueue" not in pivot or "RoundRobin" not in pivot:
            continue
        diff = (pivot["ShortestQueue"] - pivot["RoundRobin"]).dropna()
        pct = (diff / pivot["RoundRobin"].replace(0, pd.NA) * 100.0).dropna()
        if direction == "lower_is_better":
            wins = int((diff < 0).sum())
            ties = int((diff == 0).sum())
            losses = int((diff > 0).sum())
        elif direction == "higher_is_better":
            wins = int((diff > 0).sum())
            ties = int((diff == 0).sum())
            losses = int((diff < 0).sum())
        else:
            wins = ties = losses = pd.NA
        rows.append(
            {
                "comparison": "ShortestQueue - RoundRobin",
                "metric": metric,
                "metric_direction": direction,
                "paired_seed_count": int(diff.size),
                "mean_difference": float(diff.mean()) if not diff.empty else float("nan"),
                "median_difference": float(diff.median()) if not diff.empty else float("nan"),
                "std_difference": float(diff.std()) if diff.size > 1 else float("nan"),
                "ci_method": "student_t",
                "ci_level": 0.95,
                "difference_ci95_half_width": (
                    float(student_t.ppf(0.975, df=diff.size - 1) * diff.std() / math.sqrt(diff.size))
                    if diff.size > 1 else float("nan")
                ),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "mean_percent_difference": float(pct.mean()) if not pct.empty else float("nan"),
                "median_percent_difference": float(pct.median()) if not pct.empty else float("nan"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="joao/results/final_canonical_rfopt_candidate_20260717/fixed_replay",
    )
    parser.add_argument(
        "--output-dir",
        default="report/joao_resource_allocation_support",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(results_dir / "fixed_replay_raw_metrics.csv")
    aggregated = pd.read_csv(results_dir / "fixed_replay_aggregated_metrics.csv")
    paired = pd.read_csv(results_dir / "final_paired_comparisons.csv")

    (output_dir / "part_ii_a_fixed_replay_table.tex").write_text(
        fixed_replay_table(raw, aggregated),
        encoding="utf-8",
    )
    (output_dir / "part_ii_a_per_seed_table.tex").write_text(
        per_seed_table(raw),
        encoding="utf-8",
    )

    report_methods = set(pd.read_csv(results_dir / "final_report_table.csv")["Method"])
    raw_methods = set(raw["strategy"])
    active_paths = [
        str(path)
        for path in [
            results_dir / "final_report_table.csv",
            results_dir / "fixed_replay_raw_metrics.csv",
            results_dir / "final_paired_comparisons.csv",
        ]
    ]
    summary = {
        "results_dir": str(results_dir),
        "raw_rows": int(len(raw)),
        "method_count": int(raw["strategy"].nunique()),
        "seed_count_by_method": raw.groupby("strategy")["seed"].nunique().to_dict(),
        "report_methods_match_raw": sorted(report_methods) == sorted(raw_methods),
        "active_paths": active_paths,
        "contains_20260716_path": any("final_canonical_20260716" in path for path in active_paths),
        "shortest_queue_minus_round_robin": (
            paired[paired["comparison"].eq("ShortestQueue - RoundRobin")].to_dict(orient="records")
            or direct_shortest_queue_round_robin(raw)
        ),
    }
    (output_dir / "part_ii_a_summary.json").write_text(
        json.dumps(json_clean(summary), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "part_ii_a_report_consistency.json").write_text(
        json.dumps(
            json_clean({
                "ok": summary["report_methods_match_raw"] and not summary["contains_20260716_path"],
                "checks": summary,
            }),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
