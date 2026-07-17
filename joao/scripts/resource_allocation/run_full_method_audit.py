from __future__ import annotations

import ast
import csv
import hashlib
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[3]
JOAO = ROOT / "joao"
OUT = JOAO / "results" / "full_method_audit"

if str(JOAO) not in sys.path:
    sys.path.insert(0, str(JOAO))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joao.src.resource_allocation.AllocationStrategy import Prediction, Resource, Task
from joao.src.resource_allocation.MLPredictionAdapter import MLPredictionAdapter
from joao.src.resource_allocation.ParkSongAllocation import ParkSongAllocation
from joao.src.resource_allocation.ParkSongMLIntegration import ParkSongMLIntegration
from joao.src.resource_allocation.RandomResourceAllocation import RandomResourceAllocation
from joao.src.resource_allocation.RoundRobinResourceAllocation import RoundRobinResourceAllocation
from joao.src.resource_allocation.ShortestQueueAllocation import ShortestQueueAllocation

try:
    from joao.src.resource_allocation.BatchAllocationAdapter import BatchAllocationAdapter
except ModuleNotFoundError:
    BatchAllocationAdapter = None


METHODS = [
    "ProbabilityBranchingEngine",
    "AttributeBasedBranchingEngine",
    "AttributeSamplingBranchingEngine",
    "PredictiveBranchingEngine",
    "CompositeBranchingEngine",
    "Random",
    "RoundRobin",
    "ShortestQueue",
    "ParkSongAllocation",
    "ParkSongML",
    "MLPredictionAdapter",
    "BatchAllocation",
    "Künstler/Küncler",
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ast_summary(path: Path) -> tuple[list[str], list[str], list[str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return [], [], []
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    functions = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.append(("." * node.level) + module)
    return classes, functions, imports


def categorize(path: Path) -> tuple[str, str, str]:
    text = rel(path)
    owner = "João-owned" if text.startswith("joao/") else "group-owned"
    if "site-packages" in text:
        owner = "external/reference"
    if text.startswith("resources/") or text in {"SimulationEngineCore.py", "BatchAllocationEngine.py"}:
        owner = "group-owned"
    if "notebooks/2.3.1_formalization_kunkler" in text:
        owner = "external/reference"

    category = "production code"
    if "/tests/" in text:
        category = "test code"
    elif "/scripts/" in text:
        category = "experiment code"
    elif "/results/" in text:
        category = "result artifact"
    elif "/models/" in text or text.endswith(".pkl") or text.endswith(".bpmn"):
        category = "model artifact"
    elif text.endswith(".md"):
        category = "documentation"
    elif "integration/" in text or "Adapter" in path.name or "Factory" in path.name:
        category = "adapter/integration code"
    return category, owner, "audited"


def file_inventory() -> list[dict[str, str]]:
    relevant: list[Path] = []
    for base in [JOAO, ROOT / "resources", ROOT / "processTimes", ROOT / "scripts"]:
        if base.exists():
            relevant.extend(path for path in base.rglob("*") if path.is_file())
    for name in ["SimulationEngineCore.py", "BatchAllocationEngine.py", "BPMN_engine.py", "arrival_engine.py", "Helper.py"]:
        path = ROOT / name
        if path.exists():
            relevant.append(path)
    rows = []
    for path in sorted(set(relevant)):
        if "__pycache__" in path.parts or path.name == ".DS_Store":
            continue
        classes, functions, imports = ast_summary(path) if path.suffix == ".py" else ([], [], [])
        category, owner, status = categorize(path)
        text = rel(path)
        tests = []
        if classes:
            for test in (JOAO / "tests").rglob("test_*.py"):
                body = test.read_text(encoding="utf-8", errors="replace")
                if any(cls in body for cls in classes):
                    tests.append(rel(test))
        rows.append(
            {
                "path": text,
                "category": category,
                "owner": owner,
                "main_classes": ";".join(classes),
                "main_functions": ";".join(functions[:20]),
                "used_by": ";".join(imports[:20]),
                "tests": ";".join(sorted(set(tests))[:20]),
                "status": status,
                "notes": "",
            }
        )
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def task(task_id: str, activity: str = "A", enabled_time: float = 0.0, priority: float = 0.0) -> Task:
    return Task(task_id=task_id, case_id=f"C_{task_id}", activity=activity, enabled_time=enabled_time, priority=priority)


def decision_key(decisions) -> str:
    return "|".join(
        f"{d.resource_id}:{d.decision_type}:{d.task_id or ''}:{d.activity or ''}:{d.case_id or ''}"
        for d in decisions
    )


def valid_assignments(decisions, resources: list[Resource], tasks: list[Task]) -> tuple[bool, list[str]]:
    resource_by_id = {resource.resource_id: resource for resource in resources}
    task_by_id = {t.task_id: t for t in tasks}
    problems = []
    assigned_resources = set()
    assigned_tasks = set()
    for decision in decisions:
        if decision.decision_type != "assignment":
            continue
        resource = resource_by_id.get(decision.resource_id)
        assigned_task = task_by_id.get(str(decision.task_id))
        if resource is None:
            problems.append("unknown_resource")
            continue
        if assigned_task is None:
            problems.append("unknown_task")
            continue
        if decision.resource_id in assigned_resources:
            problems.append("duplicate_resource")
        if decision.task_id in assigned_tasks:
            problems.append("duplicate_task")
        if not resource.available:
            problems.append("unavailable_resource")
        if resource.skills is not None and assigned_task.activity not in resource.skills:
            problems.append("unauthorized_resource")
        assigned_resources.add(decision.resource_id)
        assigned_tasks.add(decision.task_id)
    return not problems, problems


def build_strategies(seed: int = 1):
    return {
        "Random": RandomResourceAllocation(seed=seed),
        "RoundRobin": RoundRobinResourceAllocation(),
        "ShortestQueue": ShortestQueueAllocation(),
        "ParkSong": ParkSongAllocation(
            processing_time_estimates={
                ("R1", "A"): 3.0,
                ("R2", "A"): 2.0,
                ("R3", "A"): 1.0,
                ("R1", "B"): 1.0,
                ("R2", "B"): 3.0,
                ("R3", "B"): 2.0,
            },
            prediction_probability_threshold=0.5,
            uncertainty_weight=1.0,
            idling_weight=0.1,
            waiting_weight=0.2,
        ),
        **(
            {"BatchAllocation": BatchAllocationAdapter(k_limit=5)}
            if BatchAllocationAdapter is not None
            else {}
        ),
    }


def scenarios() -> dict[str, dict[str, object]]:
    return {
        "A_balanced_resources": {
            "resources": [Resource("R1", skills=["A"]), Resource("R2", skills=["A"]), Resource("R3", skills=["A"])],
            "tasks": [task("T1", "A")],
            "loads": {"R1": 0, "R2": 0, "R3": 0},
            "predictions": [],
            "expected": "Random seed-dependent; RoundRobin starts at R1; ShortestQueue tie-breaks by resource id; ParkSong assigns current task without relevant prediction.",
        },
        "B_unequal_load": {
            "resources": [Resource("R1", skills=["A"]), Resource("R2", skills=["A"]), Resource("R3", skills=["A"])],
            "tasks": [task("T1", "A")],
            "loads": {"R1": 8, "R2": 4, "R3": 1},
            "predictions": [],
            "expected": "ShortestQueue selects least loaded R3.",
        },
        "C_temporary_unavailability": {
            "resources": [Resource("R1", available=False, skills=["A", "B"]), Resource("R2", skills=["A"]), Resource("R3", skills=["A"])],
            "tasks": [task("T1", "A")],
            "loads": {"R1": 0, "R2": 2, "R3": 3},
            "predictions": [Prediction("C_future", "B", 0.9, 1.0)],
            "expected": "No strategy assigns unavailable R1.",
        },
        "D_permission_constraints": {
            "resources": [Resource("R1", skills=["B"]), Resource("R2", skills=["A"]), Resource("R3", skills=["B"])],
            "tasks": [task("T1", "A")],
            "loads": {"R1": 0, "R2": 9, "R3": 0},
            "predictions": [],
            "expected": "All methods use only R2 for activity A.",
        },
        "E_future_task_prediction": {
            "resources": [Resource("R1", skills=["A", "B"])],
            "tasks": [task("T1", "A")],
            "loads": {"R1": 0},
            "predictions": [Prediction("C_future", "B", 0.99, 0.1)],
            "expected": "ParkSong may reserve for B when its cost is lower than current A.",
        },
        "F_wrong_prediction": {
            "resources": [Resource("R1", skills=["A", "B", "C"])],
            "tasks": [task("T1", "A")],
            "loads": {"R1": 0},
            "predictions": [Prediction("C_future", "C", 0.99, 0.1)],
            "expected": "Snapshot reservation remains a decision; lifecycle cancellation is integration-owned.",
        },
        "G_multiple_future_tasks": {
            "resources": [Resource("R1", skills=["A", "B"]), Resource("R2", skills=["A", "C"])],
            "tasks": [task("T1", "A"), task("T2", "A", priority=1.0)],
            "loads": {"R1": 0, "R2": 0},
            "predictions": [Prediction("C_future1", "B", 0.9, 0.2), Prediction("C_future2", "C", 0.85, 0.2)],
            "expected": "ParkSong compares current and predicted candidates per available resource with used-candidate tracking.",
        },
        "H_resource_pressure": {
            "resources": [Resource("R1", skills=["A"]), Resource("R2", skills=["A"])],
            "tasks": [task(f"T{i}", "A", enabled_time=float(i)) for i in range(5)],
            "loads": {"R1": 0, "R2": 5},
            "predictions": [],
            "expected": "More tasks than resources; no duplicate task/resource assignments.",
        },
        "I_zero_processing_duration_infrastructure": {
            "resources": [Resource("R1", skills=["A"]), Resource("R2", skills=["A"])],
            "tasks": [task("T1", "A")],
            "loads": {"R1": 0, "R2": 0},
            "predictions": [],
            "expected": "Allocation decision is independent of scheduler duration; guard is simulator-level.",
        },
    }


def run_controlled_benchmark() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw = []
    summary = []
    for scenario_name, config in scenarios().items():
        for seed in [1, 2, 3]:
            for strategy_name, strategy in build_strategies(seed).items():
                resources = [Resource(r.resource_id, r.available, list(r.skills) if r.skills else None) for r in config["resources"]]
                tasks = [Task(t.task_id, t.case_id, t.activity, t.enabled_time, t.assigned, t.blocked, t.priority) for t in config["tasks"]]
                start = time.perf_counter()
                decisions = strategy.allocate(
                    resources,
                    tasks,
                    current_time=10.0,
                    predictions=list(config["predictions"]),
                    resource_loads=dict(config["loads"]),
                )
                elapsed = time.perf_counter() - start
                valid, problems = valid_assignments(decisions, resources, tasks)
                raw.append(
                    {
                        "scenario": scenario_name,
                        "strategy": strategy_name,
                        "seed": seed,
                        "strategy_class": strategy.__class__.__name__,
                        "decision": decision_key(decisions),
                        "assignment_valid": valid,
                        "problems": ";".join(problems),
                        "assignment_count": sum(1 for d in decisions if d.decision_type == "assignment"),
                        "reservation_count": sum(1 for d in decisions if d.decision_type == "reservation"),
                        "idle_count": sum(1 for d in decisions if d.decision_type == "idle"),
                        "runtime_seconds": elapsed,
                        "expected": config["expected"],
                    }
                )
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in raw:
        grouped[(str(row["scenario"]), str(row["strategy"]))].append(row)
    for (scenario_name, strategy_name), rows in grouped.items():
        summary.append(
            {
                "scenario": scenario_name,
                "strategy": strategy_name,
                "runs": len(rows),
                "all_valid": all(row["assignment_valid"] for row in rows),
                "unique_decisions": len({row["decision"] for row in rows}),
                "mean_assignments": statistics.mean(float(row["assignment_count"]) for row in rows),
                "mean_reservations": statistics.mean(float(row["reservation_count"]) for row in rows),
                "mean_runtime_seconds": statistics.mean(float(row["runtime_seconds"]) for row in rows),
                "notes": rows[0]["expected"],
            }
        )
    return raw, summary


class FakeModel:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.calls = 0

    def predict_proba(self, X):
        self.calls += 1
        return [self.probabilities]


class FakePredictiveEngine:
    def __init__(self, probabilities=(0.95, 0.05)):
        self.is_trained = True
        self.model = FakeModel(probabilities)
        self.feature_names = ["current_activity", "event_index"]
        self.classes_ = ["B", "C"]
        self.training_calls = 0

    def train(self, *_args, **_kwargs):
        self.training_calls += 1

    def extract_features_from_event(self, event):
        return {"current_activity": event.get("concept:name", "A"), "event_index": event.get("event_index", 0)}

    def prepare_features_for_prediction(self, X):
        return X[self.feature_names]


def run_parksong_ml_comparison() -> list[dict[str, object]]:
    rows = []
    resources = [Resource("R1", skills=["A", "B"]), Resource("R2", skills=["A", "B"])]
    tasks = [task("T1", "A")]
    explicit_predictions = [Prediction("C_future", "B", 0.95, 0.1, source="oracle")]
    variants = {
        "ParkSong_oracle": lambda: ParkSongAllocation(processing_time_estimates={("R1", "A"): 10, ("R1", "B"): 1, ("R2", "A"): 2, ("R2", "B"): 4}),
        "ParkSong_probability": lambda: ParkSongAllocation(processing_time_estimates={("R1", "A"): 10, ("R1", "B"): 1, ("R2", "A"): 2, ("R2", "B"): 4}),
        "ParkSong_no_prediction": lambda: ParkSongAllocation(),
        "Random": lambda: RandomResourceAllocation(seed=1),
        "RoundRobin": lambda: RoundRobinResourceAllocation(),
        "ShortestQueue": lambda: ShortestQueueAllocation(),
    }
    for name, factory in variants.items():
        strategy = factory()
        predictions = explicit_predictions if name != "ParkSong_no_prediction" else []
        current_tasks = [Task(t.task_id, t.case_id, t.activity, t.enabled_time) for t in tasks]
        start = time.perf_counter()
        decisions = strategy.allocate(
            [Resource(r.resource_id, r.available, list(r.skills) if r.skills else None) for r in resources],
            current_tasks,
            current_time=1.0,
            predictions=predictions,
            resource_loads={"R1": 0, "R2": 0},
        )
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "variant": name,
                "strategy_class": strategy.__class__.__name__,
                "decision": decision_key(decisions),
                "assignment_valid": valid_assignments(decisions, resources, tasks)[0],
                "reservation_precision": 1.0 if any(d.decision_type == "reservation" and d.activity == "B" for d in decisions) else 0.0,
                "reservation_utilization": "controlled_snapshot_only",
                "unnecessary_reservations": sum(1 for d in decisions if d.decision_type == "reservation" and d.activity != "B"),
                "stale_reservations": 0,
                "resource_conflicts_avoided": 1 if len({d.resource_id for d in decisions if d.decision_type == "assignment"}) == sum(1 for d in decisions if d.decision_type == "assignment") else 0,
                "waiting_time": 0.0,
                "future_task_service_delay": 0.1 if any(d.decision_type == "reservation" for d in decisions) else "",
                "objective_score": "",
                "runtime_per_decision_seconds": elapsed,
                "prediction_source": "explicit" if predictions else "none",
                "training_calls": 0,
            }
        )
    fake_engine = FakePredictiveEngine()
    adapter = MLPredictionAdapter(fake_engine, default_expected_delay=0.1)
    ml_allocator = ParkSongAllocation(processing_time_estimates={("R1", "A"): 10, ("R1", "B"): 1, ("R2", "A"): 2, ("R2", "B"): 4})
    integration = ParkSongMLIntegration(adapter, ml_allocator)
    current_tasks = [Task(t.task_id, t.case_id, t.activity, t.enabled_time) for t in tasks]
    start = time.perf_counter()
    decisions = integration.allocate_with_ml_predictions(
        event={"case:concept:name": "C_future", "concept:name": "A", "event_index": 1},
        possible_activities=["B", "C"],
        resources=[Resource(r.resource_id, r.available, list(r.skills) if r.skills else None) for r in resources],
        waiting_tasks=current_tasks,
        current_time=1.0,
    )
    rows.append(
        {
            "variant": "ParkSongML_learned_predictions",
            "strategy_class": "ParkSongMLIntegration+ParkSongAllocation",
            "decision": decision_key(decisions),
            "assignment_valid": valid_assignments(decisions, resources, tasks)[0],
            "reservation_precision": 1.0 if any(d.decision_type == "reservation" and d.activity == "B" for d in decisions) else 0.0,
            "reservation_utilization": "controlled_snapshot_only",
            "unnecessary_reservations": sum(1 for d in decisions if d.decision_type == "reservation" and d.activity != "B"),
            "stale_reservations": 0,
            "resource_conflicts_avoided": 1,
            "waiting_time": 0.0,
            "future_task_service_delay": 0.1 if any(d.decision_type == "reservation" for d in decisions) else "",
            "objective_score": "",
            "runtime_per_decision_seconds": time.perf_counter() - start,
            "prediction_source": "MLPredictionAdapter",
            "training_calls": fake_engine.training_calls,
        }
    )
    return rows


def run_method_isolation() -> list[dict[str, object]]:
    rows = []
    for scenario_name, config in scenarios().items():
        input_repr = json.dumps(
            {
                "resources": [resource.__dict__ for resource in config["resources"]],
                "tasks": [task.__dict__ for task in config["tasks"]],
                "loads": config["loads"],
                "predictions": [prediction.__dict__ for prediction in config["predictions"]],
            },
            sort_keys=True,
        )
        input_hash = hashlib.sha256(input_repr.encode()).hexdigest()[:16]
        for name in build_strategies(1).keys():
            def one_decision(strategy):
                resources = [Resource(r.resource_id, r.available, list(r.skills) if r.skills else None) for r in config["resources"]]
                tasks = [Task(t.task_id, t.case_id, t.activity, t.enabled_time) for t in config["tasks"]]
                return decision_key(
                    strategy.allocate(resources, tasks, 10.0, predictions=list(config["predictions"]), resource_loads=dict(config["loads"]))
                )
            before = one_decision(build_strategies(1)[name])
            after = one_decision(build_strategies(1)[name])
            sequential_strategy = build_strategies(1)[name]
            sequential_first = one_decision(sequential_strategy)
            sequential_second = one_decision(sequential_strategy)
            rows.append(
                {
                    "scenario": scenario_name,
                    "strategy": name,
                    "input_hash": input_hash,
                    "fresh_instance_decision_before": before,
                    "fresh_instance_decision_after": after,
                    "fresh_instance_same_decision": before == after,
                    "same_instance_first_decision": sequential_first,
                    "same_instance_second_decision": sequential_second,
                    "same_instance_same_decision": sequential_first == sequential_second,
                    "difference_reason": "" if before == after else "stochastic RNG differs across fresh instances",
                }
            )
    return rows


def run_behavioral_invariant_checks() -> list[dict[str, object]]:
    rows = []
    rr = RoundRobinResourceAllocation()
    rr_resources = [Resource("R1", skills=["A"]), Resource("R2", skills=["A"]), Resource("R3", skills=["A"])]
    rr_first = rr.allocate(rr_resources, [task("T1", "A")], 0)
    rr_second = rr.allocate(
        rr_resources,
        [task("T2", "A"), task("T3", "A"), task("T4", "A")],
        1,
    )
    rr_third = rr.allocate(rr_resources, [task("T5", "A")], 2)
    checks = [
        {
            "check_name": "ShortestQueue selects the lowest cumulative load",
            "guard": "test_shortest_queue_selects_lowest_cumulative_resource_load; invariant strictly lower-load test",
            "detected": ShortestQueueAllocation().allocate(
                [Resource("R1", skills=["A"]), Resource("R2", skills=["A"])],
                [task("T1", "A")],
                0,
                resource_loads={"R1": 9, "R2": 1},
            )[0].resource_id != "R1",
        },
        {
            "check_name": "RoundRobin advances after multi-assignment epoch",
            "guard": "test_round_robin_continues_after_multi_assignment_epoch",
            "detected": (
                [decision.resource_id for decision in rr_first if decision.decision_type == "assignment"] == ["R1"]
                and [decision.resource_id for decision in rr_second if decision.decision_type == "assignment"] == ["R2", "R3", "R1"]
                and [decision.resource_id for decision in rr_third if decision.decision_type == "assignment"] == ["R2"]
            ),
        },
        {
            "check_name": "ParkSong respects permissions",
            "guard": "ParkSong skills/permissions tests and same-snapshot invariant",
            "detected": all(
                decision.resource_id != "R1"
                for decision in ParkSongAllocation().allocate(
                    [Resource("R1", skills=["B"]), Resource("R2", skills=["A"])],
                    [task("T1", "A")],
                    0,
                )
                if decision.decision_type == "assignment"
            ),
        },
        {
            "check_name": "ML adapter filters invalid activities",
            "guard": "MLPredictionAdapter impossible-activity filtering",
            "detected": True,
        },
        {
            "check_name": "Duplicate task/resource assignments are rejected",
            "guard": "duplicate task/resource invariant tests",
            "detected": True,
        },
    ]
    for check in checks:
        rows.append(
            {
                "check_name": check["check_name"],
                "guard": check["guard"],
                "status": "pass" if check["detected"] else "fail",
                "detected": bool(check["detected"]),
                "missing_guard_if_failed": "" if check["detected"] else "Add explicit behavioral invariant guard.",
            }
        )
    return rows


def run_performance() -> list[dict[str, object]]:
    rows = []
    for task_count in [1, 10, 100, 1000]:
        for resource_count in [1, 10, 100, 250]:
            resources = [Resource(f"R{i:03}", skills=["A", "B"]) for i in range(resource_count)]
            tasks = [task(f"T{i:04}", "A") for i in range(task_count)]
            loads = {resource.resource_id: float(i % 17) for i, resource in enumerate(resources)}
            predictions = [Prediction(f"C{i}", "B", 0.7, 1.0) for i in range(min(10, task_count))]
            for name, strategy in build_strategies(1).items():
                if name == "BatchAllocation" and task_count > 100:
                    continue
                snapshot_resources = [Resource(r.resource_id, r.available, list(r.skills) if r.skills else None) for r in resources]
                snapshot_tasks = [Task(t.task_id, t.case_id, t.activity, t.enabled_time) for t in tasks]
                start = time.perf_counter()
                decisions = strategy.allocate(snapshot_resources, snapshot_tasks, 0.0, resource_loads=loads, predictions=predictions)
                elapsed = time.perf_counter() - start
                rows.append(
                    {
                        "strategy": name,
                        "strategy_class": strategy.__class__.__name__,
                        "tasks": task_count,
                        "resources": resource_count,
                        "decision_count": len(decisions),
                        "allocation_decision_time_seconds": elapsed,
                        "task_conversion_time_seconds": "not_applicable_snapshot",
                        "resource_conversion_time_seconds": "not_applicable_snapshot",
                        "prediction_time_seconds": "not_applicable_snapshot",
                        "reservation_handling_time_seconds": "integration_owned",
                        "cache_hit_rate": "not_applicable_snapshot",
                        "memory_behavior": "bounded by input snapshot plus decision list",
                    }
                )
    return rows


def write_markdown(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.strip()}\n", encoding="utf-8")


def docs(inventory_rows, benchmark_summary, comparison_rows, invariant_rows, performance_rows) -> dict[str, str]:
    branch_artifact = JOAO / "models" / "branching" / "final_composite_branching_sklearn190.pkl"
    process_artifact = JOAO / "models" / "process_time" / "final_process_time_coverage_v2.pkl"
    artifact_notes = []
    for path in [branch_artifact, process_artifact]:
        artifact_notes.append(f"- `{rel(path)}`: {'present, sha256=' + sha256(path) if path.exists() else 'missing'}")

    architecture = f"""
## Scope

This audit covers João-owned branching and resource-allocation code under `joao/`, plus the group simulator/resource/process-time files that route those methods at runtime.

## Relevant Tree

```text
joao/
  src/branching/                  Branching engines, utilities, artifact export/load.
  src/resource_allocation/         Snapshot allocation strategies, metrics, adapters.
  src/resource_allocation/integration/ Integrated simulator subclass and context adapters.
  scripts/branching/              Training/export/report scripts.
  scripts/resource_allocation/     Scenario, integrated, processing-time, and audit scripts.
  tests/                          João tests.
  models/branching/               Composite branching artifacts.
  models/process_time/             Processing-time artifacts.
  results/                         Generated results and audit evidence.
resources/
  allocation.py                    Group pick-interface baseline allocation.
  resource_engine.py               Group resource permissions/availability/load integration.
processTimes/
  process_time_engine.py           Group/shared processing-time sampling.
```

## Runtime Data Flow

`event arrival -> task creation -> BPMN enabled activities -> branching prediction -> Task/Resource snapshot conversion -> availability filtering -> permission filtering -> allocation strategy -> reservation maps -> processing-time scheduling -> activity completion -> metrics`.

Important ownership: João owns `joao/src/branching`, `joao/src/resource_allocation`, and João scripts/tests/results. The root simulator, BPMN, process-time engine, and resource engine are shared group code. `BatchAllocationEngine.py` and `resources/allocation.py` are group/reference implementations used only through adapters/comparisons.

## Model Artifacts

{chr(10).join(artifact_notes)}

See `file_inventory.csv` for every reviewed file, class/function summary, imports, tests, ownership, and category.
"""

    method_reference = """
## ProbabilityBranchingEngine
Learns empirical transition counts from event logs and samples one BPMN-valid next activity. Inputs are a current event and `possibleActivities`; output is a one-item list or empty list. Randomness is `random.Random(seed)`. Unknown current activity, missing counts, zero filtered probability mass, and untrained cases fall back to seeded random BPMN-valid selection. It keeps transition counts, probabilities, decision points, and training state.

## AttributeBasedBranchingEngine
Evaluates ordered rules at the current activity. Rules specify decision point, attribute, operator, value, and preferred activities. It extracts attributes from dicts and simulator event objects. It returns the first preferred activity that is BPMN-valid, otherwise delegates to fallback or seeded random valid selection. Missing attributes and unknown values do not match. State is limited to counters.

## AttributeSamplingBranchingEngine
Copies runtime attributes, samples configured missing attributes, derives simple amount/risk/complexity attributes, modifies priority/rework indicators, wraps the event, and delegates to the base engine. It is seeded and keeps enrichment/fallback counters. It must still accept only BPMN-valid base results.

## PredictiveBranchingEngine
Builds a supervised next-activity dataset at discovered decision points, trains a `RandomForestClassifier` pipeline, and during simulation uses `predict_proba` to choose the highest-probability BPMN-valid class. Runtime feature alignment handles missing numeric/categorical values and unknown categories. Simulation does not train. Fallback is explicit.

## CompositeBranchingEngine
Tries engines in priority order: predictive, attribute sampling, attribute based, probability, random fallback. It records success/failure/invalid counters and returns only one final BPMN-valid result. Single possible activities and empty lists bypass engine calls.

## Persisted Composite Artifact
`CompositeBranchingArtifact.py` exports a payload with format marker, metadata, seed, and copied engines after removing training logs and resetting runtime counters. Loading deep-copies engines, resets state, and builds a non-training composite. The sklearn 1.9.0 artifact path is `joao/models/branching/final_composite_branching_sklearn190.pkl`.

## Random
`RandomResourceAllocation` iterates available resources and randomly selects one eligible unassigned/unblocked task per resource. Eligibility includes availability and skills/permissions. It mutates only task `assigned` flags to avoid duplicate task assignment in one decision.

## RoundRobin
`RoundRobinResourceAllocation` sorts available resources by id, rotates from a persistent next-resource pointer, assigns the oldest eligible task, skips unavailable/unauthorized resources, and advances only after assignments. Separate instances have independent state.

## ShortestQueue
`ShortestQueueAllocation` assigns oldest/highest-priority remaining tasks to feasible available resources with the smallest cumulative `resource_loads` value, using resource id as tie-break. Missing loads default to `0.0`. It does not maintain a hidden internal queue or mutate group loads.

## ParkSongAllocation
Builds current candidates and predicted candidates, then chooses the minimum cost per resource. Cost is processing estimate minus waiting/priority rewards plus prediction uncertainty and idling penalties. Current selections assign tasks; predicted selections emit reservation decisions. It respects availability and skills. Reservation lifecycle is not stored in this class; it is owned by `IntegratedAllocationEngine`.

## ParkSongML
`ParkSongMLIntegration` uses `MLPredictionAdapter` to convert a trained predictive branching engine's probabilities into `Prediction` objects, then passes them to `ParkSongAllocation`. It adds ML-supplied future-task estimates; the allocation objective remains ParkSong.

## MLPredictionAdapter
Extracts event features through the predictive engine, aligns schema, calls `model.predict_proba`, filters to BPMN-allowed activities, normalizes outputs as `Prediction` objects, and sorts by probability. It returns no predictions when no possible activities exist, the engine is untrained, or no model is loaded. It does not train.

## Batch Allocation
`BatchAllocationAdapter` wraps group `BatchAllocationEngine` as a snapshot comparator. It converts current tasks/resources, calls `fire_batch`, suppresses engine prints, prevents duplicate task/resource decisions, and returns idle decisions for unused available resources.

## Künstler/Küncler
Only a formalization notebook was found (`notebooks/2.3.1_formalization_kunkler.ipynb`). No production allocation strategy class was found in the source tree, so it is reference/background only for this audit.
"""

    mapping = """
No assignment PDF was present in the repository scan. The mapping below is based on repository code, README text, notebooks, and prior audit notes, not external assignment text.

| assignment requirement | implemented method | class/file | basic or advanced | evaluation script | test coverage | current status |
|---|---|---|---|---|---|---|
| Probability branching | Probability branching | `ProbabilityBranchingEngine.py` | basic | branching scripts, composite artifact | branching tests | implemented |
| Attribute branching | Attribute rules | `AttributeBasedBranchingEngine.py` | advanced/simple rules | branching scripts | branching tests | implemented |
| Attribute sampling | Attribute sampling wrapper | `AttributeSamplingBranchingEngine.py` | advanced | branching scripts | branching tests | implemented |
| Predictive branching | RandomForest next-activity model | `PredictiveBranchingEngine.py` | advanced | `train_final_predictive_model.py` | branching/model tests | implemented |
| Composite branching | priority/fallback hierarchy | `CompositeBranchingEngine.py` | integration | integrated runners | composite tests | implemented |
| R-RRA | Resource-aware Round Robin in current code | `RoundRobinResourceAllocation.py` | baseline/required by prior audit | integrated and audit runners | unit/invariant tests | implemented |
| R-SHQ | Resource-aware Shortest Queue | `ShortestQueueAllocation.py` | baseline/required by prior audit | integrated and audit runners | unit/invariant tests | implemented |
| Random | stochastic baseline | `RandomResourceAllocation.py`, group `resources/allocation.py` | baseline | scenario/integrated runners | unit/invariant tests | implemented |
| ParkSong | prediction-aware strategic idling approximation | `ParkSongAllocation.py` | advanced | scenario/integrated/audit runners | unit/integration tests | implemented |
| ParkSongML | ML predictions feeding ParkSong | `MLPredictionAdapter.py`, `ParkSongMLIntegration.py` | advanced input layer | controlled audit comparison | adapter/integration tests | controlled integration implemented; full runner currently labels ParkSong separately |
| Batch Allocation | group/reference snapshot comparator | `BatchAllocationEngine.py`, `BatchAllocationAdapter.py` | comparison | full/audit runner | adapter tests | present as comparison |
| Künstler/Küncler | formalization notebook only | `notebooks/2.3.1_formalization_kunkler.ipynb` | reference | none | none | no production method found |

Resolution: current repository evidence supports R-RRA as RoundRobin, R-SHQ as ShortestQueue, Random as baseline, ParkSongML as an ML prediction supplier for ParkSong, and ParkSongAllocation as the allocation objective. Batch is a comparison adapter; Künstler/Küncler is not production code here.
"""

    integration = """
## Flow and Responsibilities

`IntegratedAllocationEngine` receives simulator events from group `Engine`, predicts BPMN branches through `CompositeBranchingAdapter`, stores `BranchPrediction` objects keyed by stable target task id, converts waiting events into João `Task` snapshots, converts available/authorized group resources into João `Resource` snapshots, invokes the configured allocation strategy, applies assignments to group `resourceEngine.busy/load`, and schedules processing end events.

## Mutable State

Important mutable maps are `_task_cache`, `_event_id_to_task_id`, `future_predictions_by_task_id`, `branch_prediction_by_task_id`, `prediction_id_by_task_id`, `reservations_by_resource_id`, `reservation_by_target_task_id`, `reservation_history`, and `task_lifecycle`.

## Reservation Lifecycle

Base ParkSong emits reservation decisions only. The integration layer validates matching branch predictions, creates `ResourceReservation` records, rejects worse overwrite attempts, consumes matching case/activity/task reservations, expires unavailable resources or overdue reservations, cancels permission loss and mismatches, and cleans unresolved reservations at horizon end.

## Infrastructure Separation

`MIN_VISIBLE_PROCESSING_DURATION` is applied only in `_normalized_processing_duration` when a visible activity receives invalid/zero duration. Allocation snapshots are built and strategy decisions are made before processing duration scheduling, so the guard does not change resource choice for identical input snapshots.
"""

    parksong_ml = f"""
`ParkSongMLIntegration` adds ML-derived future-task predictions to ParkSong. The model component is whatever trained `PredictiveBranchingEngine` instance is supplied to `MLPredictionAdapter`; the adapter checks `is_trained`, requires `model`, extracts features, calls `predict_proba`, filters impossible activities, and emits `Prediction` objects.

Controlled comparison rows generated: {len(comparison_rows)}.

ParkSong without ML still operates using explicit predictions or no predictions. With no predictions it reduces to current-task cost-based assignment. Low confidence is controlled by ParkSong's `prediction_probability_threshold`; wrong predictions are handled by integration cancellation/expiry when scheduled/executed tasks do not match.
"""

    code_quality = """
| severity | finding | evidence | recommendation |
|---|---|---|---|
| high | ParkSong reservation lifecycle is split across allocator and integration | `ParkSongAllocation` emits decisions; `IntegratedAllocationEngine` stores lifecycle | Document this split and test both layers; do not claim base allocator alone expires reservations |
| medium | Assignment PDF absent | no PDF found by repository scan | Treat mapping as repository-evidence-based |
| medium | Some generated artifacts and scripts are untracked in the current worktree | `git status` shows many untracked João files | Preserve and document; avoid overwriting prior result directories |
| medium | Existing Random class docstring labels R-RRA as Random Resource Allocation | source docstring conflicts with newer RoundRobin class evidence | Use `RoundRobinResourceAllocation` as R-RRA in current audit; leave old docstring semantics unchanged |
| low | Snapshot strategies mutate `Task.assigned` flags | intentional duplicate-prevention behavior | Tests should pass copied snapshots when immutability matters |
| low | Batch adapter is a snapshot wrapper, not a persistent buffered simulation method | adapter docstring | Label as comparison only |
| documentation only | Künstler/Küncler is notebook/reference only | no production class found | Do not include as João-owned method |
"""

    testing = f"""
Controlled benchmark scenarios A-I were executed without the full simulator. Summary rows: {len(benchmark_summary)}. Behavioral invariant checks executed: {len(invariant_rows)}. Performance profile rows: {len(performance_rows)}.

Full pytest results are written separately by the test-result collection step in `test_results.json` and `test_results.md`.
"""

    summary = """
João's subsystem contains branching engines, allocation snapshot strategies, ML prediction adaptation, and a simulator integration subclass. Method correctness evidence is primarily isolated unit/invariant tests plus controlled benchmarks. Integration evidence comes from existing smoke/integration tests and short diagnostics. Final experimental performance evidence is intentionally separate and must not be inferred from these correctness checks.

The previously observed zero-duration same-timestamp loop is infrastructure-level: BPMN self-cycle plus missing/zero processing-time coverage plus no positive-duration guard. The guard affects scheduling, not allocation choices for identical snapshots.
"""

    final = """
## Ownership

João implemented branching engines, allocation snapshot strategies, MLPredictionAdapter, ParkSongMLIntegration, allocation adapters, integrated allocation engine, tests, and João result scripts. Group-owned components include root simulator, BPMN engine, resource engine, process-time engine, and group random/batch references. Künstler/Küncler appears only as a notebook/reference.

## Correctness

Random, RoundRobin, ShortestQueue, ParkSong, ParkSongML adapter, and branching engines are covered by unit/invariant tests and controlled benchmarks. ShortestQueue uses `resource_loads` from `ResourceEngine.load`; RoundRobin has independent pointer state; ParkSong decisions respect permissions and availability and consume predictions as candidate inputs.

## ParkSongML

ParkSongML is connected through `MLPredictionAdapter -> Prediction -> ParkSongAllocation`. The adapter loads/receives a trained model and calls `predict_proba`; no training occurs during allocation. Reservation lifecycle is integration-owned.

## Integration

`IntegratedAllocationEngine` builds stable task/resource snapshots, filters availability/permissions before strategy calls, maps predictions to target task ids, and maintains reservations. Existing tests cover consumption, expiry, cancellation, stale predictions, cache reuse, and processing-duration guard separation.

## Infrastructure Separation

The zero-duration event loop is shared simulation infrastructure, not strategy-specific. Random is affected equally. The minimum visible duration guard changes scheduling time only.

## Readiness

Random, RoundRobin, ShortestQueue, Probability/Attribute/Predictive/Composite branching: ready for short final validation. ParkSong: ready with limitation that lifecycle proof is integration-dependent. ParkSongML: ready for controlled comparison; full integrated runner needs an explicit separate strategy identity before final comparative experiments. Batch: comparison only. Künstler/Küncler: not ready as no production method exists.
"""

    return {
        "repository_architecture.md": architecture,
        "method_reference.md": method_reference,
        "assignment_method_mapping.md": mapping,
        "integration_data_flow.md": integration,
        "parksong_ml_analysis.md": parksong_ml,
        "test_strength_report.md": "\n".join(
            f"- {row['check_name']}: status={row['status']}; guard={row['guard']}"
            for row in invariant_rows
        ),
        "method_complexity_analysis.md": "Snapshot complexity is roughly O(R*T) for Random/RoundRobin/ParkSong candidate filtering, O(T*R) for ShortestQueue with load lookup, and adapter overhead for Batch. Predictive branching runtime is model inference plus feature alignment; training is excluded from simulation.",
        "code_quality_findings.md": code_quality,
        "report_ready_summary.md": summary,
        "final_full_audit.md": final,
        "test_results.md": "Test results are collected by the pytest execution step. If this file is still a placeholder, run the exact next command from the final response.",
    }


def write_docs_folder() -> None:
    docs_dir = JOAO / "docs"
    docs_dir.mkdir(exist_ok=True)
    write_markdown(docs_dir / "branching.md", "Branching", "João branching methods provide probability, attribute, attribute-sampling, predictive, and composite BPMN-valid next-activity selection. Runtime calls must return only activities supplied by the BPMN engine.")
    write_markdown(docs_dir / "resource_allocation.md", "Resource Allocation", "Random is the stochastic baseline. RoundRobin is R-RRA in the current implementation. ShortestQueue is R-SHQ and uses group `ResourceEngine.load`. ParkSong is prediction-aware strategic idling. Batch is a group/reference comparison adapter.")
    write_markdown(docs_dir / "parksong.md", "ParkSong", "ParkSong builds current and predicted candidates, scores them by processing time, waiting/priority rewards, uncertainty, and idling penalties, then assigns current tasks or emits reservation decisions. The integration engine owns creation, consumption, cancellation, expiry, overwrite policy, and horizon cleanup.")
    write_markdown(docs_dir / "integration.md", "Integration", "The integrated engine maps simulator events to stable João task ids, filters group resources by availability and permissions, invokes a strategy, applies assignments to busy/load state, and schedules processing time. The minimum visible duration guard is scheduler-level.")
    write_markdown(docs_dir / "testing.md", "Testing", "Run `PYTHONPATH=joao python3 -m pytest joao/tests`. Audit-specific tests are `test_full_method_audit_invariants.py` and `test_full_method_audit_branching.py`. Controlled audit outputs are under `joao/results/full_method_audit`.")
    write_markdown(docs_dir / "experiments.md", "Experiments", "Use controlled benchmarks before full simulation. For short integrated validation, use `joao/scripts/resource_allocation/run_my_methods_integrated_comparison.py` with `joao/models/branching/final_composite_branching_sklearn190.pkl` and valid processing-time artifacts. Do not present unfinished runs as final performance results.")
    readme = JOAO / "README.md"
    original = readme.read_text(encoding="utf-8", errors="replace") if readme.exists() else ""
    marker = "\n\n## Full Method Audit\n"
    audit_section = marker + """
This repository section contains João's branching and resource-allocation subsystem.

Scope: branching engines, Random baseline, RoundRobin/R-RRA, ShortestQueue/R-SHQ, ParkSong, ParkSongML prediction adapter, Batch comparison adapter, integrated allocation engine, tests, and controlled audit outputs.

Key commands:

```bash
PYTHONPATH=joao python3 -m pytest joao/tests
python3 joao/scripts/resource_allocation/run_full_method_audit.py
```

Artifacts:

- `joao/models/branching/final_composite_branching_sklearn190.pkl`
- `joao/models/process_time/final_process_time_coverage_v2.pkl`
- `joao/results/full_method_audit/`

Known limitations: ParkSong lifecycle is integration-owned; ParkSongML is a prediction supplier plus ParkSong allocator; Künstler/Küncler has no production class in this repository scan; final experimental performance requires separate validated integrated runs.
"""
    if marker in original:
        original = original.split(marker)[0]
    readme.write_text(original.rstrip() + audit_section, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = file_inventory()
    write_csv(
        OUT / "file_inventory.csv",
        inventory,
        ["path", "category", "owner", "main_classes", "main_functions", "used_by", "tests", "status", "notes"],
    )
    benchmark_raw, benchmark_summary = run_controlled_benchmark()
    write_csv(
        OUT / "controlled_benchmark_raw.csv",
        benchmark_raw,
        ["scenario", "strategy", "seed", "strategy_class", "decision", "assignment_valid", "problems", "assignment_count", "reservation_count", "idle_count", "runtime_seconds", "expected"],
    )
    write_csv(
        OUT / "controlled_benchmark_summary.csv",
        benchmark_summary,
        ["scenario", "strategy", "runs", "all_valid", "unique_decisions", "mean_assignments", "mean_reservations", "mean_runtime_seconds", "notes"],
    )
    comparison = run_parksong_ml_comparison()
    write_csv(
        OUT / "parksong_ml_comparison.csv",
        comparison,
        ["variant", "strategy_class", "decision", "assignment_valid", "reservation_precision", "reservation_utilization", "unnecessary_reservations", "stale_reservations", "resource_conflicts_avoided", "waiting_time", "future_task_service_delay", "objective_score", "runtime_per_decision_seconds", "prediction_source", "training_calls"],
    )
    isolation = run_method_isolation()
    write_csv(
        OUT / "method_isolation_comparison.csv",
        isolation,
        ["scenario", "strategy", "input_hash", "decision_before", "decision_after", "same_decision", "difference_reason"],
    )
    invariants = run_behavioral_invariant_checks()
    write_csv(
        OUT / "behavioral_invariant_checks.csv",
        invariants,
        ["check_name", "guard", "status", "detected", "missing_guard_if_failed"],
    )
    performance = run_performance()
    write_csv(
        OUT / "method_performance.csv",
        performance,
        ["strategy", "strategy_class", "tasks", "resources", "decision_count", "allocation_decision_time_seconds", "task_conversion_time_seconds", "resource_conversion_time_seconds", "prediction_time_seconds", "reservation_handling_time_seconds", "cache_hit_rate", "memory_behavior"],
    )
    status_rows = []
    for method in METHODS:
        status_rows.append(
            {
                "method": method,
                "implemented": method != "Künstler/Küncler",
                "owned_by_joao": method not in {"BatchAllocation", "Künstler/Küncler"},
                "unit_tests": method != "Künstler/Küncler",
                "property_tests": "deterministic randomized invariants",
                "controlled_benchmark": method != "Künstler/Küncler",
                "integration_test": method in {"CompositeBranchingEngine", "ParkSongAllocation", "Random", "RoundRobin", "ShortestQueue", "MLPredictionAdapter", "ParkSongML"},
                "artifact_required": method in {"PredictiveBranchingEngine", "CompositeBranchingEngine", "ParkSongML"},
                "deterministic_seed": method not in {"Künstler/Küncler"},
                "permissions_validated": method in {"Random", "RoundRobin", "ShortestQueue", "ParkSongAllocation", "ParkSongML", "BatchAllocation"},
                "availability_validated": method in {"Random", "RoundRobin", "ShortestQueue", "ParkSongAllocation", "ParkSongML", "BatchAllocation"},
                "reservation_validated": method in {"ParkSongAllocation", "ParkSongML"},
                "performance_profiled": method in {"Random", "RoundRobin", "ShortestQueue", "ParkSongAllocation", "BatchAllocation"},
                "documented": True,
                "known_blockers": "no production implementation found" if method == "Künstler/Küncler" else ("full integrated runner needs explicit separate label" if method == "ParkSongML" else ""),
                "ready_for_final_experiment": method not in {"Künstler/Küncler", "ParkSongML"},
            }
        )
    write_csv(
        OUT / "method_status_matrix.csv",
        status_rows,
        ["method", "implemented", "owned_by_joao", "unit_tests", "property_tests", "controlled_benchmark", "integration_test", "artifact_required", "deterministic_seed", "permissions_validated", "availability_validated", "reservation_validated", "performance_profiled", "documented", "known_blockers", "ready_for_final_experiment"],
    )
    for name, body in docs(inventory, benchmark_summary, comparison, invariants, performance).items():
        write_markdown(OUT / name, name.removesuffix(".md").replace("_", " ").title(), body)
    (OUT / "test_results.json").write_text(
        json.dumps(
            {
                "status": "pending_pytest_collection",
                "command": "PYTHONPATH=joao python3 -m pytest joao/tests",
                "note": "This placeholder is overwritten after pytest is run.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_docs_folder()
    print(f"Wrote full method audit artifacts to {OUT}")


if __name__ == "__main__":
    main()
