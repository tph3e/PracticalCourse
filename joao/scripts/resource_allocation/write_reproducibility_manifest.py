from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


HASH_FILES = [
    "fixed_replay/final_run_config.json",
    "fixed_replay/final_artifact_hashes.json",
    "fixed_replay/final_route_ids.csv",
    "fixed_replay/final_raw_metrics.csv",
    "fixed_replay/final_aggregated_metrics.csv",
    "fixed_replay/final_report_table.csv",
    "fixed_replay/final_paired_comparisons.csv",
]


def command_output(args: list[str]) -> str:
    result = subprocess.run(args, check=False, text=True, capture_output=True)
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_hashes(results_dir: Path) -> dict[str, str]:
    hashes = {}
    for relative in HASH_FILES:
        path = results_dir / relative
        if path.exists():
            hashes[relative] = sha256_file(path)
    return hashes


def package_config(results_dir: Path) -> dict:
    config_path = results_dir / "fixed_replay" / "final_run_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def package_artifacts(results_dir: Path) -> dict:
    artifact_path = results_dir / "fixed_replay" / "final_artifact_hashes.json"
    if not artifact_path.exists():
        return {}
    return json.loads(artifact_path.read_text(encoding="utf-8"))


def version_info() -> dict[str, str]:
    modules = {}
    for module_name in ["numpy", "pandas", "scipy", "sklearn", "pm4py", "pytest"]:
        try:
            module = __import__(module_name)
            modules[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - diagnostic only
            modules[module_name] = f"unavailable: {exc}"
    return {
        "python": platform.python_version(),
        **modules,
    }


def default_command(config: dict, results_dir: Path) -> str:
    output_dir = results_dir / "fixed_replay"
    seeds = ",".join(str(seed) for seed in config.get("seeds", []))
    strategies = ",".join(config.get("strategies", []))
    parksong_params = config.get("parksong_params", {})
    param_text = ",".join(f"{key}={value}" for key, value in parksong_params.items())
    parts = [
        "MPLCONFIGDIR=/tmp/matplotlib",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONPATH=joao:.",
        "python3 joao/scripts/resource_allocation/run_final_resource_allocation_evaluation.py",
        f"--data-path {config.get('data_path')}",
        f"--branching-artifact {config.get('branching_artifact')}",
        f"--processing-time-artifact {config.get('processing_time_artifact')}",
        f"--start {config.get('start')}",
        f"--end {config.get('end')}",
        f"--drain-until {config.get('drain_until')}",
        f"--seeds {seeds}",
        f"--strategies {strategies}",
        f"--route-mode {config.get('route_mode')}",
        f"--split-ratio {config.get('split_ratio')}",
        f"--parksong-processing-times {config.get('parksong_processing_times')}",
    ]
    if param_text:
        parts.append(f"--parksong-params {param_text}")
    parts.extend(
        [
            f"--reservation-expiration-multiplier {config.get('reservation_expiration_multiplier')}",
            f"--output-dir {output_dir}",
        ]
    )
    return " \\\n  ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update git metadata in an existing canonical reproducibility manifest."
    )
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    manifest_path = results_dir / "reproducibility_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    status = command_output(["git", "status", "--short"])
    config = package_config(results_dir)
    artifacts = package_artifacts(results_dir)
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["current_branch"] = command_output(["git", "branch", "--show-current"])
    manifest["current_head"] = command_output(["git", "rev-parse", "HEAD"])
    manifest["git_dirty"] = bool(status)
    manifest["git_status_short"] = status.splitlines()
    manifest["results_dir"] = str(results_dir)
    manifest["command"] = manifest.get("command") or default_command(config, results_dir)
    manifest["config"] = config
    manifest["artifacts"] = artifacts
    manifest["file_hashes"] = package_hashes(results_dir)
    manifest["versions"] = version_info()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    md_path = results_dir / "reproducibility.md"
    lines = [
        "# Part II A Fixed-Replay Reproducibility",
        "",
        f"Generated: {manifest['generated_at']}",
        f"Branch: `{manifest['current_branch']}`",
        f"HEAD: `{manifest['current_head']}`",
        f"Git dirty: `{manifest['git_dirty']}`",
        "",
        "This package was generated before the final commit. After committing, run:",
        "",
        "```bash",
        f"python3 joao/scripts/resource_allocation/write_reproducibility_manifest.py --results-dir {results_dir}",
        "```",
        "",
        "That refresh updates only git metadata unless result files changed.",
        "",
        "## Command",
        "",
        "```bash",
        manifest["command"],
        "```",
        "",
        "## Artifact Hashes",
        "",
    ]
    for key, value in artifacts.items():
        if key.endswith("_sha256"):
            lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## File Hashes", ""])
    for relative, digest in manifest["file_hashes"].items():
        lines.append(f"- `{relative}`: `{digest}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
