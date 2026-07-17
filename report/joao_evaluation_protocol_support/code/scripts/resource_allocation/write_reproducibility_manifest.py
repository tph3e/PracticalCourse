from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def command_output(args: list[str]) -> str:
    result = subprocess.run(args, check=False, text=True, capture_output=True)
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update git metadata in an existing canonical reproducibility manifest."
    )
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    manifest_path = results_dir / "reproducibility_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = command_output(["git", "status", "--short"])
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["current_branch"] = command_output(["git", "branch", "--show-current"])
    manifest["current_head"] = command_output(["git", "rev-parse", "HEAD"])
    manifest["git_dirty"] = bool(status)
    manifest["git_status_short"] = status.splitlines()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    md_path = results_dir / "reproducibility.md"
    if md_path.exists():
        lines = md_path.read_text(encoding="utf-8").splitlines()
        updated = []
        for line in lines:
            if line.startswith("Generated:"):
                updated.append(f"Generated: {manifest['generated_at']}")
            elif line.startswith("Branch:"):
                updated.append(f"Branch: `{manifest['current_branch']}`")
            elif line.startswith("HEAD:"):
                updated.append(f"HEAD: `{manifest['current_head']}`")
            elif line.startswith("Git dirty:"):
                updated.append(f"Git dirty: `{manifest['git_dirty']}`")
            else:
                updated.append(line)
        md_path.write_text("\n".join(updated) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
