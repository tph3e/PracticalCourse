# Task D/E: reproducible multi-seed integrated-simulator sweep.
# Runs scripts/compare_on_sim.py once per seed (MaskablePPO included) and writes
# results/ms_seed{s}.csv. Labels come from compare_on_sim, not a post-hoc notebook rename.
# Also records the protocol to results/ms_seed_config.json for reproducibility.
#
# Run with the sb3 venv, from the repo root:
#   PYTHONPATH=. .venv-ppo/bin/python scripts/run_sim_sweep.py
# Env knobs: SEEDS="1 2 3 4 5", ARRIVAL_SCALE (default 0.5 = congested), PPO_MODEL, RL_POLICY.

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPARE = os.path.join(REPO, "scripts", "compare_on_sim.py")
SIM_CSV = os.path.join(REPO, "results", "sim_comparison.csv")

SEEDS = [int(s) for s in os.environ.get("SEEDS", "1 2 3 4 5").split()]
ARRIVAL_SCALE = os.environ.get("ARRIVAL_SCALE", "0.5")            # <1 = congested load
PPO_MODEL = os.environ.get("PPO_MODEL", "results/ppo_model")      # include the MaskablePPO row
RL_POLICY = os.environ.get("RL_POLICY", "results/rl_policy.json")  # numpy REINFORCE row (honest label)
START = "2000-01-03 09:00"
END = "2000-01-10 00:00"


def main():
    os.makedirs(os.path.join(REPO, "results"), exist_ok=True)
    for s in SEEDS:
        env = dict(os.environ)
        env["SIM_SEED"] = str(s)
        env["ARRIVAL_SCALE"] = ARRIVAL_SCALE
        env["PPO_MODEL"] = PPO_MODEL
        env["RL_POLICY"] = RL_POLICY
        env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
        print(f"=== seed {s} (ARRIVAL_SCALE={ARRIVAL_SCALE}) ===", flush=True)
        subprocess.run([sys.executable, COMPARE], cwd=REPO, env=env, check=True)
        dst = os.path.join(REPO, "results", f"ms_seed{s}.csv")
        shutil.copyfile(SIM_CSV, dst)
        print(f"saved -> results/ms_seed{s}.csv", flush=True)

    cfg = {
        "seeds": SEEDS,
        "arrival_scale": float(ARRIVAL_SCALE),
        "start": START,
        "end": END,
        "ppo_model": PPO_MODEL,
        "rl_policy": RL_POLICY,
        "note": "congested load = interarrival gaps scaled by arrival_scale (<1 = more load)",
    }
    with open(os.path.join(REPO, "results", "ms_seed_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print("saved -> results/ms_seed_config.json")


if __name__ == "__main__":
    main()
