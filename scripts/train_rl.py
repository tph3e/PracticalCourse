# 1.1 (Part 2) advanced — train the RL allocation policy (offline) and compare.

# Reproducible offline step: builds the standalone case-based environment from the
# resource artifacts + real log, trains the REINFORCE policy (with a postpone
# action), evaluates it against random, round-robin, shortest-queue

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from resources.log_loader import load_slim_log
from optimization.environment import AllocationEnv, build_env_config
from optimization.rl_agent import train, RLAllocation
from optimization.metrics import gini

MAX_STEPS = 600
EPISODES = 500


def make_policies(net, rng):
    rr = {"c": 0}
    return {
        "random":           lambda o: int(rng.integers(o.shape[0] - 1)),
        "round-robin":      lambda o: (rr.__setitem__("c", rr["c"] + 1) or (rr["c"] - 1) % (o.shape[0] - 1)),
        "shortest-queue":   lambda o: int(np.argmin(o[:o.shape[0] - 1, 0])),   # least-load proxy
        "most-experienced": lambda o: int(np.argmax(o[:o.shape[0] - 1, 1])),   # ablation
        "RL (REINFORCE)":   lambda o: int(np.argmax(net.forward(o)[0])),       # may postpone
    }


def evaluate(cfg, policy, episodes=25, seed0=5000):
    cts, ginis, rets = [], [], []
    for ep in range(episodes):
        env = AllocationEnv(**cfg, seed=seed0 + ep, max_steps=MAX_STEPS)
        obs = env.reset()
        total = 0.0
        while obs is not None:
            obs, r, done, _ = env.step(policy(obs))
            total += r
            if done:
                break
        if env.completed_cts:
            cts.append(np.mean(env.completed_cts) / 3600.0)  # hours
        ginis.append(gini([v for v in env.load.values() if v > 0]))
        rets.append(total)
    return np.mean(cts), np.mean(ginis), np.mean(rets)


def main():
    cfg = build_env_config(load_slim_log())
    print(f"env: {len(cfg['activity_mix'])} activities, {len(cfg['calendars'])} resources")

    net, _ = train(cfg, episodes=EPISODES, hidden=24, lr=0.05,
                   ep_len=MAX_STEPS, seed=1, entropy_beta=0.01, log_every=100)

    rng = np.random.default_rng(7)
    rows = []
    for name, pol in make_policies(net, rng).items():
        ct, g, ret = evaluate(cfg, pol)
        rows.append({"method": name, "cycle_time_h": round(ct, 2), "load_gini": round(g, 3)})
    df = pd.DataFrame(rows)
    # disclosed, equal-weight multi-objective score (both normalized by their max
    # All methods scored identically on the SAME objective.
    df["combined"] = (0.5 * df.cycle_time_h / df.cycle_time_h.max()
                      + 0.5 * df.load_gini / df.load_gini.max()).round(3)
    print(df.to_string(index=False))
    df.to_csv("results/allocation_comparison.csv", index=False)
    max_cal = max(len(c) for c in cfg["calendars"].values())
    RLAllocation(net, cfg["calendars"], cfg["skill"], max_cal).save("results/rl_policy.json")
    print("saved -> results/rl_policy.json, results/allocation_comparison.csv")


if __name__ == "__main__":
    main()
