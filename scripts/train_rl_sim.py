# 1.1 (Part 2) advanced — 2C: sim-in-the-loop REINFORCE.

# Trains the allocation policy DIRECTLY on the integrated SimulationEngineCore. A recording
# allocation strategy (injected via resourceEngine.set_allocation) collects the REINFORCE
# trajectory during each forward sim run, then reinforce_update does a gradient step. The policy
# therefore sees real concurrency + suspend/resume, unlike the linear standalone env.

# Non-invasive: no core changes. One Engine is built once and reused across episodes with an
# external per-run state reset. Reward is a shaped fairness signal (allocation-count Gini),
# NOT compute_all.

from __future__ import annotations

import os
import sys
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from SimulationEngineCore import Engine, EventLogger
from resources.log_loader import load_slim_log
from optimization.environment import build_env_config, N_FEATURES
from optimization.rl_agent import PolicyNet, RecordingRLAllocation, RLAllocation, reinforce_update

DATA_PATH = "data/BPI Challenge 2017.xes"
SEED = 1
START = datetime(2000, 1, 3, 9, 0)       # Monday 09:00
END = datetime(2000, 1, 6, 0, 0)         # 3-day training window (fast episodes)
EPISODES = 300
HIDDEN = 24
LR = 0.05
GAMMA = 0.99
ENTROPY = 0.01
PROGRESS = 1.0            # per-allocation throughput reward (allocate beats postpone)
OUT = "results/rl_policy_sim.json"


def reset_engine(eng):
    # External per-episode reset (verified). Clears every per-run mutable attribute, keeps the
    # trained sub-engines. case_markings.clear() is MANDATORY (else case-id/marking collisions
    # after case_counter is reset to 0).
    eng.event_counter = 0
    eng.case_counter = 0
    eng.event_queue = []
    eng.cases = []
    eng.waiting_processes = []
    eng.simulation_time = None
    eng.logger = EventLogger()
    eng.bpmnEngine.case_markings.clear()
    eng.resourceEngine.busy.clear()
    eng.resourceEngine.load.clear()


def main():
    # calendars + skill + max_cal for the policy features (same source RLAllocation uses).
    cfg = build_env_config(load_slim_log(), faithful=False)
    calendars = cfg["calendars"]
    skill = cfg["skill"]
    max_cal = max((len(c) for c in calendars.values()), default=1)

    eng = Engine(DATA_PATH, seed=SEED)

    net = PolicyNet(N_FEATURES, HIDDEN, SEED)
    strat = RecordingRLAllocation(net, calendars, skill, max_cal, np.random.default_rng(SEED),
                                  progress_reward=PROGRESS)
    eng.resourceEngine.set_allocation(strat)

    baseline = None
    history = []
    for ep in range(EPISODES):
        random.seed(10_000 + ep)          # per-episode diversity + reproducibility (sim RNGs)
        np.random.seed(10_000 + ep)
        reset_engine(eng)
        strat.reset()
        eng.run(START, END, format_type=[])
        traj = strat.trajectory
        baseline = reinforce_update(net, traj, baseline, LR, GAMMA, ENTROPY)
        history.append(sum(t[4] for t in traj))
        if (ep + 1) % 25 == 0:
            print(f"  ep {ep+1:4d} | decisions {len(traj):5d} | avg return {np.mean(history[-25:]):.2f}")

    RLAllocation(net, calendars, skill, max_cal).save(OUT)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
