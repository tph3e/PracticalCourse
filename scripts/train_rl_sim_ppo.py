# Middelhuis "PPO on the simulator", non-invasive: the recording strategy collects the trajectory
# during a normal Engine.run (no gym control-inversion), and a PPO-style update (critic + GAE +
# clipped surrogate, optimization/ppo_numpy.py) replaces the weak vanilla-REINFORCE update used in 2C.
# The policy is PRETRAINED from the (a) standalone policy (results/rl_policy.json, same PolicyNet).

from __future__ import annotations

import os
import sys
import json
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from SimulationEngineCore import Engine, EventLogger
from resources.log_loader import load_slim_log
from optimization.environment import build_env_config
from optimization.rl_agent import PolicyNet, RecordingRLAllocation, RLAllocation
from optimization.ppo_numpy import ValueNet, ppo_update

DATA_PATH = "data/BPI Challenge 2017.xes"
SEED = 1
START = datetime(2000, 1, 3, 9, 0)
END = datetime(2000, 1, 6, 0, 0)          # 3-day collection window
ITERS = 200                                # PPO iterations (1 sim episode each)
PRETRAIN = "results/rl_policy.json"        # (a) standalone policy -> warm start
OUT = "results/rl_policy_sim_ppo.json"


def reset_engine(eng):
    # External per-episode reset (same as train_rl_sim). case_markings.clear() is mandatory.
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
    cfg = build_env_config(load_slim_log(), faithful=False)
    calendars, skill = cfg["calendars"], cfg["skill"]
    max_cal = max((len(c) for c in calendars.values()), default=1)

    # pretrain: warm-start the policy from the (a) standalone net (same architecture)
    if os.path.exists(PRETRAIN):
        policy = PolicyNet.from_dict(json.load(open(PRETRAIN))["net"])
        print(f"pretrained policy from {PRETRAIN}")
    else:
        policy = PolicyNet(seed=SEED)
        print("no pretrain policy found, random init")
    value = ValueNet(seed=SEED)

    eng = Engine(DATA_PATH, seed=SEED)
    strat = RecordingRLAllocation(policy, calendars, skill, max_cal, np.random.default_rng(SEED))
    eng.resourceEngine.set_allocation(strat)

    history = []
    for it in range(ITERS):
        random.seed(10_000 + it)
        np.random.seed(10_000 + it)
        reset_engine(eng)
        strat.reset()
        eng.run(START, END, format_type=[])
        info = ppo_update(policy, value, strat.trajectory)
        history.append(sum(t[4] for t in strat.trajectory))
        if (it + 1) % 25 == 0:
            print(f"  it {it+1:4d} | decisions {len(strat.trajectory):5d} | "
                  f"avg return {np.mean(history[-25:]):.1f} | value_mse {info.get('value_mse', 0):.2f}")

    RLAllocation(policy, calendars, skill, max_cal).save(OUT)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
