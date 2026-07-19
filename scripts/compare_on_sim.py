# 1.2 (Part 2) F bridge: cross-method allocation comparison on the INTEGRATED simulator.
# Runs the sim core once per strategy on an identical seed and window so only pick() differs,
# then scores each log with optimization.metrics.compare_on_sim.

from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from SimulationEngineCore import Engine
from resources.allocation import AllocationStrategy, RandomAllocation
from resources.log_loader import load_slim_log
from optimization.environment import build_env_config
from optimization.metrics import compare_on_sim
from optimization.rl_agent import RLAllocation

DATA_PATH = "data/BPI Challenge 2017.xes"
SEED = int(os.environ.get("SIM_SEED", "1"))   
# Monday 09:00 start 
START = datetime(2000, 1, 3, 9, 0)
END = datetime(2000, 1, 10, 0, 0)          # one-week window
RL_POLICY = os.environ.get("RL_POLICY", "results/rl_policy.json")  
PPO_MODEL = os.environ.get("PPO_MODEL")  


def _rl_label(path: str) -> str:
    if os.environ.get("RL_LABEL"):
        return os.environ["RL_LABEL"]
    return "RL (REINFORCE)"


RL_LABEL = _rl_label(RL_POLICY)
OUT_CSV = "results/sim_comparison.csv"
# 1.0 = real BPIC arrival rate. 
ARRIVAL_SCALE = float(os.environ.get("ARRIVAL_SCALE", "1.0"))

# temporary stand-in, not part of the deliverable. Compensates for ProcessTimeEngine
# bugs (task 1.3) so the comparison can run
USE_ACTIVITY_FALLBACK = False
_KEY_RE = re.compile(r"^(?P<act>.+)_(?P<res>User_\d+)_(?P<kind>processing|waiting)$")
_MAX_DURATION_S = 24 * 3600.0            


def install_activity_level_fallback(pte) -> int:
    # Replace sampleTime_basic with an activity-level
    mb = pte.models_basic
    representative: dict[tuple[str, str], str] = {}
    for key in mb:
        m = _KEY_RE.match(key)
        if not m:
            continue
        ak = (m.group("act"), m.group("kind"))
        if ak not in representative or key < representative[ak]:
            representative[ak] = key

    def sample_seconds(model):
        if np.random.rand() < model.get("0-proportion", 0.0):
            return timedelta(0)
        td = pte.sample_distrib(model["distribution"], model["parameters"])
        raw = td.total_seconds() / 86400.0        
        return timedelta(seconds=min(raw, _MAX_DURATION_S))

    def patched(activity, resource="", kind="processing"):
        model = mb.get(f"{activity}_{resource}_{kind}") or (
            mb.get(representative.get((activity, kind))) if representative.get((activity, kind)) else None
        )
        return sample_seconds(model) if model is not None else timedelta(0)

    pte.sampleTime_basic = patched
    return len({a for a, _ in representative})


class ShortestQueue(AllocationStrategy):
    # Load-balancing baseline: pick resource with least load.
    def pick(self, candidates, context=None):
        if not candidates:
            return None
        load = context.load if context is not None else {}
        return min(sorted(candidates), key=lambda r: load.get(r, 0.0))


class MostExperienced(AllocationStrategy):
    # Skill baseline: pick candidate that ran activity most in the real log.
    def __init__(self, skill):
        self.skill = skill

    def pick(self, candidates, context=None):
        if not candidates:
            return None
        event = getattr(context, "event", None) if context is not None else None
        activity = getattr(event, "activity", None)
        sk = self.skill.get(activity, {})
        return max(sorted(candidates), key=lambda r: sk.get(r, 0.0))


def run_strategy(name, make_strategy):
    # Fresh engine per strategy: same seed -> identical arrivals/processing, only pick differs.
    eng = Engine(DATA_PATH, seed=SEED)
    if USE_ACTIVITY_FALLBACK:
        install_activity_level_fallback(eng.processTimeEngine)
    if ARRIVAL_SCALE != 1.0:                        
        _orig = eng.arrivalEngine.nextArrivalTime
        eng.arrivalEngine.nextArrivalTime = lambda ct: timedelta(
            seconds=_orig(ct).total_seconds() * ARRIVAL_SCALE)
    eng.resourceEngine.set_allocation(make_strategy(eng))
    eng.run(START, END, format_type=[])            
    log = eng.logger.get_log()
    lc = log["lifecycle:transition"].value_counts().to_dict() if not log.empty else {}
    n_cases = log["case:concept:name"].nunique() if not log.empty else 0
    print(f"  [{name}] {len(log)} events, {n_cases} cases, "
          f"start={lc.get('start', 0)} resume={lc.get('resume', 0)} "
          f"suspend={lc.get('suspend', 0)} complete={lc.get('complete', 0)}")
    return log, eng.resourceEngine.availability


def main():
    if USE_ACTIVITY_FALLBACK:
        print("NOTE: activity-level processing-time fallback ACTIVE (temporary stand-in "
              "for K1's ProcessTimeEngine; numbers are provisional).\n")

    cfg = build_env_config(load_slim_log(), faithful=False)   
    skill = cfg["skill"]
    if ARRIVAL_SCALE != 1.0:
        print(f"NOTE: ARRIVAL_SCALE={ARRIVAL_SCALE} (congestion experiment, driver monkeypatch).\n")

    strategies = {
        "random": lambda eng: RandomAllocation(SEED),
        "shortest-queue": lambda eng: ShortestQueue(),
        "most-experienced": lambda eng: MostExperienced(skill),
    }
    if os.path.exists(RL_POLICY):
        strategies[RL_LABEL] = lambda eng: RLAllocation.load(RL_POLICY)
    else:
        print(f"  (skipping RL: {RL_POLICY} not found)")
    if PPO_MODEL:
        from optimization.ppo_agent import PPOAllocation      
        strategies["PPO (MaskablePPO)"] = lambda eng: PPOAllocation.load(
            PPO_MODEL, cfg["calendars"], cfg["skill"], cfg["activity_mix"])

    sim_logs = {}
    availability = None
    for name, make in strategies.items():
        print(f"running {name} ...")
        log, availability = run_strategy(name, make)
        sim_logs[name] = log

    table = compare_on_sim(sim_logs, availability=availability)
    os.makedirs("results", exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    print()
    print(table.to_string(index=False))
    print(f"\nsaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
