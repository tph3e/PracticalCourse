# 1.2 (Part 2) F bridge — cross-method allocation comparison on the INTEGRATED simulator.

# Runs the team simulation core once per allocation strategy on an identical seed and
# window, so only the 1.8 pick() strategy differs (arrivals, case data and processing
# times are the same stream across methods). Each per-method output log is scored with
# the same optimization.metrics.compare_on_sim, giving the honest RL-vs-baseline numbers
# for the D evaluation (Aldo's section) and the compare_on_sim data F (K3) consumes.

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
SEED = int(os.environ.get("SIM_SEED", "1"))   # vary for multi-seed statistical comparison
# Monday 09:00 start so the working-hour calendars are active from the first arrival.
START = datetime(2000, 1, 3, 9, 0)
END = datetime(2000, 1, 10, 0, 0)          # one-week window
RL_POLICY = os.environ.get("RL_POLICY", "results/rl_policy.json")  # override to eval the 2C policy
PPO_MODEL = os.environ.get("PPO_MODEL")  # set (e.g. results/ppo_model) to add PPO (needs sb3 venv)


def _rl_label(path: str) -> str:
    # The row name must follow the policy that is actually loaded, otherwise the output CSV lies.
    if os.environ.get("RL_LABEL"):
        return os.environ["RL_LABEL"]
    base = os.path.basename(path)
    if base == "rl_policy_sim_ppo.json":
        return "PPO on simulator"
    if base == "rl_policy_sim.json":
        return "RL (sim-in-the-loop)"
    return "RL (REINFORCE)"


RL_LABEL = _rl_label(RL_POLICY)
OUT_CSV = "results/sim_comparison.csv"
# Load knob for congestion experiments: scales every interarrival gap (<1 = more load).
# 1.0 = real BPIC arrival rate. A core-level knob would be K1's; here a driver monkeypatch.
ARRIVAL_SCALE = float(os.environ.get("ARRIVAL_SCALE", "1.0"))

# TEMPORARY stand-in, NOT part of the deliverable. It compensates for two bugs in
# K1's ProcessTimeEngine (task 1.3) so the allocation comparison can run on the sim:
#   (1) UNITS: the basic path wraps fitted values as bare timedelta(x) = DAYS, but the
#       models are fit in SECONDS (the advanced path uses timedelta(seconds=...)). A
#       ~10 h duration comes out as ~105 years, so any case that hits a real model
#       schedules its END decades out and never completes in-window.
#   (2) MISS -> 0: an (activity, resource) pair never seen in the real log returns
#       timedelta(0). Under free Part-2 allocation ~98 % of pairs are unseen, so
#       durations collapse to 0 and every timestamp coincides.
# Here every (activity, kind) uses a seen per-activity model (representative resource)
# read in the correct unit. Set to False once K1 fixes the engine (timedelta(seconds=)
# + an activity-level fallback).
USE_ACTIVITY_FALLBACK = False
_KEY_RE = re.compile(r"^(?P<act>.+)_(?P<res>User_\d+)_(?P<kind>processing|waiting)$")
_MAX_DURATION_S = 24 * 3600.0            # clip pathological heavy-tail samples to 24 h


def install_activity_level_fallback(pte) -> int:
    # Replace sampleTime_basic with an activity-level, correct-unit sampler built only
    # from K1's own fitted distributions. Returns the number of activities covered.
    mb = pte.models_basic
    representative: dict[tuple[str, str], str] = {}
    for key in mb:
        m = _KEY_RE.match(key)
        if not m:
            continue
        ak = (m.group("act"), m.group("kind"))
        # deterministic representative: lexicographically smallest matching key
        if ak not in representative or key < representative[ak]:
            representative[ak] = key

    def sample_seconds(model):
        if np.random.rand() < model.get("0-proportion", 0.0):
            return timedelta(0)
        td = pte.sample_distrib(model["distribution"], model["parameters"])
        raw = td.total_seconds() / 86400.0        # undo the timedelta(days) misread -> fitted seconds
        return timedelta(seconds=min(raw, _MAX_DURATION_S))

    def patched(activity, resource="", kind="processing"):
        model = mb.get(f"{activity}_{resource}_{kind}") or (
            mb.get(representative.get((activity, kind))) if representative.get((activity, kind)) else None
        )
        return sample_seconds(model) if model is not None else timedelta(0)

    pte.sampleTime_basic = patched
    return len({a for a, _ in representative})


class ShortestQueue(AllocationStrategy):
    # Load-balancing baseline: pick the permitted-and-free resource with least load.
    def pick(self, candidates, context=None):
        if not candidates:
            return None
        load = context.load if context is not None else {}
        return min(sorted(candidates), key=lambda r: load.get(r, 0.0))


class MostExperienced(AllocationStrategy):
    # Skill baseline: pick the candidate that ran this activity most in the real log.
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
    if ARRIVAL_SCALE != 1.0:                        # congestion experiment (driver monkeypatch)
        _orig = eng.arrivalEngine.nextArrivalTime
        eng.arrivalEngine.nextArrivalTime = lambda ct: timedelta(
            seconds=_orig(ct).total_seconds() * ARRIVAL_SCALE)
    eng.resourceEngine.set_allocation(make_strategy(eng))
    eng.run(START, END, format_type=[])            # no file output, keep the log in memory
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

    cfg = build_env_config(load_slim_log(), faithful=False)   # only skill needed here
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
        from optimization.ppo_agent import PPOAllocation      # lazy: needs the sb3 venv
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
