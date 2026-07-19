# 1.1 advanced: standalone case-based allocation environment for RL.

# Compact event-driven model of the resource-allocation decision, aligned with Middelhuis
# et al. (2025). Built from resource artifacts (1.6 calendars, 1.7 roles) and per-activity
# BPIC-17 statistics, so the agent trains without the integrated team simulator.

from __future__ import annotations

import heapq
import json
from collections import deque

import numpy as np
import pandas as pd

from .metrics import gini as _gini

N_FEATURES = 5  # load_norm, experience, availability_breadth, busy_fraction, is_postpone
SECONDS_PER_WEEK = 7 * 24 * 3600
W_FAIR = 5.0  # weight balancing the fairness 


def build_features(candidates, activity, load, calendars, skill, max_cal, busy_fraction=0.0):
    # Feature matrix [n_candidates + 1, N_FEATURES]. Last row = postpone action.

    # Shared by the env (training) and RLAllocation (inference) so the policy sees identical
    # inputs.
    
    n = len(candidates)
    feats = np.zeros((n + 1, N_FEATURES))
    max_load = (max(load.values()) if load else 0.0) or 1.0
    for i, r in enumerate(candidates):
        feats[i, 0] = load.get(r, 0.0) / max_load
        feats[i, 1] = skill.get(activity, {}).get(r, 0.0)
        feats[i, 2] = len(calendars.get(r, ())) / max_cal
        feats[i, 3] = busy_fraction
    if n > 0:
        feats[n, 0] = feats[:n, 0].min()   # best (lowest) candidate load
        feats[n, 1] = feats[:n, 1].max()   # best candidate experience
        feats[n, 2] = feats[:n, 2].mean()
        feats[n, 3] = busy_fraction
    feats[n, 4] = 1.0                       # is_postpone flag
    return feats


# faithful dynamics helpers (2A/2B), shared by AllocationEnv and GymAllocationEnv

def sample_interarrival(arrival, rng, t, mean_interarrival_s, scale=1.0):
    # 2B: real ArrivalEngine gap if available, else the exponential fallback. scale = load knob.
    if arrival is not None:
        return arrival.nextArrivalTime(t).total_seconds() * scale
    return float(rng.exponential(mean_interarrival_s)) * scale


def sample_processing(proc, skill, base_proc_s, act, r):
    # 2B: real per-(activity, resource) processing time if the engine is available.
    if proc is not None:
        return max(0.0, proc.sampleTime_basic(act, r, "processing").total_seconds())
    return base_proc_s[act] * (1.0 - 0.3 * skill.get(act, {}).get(r, 0.0))


def sample_waiting(proc, act, r):
    # 2B: real inter-activity waiting time (0 if no engine -> immediate readiness).
    if proc is not None:
        return max(0.0, proc.sampleTime_basic(act, r, "waiting").total_seconds())
    return 0.0


def build_case_pool(n=2000, seed=0, cap=200):
    # 2A: pre-generate valid linear activity sequences via the SAME mechanism the eval
    # simulator uses (BPMN v4 model + CompositeBranchingEngine). Parallel branches are linearised.
    from datetime import datetime
    import random as _random
    from BPMN_engine import BPMNEngine
    from joao.src.branching.CompositeBranchingEngine import CompositeBranchingEngine as BranchingEngine
    from Helper import Event, Case, EventType

    bpmn = BPMNEngine()
    branch = BranchingEngine()
    rng = _random.Random(seed)
    t0 = datetime(2000, 1, 3, 9, 0)

    pool = []
    for i in range(n):
        cid = f"pool{i}"
        bpmn.initialize_case(cid)
        current = bpmn.getStartActivity({})
        if current is None:
            continue
        seq = [current]
        for _ in range(cap):
            bpmn.fire_activity(current, cid)
            cands = bpmn.getPossibleNextActivities(current, case_id=cid)
            if not cands:
                break
            ev = Event(EventType.ACTIVITY_START, current, t0, 0, Case(cid), None)
            try:
                nxt = branch.getNextActivities(ev, cands)
            except Exception:
                nxt = cands
            if not nxt:
                break
            current = rng.choice(nxt if isinstance(nxt, list) else [nxt])
            seq.append(current)
        if len(seq) >= 2:
            pool.append(seq)
    return pool


class AllocationEnv:
    def __init__(
        self,
        permitted: dict,
        calendars: dict,
        skill: dict,
        base_proc_s: dict,
        activity_mix: dict,
        arrival=None,           # ArrivalEngine 
        proc=None,              # ProcessTimeEngine 
        case_pool=None,         # pool of valid BPMN case sequences
        seed: int = 0,
        max_steps: int = 600,
        mean_interarrival_s: float = 1000.0,   # calibrated to real BPIC-17 (1000 s)
        min_case_len: int = 8,                  
        max_case_len: int = 21,
        max_postpone: int = 3,
        ct_scale_s: float = 3600.0,
        interarrival_scale: float = 1.0,        
        progress_reward: float = 1.0,           
        w_fair: float = 1.0,                    # dense scale-invariant fairness weight
        w_ct: float = 1.0,                      # cycle-time weight
    ):
        self.permitted = {a: list(rs) for a, rs in permitted.items()}
        self.calendars = calendars
        self.skill = skill
        self.base_proc_s = base_proc_s
        self.arrival = arrival
        self.proc = proc
        self.case_pool = case_pool
        self.activities = list(activity_mix.keys())
        self.activity_p = np.array([activity_mix[a] for a in self.activities], float)
        self.activity_p /= self.activity_p.sum()
        self.resources = sorted(calendars.keys())
        self.max_cal = max((len(c) for c in calendars.values()), default=1)
        self._seed = seed
        self.max_steps = max_steps
        self.mean_interarrival_s = mean_interarrival_s
        self.min_case_len = min_case_len
        self.max_case_len = max_case_len
        self.max_postpone = max_postpone
        self.ct_scale_s = ct_scale_s
        self.interarrival_scale = interarrival_scale
        self.progress_reward = progress_reward
        self.w_fair = w_fair
        self.w_ct = w_ct

    def reset(self):
        self.rng = np.random.default_rng(self._seed)
        self.t = 0.0
        self.step_i = 0
        self._seq = 0
        self.evq = []
        self.busy_until = {}
        self.load = {r: 0.0 for r in self.resources}        # busy seconds (reward/eval objective)
        self.alloc_count = {r: 0 for r in self.resources}   # allocation count = inference-side load signal (ResourceEngine.load)
        self.ready = deque()
        self.cases = {}
        self.case_counter = 0
        self.completed_cts = []
        self._pending_reward = 0.0
        self._schedule_arrival(0.0)
        return self._advance_to_decision()

    def _push(self, time, kind, payload):
        heapq.heappush(self.evq, (time, self._seq, kind, payload))
        self._seq += 1

    def _schedule_arrival(self, t):
        self._push(t, "ARRIVAL", None)

    def _sample_case(self):
        if self.case_pool:                                  # 2A: valid BPMN sequence
            return self.case_pool[int(self.rng.integers(len(self.case_pool)))]
        # fallback: iid activity bag (surrogate behaviour)
        n = int(self.rng.integers(self.min_case_len, self.max_case_len + 1))
        idx = self.rng.choice(len(self.activities), size=n, p=self.activity_p)
        return [self.activities[i] for i in idx]

    def _available_now(self, r):
        hours = self.t / 3600.0
        weekday = int((hours // 24) % 7)
        hour = int(hours % 24)
        cal = self.calendars.get(r)
        return cal is None or (weekday, hour) in cal

    def _candidates(self, activity):
        return [
            r for r in self.permitted.get(activity, ())
            if self.busy_until.get(r, 0.0) <= self.t and self._available_now(r)
        ]

    def _process_event(self):
        t, _, kind, payload = heapq.heappop(self.evq)
        self.t = t
        if kind == "ARRIVAL":
            cid = self.case_counter
            self.case_counter += 1
            self.cases[cid] = {"acts": self._sample_case(), "idx": 0, "start": t, "postpone": 0}
            self.ready.append(cid)
            gap = sample_interarrival(self.arrival, self.rng, t,
                                      self.mean_interarrival_s, self.interarrival_scale)
            self._schedule_arrival(t + gap)
        elif kind == "COMPLETE":
            cid, r = payload
            case = self.cases.get(cid)
            if case is None:
                return
            completed_act = case["acts"][case["idx"]]
            case["idx"] += 1
            if case["idx"] >= len(case["acts"]):
                ct = t - case["start"]
                self.completed_cts.append(ct)
                self._pending_reward += self.w_ct * 1.0 / (ct / self.ct_scale_s + 1.0)  # inverse cycle time
                del self.cases[cid]
            else:
                case["postpone"] = 0
                wait = sample_waiting(self.proc, completed_act, r)   # 2B: inter-activity waiting
                if wait > 0:
                    self._push(t + wait, "READY", cid)
                else:
                    self.ready.append(cid)
        elif kind == "READY":
            cid = payload
            if cid in self.cases:
                self.ready.append(cid)

    def _current_ready(self):
        for cid in list(self.ready):
            act = self.cases[cid]["acts"][self.cases[cid]["idx"]]
            cands = self._candidates(act)
            if cands:
                return cid, act, cands
        return None

    def _advance_to_decision(self):
        while self.step_i < self.max_steps:
            cur = self._current_ready()
            if cur is not None:
                self._cur = cur
                cid, act, cands = cur
                busy = sum(1 for r in self.resources if self.busy_until.get(r, 0.0) > self.t)
                # (a) feature uses allocation count (matches ResourceEngine.load at inference)
                return build_features(cands, act, self.alloc_count, self.calendars, self.skill,
                                      self.max_cal, busy / len(self.resources))
            if not self.evq:
                return None
            self._process_event()
        return None

    def step(self, action: int):
        cid, act, cands = self._cur
        n = len(cands)
        self._pending_reward = 0.0

        if action == n:  # postpone
            self.cases[cid]["postpone"] += 1
            if self.cases[cid]["postpone"] > self.max_postpone:
                action = int(np.argmin([self.load[r] for r in cands]))  # forced assignment
            else:
                self.ready.remove(cid)
                self.ready.append(cid)              # rotate to back
                self._pending_reward -= 0.5          # strong cost: postpone must not dodge the overload penalty (collapsed training)
                if self.evq:
                    self._process_event()            # advance time so state changes
                self.step_i += 1
                obs = self._advance_to_decision()
                return obs, self._pending_reward, obs is None, {"postpone": True}

        r = cands[action]
        proc = sample_processing(self.proc, self.skill, self.base_proc_s, act, r)  # 2B
        gini_before = _gini(self.load.values())
        self.busy_until[r] = self.t + proc
        self.load[r] += proc
        self.alloc_count[r] += 1                      # (a) count = inference-side load signal
        self.ready.remove(cid)                       # in progress
        self._push(self.t + proc, "COMPLETE", (cid, r))

        # Telescoping fairness potential on the load Gini, for the 600-step standalone scale.
        # The dense/scale-invariant variant is applied in 2C where episodes are ~40k steps.
        self._pending_reward += W_FAIR * (gini_before - _gini(self.load.values()))
        self.step_i += 1
        obs = self._advance_to_decision()
        return obs, self._pending_reward, obs is None, {"resource": r, "activity": act}


def build_env_config(
    slim_log,
    permissions_path: str = "results/permissions_roles.json",
    calendars_path: str = "results/availability_calendars.json",
    case_pool_n: int = 2000,
    pool_seed: int = 0,
    faithful: bool = True,
):
    from .metrics import activity_durations

    permitted = json.load(open(permissions_path))
    raw_cal = json.load(open(calendars_path))
    calendars = {r: {(int(wd), int(h)) for wd, h in b} for r, b in raw_cal.items()}

    dur = activity_durations(slim_log)
    med = dur.groupby("concept:name")["duration_s"].median()
    base_proc_s = med.clip(lower=300.0, upper=4 * 3600.0).to_dict()
    DEFAULT = 1200.0

    ct = pd.crosstab(slim_log["concept:name"].astype(str), slim_log["org:resource"].astype(str))
    skill_df = ct.div(ct.max(axis=1).replace(0, 1), axis=0)
    skill = {a: skill_df.loc[a][skill_df.loc[a] > 0].to_dict() for a in skill_df.index}

    mix = slim_log["concept:name"].astype(str).value_counts(normalize=True).to_dict()
    activity_mix = {a: p for a, p in mix.items() if a in permitted}
    base_proc_s = {a: base_proc_s.get(a, DEFAULT) for a in activity_mix}

    cfg = dict(
        permitted=permitted,
        calendars=calendars,
        skill=skill,
        base_proc_s=base_proc_s,
        activity_mix=activity_mix,
    )

    # 2A/2B: real process dynamics. Off (faithful=False) restores the surrogate.
    if faithful:
        from arrival_engine import ArrivalEngine
        from processTimes import ProcessTimeEngine
        cfg["arrival"] = ArrivalEngine(slim_log)          # 2B interarrival
        cfg["proc"] = ProcessTimeEngine()                 # 2B processing/waiting (loads pkl)
        cfg["case_pool"] = build_case_pool(case_pool_n, pool_seed)  # 2A valid sequences

    return cfg
