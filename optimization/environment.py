# 1.1 advanced — standalone case-based allocation environment for RL.

# A compact event-driven environment that captures the resource-allocation decision
# with the mechanics that matter for the objective, aligned with Middelhuis et al.
# (2025)

# Built from the resource artifacts (1.6 calendars, 1.7 roles) and per-activity
# statistics from the real BPIC-17 log, so the agent can be trained without the
# integrated team simulator. 

from __future__ import annotations

import heapq
import json
from collections import deque

import numpy as np
import pandas as pd

from .metrics import gini as _gini

N_FEATURES = 5  # load_norm, experience, availability_breadth, busy_fraction, is_postpone
SECONDS_PER_WEEK = 7 * 24 * 3600
W_FAIR = 5.0  # weight balancing the fairness objective against the CT reward


def build_features(candidates, activity, load, calendars, skill, max_cal, busy_fraction=0.0):
    # Feature matrix [n_candidates + 1, N_FEATURES]; last row = postpone action.

    # Shared by the env (training) and RLAllocation (inference) so the policy sees
    # identical inputs. busy_fraction (share of resources currently occupied) is a
    # congestion signal, reconstructable at runtime from the AllocationContext. The
    # postpone row summarizes the best available alternative, so the agent can learn
    # to wait when candidates are poor or the system is congested.
    
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


class AllocationEnv:
    def __init__(
        self,
        permitted: dict,
        calendars: dict,
        skill: dict,
        base_proc_s: dict,
        activity_mix: dict,
        seed: int = 0,
        max_steps: int = 600,
        mean_interarrival_s: float = 1000.0,   # calibrated to real BPIC-17 (~1002 s)
        min_case_len: int = 8,                  # calibrated: real ~15 activities/case
        max_case_len: int = 21,
        max_postpone: int = 3,
        ct_scale_s: float = 3600.0,
    ):
        self.permitted = {a: list(rs) for a, rs in permitted.items()}
        self.calendars = calendars
        self.skill = skill
        self.base_proc_s = base_proc_s
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

    def reset(self):
        self.rng = np.random.default_rng(self._seed)
        self.t = 0.0
        self.step_i = 0
        self._seq = 0
        self.evq = []
        self.busy_until = {}
        self.load = {r: 0.0 for r in self.resources}
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
            self._schedule_arrival(t + self.rng.exponential(self.mean_interarrival_s))
        elif kind == "COMPLETE":
            cid, r = payload
            case = self.cases.get(cid)
            if case is None:
                return
            case["idx"] += 1
            if case["idx"] >= len(case["acts"]):
                ct = t - case["start"]
                self.completed_cts.append(ct)
                self._pending_reward += 1.0 / (ct / self.ct_scale_s + 1.0)  # inverse cycle time
                del self.cases[cid]
            else:
                case["postpone"] = 0
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
                return build_features(cands, act, self.load, self.calendars, self.skill,
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
                self._pending_reward -= 0.5          # strong cost: postpone must not be a way to dodge
                                                      # the overload penalty (that collapsed training)
                if self.evq:
                    self._process_event()            # advance time so state changes
                self.step_i += 1
                obs = self._advance_to_decision()
                return obs, self._pending_reward, obs is None, {"postpone": True}

        r = cands[action]
        skill_val = self.skill.get(act, {}).get(r, 0.0)
        proc = self.base_proc_s[act] * (1.0 - 0.3 * skill_val)
        gini_before = _gini(self.load.values())
        self.busy_until[r] = self.t + proc
        self.load[r] += proc
        self.ready.remove(cid)                       # in progress
        self._push(self.t + proc, "COMPLETE", (cid, r))

        # Faithful fairness shaping (potential-based on the Gini of loads). Summed
        # over an episode it telescopes to -W_FAIR * final_Gini, so it rewards
        # exactly the fairness metric we evaluate.

        self._pending_reward += W_FAIR * (gini_before - _gini(self.load.values()))
        self.step_i += 1
        obs = self._advance_to_decision()
        return obs, self._pending_reward, obs is None, {"resource": r, "activity": act}


def build_env_config(
    slim_log,
    permissions_path: str = "results/permissions_roles.json",
    calendars_path: str = "results/availability_calendars.json",
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

    return dict(
        permitted=permitted,
        calendars=calendars,
        skill=skill,
        base_proc_s=base_proc_s,
        activity_mix=activity_mix,
    )
