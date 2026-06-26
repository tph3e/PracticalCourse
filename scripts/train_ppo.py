# 1.1 (Part 2) advanced — PPO allocation policy (Middelhuis et al. (2025) Paper, isolated venv).

# This is the state-of-the-art variant: a gymnasium environment with a fixed action
# space over all resources + a postpone action, infeasible actions removed by an
# action mask, trained with MaskablePPO (sb3-contrib).

from __future__ import annotations

import heapq
import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from resources.log_loader import load_slim_log
from optimization.environment import build_env_config
from optimization.metrics import gini


W_FAIR = 5.0  # weight balancing the (telescoped) fairness objective vs. the CT reward


class GymAllocationEnv(gym.Env):
    def __init__(self, cfg, max_steps=600, mean_interarrival_s=1000.0,  # calibrated to real ~1002 s
                 min_case_len=8, max_case_len=21, max_postpone=3, ct_scale_s=3600.0, seed=0):
        super().__init__()
        self.permitted = {a: list(rs) for a, rs in cfg["permitted"].items()}
        self.calendars = cfg["calendars"]
        self.skill = cfg["skill"]
        self.base_proc_s = cfg["base_proc_s"]
        self.activities = list(cfg["activity_mix"].keys())
        self.act_index = {a: i for i, a in enumerate(self.activities)}
        self.activity_p = np.array([cfg["activity_mix"][a] for a in self.activities], float)
        self.activity_p /= self.activity_p.sum()
        self.resources = sorted(self.calendars.keys())
        self.res_index = {r: i for i, r in enumerate(self.resources)}
        self.R = len(self.resources)
        self.A = len(self.activities)
        self.max_cal = max((len(c) for c in self.calendars.values()), default=1)
        self.max_steps = max_steps
        self.mean_interarrival_s = mean_interarrival_s
        self.min_case_len = min_case_len
        self.max_case_len = max_case_len
        self.max_postpone = max_postpone
        self.ct_scale_s = ct_scale_s
        self._seed = seed

        self.action_space = spaces.Discrete(self.R + 1)              # resource i, or postpone (=R)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(3 * self.R + self.A + 1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(self._seed if seed is None else seed)
        self.t = 0.0; self.step_i = 0; self._seq = 0
        self.evq = []; self.busy_until = {}; self.load = {r: 0.0 for r in self.resources}
        self.ready = deque(); self.cases = {}; self.case_counter = 0
        self.completed_cts = []
        self._push(0.0, "ARRIVAL", None)
        self._advance()
        return self._obs(), {}

    def _push(self, t, kind, payload):
        heapq.heappush(self.evq, (t, self._seq, kind, payload)); self._seq += 1

    def _available_now(self, r):
        h = self.t / 3600.0
        cal = self.calendars.get(r)
        return cal is None or (int((h // 24) % 7), int(h % 24)) in cal

    def _candidates(self, activity):
        return [r for r in self.permitted.get(activity, ())
                if self.busy_until.get(r, 0.0) <= self.t and self._available_now(r)]

    def _process_event(self):
        t, _, kind, payload = heapq.heappop(self.evq)
        self.t = t
        if kind == "ARRIVAL":
            cid = self.case_counter; self.case_counter += 1
            n = int(self.rng.integers(self.min_case_len, self.max_case_len + 1))
            acts = [self.activities[i] for i in self.rng.choice(self.A, size=n, p=self.activity_p)]
            self.cases[cid] = {"acts": acts, "idx": 0, "start": t, "postpone": 0}
            self.ready.append(cid)
            self._push(t + self.rng.exponential(self.mean_interarrival_s), "ARRIVAL", None)
        elif kind == "COMPLETE":
            cid, r = payload
            case = self.cases.get(cid)
            if case is None:
                return
            case["idx"] += 1
            if case["idx"] >= len(case["acts"]):
                ct = t - case["start"]; self.completed_cts.append(ct)
                self._pending += 1.0 / (ct / self.ct_scale_s + 1.0)
                del self.cases[cid]
            else:
                case["postpone"] = 0; self.ready.append(cid)

    def _current(self):
        for cid in list(self.ready):
            act = self.cases[cid]["acts"][self.cases[cid]["idx"]]
            cands = self._candidates(act)
            if cands:
                return cid, act, cands
        return None

    def _advance(self):
        self._cur = None
        while self.step_i < self.max_steps:
            cur = self._current()
            if cur is not None:
                self._cur = cur; return
            if not self.evq:
                return
            self._process_event()

    def _obs(self):
        obs = np.zeros(3 * self.R + self.A + 1, dtype=np.float32)
        if self._cur is None:
            return obs
        cid, act, cands = self._cur
        max_load = (max(self.load.values()) if self.load else 0.0) or 1.0
        skill_a = self.skill.get(act, {})
        for r in self.resources:
            i = self.res_index[r]
            obs[i] = self.load[r] / max_load
            obs[self.R + i] = 1.0 if self._available_now(r) else 0.0
            obs[2 * self.R + i] = skill_a.get(r, 0.0)
        obs[3 * self.R + self.act_index[act]] = 1.0
        busy = sum(1 for r in self.resources if self.busy_until.get(r, 0.0) > self.t)
        obs[3 * self.R + self.A] = busy / self.R          # congestion (busy fraction)
        return obs

    def action_masks(self):
        mask = np.zeros(self.R + 1, dtype=bool)
        mask[self.R] = True  # postpone always feasible
        if self._cur is not None:
            for r in self._cur[2]:
                mask[self.res_index[r]] = True
        return mask

    def step(self, action):
        self._pending = 0.0
        cid, act, cands = self._cur
        if action == self.R:  # postpone
            self.cases[cid]["postpone"] += 1
            if self.cases[cid]["postpone"] > self.max_postpone:
                action = self.res_index[min(cands, key=lambda r: self.load[r])]
            else:
                self.ready.remove(cid); self.ready.append(cid)
                self._pending -= 0.5  # strong cost: postpone is not a way to dodge the fairness penalty
                if self.evq:
                    self._process_event()
                self.step_i += 1
                self._advance()
                done = self._cur is None
                return self._obs(), self._pending, done, False, {}
        r = self.resources[action]
        skill_val = self.skill.get(act, {}).get(r, 0.0)
        proc = self.base_proc_s[act] * (1.0 - 0.3 * skill_val)
        gini_before = gini(self.load.values())
        self.busy_until[r] = self.t + proc; self.load[r] += proc
        self.ready.remove(cid)
        self._push(self.t + proc, "COMPLETE", (cid, r))
        # faithful fairness shaping (potential-based on Gini) 
        # CT objective via the per-case completion reward. See environment.py for the rationale.
        self._pending += W_FAIR * (gini_before - gini(self.load.values()))
        self.step_i += 1
        self._advance()
        done = self._cur is None
        return self._obs(), self._pending, done, False, {}


def evaluate(make_env, select, episodes=25, seed0=5000):
    cts, ginis = [], []
    for ep in range(episodes):
        env = make_env(seed0 + ep)
        obs, _ = env.reset(seed=seed0 + ep)
        while env._cur is not None:
            a = select(env, obs)
            obs, r, done, trunc, _ = env.step(a)
            if done:
                break
        if env.completed_cts:
            cts.append(np.mean(env.completed_cts) / 3600.0)
        ginis.append(gini([v for v in env.load.values() if v > 0]))
    return float(np.mean(cts)), float(np.mean(ginis))


def main():
    cfg = build_env_config(load_slim_log())
    make = lambda s: GymAllocationEnv(cfg, seed=s)
    env = make(0)
    print(f"PPO env: {env.R} resources, {env.A} activities, obs dim {env.observation_space.shape[0]}")

    model = MaskablePPO("MlpPolicy", env, n_steps=2048, batch_size=256, gamma=0.99,
                        policy_kwargs=dict(net_arch=[64, 64]), seed=1, verbose=0)
    model.learn(total_timesteps=600_000)
    model.save("results/ppo_model")

    rng = np.random.default_rng(7)
    rr = {"c": 0}

    def cand_idx(env):
        return [env.res_index[r] for r in env._cur[2]]

    def sel_random(env, obs):
        return int(rng.choice(cand_idx(env)))

    def sel_rr(env, obs):
        ci = cand_idx(env); a = ci[rr["c"] % len(ci)]; rr["c"] += 1; return a

    def sel_sq(env, obs):
        ci = env._cur[2]; return env.res_index[min(ci, key=lambda r: env.load[r])]

    def sel_exp(env, obs):
        ci = env._cur[2]; act = env._cur[1]
        return env.res_index[max(ci, key=lambda r: env.skill.get(act, {}).get(r, 0.0))]

    def sel_ppo(env, obs):
        a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        return int(a)

    import pandas as pd
    rows = []
    for name, sel in [("random", sel_random), ("round-robin", sel_rr),
                      ("shortest-queue", sel_sq), ("most-experienced", sel_exp),
                      ("PPO (MaskablePPO)", sel_ppo)]:
        ct, g = evaluate(make, sel)
        rows.append({"method": name, "cycle_time_h": round(ct, 2), "load_gini": round(g, 3)})
    df = pd.DataFrame(rows)
    # disclosed, equal-weight multi-objective score (same objective for all methods)
    df["combined"] = (0.5 * df.cycle_time_h / df.cycle_time_h.max()
                      + 0.5 * df.load_gini / df.load_gini.max()).round(3)
    print(df.to_string(index=False))
    df.to_csv("results/ppo_comparison.csv", index=False)
    print("saved -> results/ppo_comparison.csv, results/ppo_model.zip")


if __name__ == "__main__":
    main()
