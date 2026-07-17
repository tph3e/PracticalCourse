# 1.1 (Part 2) advanced: paper-faithful PPO allocation policy (Middelhuis et al. 2025), trained
# with MaskablePPO (sb3-contrib) on a discrete-event BPIC-17 simulation (state 2|R|+|A|, action
# |R|*|A|+1 with masking, cycle-time reward plus a potential-based load-Gini fairness shaping).
# Deployed via optimization.ppo_agent.PPOAllocation onto the integrated SimulationEngineCore.

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
from stable_baselines3.common.callbacks import BaseCallback

from resources.log_loader import load_slim_log
from optimization.environment import (
    build_env_config, sample_interarrival, sample_processing, sample_waiting,
)
from optimization.metrics import gini


class GymAllocationEnv(gym.Env):
    # Paper-faithful MDP (state 2R+A, action R*A+1 with masking, cycle-time reward).
    def __init__(self, cfg, max_steps=800, mean_interarrival_s=1000.0,  # calibrated to real ~1002 s
                 min_case_len=8, max_case_len=21, max_postpone=3, ct_scale_s=3600.0, seed=0,
                 interarrival_scale=1.0, w_fair=1.0):
        super().__init__()
        self.permitted = {a: list(rs) for a, rs in cfg["permitted"].items()}
        self.calendars = cfg["calendars"]
        self.skill = cfg["skill"]
        self.base_proc_s = cfg["base_proc_s"]
        self.arrival = cfg.get("arrival")          # 2B
        self.proc = cfg.get("proc")                # 2B
        self.case_pool = cfg.get("case_pool")      # 2A
        self.interarrival_scale = interarrival_scale
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
        self.w_fair = w_fair
        self._seed = seed

        # action = resource i -> activity j  (index i*A + j), or postpone (index R*A)
        self.action_space = spaces.Discrete(self.R * self.A + 1)
        # state = delta (R) | eta (R) | kappa (A)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(2 * self.R + self.A,), dtype=np.float32)

    # event-driven dynamics
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(self._seed if seed is None else seed)
        self.t = 0.0; self.step_i = 0; self._seq = 0
        self.evq = []; self.busy_until = {}; self.load = {r: 0.0 for r in self.resources}
        self.alloc_count = {r: 0 for r in self.resources}   # inference-side load signal (count)
        self.busy_activity = {}                             # resource -> activity it is executing (eta)
        self.ready = deque(); self.cases = {}; self.case_counter = 0
        self.completed_cts = []
        self.postpones = 0
        self._pending = 0.0
        self._decision = None
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
        # cycle-time reward: penalize time-in-system integrated over active cases (telescopes to -CT)
        self._pending += -(t - self.t) * len(self.cases) / self.ct_scale_s
        self.t = t
        if kind == "ARRIVAL":
            cid = self.case_counter; self.case_counter += 1
            if self.case_pool:                                  # 2A: valid BPMN sequence
                acts = self.case_pool[int(self.rng.integers(len(self.case_pool)))]
            else:
                n = int(self.rng.integers(self.min_case_len, self.max_case_len + 1))
                acts = [self.activities[i] for i in self.rng.choice(self.A, size=n, p=self.activity_p)]
            self.cases[cid] = {"acts": acts, "idx": 0, "start": t, "postpone": 0}
            self.ready.append(cid)
            gap = sample_interarrival(self.arrival, self.rng, t,
                                      self.mean_interarrival_s, self.interarrival_scale)
            self._push(t + gap, "ARRIVAL", None)
        elif kind == "COMPLETE":
            cid, r = payload
            self.busy_activity.pop(r, None)
            case = self.cases.get(cid)
            if case is None:
                return
            completed_act = case["acts"][case["idx"]]
            case["idx"] += 1
            if case["idx"] >= len(case["acts"]):
                ct = t - case["start"]; self.completed_cts.append(ct)
                del self.cases[cid]
            else:
                wait = sample_waiting(self.proc, completed_act, r)   # 2B
                if wait > 0:
                    self._push(t + wait, "READY", cid)
                else:
                    self.ready.append(cid)
        elif kind == "READY":
            cid = payload
            if cid in self.cases:
                self.ready.append(cid)

    def _current(self):
        # first ready-unassigned case that has at least one feasible resource (defines that a
        # decision exists). Also the FIFO task the heuristic baselines serve.
        for cid in list(self.ready):
            act = self.cases[cid]["acts"][self.cases[cid]["idx"]]
            cands = self._candidates(act)
            if cands:
                return cid, act, cands
        return None

    def _advance(self):
        # roll the simulation forward until at least one feasible (r,a) assignment exists.
        while self.step_i < self.max_steps:
            cur = self._current()
            if cur is not None:
                self._decision = cur
                return
            if not self.evq:
                self._decision = None
                return
            self._process_event()
        self._decision = None

    def _pick_ready_case(self, activity):
        # FIFO among ready cases whose current activity is `activity` (instances are interchangeable).
        for cid in self.ready:
            if self.cases[cid]["acts"][self.cases[cid]["idx"]] == activity:
                return cid
        return None

    def _assign(self, r, a, cid):
        proc = sample_processing(self.proc, self.skill, self.base_proc_s, a, r)  # 2B
        self.busy_until[r] = self.t + proc
        self.load[r] += proc
        self.alloc_count[r] += 1
        self.busy_activity[r] = a
        if cid in self.ready:
            self.ready.remove(cid)
        self._push(self.t + proc, "COMPLETE", (cid, r))

    def _episode_done(self):
        return self.step_i >= self.max_steps or (self._decision is None and not self.evq)

    # observation & mask
    def _obs(self):
        R, A = self.R, self.A
        obs = np.zeros(2 * R + A, dtype=np.float32)
        t = self.t
        for r in self.resources:
            i = self.res_index[r]
            if self.busy_until.get(r, 0.0) > t:               # delta: busy
                obs[i] = 1.0
                ba = self.busy_activity.get(r)                # eta: which activity
                if ba in self.act_index:
                    obs[R + i] = (self.act_index[ba] + 1) / A
        counts = {}                                           # kappa: unassigned queue per activity
        for cid in self.ready:
            a = self.cases[cid]["acts"][self.cases[cid]["idx"]]
            counts[a] = counts.get(a, 0) + 1
        for a, c in counts.items():
            j = self.act_index.get(a)
            if j is not None:
                obs[2 * R + j] = min(c / 100.0, 1.0)
        return obs

    def action_masks(self):
        R, A = self.R, self.A
        mask = np.zeros(R * A + 1, dtype=bool)
        mask[R * A] = True                                    # postpone always feasible
        ready_acts = set()
        for cid in self.ready:
            ready_acts.add(self.cases[cid]["acts"][self.cases[cid]["idx"]])
        for a in ready_acts:
            j = self.act_index.get(a)
            if j is None:
                continue
            for r in self._candidates(a):
                mask[self.res_index[r] * A + j] = True
        return mask

    # step
    def step(self, action):
        self._pending = 0.0
        R, A = self.R, self.A

        if action == R * A:                                   # postpone
            self.postpones += 1
            if self._decision is not None and self.postpones > self.max_postpone:
                cid, a, cands = self._decision                # safeguard: force least-loaded assignment
                r = min(cands, key=lambda r: self.load[r])
                self._assign(r, a, cid)
                self.postpones = 0
            elif self.evq:
                self._process_event()                         # advance time so the state changes
            self.step_i += 1
            self._advance()
            return self._obs(), self._pending, self._episode_done(), False, {"postpone": True}

        i, j = divmod(action, A)
        r = self.resources[i]; a = self.activities[j]
        cid = self._pick_ready_case(a)
        if cid is None or r not in self._candidates(a):       # infeasible (mask should prevent this)
            self.step_i += 1
            self._advance()
            return self._obs(), self._pending, self._episode_done(), False, {"noop": True}

        gini_before = gini(self.load.values())
        self._assign(r, a, cid)
        self.postpones = 0
        self._pending += self.w_fair * (gini_before - gini(self.load.values()))  # potential-based shaping
        self.step_i += 1
        self._advance()
        return self._obs(), self._pending, self._episode_done(), False, {"resource": r, "activity": a}


class CurveCallback(BaseCallback):
    # Record the true MaskablePPO episode-return curve during the training run.
    def __init__(self):
        super().__init__()
        self.ep_returns = []
        self._cur = 0.0

    def _on_step(self) -> bool:
        self._cur += float(self.locals["rewards"][0])
        if bool(self.locals["dones"][0]):
            self.ep_returns.append(self._cur)
            self._cur = 0.0
        return True


def _feasible_pairs_present(env):
    return env._decision is not None


def evaluate(make_env, select, episodes=25, seed0=5000):
    cts, ginis = [], []
    for ep in range(episodes):
        env = make_env(seed0 + ep)
        obs, _ = env.reset(seed=seed0 + ep)
        while not env._episode_done():
            a = select(env, obs)
            obs, r, done, trunc, _ = env.step(a)
            if done:
                break
        if env.completed_cts:
            cts.append(np.mean(env.completed_cts) / 3600.0)
        ginis.append(gini([v for v in env.load.values() if v > 0]))
    return float(np.mean(cts)) if cts else float("nan"), float(np.mean(ginis))


def _build_cfg():
    # Calibrated DES config: prefer the faithful engines, but if the committed ProcessTimeEngine
    # pickle cannot be unpickled here (fitted with an older scikit-learn) fall back to the arrival
    # engine plus per-activity median processing and BPMN-valid case sequences.
    slim = load_slim_log()
    try:
        return build_env_config(slim)                      # faithful=True
    except Exception as e:
        print(f"[warn] faithful ProcessTimeEngine unavailable ({e.__class__.__name__}); "
              f"using arrival engine + per-activity median processing + BPMN case pool.")
        from arrival_engine import ArrivalEngine
        from optimization.environment import build_case_pool
        cfg = build_env_config(slim, faithful=False)
        cfg["arrival"] = ArrivalEngine(slim)               # real interarrival distribution
        cfg["case_pool"] = build_case_pool(2000, 0)        # BPMN-valid case sequences
        return cfg


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    timesteps = int(os.environ.get("PPO_TIMESTEPS", "600000"))

    cfg = _build_cfg()
    make = lambda s: GymAllocationEnv(cfg, seed=s)
    env = make(0)
    print(f"PPO env (paper MDP): {env.R} resources, {env.A} activities, "
          f"obs dim {env.observation_space.shape[0]} (=2R+A), "
          f"action dim {env.action_space.n} (=R*A+1)")

    model = MaskablePPO("MlpPolicy", env, n_steps=2048, batch_size=256, gamma=0.99,
                        policy_kwargs=dict(net_arch=[128, 128]), seed=1, verbose=0)
    curve = CurveCallback()
    model.learn(total_timesteps=timesteps, callback=curve)
    os.makedirs("results", exist_ok=True)
    model.save("results/ppo_model")

    # learning curve from the real run
    if curve.ep_returns:
        h = np.array(curve.ep_returns)
        w = max(1, min(20, len(h) // 5))
        sm = np.convolve(h, np.ones(w) / w, mode="valid")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(h, alpha=0.3, label="episode return")
        ax.plot(range(len(sm)), sm, label=f"{w}-episode moving avg")
        ax.set_xlabel("episode"); ax.set_ylabel("return (cycle-time reward)")
        ax.set_title("MaskablePPO learning curve"); ax.legend()
        fig.tight_layout(); fig.savefig("results/rl_learning_curve.png", dpi=120)
        print(f"saved -> results/rl_learning_curve.png ({len(h)} episodes)")

    rng = np.random.default_rng(7)
    rr = {"c": 0}

    def _decision(env):
        return env._decision  # (cid, activity, candidates)

    def _act(env, r, a):
        return env.res_index[r] * env.A + env.act_index[a]

    def sel_random(env, obs):
        cid, a, cands = _decision(env)
        return _act(env, cands[int(rng.integers(len(cands)))], a)

    def sel_rr(env, obs):
        cid, a, cands = _decision(env)
        cs = sorted(cands); r = cs[rr["c"] % len(cs)]; rr["c"] += 1
        return _act(env, r, a)

    def sel_sq(env, obs):
        cid, a, cands = _decision(env)
        return _act(env, min(cands, key=lambda r: env.load[r]), a)

    def sel_exp(env, obs):
        cid, a, cands = _decision(env)
        sk = env.skill.get(a, {})
        return _act(env, max(cands, key=lambda r: sk.get(r, 0.0)), a)

    def sel_ppo(env, obs):
        act, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        return int(np.asarray(act).flatten()[0])

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
