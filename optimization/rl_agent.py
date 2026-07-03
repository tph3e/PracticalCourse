# 1.1 advanced — neural policy-gradient (reinforce) allocation agent.

# A compact per-candidate scoring policy (a small MLP) trained with reinforce on
# the standalone AllocationEnv. The policy scores each candidate's feature vector

# Everything is pure numpy: training and the runtime forward pass

from __future__ import annotations

import json

import numpy as np

from resources.allocation import AllocationStrategy
from .environment import AllocationEnv, build_features, N_FEATURES


# policy network (numpy MLP: features -> hidden(tanh) -> scalar score)
class PolicyNet:
    def __init__(self, n_features: int = N_FEATURES, hidden: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.5, (n_features, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.5, (hidden, 1))
        self.b2 = np.zeros(1)

    def forward(self, X: np.ndarray):
    #    X: [n_candidates, n_features] -> (probs[n], hidden[n,h]).
        H = np.tanh(X @ self.W1 + self.b1)
        s = (H @ self.W2 + self.b2).ravel()
        s = s - s.max()
        p = np.exp(s)
        p /= p.sum()
        return p, H

    def to_dict(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("W1", "b1", "W2", "b2")}

    @classmethod
    def from_dict(cls, d: dict) -> "PolicyNet":
        net = cls()
        for k in ("W1", "b1", "W2", "b2"):
            setattr(net, k, np.asarray(d[k], dtype=float))
        return net


# REINFORCE training
def train(cfg, episodes=300, hidden=16, lr=0.02, gamma=0.99, ep_len=1000, seed=0,
          entropy_beta=0.01, log_every=0):
    net = PolicyNet(N_FEATURES, hidden, seed)
    rng = np.random.default_rng(seed)
    baseline = 0.0
    history = []

    for ep in range(episodes):
        env = AllocationEnv(**cfg, seed=10_000 + ep, max_steps=ep_len)
        obs = env.reset()
        traj = []  # (X, action, hidden, probs, reward)
        total = 0.0
        while obs is not None:
            p, H = net.forward(obs)
            a = int(rng.choice(len(p), p=p))
            X = obs
            obs, r, done, _ = env.step(a)
            traj.append((X, a, H, p, r))
            total += r
            if done:
                break

        # returns-to-go
        returns = np.empty(len(traj))
        G = 0.0
        for i in range(len(traj) - 1, -1, -1):
            G = traj[i][4] + gamma * G
            returns[i] = G

        baseline = returns.mean() if ep == 0 else 0.95 * baseline + 0.05 * returns.mean()
        adv = returns - baseline
        if adv.std() > 1e-8:
            adv = adv / (adv.std() + 1e-8)

        gW1 = np.zeros_like(net.W1); gb1 = np.zeros_like(net.b1)
        gW2 = np.zeros_like(net.W2); gb2 = np.zeros_like(net.b2)
        for (X, a, H, p, _), A in zip(traj, adv):
            ds = -p.copy(); ds[a] += 1.0          # d log pi(a) / d scores
            ds *= A                               # weight by advantage
            if entropy_beta and len(p) > 1:       # entropy bonus -> exploration
                Hp = -(p * np.log(p + 1e-9)).sum()
                ds += entropy_beta * (-(np.log(p + 1e-9) + Hp) * p)
            ds_col = ds.reshape(-1, 1)
            gW2 += H.T @ ds_col
            gb2 += ds.sum()
            dH = (ds_col @ net.W2.T) * (1 - H ** 2)
            gW1 += X.T @ dH
            gb1 += dH.sum(axis=0)

        # gradient ASCENT on E[log pi * advantage]
        net.W1 += lr * gW1; net.b1 += lr * gb1
        net.W2 += lr * gW2; net.b2 += lr * gb2
        history.append(total)
        if log_every and (ep + 1) % log_every == 0:
            print(f"  ep {ep+1:4d} | avg return {np.mean(history[-log_every:]):.1f}")

    return net, history


# runtime strategy
class RLAllocation(AllocationStrategy):
    # Trained policy as a swappable allocation strategy (greedy at inference)

    def __init__(self, net: PolicyNet, calendars: dict, skill: dict, max_cal: int):
        self.net = net
        self.calendars = calendars
        self.skill = skill
        self.max_cal = max_cal

    def pick(self, candidates, context=None):
        if not candidates:
            return None
        cands = sorted(candidates)
        event = getattr(context, "event", None) if context else None
        activity = getattr(event, "activity", None)
        load = context.load if context is not None else {}
        n_res = len(self.calendars) or 1
        busy_fraction = len(getattr(context, "busy", ())) / n_res if context is not None else 0.0
        X = build_features(cands, activity, load, self.calendars, self.skill, self.max_cal, busy_fraction)
        p, _ = self.net.forward(X)
        a = int(np.argmax(p))  # greedy
        if a == len(cands):    # postpone -> return None so the core suspends/retries
            return None
        return cands[a]

    def save(self, path: str) -> None:
        bundle = {
            "net": self.net.to_dict(),
            "max_cal": self.max_cal,
            "cal_sizes": {r: len(c) for r, c in self.calendars.items()},
            "skill": {a: dict(rs) for a, rs in self.skill.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bundle, f)

    @classmethod
    def load(cls, path: str) -> "RLAllocation":
        with open(path, encoding="utf-8") as f:
            b = json.load(f)
        net = PolicyNet.from_dict(b["net"])
        # only calendar sizes are needed for the availability feature
        calendars = {r: range(n) for r, n in b["cal_sizes"].items()}
        return cls(net, calendars, b["skill"], int(b["max_cal"]))
