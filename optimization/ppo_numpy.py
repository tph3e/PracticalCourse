# PPO-style update (numpy) on trajectories collected FROM the integrated simulator.

# The recording strategy (RecordingRLAllocation) collects (X, action, hidden, p_old, reward) per
# decision during a normal Engine.run. This module upgrades the weak vanilla-REINFORCE update to a
# PPO-style one: a value critic + GAE(lambda) advantages + a clipped surrogate objective trained for
# several epochs over minibatches. This is the faithful Middelhuis "PPO on the simulator" learner,
# obtained WITHOUT gym control-inversion (the sim still runs forward; only the update is stronger).

# Pure numpy, matching PolicyNet (optimization.rl_agent): features -> tanh hidden -> per-action score.

from __future__ import annotations

import numpy as np

STATE_DIM = 8   # pool_state output size


def pool_state(X) -> np.ndarray:
    # Fixed-size critic input from the variable-size per-candidate feature matrix X
    # ([n_candidates + 1, N_FEATURES]; last row = postpone). Columns of X:
    # 0 load_norm, 1 skill, 2 availability_breadth, 3 busy_fraction, 4 is_postpone.
    n = X.shape[0] - 1
    cand = X[:n] if n > 0 else X[:1]
    return np.array([
        X[0, 3],                      # busy fraction (congestion)
        min(n / 10.0, 1.0),           # candidate-set size (normalised)
        cand[:, 0].mean(), cand[:, 0].min(), cand[:, 0].max(),   # load stats
        cand[:, 1].mean(), cand[:, 1].max(),                     # skill stats
        cand[:, 2].mean(),                                       # availability breadth
    ], dtype=float)


class ValueNet:
    # Small critic MLP: state -> tanh(hidden) -> scalar value.
    def __init__(self, n_in: int = STATE_DIM, hidden: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.3, (n_in, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.3, (hidden, 1))
        self.b2 = np.zeros(1)

    def forward(self, S: np.ndarray) -> np.ndarray:
        H = np.tanh(S @ self.W1 + self.b1)
        return (H @ self.W2 + self.b2).ravel()

    def sgd_step(self, S: np.ndarray, targets: np.ndarray, lr: float) -> float:
        # One MSE gradient step towards the GAE returns. Returns the mean squared error.
        H = np.tanh(S @ self.W1 + self.b1)
        v = (H @ self.W2 + self.b2).ravel()
        diff = v - targets
        dv = (diff / len(S)).reshape(-1, 1)
        gW2 = H.T @ dv
        gb2 = dv.sum()
        dH = (dv @ self.W2.T) * (1 - H ** 2)
        gW1 = S.T @ dH
        gb1 = dH.sum(axis=0)
        self.W1 -= lr * gW1; self.b1 -= lr * gb1
        self.W2 -= lr * gW2; self.b2 -= lr * gb2
        return float(np.mean(diff ** 2))


def compute_gae(rewards: np.ndarray, values: np.ndarray, gamma: float = 0.99, lam: float = 0.95):
    # GAE(lambda) for a single episode (bootstrap 0 at the end). Returns (advantages, returns).
    n = len(rewards)
    adv = np.zeros(n)
    last = 0.0
    for t in range(n - 1, -1, -1):
        next_v = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values


def ppo_update(policy, value, traj, clip: float = 0.2, epochs: int = 4, minibatch: int = 512,
               lr: float = 0.01, v_lr: float = 0.01, gamma: float = 0.99, lam: float = 0.95,
               entropy_beta: float = 0.01):
    # One PPO iteration over a collected trajectory of (X, a, hidden, p_old, reward).
    if not traj:
        return {}

    Xs = [t[0] for t in traj]
    acts = np.array([t[1] for t in traj])
    p_olds = [t[3] for t in traj]
    rewards = np.array([t[4] for t in traj], dtype=float)

    states = np.array([pool_state(X) for X in Xs])
    values = value.forward(states)
    adv, returns = compute_gae(rewards, values, gamma, lam)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    n = len(traj)
    idx = np.arange(n)
    for _ in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, n, minibatch):
            mb = idx[s:s + minibatch]
            gW1 = np.zeros_like(policy.W1); gb1 = np.zeros_like(policy.b1)
            gW2 = np.zeros_like(policy.W2); gb2 = np.zeros_like(policy.b2)
            for j in mb:
                X = Xs[j]; a = acts[j]; A = adv[j]
                p_new, H = policy.forward(X)
                ratio = p_new[a] / (p_olds[j][a] + 1e-9)
                unclipped = ratio * A
                clipped = np.clip(ratio, 1 - clip, 1 + clip) * A
                weight = ratio * A if unclipped <= clipped else 0.0   # clipped surrogate gradient gate
                ds = -p_new.copy(); ds[a] += 1.0        # d log pi(a) / d scores
                ds *= weight
                if entropy_beta and len(p_new) > 1:     # entropy bonus -> exploration
                    Hp = -(p_new * np.log(p_new + 1e-9)).sum()
                    ds += entropy_beta * (-(np.log(p_new + 1e-9) + Hp) * p_new)
                ds_col = ds.reshape(-1, 1)
                gW2 += H.T @ ds_col
                gb2 += ds.sum()
                dH = (ds_col @ policy.W2.T) * (1 - H ** 2)
                gW1 += X.T @ dH
                gb1 += dH.sum(axis=0)
            m = max(len(mb), 1)
            # gradient ASCENT on the clipped surrogate (mean over the minibatch)
            policy.W1 += lr * gW1 / m; policy.b1 += lr * gb1 / m
            policy.W2 += lr * gW2 / m; policy.b2 += lr * gb2 / m
            value.sgd_step(states[mb], returns[mb], v_lr)

    return {"value_mse": float(np.mean((value.forward(states) - returns) ** 2)),
            "mean_return": float(returns.mean())}
