# w_fair sensitivity: the numpy REINFORCE and MaskablePPO trained on the SAME paper-faithful
# GymAllocationEnv at w_fair in {0, 1, 5}, so both learned methods and the heuristic baselines are
# directly comparable (identical dynamics and reward). REINFORCE keeps its per-candidate policy
# (PolicyNet/reinforce_update reused unchanged) and acts through the env decision interface
# (_decision -> global action), which avoids learning the raw R*A+1 global softmax. w_fair enters
# only the reward, so the baselines are w_fair-invariant and evaluation is scale-consistent.

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from optimization.rl_agent import PolicyNet, reinforce_update
from optimization.environment import build_features, N_FEATURES
from scripts.train_ppo import GymAllocationEnv, _build_cfg, evaluate

W_FAIRS = [0.0, 1.0, 5.0]


# shared helpers: turn the current env decision into per-candidate features and a global action
def _busy_fraction(env):
    return sum(1 for r in env.resources if env.busy_until.get(r, 0.0) > env.t) / max(1, env.R)


def _features(env, cands, act):
    return build_features(cands, act, env.alloc_count, env.calendars, env.skill,
                          env.max_cal, _busy_fraction(env))


def _global_action(env, idx, cands, act):
    if idx == len(cands):                       # postpone row -> global postpone action
        return env.R * env.A
    r = cands[idx]
    return env.res_index[r] * env.A + env.act_index[act]


# REINFORCE training on GymAllocationEnv (mirrors optimization.rl_agent.train, decision interface)
def train_reinforce_on_gym(cfg, w_fair, episodes=500, hidden=24, lr=0.05, gamma=0.99,
                           max_steps=800, seed=1, entropy_beta=0.01, log_every=100):
    net = PolicyNet(N_FEATURES, hidden, seed)
    rng = np.random.default_rng(seed)
    baseline = None
    history = []
    for ep in range(episodes):
        env = GymAllocationEnv(cfg, seed=10_000 + ep, w_fair=w_fair, max_steps=max_steps)
        env.reset(seed=10_000 + ep)
        traj = []
        total = 0.0
        while not env._episode_done():
            if env._decision is None:
                break
            _cid, act, cands = env._decision
            cands = list(cands)
            X = _features(env, cands, act)
            p, H = net.forward(X)
            idx = int(rng.choice(len(p), p=p))
            _obs, r, done, _trunc, _info = env.step(_global_action(env, idx, cands, act))
            traj.append((X, idx, H, p, r))
            total += r
            if done:
                break
        baseline = reinforce_update(net, traj, baseline, lr, gamma, entropy_beta)
        history.append(total)
        if log_every and (ep + 1) % log_every == 0:
            print(f"  [w_fair={w_fair}] ep {ep + 1:4d} | avg return "
                  f"{np.mean(history[-log_every:]):.2f}", flush=True)
    return net, history


def make_sel_reinforce(net):
    def sel(env, _obs):
        _cid, act, cands = env._decision
        cands = list(cands)
        p, _ = net.forward(_features(env, cands, act))
        return _global_action(env, int(np.argmax(p)), cands, act)   # greedy
    return sel


# heuristic baselines (mirror scripts/train_ppo main selectors; reward-independent)
def make_baseline_selectors():
    rng = np.random.default_rng(7)
    rr = {"c": 0}

    def _act(env, r, a):
        return env.res_index[r] * env.A + env.act_index[a]

    def sel_random(env, _obs):
        _c, a, cands = env._decision
        return _act(env, cands[int(rng.integers(len(cands)))], a)

    def sel_rr(env, _obs):
        _c, a, cands = env._decision
        cs = sorted(cands); r = cs[rr["c"] % len(cs)]; rr["c"] += 1
        return _act(env, r, a)

    def sel_sq(env, _obs):
        _c, a, cands = env._decision
        return _act(env, min(cands, key=lambda r: env.load[r]), a)

    def sel_exp(env, _obs):
        _c, a, cands = env._decision
        sk = env.skill.get(a, {})
        return _act(env, max(cands, key=lambda r: sk.get(r, 0.0)), a)

    return [("random", sel_random), ("round-robin", sel_rr),
            ("shortest-queue", sel_sq), ("most-experienced", sel_exp)]


def train_ppo_on_gym(cfg, w_fair, timesteps, seed=1):
    from sb3_contrib import MaskablePPO
    env = GymAllocationEnv(cfg, seed=0, w_fair=w_fair)
    model = MaskablePPO("MlpPolicy", env, n_steps=2048, batch_size=256, gamma=0.99,
                        policy_kwargs=dict(net_arch=[128, 128]), seed=seed, verbose=0)
    model.learn(total_timesteps=timesteps)
    return model


def make_sel_ppo(model):
    def sel(env, obs):
        act, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        return int(np.asarray(act).flatten()[0])
    return sel


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    ppo_timesteps = int(os.environ.get("PPO_TIMESTEPS", "600000"))
    rl_episodes = int(os.environ.get("RL_EPISODES", "500"))
    cfg = _build_cfg()
    probe = GymAllocationEnv(cfg, seed=0)
    print(f"env: {probe.R} resources, {probe.A} activities, obs {2 * probe.R + probe.A}, "
          f"actions {probe.R * probe.A + 1}; PPO_TIMESTEPS={ppo_timesteps}, RL_EPISODES={rl_episodes}",
          flush=True)

    make_eval = lambda s: GymAllocationEnv(cfg, seed=s, w_fair=1.0)   # reward-invariant measurement
    rows = []

    # baselines are w_fair-invariant (they ignore the reward) -> evaluate once, replicate across w
    for name, sel in make_baseline_selectors():
        ct, g = evaluate(make_eval, sel)
        for w in W_FAIRS:
            rows.append({"w_fair": w, "method": name,
                         "cycle_time_h": round(ct, 2), "load_gini": round(g, 3)})
        print(f"baseline {name:16s} ct={ct:.2f} gini={g:.3f}", flush=True)

    for w in W_FAIRS:
        print(f"=== training w_fair={w} ===", flush=True)
        net, _ = train_reinforce_on_gym(cfg, w, episodes=rl_episodes)
        ct, g = evaluate(make_eval, make_sel_reinforce(net))
        rows.append({"w_fair": w, "method": "REINFORCE",
                     "cycle_time_h": round(ct, 2), "load_gini": round(g, 3)})
        print(f"  REINFORCE          ct={ct:.2f} gini={g:.3f}", flush=True)

        model = train_ppo_on_gym(cfg, w, ppo_timesteps)
        ct, g = evaluate(make_eval, make_sel_ppo(model))
        rows.append({"w_fair": w, "method": "PPO (MaskablePPO)",
                     "cycle_time_h": round(ct, 2), "load_gini": round(g, 3)})
        print(f"  PPO (MaskablePPO)  ct={ct:.2f} gini={g:.3f}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/wfair_sensitivity.csv", index=False)
    print(df.to_string(index=False), flush=True)

    # figure: cycle time and load Gini vs training w_fair, with the REINFORCE degenerate-collapse
    # (no completed cases -> NaN cycle time at w_fair >= 1) annotated honestly.
    base = ["random", "round-robin", "shortest-queue", "most-experienced"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, col, title in [(axes[0], "cycle_time_h", "cycle time [h] (lower better)"),
                           (axes[1], "load_gini", "load Gini (lower fairer)")]:
        for m in base:
            ax.axhline(df[df.method == m][col].iloc[0], ls="--", lw=0.8, alpha=0.55, label=m)
        for m, c in [("REINFORCE", "tab:green"), ("PPO (MaskablePPO)", "tab:red")]:
            sub = df[df.method == m].sort_values("w_fair")
            ax.plot(sub.w_fair, sub[col], marker="o", color=c, label=m)
        ax.set_xlabel("training $w_{\\mathrm{fair}}$"); ax.set_title(title); ax.set_xticks(W_FAIRS)
    rf = df[df.method == "REINFORCE"].set_index("w_fair")
    for w in W_FAIRS:
        if np.isnan(rf.loc[w, "cycle_time_h"]):
            axes[1].annotate("no\ncompletions", (w, rf.loc[w, "load_gini"]), fontsize=6,
                             ha="center", va="bottom", color="tab:green",
                             xytext=(0, 6), textcoords="offset points")
    axes[0].text(0.30, 0.97, "REINFORCE completes no cases at $w_{\\mathrm{fair}}\\geq1$\n"
                 "(degenerate; its low Gini is not a real fairness gain)",
                 transform=axes[0].transAxes, fontsize=6, va="top",
                 bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    axes[1].legend(fontsize=6, loc="center right")
    fig.tight_layout(); fig.savefig("results/wfair_sensitivity.png", dpi=120)
    print("saved -> results/wfair_sensitivity.csv, results/wfair_sensitivity.png", flush=True)


if __name__ == "__main__":
    main()
