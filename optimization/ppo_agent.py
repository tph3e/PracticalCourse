# Option B — deploy a MaskablePPO policy (trained on GymAllocationEnv, scripts/train_ppo.py) as a
# swappable allocation strategy on the INTEGRATED simulator. It rebuilds the exact GymAllocationEnv
# observation (3R + A + 1) and action mask from the AllocationContext, so the policy sees identical
# inputs at training and at inference. sb3 is imported lazily (only needed to load/predict).

from __future__ import annotations

import numpy as np

from resources.allocation import AllocationStrategy


class PPOAllocation(AllocationStrategy):
    def __init__(self, model, calendars: dict, skill: dict, activity_mix: dict):
        self.model = model
        self.calendars = calendars
        self.skill = skill
        self.resources = sorted(calendars.keys())
        self.res_index = {r: i for i, r in enumerate(self.resources)}
        self.activities = list(activity_mix.keys())
        self.act_index = {a: i for i, a in enumerate(self.activities)}
        self.R = len(self.resources)
        self.A = len(self.activities)

    def _available_now(self, r, time) -> bool:
        # Same semantics as GymAllocationEnv._available_now, but from the real datetime.
        cal = self.calendars.get(r)
        return cal is None or (time.weekday(), time.hour) in cal

    def pick(self, candidates, context=None):
        if not candidates or context is None:
            return None
        event = getattr(context, "event", None)
        act = getattr(event, "activity", None)
        time = context.time
        load = context.load or {}                 # ResourceEngine.load = cumulative allocation count
        busy = getattr(context, "busy", ()) or ()

        # rebuild GymAllocationEnv._obs exactly (load feature = alloc count, per (a))
        obs = np.zeros(3 * self.R + self.A + 1, dtype=np.float32)
        max_count = (max(load.values()) if load else 0) or 1
        skill_a = self.skill.get(act, {})
        for r in self.resources:
            i = self.res_index[r]
            obs[i] = load.get(r, 0) / max_count
            obs[self.R + i] = 1.0 if self._available_now(r, time) else 0.0
            obs[2 * self.R + i] = skill_a.get(r, 0.0)
        if act in self.act_index:
            obs[3 * self.R + self.act_index[act]] = 1.0
        obs[3 * self.R + self.A] = (len(busy) / self.R) if self.R else 0.0

        # action mask: candidate resource indices + postpone (index R)
        mask = np.zeros(self.R + 1, dtype=bool)
        mask[self.R] = True
        for r in candidates:
            j = self.res_index.get(r)
            if j is not None:
                mask[j] = True

        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        a = int(np.asarray(action).flatten()[0])
        if a == self.R:                           # postpone -> core suspends/retries
            return None
        chosen = self.resources[a]
        return chosen if chosen in candidates else None

    @classmethod
    def load(cls, model_path: str, calendars: dict, skill: dict, activity_mix: dict) -> "PPOAllocation":
        from sb3_contrib import MaskablePPO       # lazy: sb3 only required when PPO is used
        return cls(MaskablePPO.load(model_path), calendars, skill, activity_mix)
