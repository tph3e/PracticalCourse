# # 1.1 advanced: Deploy the paper-faithful MaskablePPO policy (trained on GymAllocationEnv, scripts/train_ppo.py)
# as a swappable allocation strategy on the integrated simulator.
#
# Training MDP: global (resource, activity)-pair action space, 2R+A state (delta | eta | kappa).
# The engine's pick(candidates, context) is per-task.
#
# delta and eta are reconstructed from the AllocationContext. kappa is not observable at a single pick(),
# so it is reduced to the one locally


from __future__ import annotations

import numpy as np

from resources.allocation import AllocationStrategy


class PPOAllocation(AllocationStrategy):
    def __init__(self, model, calendars: dict, skill: dict, activity_mix: dict):
        self.model = model
        self.calendars = calendars
        self.skill = skill                       # kept for signature compatibility (paper obs has no skill)
        self.resources = sorted(calendars.keys())
        self.res_index = {r: i for i, r in enumerate(self.resources)}
        self.activities = list(activity_mix.keys())
        self.act_index = {a: i for i, a in enumerate(self.activities)}
        self.R = len(self.resources)
        self.A = len(self.activities)

    def pick(self, candidates, context=None):
        if not candidates or context is None:
            return None
        event = getattr(context, "event", None)
        act = getattr(event, "activity", None)
        if act not in self.act_index:
            return None
        busy = getattr(context, "busy", ()) or ()
        busy_activity = getattr(context, "busy_activity", {}) or {}
        R, A = self.R, self.A

        # rebuild the paper state 2R+A = delta | eta | kappa
        obs = np.zeros(2 * R + A, dtype=np.float32)
        for r in busy:
            i = self.res_index.get(r)
            if i is None:
                continue
            obs[i] = 1.0                                    # delta
            ba = busy_activity.get(r)                       # eta
            if ba in self.act_index:
                obs[R + i] = (self.act_index[ba] + 1) / A
        # kappa 
        obs[2 * R + self.act_index[act]] = min(1 / 100.0, 1.0)

        # per-task reduction
        j = self.act_index[act]
        mask = np.zeros(R * A + 1, dtype=bool)
        mask[R * A] = True                                  # postpone
        for r in candidates:
            k = self.res_index.get(r)
            if k is not None:
                mask[k * A + j] = True

        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        a = int(np.asarray(action).flatten()[0])
        if a == R * A:                                      # postpone 
            return None
        i = a // A
        chosen = self.resources[i] if 0 <= i < R else None
        return chosen if chosen in candidates else None

    @classmethod
    def load(cls, model_path: str, calendars: dict, skill: dict, activity_mix: dict) -> "PPOAllocation":
        from sb3_contrib import MaskablePPO       
        return cls(MaskablePPO.load(model_path), calendars, skill, activity_mix)
