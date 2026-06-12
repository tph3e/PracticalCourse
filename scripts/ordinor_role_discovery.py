# 1.7 advanced — role discovery with OrdinoR proper (reference [3])

from __future__ import annotations

import json
import os

import pandas as pd

from ordinor.execution_context import ATonlyMiner
from ordinor.org_model_miner.resource_features import direct_count
from ordinor.org_model_miner.group_discovery import ahc
from ordinor.org_model_miner.group_profiling import full_recall

SLIM = "data/bpic17_slim.parquet"
OUT = "results/permissions_roles.json"
N_GROUPS = 14  # comparable to the 14 roles found in the first-assignment analysis


def main() -> None:
    full = pd.read_parquet(SLIM)
    # floor: actual performers per activity over ALL lifecycle events, so every
    # activity (incl. the few without 'complete' events) stays covered and the
    # role model never permits fewer resources than were observed.
    floor = (
        full[["concept:name", "org:resource"]]
        .dropna()
        .astype(str)
        .groupby("concept:name")["org:resource"]
        .agg(set)
        .to_dict()
    )

    # an "execution" of an activity = its completion event (OrdinoR input)
    df = full[full["lifecycle:transition"] == "complete"][
        ["case:concept:name", "concept:name", "org:resource", "time:timestamp"]
    ].dropna()

    cxt = ATonlyMiner(df)
    rl = cxt.derive_resource_log(df)
    profiles = direct_count(rl)
    groups = ahc(profiles, n_groups=N_GROUPS)
    om = full_recall(groups, rl)

    # execution context = (case_type, activity_type, time_type). For ATonlyMiner
    # only the activity type is informative -> map it back to activity name(s).
    activity_to_resources: dict[str, set[str]] = {}
    for gid, members in om.find_all_groups():
        members = set(map(str, members))
        for (_, at, _) in om.find_group_execution_contexts(gid):
            for activity in cxt.get_values_by_type(at):
                activity_to_resources.setdefault(str(activity), set()).update(members)

    # union the discovered role permissions with the observed-performer floor
    for activity, performers in floor.items():
        activity_to_resources.setdefault(activity, set()).update(performers)

    mapping = {a: sorted(rs) for a, rs in activity_to_resources.items() if rs}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(mapping, f)

    n_act = full["concept:name"].nunique()
    print(f"OrdinoR: {len(groups)} groups, {len(mapping)}/{n_act} activities covered")
    print(f"written -> {OUT}")


if __name__ == "__main__":
    main()
