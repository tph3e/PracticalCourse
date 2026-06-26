# 1.7 advanced — OrdinoR-style role discovery (fallback implementation)

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

DEFAULT_ARTIFACT = "results/permissions_roles.json"


def discover_roles(
    slim_df: pd.DataFrame,
    n_groups: int = 14,
    mode_threshold: float = 0.05,
    activity_col: str = "concept:name",
    resource_col: str = "org:resource",
) -> tuple[dict[str, list[str]], dict]:
    
    # Discover resource groups and derive an activity->resources permission map.
    df = slim_df[[resource_col, activity_col]].dropna().astype(str)

    # resource x activity count matrix -> row-normalized profiles
    counts = pd.crosstab(df[resource_col], df[activity_col])
    profiles = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)

    resources = profiles.index.to_list()
    activities = profiles.columns.to_list()
    k = min(n_groups, len(resources))
    labels = AgglomerativeClustering(n_clusters=k).fit_predict(profiles.values)

    # floor: a resource that demonstrably performed an activity is always
    # permitted it, so the role model never drops below observed reality (no
    # activity ends up with zero permitted resources -> no deadlock). Roles only
    # generalize beyond this floor to other members of the same group.
    activity_to_resources: dict[str, set[str]] = {
        a: set(counts.index[counts[a] > 0].astype(str)) for a in activities
    }
    groups_meta = []
    for g in sorted(set(labels)):
        members = [resources[i] for i in range(len(resources)) if labels[i] == g]
        group_counts = counts.loc[members].sum(axis=0)
        total = group_counts.sum()
        shares = group_counts / total if total else group_counts
        mode = [a for a in activities if shares[a] >= mode_threshold]
        for a in mode:
            activity_to_resources[a].update(members)
        groups_meta.append(
            {"group": int(g), "n_members": len(members), "execution_mode": mode}
        )

    mapping = {a: sorted(rs) for a, rs in activity_to_resources.items() if rs}
    meta = {
        "method": "ordinor-style fallback (agglomerative clustering on resource profiles)",
        "n_groups": k,
        "mode_threshold": mode_threshold,
        "n_resources": len(resources),
        "n_activities": len(activities),
        "groups": groups_meta,
    }
    return mapping, meta


def save_roles(mapping: dict[str, list[str]], path: str = DEFAULT_ARTIFACT) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f)
