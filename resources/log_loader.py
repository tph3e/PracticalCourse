# Slim log cache for the resource models (offline preprocessing).

# Parsing the 578 MB BPIC-17 XES takes minutes, so parse it once and cache a slim parquet 

from __future__ import annotations

import os

import pandas as pd

# columns the resource models need (1.6 timestamps, 1.7 resource–activity, lifecycle)
SLIM_COLUMNS = [
    "case:concept:name",
    "concept:name",
    "org:resource",
    "time:timestamp",
    "lifecycle:transition",
]

DEFAULT_XES = "data/BPI Challenge 2017.xes"
DEFAULT_CACHE = "data/bpic17_slim.parquet"


def load_slim_log(
    xes_path: str = DEFAULT_XES,
    cache_path: str = DEFAULT_CACHE,
    force_reparse: bool = False,
) -> pd.DataFrame:

    if not force_reparse and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    import pm4py  

    df = pm4py.read_xes(xes_path)
    # keep only the slim columns that are actually present
    cols = [c for c in SLIM_COLUMNS if c in df.columns]
    slim = df[cols].copy()
    slim.to_parquet(cache_path, index=False)
    return slim
