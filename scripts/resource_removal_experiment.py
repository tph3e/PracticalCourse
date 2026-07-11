# Compares baseline resource pool vs. pool with two resources removed,
# on the integrated simulator. Same structure as compare_on_sim.py, but
# varies the resource pool instead of the allocation strategy.

import sys, os
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from SimulationEngineCore import Engine
from optimization.metrics import compute_all

SEED = 1
START = datetime(2000, 1, 3, 9, 0)   # matches compare_on_sim.py's window
END = datetime(2000, 1, 10, 0, 0)
REMOVE = {"User_76", "User_108"}
OUT_CSV = "results/resource_removal_comparison.csv"


def run(remove_users, label):
    print(f"running '{label}' ...")
    eng = Engine("data/BPI Challenge 2017.xes", seed=SEED)

    av = eng.resourceEngine.availability
    if remove_users:
        av._all_resources -= remove_users
        if av.calendars is not None:
            for u in remove_users:
                av.calendars.pop(u, None)

    eng.run(START, END, format_type=[])
    log = eng.logger.get_log()
    n_cases = log["case:concept:name"].nunique() if not log.empty else 0
    print(f"  [{label}] {len(log)} events, {n_cases} cases")

    m = compute_all(log, availability=av)
    return {
        "scenario": label,
        "cycle_time_days": round(m["cycle_time"]["mean_days"], 2),
        "occupation": round(m["occupation"]["mean"], 3),
        "fairness_gini": round(m["fairness"]["gini"], 3),
    }


def main():
    rows = [
        run(set(), "baseline (149 resources)"),
        run(REMOVE, "after removal (147 resources)"),
    ]
    table = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    print()
    print(table.to_string(index=False))
    print(f"\nsaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()