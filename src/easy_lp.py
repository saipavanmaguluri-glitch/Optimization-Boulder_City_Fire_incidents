"""
easy_lp.py
----------
Step 6: EASY problem -- LP-relaxed facility location, solved on the real
Boulder travel-time matrix from Step 5.

For each candidate station i and demand point j:
  x_i  in [0,1]  -- how "open" station i is (fractional, not yet binary)
  y_ij in [0,1]  -- fraction of demand point j's incidents served by station i

Minimize total weighted travel time, subject to:
  sum_i x_i <= p        -- open at most p stations
  sum_i y_ij = 1          -- every demand point fully served (fractionally)
  y_ij <= x_i             -- can only assign to an open station

Solved for several values of p so you can plot objective vs. p and
compare directly against the hard MIP version later.

Reads:
  data/travel_time_matrix.npy
  data/site_ids.npy
  data/demand_ids.npy
  data/demand_points.csv   (for the weight column)

Produces:
  data/easy_results.csv    -- one row per p: objective value, runtime
  data/easy_x_p{p}.npy     -- station "openness" vector for each p tried

Install first:
    pip install cvxpy numpy pandas
"""

import time
import numpy as np
import pandas as pd
import cvxpy as cp

MATRIX_PATH = "../data/travel_time_matrix.npy"
SITE_IDS_PATH = "../data/site_ids.npy"
DEMAND_IDS_PATH = "../data/demand_ids.npy"
DEMAND_CSV = "../data/demand_points.csv"

RESULTS_OUT = "../data/easy_results.csv"

P_VALUES = [2, 3, 4, 5, 6, 8, 10]  # number of stations to try opening


def load_data():
    T = np.load(MATRIX_PATH)               # shape (n_sites, n_demand), seconds
    site_ids = np.load(SITE_IDS_PATH)
    demand_ids = np.load(DEMAND_IDS_PATH)
    demand_df = pd.read_csv(DEMAND_CSV)

    # match weight order to demand_ids order -- don't assume the CSV row
    # order already matches the .npy column order
    weight_lookup = demand_df.set_index("demand_id")["weight"]
    w = weight_lookup.loc[demand_ids].values.astype(float)

    print(f"Loaded matrix {T.shape}, {len(site_ids)} sites, {len(demand_ids)} demand points")
    print(f"Weight sum: {w.sum()} (should match total cleaned incident count)")

    n_nan = np.isnan(T).sum()
    if n_nan > 0:
        print(f"WARNING: {n_nan} NaN entries found -- replacing with a "
              f"large penalty travel time so the solver avoids them.")
        T = np.nan_to_num(T, nan=1e6)

    return T, site_ids, demand_ids, w


def solve_easy(T, w, p):
    n_sites, n_demand = T.shape

    x = cp.Variable(n_sites, nonneg=True)
    y = cp.Variable((n_sites, n_demand), nonneg=True)

    objective = cp.Minimize(cp.sum(cp.multiply(w[np.newaxis, :] * T, y)))

    constraints = [
        x <= 1,
        y <= 1,
        cp.sum(x) <= p,
        cp.sum(y, axis=0) == 1,
    ]
    for i in range(n_sites):
        constraints.append(y[i, :] <= x[i])

    problem = cp.Problem(objective, constraints)
    t0 = time.time()
    problem.solve()
    elapsed = time.time() - t0

    return {
        "status": problem.status,
        "objective": problem.value,
        "x": x.value,
        "y": y.value,
        "solve_time_s": elapsed,
    }


def main():
    T, site_ids, demand_ids, w = load_data()

    results_rows = []
    for p in P_VALUES:
        print(f"\nSolving easy LP for p={p}...")
        result = solve_easy(T, w, p)

        if result["status"] != "optimal":
            print(f"  WARNING: solver status is '{result['status']}', not 'optimal' "
                  f"-- treat this result with caution.")

        n_open = (result["x"] > 1e-4).sum()
        print(f"  status={result['status']}  "
              f"objective={result['objective']:.1f}  "
              f"stations materially open={n_open}  "
              f"solve_time={result['solve_time_s']:.2f}s")

        np.save(f"../data/easy_x_p{p}.npy", result["x"])

        results_rows.append({
            "p": p,
            "status": result["status"],
            "objective": result["objective"],
            "n_stations_open": n_open,
            "solve_time_s": result["solve_time_s"],
        })

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(RESULTS_OUT, index=False)
    print(f"\nSaved summary to {RESULTS_OUT}")
    print(results_df)


if __name__ == "__main__":
    main()