"""
travel_matrix.py
-----------------
Step 5: build a real road-network travel-time matrix between every
candidate station site and every demand point.

Reads:
  data/candidate_sites.csv
  data/demand_points.csv

Produces:
  data/travel_time_matrix.npy   -- shape (n_sites, n_demand), seconds
  data/site_ids.npy             -- site_id for each matrix row, in order
  data/demand_ids.npy           -- demand_id for each matrix column, in order

Install first:
    pip install osmnx networkx numpy pandas
"""

import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
import time

SITES_CSV = "../data/candidate_sites.csv"
DEMAND_CSV = "../data/demand_points.csv"

MATRIX_OUT = "../data/travel_time_matrix.npy"
SITE_IDS_OUT = "../data/site_ids.npy"
DEMAND_IDS_OUT = "../data/demand_ids.npy"

# Padding (in degrees) added around the bounding box of all points before
# downloading the road network. ~0.05 deg is roughly 5km at this latitude.
BBOX_PADDING = 0.05


def load_points():
    sites = pd.read_csv(SITES_CSV)
    demand = pd.read_csv(DEMAND_CSV)
    print(f"Loaded {len(sites)} candidate sites, {len(demand)} demand points")
    return sites, demand


def build_graph(sites, demand):
    """
    Downloads the drivable road network covering all sites + demand
    points, with padding. If any demand points are far outliers, the
    bounding box -- and download size/runtime -- can balloon. Check the
    printed bbox size before waiting on a huge download.
    """
    all_lats = pd.concat([sites["lat"], demand["lat"]])
    all_lons = pd.concat([sites["lon"], demand["lon"]])

    north, south = all_lats.max() + BBOX_PADDING, all_lats.min() - BBOX_PADDING
    east, west = all_lons.max() + BBOX_PADDING, all_lons.min() - BBOX_PADDING

    print(f"Bounding box: lat [{south:.3f}, {north:.3f}], "
          f"lon [{west:.3f}, {east:.3f}]")
    print(f"  (~{(north-south)*111:.1f} km tall x "
          f"~{(east-west)*85:.1f} km wide at this latitude)")
    print("Downloading road network from OpenStreetMap -- this can take "
          "a few minutes for a large box...")

    max_retries = 5
    wait = 15  # seconds, doubles each retry
    G = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt}/{max_retries}...")
            G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type="drive")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            if attempt < max_retries:
                print(f"  Waiting {wait}s before retry...")
                time.sleep(wait)
                wait *= 2
            else:
                raise

    if G is None:
        raise RuntimeError("Could not download road network after retries.")
        print(f"Graph has {len(G.nodes)} nodes, {len(G.edges)} edges")

    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    return G


def snap_to_nodes(G, df):
    """Finds the nearest road-network node to each point. Returns a
    list of node ids, same order as the input dataframe's rows."""
    nodes = ox.distance.nearest_nodes(G, X=df["lon"].values, Y=df["lat"].values)
    return list(nodes)


def build_travel_time_matrix(G, site_nodes, demand_nodes):
    """
    For each candidate site, run one Dijkstra pass (single-source
    shortest path to every reachable node) and read off the travel
    time to each demand point's node.
    """
    n_sites, n_demand = len(site_nodes), len(demand_nodes)
    T = np.full((n_sites, n_demand), np.nan)

    for i, s_node in enumerate(site_nodes):
        lengths = nx.single_source_dijkstra_path_length(G, s_node, weight="travel_time")
        for j, d_node in enumerate(demand_nodes):
            if d_node in lengths:
                T[i, j] = lengths[d_node]
        if (i + 1) % 10 == 0 or i == n_sites - 1:
            print(f"  computed site {i+1}/{n_sites}")

    unreachable = np.isnan(T).sum()
    if unreachable > 0:
        print(f"\nWARNING: {unreachable} of {n_sites * n_demand} "
              f"(site, demand) pairs are UNREACHABLE (no road path found). "
              f"These will show as NaN -- treat NaN as 'cannot serve this "
              f"demand point from this station', not as 0, in later steps.")

    return T


def main():
    sites, demand = load_points()
    G = build_graph(sites, demand)

    print("\nSnapping sites and demand points to nearest road nodes...")
    site_nodes = snap_to_nodes(G, sites)
    demand_nodes = snap_to_nodes(G, demand)

    print("\nComputing travel-time matrix (this is the slow step)...")
    T = build_travel_time_matrix(G, site_nodes, demand_nodes)

    print(f"\nMatrix shape: {T.shape}")
    valid = T[~np.isnan(T)]
    if len(valid) > 0:
        print(f"Travel time range (valid entries): "
              f"{valid.min():.0f}s to {valid.max():.0f}s "
              f"({valid.min()/60:.1f} to {valid.max()/60:.1f} minutes)")

    np.save(MATRIX_OUT, T)
    np.save(SITE_IDS_OUT, sites["site_id"].values)
    np.save(DEMAND_IDS_OUT, demand["demand_id"].values)
    print(f"\nSaved {MATRIX_OUT}, {SITE_IDS_OUT}, {DEMAND_IDS_OUT}")


if __name__ == "__main__":
    main()