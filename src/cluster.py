"""
cluster.py
----------
Step 4: shrink data/incidents_clean.csv (30k+ individual incidents) down
to a solvable number of demand points, and derive a set of candidate
station sites.

Produces:
  data/demand_points.csv    -- cluster centroids + weight (incident count)
  data/candidate_sites.csv  -- candidate station locations
  data/clusters_map.html    -- a folium map to sanity-check visually

Install first:
    pip install scikit-learn folium pandas numpy
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import folium

INPUT_CSV = "../data/incidents_clean.csv"
DEMAND_OUT = "../data/demand_points.csv"
SITES_OUT = "../data/candidate_sites.csv"
MAP_OUT = "../data/clusters_map.html"

N_DEMAND_CLUSTERS = 200     # how many demand points to collapse incidents into
N_EXTRA_CANDIDATE_SITES = 40  # extra candidate stations beyond real ones

# Keep this out of git -- regenerate with your own key if needed.
CARTO_API_KEY = "cb1_560q_1_48580a49ee46aaf8ce6a53fb"


def build_demand_points(df, k=N_DEMAND_CLUSTERS):
    """
    Cluster all incident lat/lon into k demand points. Each point's
    weight = how many incidents fell in that cluster.
    """
    coords = df[["lat", "lon"]].values

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(coords)

    df = df.copy()
    df["cluster"] = labels

    weight = df.groupby("cluster").size().rename("weight")

    centroids = pd.DataFrame(km.cluster_centers_, columns=["lat", "lon"])
    centroids["cluster"] = range(k)
    centroids = centroids.merge(weight, on="cluster")
    centroids = centroids.rename(columns={"cluster": "demand_id"})

    return centroids[["demand_id", "lat", "lon", "weight"]]


def build_candidate_sites(df, n_extra=N_EXTRA_CANDIDATE_SITES):
    """
    Candidate sites = (a) real station-adjacent locations already
    present in the data, from where fire units were dispatched, plus
    (b) extra candidate points from clustering the same incident data
    at a coarser resolution.
    """
    real_sites = None
    if "LOCATIONATASSIGNFIRE" in df.columns and "LATATASSIGNFIRE" in df.columns:
        site_counts = (
            df.dropna(subset=["LOCATIONATASSIGNFIRE", "LATATASSIGNFIRE", "LONGATASSIGNFIRE"])
            .groupby("LOCATIONATASSIGNFIRE")
            .agg(
                lat=("LATATASSIGNFIRE", "median"),
                lon=("LONGATASSIGNFIRE", "median"),
                n=("LOCATIONATASSIGNFIRE", "size"),
            )
            .reset_index()
        )
        # if values look like scaled integers (huge numbers), rescale.
        # LONGATASSIGNFIRE is stored as a positive magnitude (no sign),
        # so force it negative -- Boulder is West of the prime meridian.
        if site_counts["lat"].abs().median() > 1000:
            site_counts["lat"] = site_counts["lat"] / 1e6
            site_counts["lon"] = -site_counts["lon"].abs() / 1e6

        real_sites = site_counts[site_counts["n"] >= 20].copy()
        real_sites["source"] = "real_station"
        real_sites = real_sites[["lat", "lon", "source"]]
        print(f"Found {len(real_sites)} likely real station locations "
              f"(dispatched from 20+ times)")

    coords = df[["lat", "lon"]].values
    km = KMeans(n_clusters=n_extra, random_state=7, n_init=10)
    km.fit(coords)
    extra_sites = pd.DataFrame(km.cluster_centers_, columns=["lat", "lon"])
    extra_sites["source"] = "candidate_grid"

    if real_sites is not None and len(real_sites) > 0:
        sites = pd.concat([real_sites, extra_sites], ignore_index=True)
    else:
        print("No usable LOCATIONATASSIGNFIRE data found -- using only "
              "the candidate_grid sites.")
        sites = extra_sites

    sites = sites.reset_index(drop=True)
    sites["site_id"] = sites.index
    return sites[["site_id", "lat", "lon", "source"]]


def plot_sanity_check(demand, sites, out_path=MAP_OUT):
    """Draws demand points (sized by weight) and candidate sites on a
    real map with actual street tiles, for a visual sanity check."""
    center = [demand["lat"].mean(), demand["lon"].mean()]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    max_w = demand["weight"].max()
    for _, row in demand.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=2 + 8 * (row["weight"] / max_w),
            color="crimson", fill=True, fill_opacity=0.5,
            popup=f"demand {row['demand_id']}: weight {row['weight']}",
        ).add_to(m)

    for _, row in sites.iterrows():
        color = "navy" if row["source"] == "real_station" else "green"
        folium.Marker(
            location=[row["lat"], row["lon"]],
            icon=folium.Icon(color=color, icon="home"),
            popup=f"site {row['site_id']} ({row['source']})",
        ).add_to(m)

    m.save(out_path)
    print(f"Saved sanity-check map to {out_path}")


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} cleaned incidents")

    demand = build_demand_points(df)
    print(f"\nBuilt {len(demand)} demand points")
    print(f"Weight range: {demand['weight'].min()} to {demand['weight'].max()}")
    print(f"Total incidents represented: {demand['weight'].sum()} (should equal {len(df)})")

    sites = build_candidate_sites(df)
    print(f"\nBuilt {len(sites)} candidate sites "
          f"({(sites['source'] == 'real_station').sum()} real, "
          f"{(sites['source'] == 'candidate_grid').sum()} grid-derived)")

    demand.to_csv(DEMAND_OUT, index=False)
    sites.to_csv(SITES_OUT, index=False)
    print(f"\nSaved {DEMAND_OUT}")
    print(f"Saved {SITES_OUT}")

    plot_sanity_check(demand, sites)


if __name__ == "__main__":
    main()