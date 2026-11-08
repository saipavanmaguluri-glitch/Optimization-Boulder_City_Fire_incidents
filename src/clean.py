"""
clean.py
--------
Cleans data/incidents_raw.csv (from merge.py) into data/incidents_clean.csv.

What it does:
  1. Converts RESPONSEDATE from milliseconds to a real datetime.
  2. Drops rows with missing lat/lon.
  3. Finds and removes the "fallback" coordinate (Boulder CAD's dump
     point for un-geocodable incidents, inside the city bounding box).
  4. Drops rows with impossible/negative/zero elapsed times.
  5. Prints a summary so you can sanity-check before moving on.

Run: python clean.py
"""

import pandas as pd

INPUT_CSV = "../data/incidents_raw.csv"
OUTPUT_CSV = "../data/incidents_clean.csv"

ELAPSED_FIELD_CANDIDATES = [
    "ELAPSEDCALLRCVD2FIRSTASSIGNMIN",
    "ELAPSEDASSIGNED2FIRSTENROUTEMI",
    "ELAPSEDENROUTE2FIRSTATSCENEMIN",
    "ELAPSEDASSIGNED1STSCENEMIN",
]


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} raw rows")
    print(f"Columns: {list(df.columns)}\n")

    df["when"] = pd.to_datetime(df["RESPONSEDATE"], unit="ms", utc=True)
    df["when"] = df["when"].dt.tz_convert("America/Denver")

    before = len(df)
    df = df.dropna(subset=["lat", "lon"])
    print(f"Dropped {before - len(df)} rows with missing lat/lon")

    coord_counts = df.groupby(["lat", "lon"]).size().sort_values(ascending=False)
    print("\nTop 5 most frequent coordinates (the fallback point, if "
          "present, will stand out with a huge count):")
    print(coord_counts.head(5))

    if len(coord_counts) > 1 and coord_counts.iloc[0] > coord_counts.iloc[1] * 5:
        fallback_lat, fallback_lon = coord_counts.index[0]
        before = len(df)
        df = df[~((df["lat"] == fallback_lat) & (df["lon"] == fallback_lon))]
        print(f"\nDetected and removed fallback point at "
              f"({fallback_lat}, {fallback_lon}) -- {before - len(df)} rows")
    else:
        print("\nNo obvious fallback point detected automatically -- "
              "check the top-5 list above yourself.")

    present = [c for c in ELAPSED_FIELD_CANDIDATES if c in df.columns]
    print(f"\nElapsed-time fields found: {present}")
    for col in present:
        before = len(df)
        df = df[df[col].isna() | ((df[col] > 0) & (df[col] < 60))]
        dropped = before - len(df)
        if dropped:
            print(f"  {col}: dropped {dropped} rows with impossible values")

    print(f"\nFinal clean row count: {len(df)}")
    print(f"Date range: {df['when'].min()} to {df['when'].max()}")
    print(f"Lat range: {df['lat'].min()} to {df['lat'].max()}")
    print(f"Lon range: {df['lon'].min()} to {df['lon'].max()}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()