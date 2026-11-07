"""
merge.py
--------
Reads all the manually-saved page_0.json ... page_16.json files and
stitches them into one clean CSV. No network calls.

Run from the folder containing your page_*.json files:
    python merge.py
"""

import glob
import json
import re
import pandas as pd

PAGE_FOLDER = "."          # change if your pages are in a subfolder
OUTPUT_CSV = "../data/incidents_raw.csv"


def natural_key(path):
    """Sorts page_0, page_1, ..., page_16 in numeric order, not string
    order (string order would put page_10 before page_2)."""
    m = re.search(r"page_(\d+)\.json", path)
    return int(m.group(1)) if m else -1


def main():
    files = sorted(glob.glob(f"{PAGE_FOLDER}/page_*.json"), key=natural_key)
    print(f"Found {len(files)} page files:")
    for f in files:
        print(" ", f)

    all_rows = []
    seen_offsets = set()

    for path in files:
        n = natural_key(path)
        if n in seen_offsets:
            print(f"WARNING: duplicate page number {n} -- skipping {path}")
            continue
        seen_offsets.add(n)

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        feats = data.get("features", [])
        print(f"  {path}: {len(feats)} records")

        if len(feats) != 2000 and n != max(seen_offsets):
            print(f"    NOTE: this page has {len(feats)} records, not 2000 -- "
                  f"only expected on the LAST page. If this isn't the last "
                  f"page, you may be missing data.")

        for f in feats:
            row = f.get("properties", {})
            geom = f.get("geometry")
            if geom and geom.get("coordinates"):
                row["lon"], row["lat"] = geom["coordinates"][:2]
            else:
                row["lon"], row["lat"] = None, None
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    print(f"\nTotal records before dedup: {len(df)}")

    id_col = "OBJECTID" if "OBJECTID" in df.columns else None
    if id_col:
        before = len(df)
        df = df.drop_duplicates(subset=[id_col])
        if len(df) != before:
            print(f"Dropped {before - len(df)} duplicate rows based on {id_col}")
    else:
        print("Could not find an OBJECTID column to dedupe on.")

    print(f"Total records after dedup: {len(df)}")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()