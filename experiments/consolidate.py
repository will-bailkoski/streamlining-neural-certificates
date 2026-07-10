"""
Consolidate every campaign's compiled.csv into one queryable table.

Results live per campaign as  results/<campaign>/<tag>/compiled.csv  (and cluster runs
produce the same layout under a synced tree). Each row already carries a deterministic
`run_id` (a hash of the run's args), so this globs every compiled.csv, tags each row
with its campaign / tag / source file, UNIONS the heterogeneous columns (engines record
different stats), DE-DUPLICATES by run_id (a run present both locally and from the
cluster appears once), and writes one results/all_results.csv for analysis and plots.

Local + cluster: bring the cluster results in first, then point --roots at both, e.g.
    rsync -av user@cluster:/path/to/project/results/  results_cluster/
    python -m experiments.consolidate --roots results results_cluster

    python -m experiments.consolidate                       # -> results/all_results.csv
    python -m experiments.consolidate --glob "**/compiled.csv" "**/results.csv"
    python -m experiments.consolidate --out results/all_results.csv
"""
from __future__ import annotations
import argparse
import glob
import os

import pandas as pd

# readable leading columns when present; everything else follows
_LEAD = ["campaign", "tag", "engine", "env", "seed", "refuter", "cert_structure",
         "noise_disc", "verdict", "rounds", "total_time", "verifier_time",
         "refuter_time", "n_verifier_calls", "run_id"]


def consolidate(roots, patterns, out):
    frames = []
    for root in roots:
        if not os.path.isdir(root):
            print(f"  (root not found: {root})")
            continue
        seen = set()
        for pat in patterns:
            for csv in sorted(glob.glob(os.path.join(root, pat), recursive=True)):
                if csv in seen:
                    continue
                seen.add(csv)
                try:
                    df = pd.read_csv(csv)
                except Exception as e:  # empty / malformed file -> skip, don't abort
                    print(f"  skip {csv}: {e}")
                    continue
                if df.empty:
                    continue
                rel = os.path.relpath(csv, root).split(os.sep)
                df["campaign"] = rel[0]
                df["tag"] = rel[1] if len(rel) > 2 else ""
                df["source_root"] = root
                df["source_csv"] = csv
                frames.append(df)
                print(f"  + {csv:<60} {len(df):>4} rows  {df.shape[1]:>3} cols")

    if not frames:
        print("no result CSVs found")
        return None

    master = pd.concat(frames, ignore_index=True, sort=False)   # union of columns, NaN-filled
    n_raw = len(master)
    if "run_id" in master.columns:
        # keep the fullest row per run_id (fewest NaNs), so a richer local run beats a
        # sparse cluster copy (or vice-versa) rather than an arbitrary "first".
        master = (master.assign(_nan=master.isna().sum(axis=1))
                        .sort_values("_nan")
                        .drop_duplicates(subset="run_id", keep="first")
                        .drop(columns="_nan"))

    lead = [c for c in _LEAD if c in master.columns]
    master = master[lead + [c for c in master.columns if c not in lead]]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    master.to_csv(out, index=False)

    print(f"\nconsolidated {len(frames)} files: {n_raw} rows -> {len(master)} unique -> {out}")
    if {"campaign", "verdict"} <= set(master.columns):
        print("\nverdicts by campaign:")
        tbl = master.groupby(["campaign", "verdict"]).size().unstack(fill_value=0)
        print(tbl.to_string())
    return master


def main():
    p = argparse.ArgumentParser(description="Merge every campaign's compiled.csv into one table")
    p.add_argument("--roots", nargs="+", default=["results"],
                   help="result trees to scan (add your synced cluster tree here)")
    p.add_argument("--glob", dest="patterns", nargs="+", default=["**/compiled.csv"],
                   help="glob(s) under each root (e.g. also **/results.csv)")
    p.add_argument("--out", default="results/all_results.csv")
    a = p.parse_args()
    consolidate(a.roots, a.patterns, a.out)


if __name__ == "__main__":
    main()
