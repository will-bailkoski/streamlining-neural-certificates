"""
Compile one campaign's runs into compiled.csv + print a comparison summary.

    python -m results.analysis.compile results/cegis_verify_smt/<tag>
    python -m results.analysis.compile results/refute_bbob_whale/<tag>

Takes a campaign tag directory (the one holding runs/), walks every run's
manifest + stats, and writes `compiled.csv` IN that directory (overwrite, so it
can be re-run as results stream in). The campaign kind is inferred from the title
(the parent directory name) to pick the comparison axis and the metrics shown in
the console summary. Stdlib only.
"""

from __future__ import annotations
import argparse
import csv
import statistics
from pathlib import Path

from experiments.common.array import compile_campaign

# title prefix -> (group-by column, metric columns to summarise as medians)
_KINDS = {
    "cegis_verify": ("env", ["total_time", "verifier_time", "rounds"]),
    "cegis_improve": ("refuter", ["total_time", "verifier_time", "refuter_time",
                                  "n_counterexamples", "rounds"]),
    "refute_bbob": ("function", ["final_value", "time"]),
    "refute_certs": ("refuter", ["refuter_time", "verifier_time", "speedup"]),
}


def _kind(title: str) -> tuple[str, list[str]]:
    for prefix, spec in _KINDS.items():
        if title.startswith(prefix):
            return spec
    return ("experiment", ["total_time", "time"])  # generic fallback


def _summary(csv_path: Path, group_key: str, metrics: list[str]) -> None:
    rows = list(csv.DictReader(open(csv_path, newline="")))
    if not rows:
        return
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(str(r.get(group_key, "?")), []).append(r)
    cols = [m for m in metrics if any(r.get(m) not in (None, "") for r in rows)]
    print(f"\n  median by {group_key}:")
    for g, rs in sorted(groups.items()):
        line = f"    {group_key}={g:<14} n={len(rs):<3}"
        for m in cols:
            vals = []
            for r in rs:
                v = r.get(m)
                if v in (None, "", "nan"):
                    continue
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
            if vals:
                line += f"  {m}~{statistics.median(vals):.4g}"
        print(line)


def main():
    ap = argparse.ArgumentParser(description="Compile a campaign -> compiled.csv + summary")
    ap.add_argument("campaign_dir", help="results/<campaign_title>/<tag>/ (holds runs/)")
    ap.add_argument("--iters", action="store_true",
                    help="one row per iterations.csv entry")
    a = ap.parse_args()

    cdir = Path(a.campaign_dir)
    if not (cdir / "runs").is_dir():
        raise SystemExit(f"{cdir} has no runs/ — point at a campaign tag directory")

    title = cdir.parent.name  # results/<title>/<tag> -> <title>
    group_key, metrics = _kind(title)
    iters = a.iters

    out = compile_campaign(cdir, iters=iters)
    if out.exists():
        _summary(out, group_key, metrics)


if __name__ == "__main__":
    main()
