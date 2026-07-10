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
    "lirpa_bench": ("method", ["time", "total_boxes", "peak_boxes"]),
    "gpu_check": ("host", ["torch_pass", "jax_pass", "gpu_speedup_torch",
                           "gpu_speedup_jax", "passed"]),
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


def _feasibility_summary(csv_path: Path) -> None:
    """Per-env frontier: did anything verify, and the easiest config that did."""
    rows = list(csv.DictReader(open(csv_path, newline="")))
    if not rows:
        return
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(str(r.get("env", "?")), []).append(r)

    def fnum(r, k, default):
        try:
            return float(r.get(k))
        except (TypeError, ValueError):
            return default

    def neurons(r):
        return sum(int(x) for x in str(r.get("hidden_layers", "")).split(",") if x.strip().isdigit())

    print("\n  feasibility frontier (verified configs per env):")
    for env, rs in sorted(groups.items()):
        ver = [r for r in rs if r.get("verdict") == "verified"]
        if not ver:
            print(f"    {env:<14} NONE verified ({len(rs)} tried) — widen epsilon / neurons / caps")
            continue
        fastest = min(ver, key=lambda r: fnum(r, "total_time", float("inf")))
        print(f"    {env:<14} VERIFIED  max_eps={max(fnum(r,'epsilon',0) for r in ver):g}  "
              f"min_neurons={min(neurons(r) for r in ver)}  "
              f"min_noise_disc={min(int(fnum(r,'noise_disc',1)) for r in ver)}  "
              f"fastest={fnum(fastest,'total_time',0):.1f}s "
              f"(eps={fastest.get('epsilon')}, hidden={fastest.get('hidden_layers')}, "
              f"disc={fastest.get('noise_disc')})")


def main():
    ap = argparse.ArgumentParser(description="Compile a campaign -> compiled.csv + summary")
    ap.add_argument("campaign_dir", help="results/<campaign_title>/<tag>/ (holds runs/)")
    ap.add_argument("--iters", action="store_true",
                    help="one row per iterations.csv entry (auto for lirpa_bench)")
    a = ap.parse_args()

    cdir = Path(a.campaign_dir)
    if not (cdir / "runs").is_dir():
        raise SystemExit(f"{cdir} has no runs/ — point at a campaign tag directory")

    title = cdir.parent.name  # results/<title>/<tag> -> <title>
    group_key, metrics = _kind(title)
    iters = a.iters or title.startswith("lirpa_bench")

    out = compile_campaign(cdir, iters=iters)
    if out.exists():
        if title.startswith("feasibility"):
            _feasibility_summary(out)
        else:
            _summary(out, group_key, metrics)


if __name__ == "__main__":
    main()
