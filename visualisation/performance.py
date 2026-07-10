"""
Performance-comparison figures from compiled.csv files.

Point it at one or more campaign dirs / compiled.csv files (or a parent like
results/ that holds several campaigns); it concatenates them and draws the
comparison for the detected experiment kind, written to
results/figures/performance/:

  * cegis_verify   verifier wall-time per engine (box) + verdict mix per engine
  * refute_bbob    final value per refuter (box) + wall-time per refuter (box)
  * refute_certs   speedup over the verifier per refuter (box) + refuter-vs-
                   verifier time scatter
  * cegis_improve  total time per refuter (box) + speed-vs-sample-efficiency
                   scatter (counterexamples vs total time)

    python -m visualisation.performance results/cegis_verify_smt/<tag>
    python -m visualisation.performance results          # every campaign found

Stdlib + numpy + matplotlib only (no pandas dependency).
"""

from __future__ import annotations
import argparse
import csv
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.plotting.systems import PUB_RC


def _gather(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        p = Path(path)
        csvs = [p] if p.suffix == ".csv" else sorted(p.glob("**/compiled.csv"))
        for c in csvs:
            with open(c, newline="") as f:
                rows.extend(csv.DictReader(f))
    return rows


def _num(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r.get(key)))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.asarray(out)


def _groups(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[r.get(key, "?")].append(r)
    return g


def _box_by(ax, rows, group_key, metric, ylabel):
    g = _groups(rows, group_key)
    labels = sorted(g)
    data = [_num(g[k], metric)[~np.isnan(_num(g[k], metric))] for k in labels]
    data = [d if len(d) else np.array([np.nan]) for d in data]
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(group_key)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def _verdict_bar(ax, rows, group_key):
    g = _groups(rows, group_key)
    labels = sorted(g)
    verdicts = sorted({r.get("verdict", "?") for r in rows})
    bottoms = np.zeros(len(labels))
    for v in verdicts:
        counts = np.array([sum(1 for r in g[k] if r.get("verdict") == v) for k in labels],
                          dtype=float)
        ax.bar(labels, counts, bottom=bottoms, label=v)
        bottoms += counts
    ax.set_ylabel("runs")
    ax.legend(fontsize=7, title="verdict")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def _scatter(ax, rows, xk, yk, hue, xlabel, ylabel):
    for k in sorted(_groups(rows, hue)):
        rs = _groups(rows, hue)[k]
        ax.scatter(_num(rs, xk), _num(rs, yk), s=18, alpha=0.7, label=k)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, title=hue)


def _kind(rows) -> str:
    exps = {r.get("experiment") for r in rows}
    if "refute_cert" in exps:
        return "refute_certs"
    if "refute_bbob" in exps:
        return "refute_bbob"
    if "cegis" in exps:
        refuters = {r.get("refuter") for r in rows}
        return "cegis_improve" if refuters - {"none", None, ""} else "cegis_verify"
    return "unknown"


def _figure(rows, kind, out: Path):
    with plt.rc_context(PUB_RC):
        fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
        if kind == "cegis_verify":
            _box_by(a, rows, "engine", "total_time", "total time (s)")
            _verdict_bar(b, rows, "engine")
        elif kind == "refute_bbob":
            _box_by(a, rows, "refuter", "final_value", "final value")
            _box_by(b, rows, "refuter", "time", "wall-clock (s)")
        elif kind == "refute_certs":
            _box_by(a, rows, "refuter", "speedup", "speedup over verifier (x)")
            a.axhline(1.0, color="0.4", ls="--", lw=0.8)
            _scatter(b, rows, "verifier_time", "refuter_time", "refuter",
                     "verifier time (s)", "refuter time (s)")
        elif kind == "cegis_improve":
            _box_by(a, rows, "refuter", "total_time", "total time (s)")
            _scatter(b, rows, "n_counterexamples", "total_time", "refuter",
                     "counterexamples (sample cost)", "total time (s)")
        else:
            a.text(0.5, 0.5, "unknown campaign kind", ha="center", va="center")
        fig.suptitle(f"{kind} — performance", fontsize=13)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{kind}.pdf")
        fig.savefig(out / f"{kind}.png")
        plt.close(fig)
        print(f"  -> {kind}.pdf / .png  ({len(rows)} rows)")


def main():
    p = argparse.ArgumentParser(description="Performance figures from compiled.csv files")
    p.add_argument("paths", nargs="+", help="campaign dirs / compiled.csv files / a parent dir")
    p.add_argument("--out", default="results/figures/performance")
    a = p.parse_args()

    rows = _gather(a.paths)
    if not rows:
        raise SystemExit("no compiled.csv rows found — run a campaign's --aggregate first")
    kind = _kind(rows)
    print(f"{kind}: {len(rows)} rows")
    _figure(rows, kind, Path(a.out))


if __name__ == "__main__":
    main()
