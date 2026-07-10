"""
Thesis figure: refuter quality vs wall-clock cost at best hyperparameters
(refutation chapter, Figures/bbob_quality_vs_time.png).

Paired experiment: experiments/refute_bbob/* launchers (campaigns refute_bbob_<r>)
Data read:         results/refute_bbob_<refuter>/<tag>/compiled.csv (all refuters)

Log-log scatter: one point per (refuter, function) at the refuter's best HPs —
x = mean wall time per search, y = mean normalised regret. Colour = refuter,
marker = function. The refuter we want inside CEGIS lives in the bottom-left
corner: near-zero regret at millisecond cost.

    python -m visualisation.bbob_quality_vs_time
    python -m visualisation.bbob_quality_vs_time --tag hpsweep --doc
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from visualisation.common import (PUB_RC, REFUTER_ORDER, REFUTER_COLOR,
                                  FUNCTION_MARKERS, save, base_parser)
from visualisation._bbob import load_rows, normalised_regret, best_hp_rows

DOC_NAME = "bbob_quality_vs_time.png"


def make_figure(best: dict[str, list[dict]]):
    functions = sorted({r["function"] for rows in best.values() for r in rows})
    refuters = [r for r in REFUTER_ORDER if r in best] + sorted(set(best) - set(REFUTER_ORDER))
    marker = {fn: FUNCTION_MARKERS[i % len(FUNCTION_MARKERS)] for i, fn in enumerate(functions)}

    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(7.6, 5.4), constrained_layout=True)
        for name in refuters:
            for fn in functions:
                rs = [r for r in best[name] if r["function"] == fn]
                if not rs:
                    continue
                t = np.mean([float(r["time"]) for r in rs])
                q = np.mean([r["nregret"] for r in rs])
                ax.scatter(t, q, s=64, marker=marker[fn],
                           color=REFUTER_COLOR.get(name, "#777"),
                           edgecolor="k", linewidths=0.5, alpha=0.9, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.2)
        ax.set_xlabel("wall time per search (s, log)")
        ax.set_ylabel("normalised regret (log; lower = better)")
        ax.set_title("Quality vs time at best HPs (colour = refuter, marker = function)",
                     fontsize=12)
        leg1 = ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=8,
                                         color=REFUTER_COLOR.get(n, "#777"), label=n)
                                  for n in refuters],
                         title="refuter", loc="upper left",
                         bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=8.5)
        ax.add_artist(leg1)
        ax.legend(handles=[Line2D([], [], marker=marker[fn], ls="", ms=8,
                                  color="0.55", label=fn) for fn in functions],
                  title="function", loc="lower left",
                  bbox_to_anchor=(1.01, 0.0), borderaxespad=0.0, fontsize=8.5)
    return fig


def main():
    p = base_parser("Refuter quality vs wall-clock at best HPs")
    p.add_argument("--tag", default=None, help="campaign tag (default: latest per refuter)")
    a = p.parse_args()

    rows = load_rows(a.tag)
    if not rows:
        raise SystemExit("no refute_bbob_* compiled.csv rows found — run the sweep first")
    normalised_regret(rows)
    best = best_hp_rows(rows)
    fig = make_figure(best)
    save(fig, "bbob_quality", "bbob_quality_vs_time", DOC_NAME if a.doc else None)


if __name__ == "__main__":
    main()
