"""
Thesis figure: refuter quality per BBOB function at best hyperparameters
(refutation chapter, Figures/bbob_quality_per_function.png).

Paired experiment: experiments/refute_bbob/* launchers (campaigns refute_bbob_<r>)
Data read:         results/refute_bbob_<refuter>/<tag>/compiled.csv (all refuters)

Grouped bars of normalised regret (log; floored at 1e-7 = "solved") per
function, one colour per refuter, error bars = std over seeds. This is the
head-to-head evidence for which refuter to put inside the CEGIS loop.

    python -m visualisation.bbob_quality_per_function
    python -m visualisation.bbob_quality_per_function --tag hpsweep --doc
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from visualisation.common import (PUB_RC, REFUTER_ORDER, REFUTER_COLOR,
                                  save, base_parser)
from visualisation._bbob import (load_rows, normalised_regret, best_hp_rows,
                                 REGRET_FLOOR)

DOC_NAME = "bbob_quality_per_function.png"


def make_figure(best: dict[str, list[dict]]):
    functions = sorted({r["function"] for rows in best.values() for r in rows})
    refuters = [r for r in REFUTER_ORDER if r in best] + sorted(set(best) - set(REFUTER_ORDER))
    width = 0.8 / len(refuters)

    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(10.5, 4.4), constrained_layout=True)
        x = np.arange(len(functions))
        for i, name in enumerate(refuters):
            mean, std = [], []
            for fn in functions:
                v = np.array([r["nregret"] for r in best[name] if r["function"] == fn])
                mean.append(v.mean() if v.size else np.nan)
                std.append(v.std() if v.size else 0.0)
            pos = x + (i - (len(refuters) - 1) / 2) * width
            ax.bar(pos, mean, width * 0.94, yerr=std, color=REFUTER_COLOR.get(name, "#777"),
                   error_kw=dict(lw=0.9, capsize=0), label=name)
        budget = best[refuters[0]][0].get("budget", "?")
        ax.set_yscale("log")
        ax.set_ylim(bottom=REGRET_FLOOR * 0.6)
        ax.axhline(REGRET_FLOOR, color="0.6", lw=0.7, ls=":")
        ax.set_xticks(x, functions)
        ax.set_ylabel("normalised regret  (log; lower = better)")
        ax.set_title(f"Refuters at their best HPs, per BBOB function "
                     f"(budget {budget} evals)", fontsize=12)
        ax.legend(ncols=len(refuters), fontsize=8.5, loc="upper right", framealpha=0.9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    return fig


def main():
    p = base_parser("Per-function refuter quality at best HPs")
    p.add_argument("--tag", default=None, help="campaign tag (default: latest per refuter)")
    a = p.parse_args()

    rows = load_rows(a.tag)
    if not rows:
        raise SystemExit("no refute_bbob_* compiled.csv rows found — run the sweep first")
    normalised_regret(rows)
    best = best_hp_rows(rows)
    fig = make_figure(best)
    save(fig, "bbob_quality", "bbob_quality_per_function", DOC_NAME if a.doc else None)


if __name__ == "__main__":
    main()
