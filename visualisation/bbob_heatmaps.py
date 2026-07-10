"""
Appendix figure: heatmaps of the BBOB test functions (refutation chapter
appendix).

Paired experiment: the refute_bbob campaigns (the functions those campaigns
attack). No result data needed — the suite is analytic, so this renders
straight from experiments/bbob_functions.make_suite.

One row of clean landscape heatmaps, global optimum marked (+). Contour lines
kept subtle so the multimodality (rastrigin/schwefel/gallagher) reads clearly.

    python -m visualisation.bbob_heatmaps
    python -m visualisation.bbob_heatmaps --functions rastrigin,schwefel --doc
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from visualisation.common import PUB_RC, save, base_parser
from src.plotting.refuter_search import _function_heatmap

DOC_NAME = "bbob_landscapes.png"


def make_figure(functions, optimum=True):
    with plt.rc_context(PUB_RC):
        n = len(functions)
        fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4),
                                 constrained_layout=True, squeeze=False)
        for ax, fn in zip(axes[0], functions):
            _function_heatmap(ax, fn)
            if optimum:
                xo = np.asarray(fn.x_opt)
                ax.plot(xo[0], xo[1], "+", color="lime", ms=12, mew=2.5, zorder=6)
            ax.set_title(fn.name)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("BBOB landscapes" + ("  (+ global optimum)" if optimum else ""),
                     fontsize=13)
    return fig


def main():
    p = base_parser("BBOB function heatmaps (appendix)")
    p.add_argument("--functions", default=None, help="comma list (default: whole suite)")
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--no_optimum", action="store_true")
    a = p.parse_args()

    from experiments.bbob_functions import make_suite
    suite = make_suite(a.dim)
    pick = a.functions.split(",") if a.functions else None
    functions = [f for f in suite if pick is None or f.name in pick]

    fig = make_figure(functions, optimum=not a.no_optimum)
    save(fig, "bbob_heatmaps", "bbob_landscapes", DOC_NAME if a.doc else None)


if __name__ == "__main__":
    main()
