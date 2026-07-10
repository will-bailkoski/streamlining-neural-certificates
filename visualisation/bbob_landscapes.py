"""
Publication figures of the BBOB benchmark functions.

Two views, written to results/figures/bbob/:

  * landscapes   a clean heatmap of each function, optionally with the global
                 optimum marked (+)
  * search       a refuters x functions grid of evaluation patterns — how each
                 optimiser explored the landscape (points / agent trajectories /
                 DIRECT boxes), with the optimum (+) and the incumbent (*)

    python -m visualisation.bbob_landscapes
    python -m visualisation.bbob_landscapes --refuters whale,direct --functions ackley
    python -m visualisation.bbob_landscapes --no_optimum

Reuses src/plotting/refuter_search (the search-pattern renderer that runs each
refuter with record=True) so the eval-pattern figures match the experiments.
"""

from __future__ import annotations
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.bbob_functions import make_suite
from src.refuters.base import DIRECTORY as REFUTERS
import src.refuters  # noqa: F401  (register refuters)
from src.plotting.systems import PUB_RC
from src.plotting.refuter_search import plot_grid, _function_heatmap


def draw_landscapes(functions, out: Path, optimum: bool = True):
    with plt.rc_context(PUB_RC):
        n = len(functions)
        fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2),
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
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "landscapes.pdf")
        fig.savefig(out / "landscapes.png")
        plt.close(fig)
        print(f"  -> landscapes.pdf / .png")


def main():
    p = argparse.ArgumentParser(description="BBOB landscape + search-pattern figures")
    p.add_argument("--refuters", default=",".join(sorted(REFUTERS)),
                   help="comma list of refuters for the search grid")
    p.add_argument("--functions", default=None, help="comma list of functions (default all)")
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--budget", type=int, default=8192)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--no_optimum", action="store_true", help="omit the optimum marker")
    p.add_argument("--out", default="results/figures/bbob")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    suite = make_suite(a.dim)
    pick = [f for f in a.functions.split(",")] if a.functions else None
    functions = [f for f in suite if pick is None or f.name in pick]
    refuters = [r for r in a.refuters.split(",") if r]
    out = Path(a.out)

    draw_landscapes(functions, out, optimum=not a.no_optimum)

    fig = plot_grid(refuters, functions, budget=a.budget, batch_size=a.batch_size,
                    seed=a.seed, out_path=str(out / "search_patterns.pdf"))
    fig.savefig(out / "search_patterns.png")
    plt.close(fig)
    print(f"  -> search_patterns.pdf / .png")
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
