"""
Thesis figure: uniform-refinement cost of verification (Lipschitz chapter,
Figures/lipschitz_gridding.png).

Purely conceptual — no experiment data. Left: sound gridding needs cells of
width <= margin/L_D, so a drift-Lipschitz constant twice as loose halves the
cell size and doubles the grid per axis (shown without cell counts). Right: the
box count (Diam(X) * L_D / margin)^d on a log axis per dimension — a looser L_D
compounds as its d-th power, which is what motivates the joint-enclosure bound.

    python -m visualisation.lipschitz_gridding
    python -m visualisation.lipschitz_gridding --doc
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from visualisation.common import PUB_RC, GREEN, CORAL, MUT, INK, VIOLET, save, base_parser

DOC_NAME = "lipschitz_gridding.png"


def _schematic_grid(ax, n, color, title, sub):
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec=INK, lw=2.0))
    for i in range(1, n):
        ax.plot([i / n, i / n], [0, 1], color=color, lw=1.1)
        ax.plot([0, 1], [i / n, i / n], color=color, lw=1.1)
    ax.text(0.5, -0.12, title, ha="center", va="top", color=color, fontsize=10,
            transform=ax.transAxes)
    ax.text(0.5, -0.24, sub, ha="center", va="top", color=color, fontsize=10,
            transform=ax.transAxes)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")


def make_figure(margin: float, domain_width: float,
                dims=(1, 2, 3), coarse_n=4, fine_n=None):
    # "twice as loose L_D" -> cell size halves -> twice as many cells per axis
    if fine_n is None:
        fine_n = 2 * coarse_n
    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(10.5, 4.2), constrained_layout=True)
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 2.2])
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[0, 2])

        # left: the same domain gridded under a tight L_D vs one twice as loose
        # (no cell counts --- purely conceptual)
        _schematic_grid(ax_a, coarse_n, GREEN, r"tight $L_D$", "")
        _schematic_grid(ax_b, fine_n, CORAL, r"$2\times$ looser $L_D$", "")
        ax_a.set_title(r"cell size $\leq$ margin$/L_D$", fontsize=11, pad=14)
        ax_a.annotate("", xy=(1.42, 0.5), xytext=(1.08, 0.5),
                      xycoords="axes fraction",
                      arrowprops=dict(arrowstyle="->", color=MUT, lw=1.6))
        ax_a.text(1.25, 0.56, r"$L_D \to 2L_D$", ha="center", color=MUT, fontsize=10,
                  transform=ax_a.transAxes)

        # right: box count vs L_D per dimension
        Ls = np.linspace(0.2, 2.0, 200)
        greys = {1: "#9a9a94", 2: VIOLET, 3: INK}
        for d in dims:
            ax_c.plot(Ls, (domain_width * Ls / margin) ** d, color=greys.get(d, MUT),
                      lw=2.4, label=rf"$d={d}$")
        ax_c.text(0.03, 0.92,
                  r"$2\times$ looser $L_D \Rightarrow 2^{d}$ more boxes",
                  ha="left", va="top", fontsize=9.5, color=INK,
                  transform=ax_c.transAxes)
        ax_c.set_yscale("log")
        ax_c.set_xlabel(r"drift-Lipschitz constant $L_D$")
        ax_c.set_ylabel(r"uniform grid boxes  $(\mathrm{Diam}(\mathcal{X})\,L_D/\mathrm{margin})^{d}$")
        ax_c.set_title(r"Tightening $L_D$ compounds with dimension", fontsize=11)
        ax_c.legend(title="dimension", frameon=False, loc="lower right", fontsize=9)
        for s in ("top", "right"):
            ax_c.spines[s].set_visible(False)

        fig.suptitle("The Lipschitz constant dictates verification cost", fontsize=13)
    return fig


def main():
    p = base_parser("Uniform-refinement gridding cost figure (no data needed)")
    p.add_argument("--margin", type=float, default=0.01, help="demanded drift margin")
    p.add_argument("--domain_width", type=float, default=1.0, help="domain size D")
    a = p.parse_args()

    fig = make_figure(a.margin, a.domain_width)
    save(fig, "lipschitz_gridding", "lipschitz_gridding", DOC_NAME if a.doc else None)


if __name__ == "__main__":
    main()
