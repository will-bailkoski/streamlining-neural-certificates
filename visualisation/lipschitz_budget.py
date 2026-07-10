"""
Thesis figures: the epsilon-L_V budget story (Lipschitz chapter).

Paired experiment: experiments/lipschitz_budget.py  (campaign "lipschitz_budget")
Data read:         results/lipschitz_budget/results_seed<N>.json

One file because the three views tell one story — "the certifiable drift margin
is a budget set by the dynamics":

  E1 budget curve : E_w||x'-x|| vs radius; the floor delta_min IS the budget.
  E2 eps frontier : demanding a bigger fixed margin only inflates L_V; the
                    achieved margin hugs the ceiling delta_min * L_V.
  E3 profiles     : how each decrease condition SPENDS the budget over the
                    domain — fixed eps is flat (wasteful), adaptive tracks E[d].

Default output is the combined 3-panel figure; --separate re-emits the three
standalone panels (the thesis includes the budget curve on its own as
Figures/lipschitz_budget_curve.png).

    python -m visualisation.lipschitz_budget
    python -m visualisation.lipschitz_budget --separate --doc
"""

from __future__ import annotations
import json

import numpy as np
import matplotlib.pyplot as plt

from visualisation.common import (PUB_RC, VIOLET, CORAL, GREEN, MUT, INK,
                                  campaign_dir, save, base_parser)

CAMPAIGN = "lipschitz_budget"          # == experiments.lipschitz_budget.CAMPAIGN_TITLE
DOC_NAME_CURVE = "lipschitz_budget_curve.png"


def load_report(seed: int) -> dict:
    path = campaign_dir(CAMPAIGN) / f"results_seed{seed}.json"
    if not path.exists():
        raise SystemExit(f"{path} not found — run `python -m experiments.lipschitz_budget"
                         f" --seed {seed}` first")
    return json.loads(path.read_text())


# ----------------------------------------------------------------------
# panels (each draws into an ax so combined + separate reuse them)
# ----------------------------------------------------------------------
def panel_curve(ax, rep):
    for name, col in zip(rep["E1"], (VIOLET, CORAL)):
        d = rep["E1"][name]
        ax.plot(d["radius"], d["Ed"], "-", color=col, lw=2.4,
                label=f"{name}  ($L_f$={d['lip_f']:.2f})")
        ax.axhline(d["delta_min"], color=col, ls="--", lw=1.0, alpha=0.55)
        ax.axvline(d["eq_r"], color=col, ls=":", lw=1.1, alpha=0.7)
    ax.set_xlabel(r"radius $\|x\|$   (equilibrium $\to$ domain edge)")
    ax.set_ylabel(r"$\mathbb{E}_w\,\|x'-x\|$   (observed displacement)")
    ax.set_title("E1 — the budget is set by the dynamics", fontsize=10.5)
    ax.text(0.98, 0.05, r"dashed $=$ displacement floor   dotted $=$ eq radius",
            transform=ax.transAxes, fontsize=8, color=MUT, ha="right")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")


def panel_frontier(ax, rep):
    e2 = rep["E2"]
    dm, pts = e2["delta_min"], e2["points"]
    Lv = np.array([p["L_V"] for p in pts])
    eps = np.array([p["eps"] for p in pts])
    ach = np.array([p["achieved"] for p in pts])
    xx = np.linspace(0, Lv.max() * 1.05, 50)
    ax.plot(xx, dm * xx, "-", color=INK, lw=1.7, label=r"ceiling $\delta_{\min}L_V$")
    ax.fill_between(xx, dm * xx, (dm * xx.max()) * 1.4, color=CORAL, alpha=0.05)
    ax.plot(Lv, eps, "o", mfc="none", mec=MUT, ms=7, label=r"demanded $\varepsilon$")
    ax.scatter(Lv, ach, color=GREEN, s=44, zorder=5, label="achieved margin")
    for p in pts:
        ax.annotate(rf"{p['eps']:g}", (p["L_V"], p["eps"]), fontsize=7,
                    textcoords="offset points", xytext=(5, 3), color=MUT)
        if not p["converged"]:
            ax.annotate("cliff", (p["L_V"], p["eps"]), fontsize=7.5, color=CORAL,
                        textcoords="offset points", xytext=(4, -12))
    ax.set_xlabel(r"certificate Lipschitz constant $L_V$  (LipBaB)")
    ax.set_ylabel("decrease margin")
    ax.set_title(r"E2 — more $\varepsilon$ only buys a steeper $V$", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")


def panel_profiles(ax, rep):
    P = rep["E3"]
    r, Ed = np.array(P["radius"]), np.array(P["Ed"])
    ax.plot(r, Ed, "-", color=INK, lw=2.2,
            label=r"ceiling $\mathbb{E}[d]$ (per unit $L_V$)")
    styles = dict(fixed=(CORAL, r"fixed $\varepsilon$"),
                  geometric=(VIOLET, r"geometric $\rho V$"),
                  adaptive=(GREEN, r"adaptive $\gamma\,\mathbb{E}[d]$"))
    norm = {}
    for m, (col, lab) in styles.items():
        norm[m] = np.array(P[m]["demand"]) / P[m]["L_V"]
        ax.plot(r, norm[m], "-", color=col, lw=2.3, label=f"{lab} (mp={P[m]['mp']:g})")
    ax.fill_between(r, norm["fixed"], Ed, color=CORAL, alpha=0.06)
    ax.text(0.62, 0.5, "wasted budget", transform=ax.transAxes,
            color="#993C1D", fontsize=8.5, ha="center")
    ax.set_xlabel(r"radius $\|x\|$   (equilibrium $\to$ domain edge)")
    ax.set_ylabel(r"demanded margin / $L_V$")
    ax.set_title("E3 — fixed is flat, adaptive tracks the dynamics", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


_PANELS = {"curve": panel_curve, "frontier": panel_frontier, "profiles": panel_profiles}
_SINGLE_SIZE = {"curve": (6.0, 3.9), "frontier": (6.0, 3.9), "profiles": (6.4, 4.0)}


def main():
    p = base_parser("epsilon-L_V budget figures (reads the lipschitz_budget campaign)")
    p.add_argument("--seed", type=int, default=0, help="which results_seed<N>.json")
    p.add_argument("--separate", action="store_true",
                   help="also emit the three standalone panels")
    a = p.parse_args()
    rep = load_report(a.seed)

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1), constrained_layout=True)
        for ax, draw in zip(axes, _PANELS.values()):
            draw(ax, rep)
        fig.suptitle(r"The certifiable margin is a budget: "
                     r"$\varepsilon \leq L_V\,\delta_{\min}$", fontsize=13)
        save(fig, "lipschitz_budget", "lipschitz_budget_story")

        if a.separate or a.doc:
            for name, draw in _PANELS.items():
                f, ax = plt.subplots(figsize=_SINGLE_SIZE[name], constrained_layout=True)
                draw(ax, rep)
                doc = DOC_NAME_CURVE if (a.doc and name == "curve") else None
                save(f, "lipschitz_budget", f"lipschitz_{name}", doc)


if __name__ == "__main__":
    main()
