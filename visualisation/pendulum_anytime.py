"""
Thesis figures: anytime tightness-vs-cost of the bound-propagation methods on
verified pendulum certificates (verification chapter).

Paired experiment: experiments/pendulum_anytime.py  (campaign "pendulum_anytime")
Data read:         results/pendulum_anytime/results.json

The certificates are VERIFIED supermartingales, so each method's sound drift
upper bound must eventually cross <= 0; what differs is the price. Two
figures, same traces:

  A  bound vs BOXES BOUNDED (compute): how much partitioning each relaxation
     needs — IBP splits thousands of boxes where alpha-CROWN needs dozens.
  B  bound vs WALL TIME: the practical ranking, which INVERTS the per-box
     ranking (alpha-CROWN's per-box optimisation is orders slower).

    python -m visualisation.pendulum_anytime
    python -m visualisation.pendulum_anytime --doc
"""

from __future__ import annotations
import json

import matplotlib.pyplot as plt

from visualisation.common import PUB_RC, ENGINE_COLOR, campaign_dir, save, base_parser

CAMPAIGN = "pendulum_anytime"      # == experiments.pendulum_anytime.CAMPAIGN
DOC_NAME_BOXES = "pendulum_anytime_boxes.png"
DOC_NAME_TIME = "pendulum_anytime_time.png"


def load_results() -> dict:
    path = campaign_dir(CAMPAIGN) / "results.json"
    if not path.exists():
        raise SystemExit(f"{path} not found — run `python -m experiments.pendulum_anytime` first")
    return json.loads(path.read_text())


def _figure(results: dict, x_of, xlabel: str, title: str):
    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
        seen = set()
        for r in results["runs"]:
            tr = r["trace"]
            if not tr:
                continue
            xs = [x_of(p) for p in tr]
            ys = [p[2] for p in tr]
            c = ENGINE_COLOR.get(r["method"], "#444")
            lbl = r["method"] if r["method"] not in seen else None
            seen.add(r["method"])
            ax.plot(xs, ys, "-o", color=c, ms=3.5, lw=1.5, alpha=0.85, label=lbl)
        ax.axhline(0.0, ls="--", color="k", lw=1, alpha=0.6)
        ax.text(0.02, 0.02, r"verified once bound $\leq 0$", transform=ax.transAxes,
                fontsize=8, va="bottom")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("sound drift upper bound")
        nd = results["meta"].get("noise_disc")
        ax.set_title(f"{title}\n(pendulum_lqr, verified certificates, noise_disc={nd})",
                     fontsize=11)
        ax.grid(True, which="both", alpha=0.22)
        ax.legend(title="bound method", fontsize=9)
    return fig


def main():
    p = base_parser("Anytime bound-prop tightness figures (pendulum)")
    a = p.parse_args()
    results = load_results()

    fig = _figure(results, lambda p_: max(p_[0], 1), "boxes bounded (log)",
                  "Tightness vs compute: loose methods buy soundness with splits")
    save(fig, "pendulum_anytime", "pendulum_anytime_boxes", DOC_NAME_BOXES if a.doc else None)

    fig = _figure(results, lambda p_: max(p_[1], 1e-3), "wall time (s, log)",
                  "Tightness vs time: the per-box cost inverts the ranking")
    save(fig, "pendulum_anytime", "pendulum_anytime_time", DOC_NAME_TIME if a.doc else None)


if __name__ == "__main__":
    main()
