"""
Thesis figure: refuting faux certificates (refutation chapter, Sec.~fauxcert).

The faux certificates are the candidates a sound verifier rejected during the
CEGIS verification campaigns; each is stored with the verifier's own wall-clock
to a counterexample. Every faux cert is replayed with each BBOB-tuned refuter,
and we compare them to the sound verifier they stand in for on the task that
matters --- finding a genuine counterexample --- on reliability, cost, and the
quality of the counterexample returned.

Data read: results/refute_certs_*/<tag>/compiled.csv (one campaign per refuter)
  columns: refuter, env, verifier_time, refuter_time, found, best_obj, speedup

Two panels over the completed replays:
  left   fraction of faux certs each refuter actually refutes (reliability) ---
         whale near-perfect, DIRECT the outlier;
  right  per-refuter speed-up over the sound verifier (log) --- every refuter is
         one-to-three orders of magnitude cheaper than a sound call.

    python -m visualisation.faux_certs
    python -m visualisation.faux_certs --doc
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from visualisation.common import (PUB_RC, MUT, INK, REFUTER_ORDER, REFUTER_COLOR,
                                  glob_campaigns, load_compiled, fnum, save,
                                  base_parser)

DOC_NAME = "faux_certs.png"


def load_rows(tag: str | None):
    rows = []
    for title in glob_campaigns("refute_certs_"):
        rows.extend(load_compiled(title, tag))
    return [r for r in rows if r.get("status") in (None, "", "completed")]


def make_figure(rows: list[dict]):
    refs = [r for r in REFUTER_ORDER if any(x.get("refuter") == r for x in rows)]
    def found(r): return str(r.get("found")).lower() == "true"

    hit, spd, ncerts = {}, {}, {}
    for ref in refs:
        g = [r for r in rows if r.get("refuter") == ref]
        gf = [r for r in g if found(r)]
        ncerts[ref] = len(g)
        hit[ref] = 100.0 * len(gf) / max(len(g), 1)
        spd[ref] = [fnum(r, "speedup") for r in gf
                    if np.isfinite(fnum(r, "speedup"))]

    with plt.rc_context(PUB_RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2),
                                       constrained_layout=True)

        # left: hit rate (reliability)
        x = np.arange(len(refs))
        ax1.bar(x, [hit[r] for r in refs], 0.62,
                color=[REFUTER_COLOR[r] for r in refs])
        for xi, r in zip(x, refs):
            ax1.text(xi, hit[r] + 1.5, f"{hit[r]:.0f}\\%", ha="center",
                     va="bottom", fontsize=8.5)
        ax1.set_xticks(x, refs, rotation=20, ha="right")
        ax1.set_ylabel("faux certs refuted (\\%)")
        ax1.set_ylim(0, 108)
        ax1.set_title("reliability", fontsize=11)

        # right: speedup over the sound verifier (log)
        data = [spd[r] if spd[r] else [np.nan] for r in refs]
        bp = ax2.boxplot(data, tick_labels=refs, showfliers=False,
                         patch_artist=True, widths=0.6)
        for patch, r in zip(bp["boxes"], refs):
            patch.set(facecolor=REFUTER_COLOR[r], alpha=0.45)
        for med in bp["medians"]:
            med.set(color=INK, lw=1.3)
        ax2.set_yscale("log")
        ax2.axhline(1.0, color=MUT, ls="--", lw=0.8)
        ax2.set_ylabel(r"speed-up over sound verifier ($\times$)")
        plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")
        ax2.set_title("cost saving", fontsize=11)

        for ax in (ax1, ax2):
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        total = sum(ncerts.values())
        fig.suptitle(f"Refuting faux certificates: six refuters vs the sound "
                     f"verifier ({total} replays)", fontsize=13)
    return fig


def main():
    p = base_parser("Faux-certificate refutation: refuters vs sound verifier")
    p.add_argument("--tag", default=None, help="campaign tag (default: latest)")
    a = p.parse_args()
    rows = load_rows(a.tag)
    if not rows:
        raise SystemExit("no compiled rows under results/refute_certs_*/")
    fig = make_figure(rows)
    save(fig, "faux_certs", "faux_certs", DOC_NAME if a.doc else None)


if __name__ == "__main__":
    main()
