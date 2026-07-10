"""
Thesis figure: verifier-only CEGIS vs CEGIS + refuter pre-screening
(refutation / benchmarking chapter).

Paired experiments: the cegis_verify_* campaigns (refuter="none": every
counterexample costs a full sound-verifier call) and the cegis_improve_*
campaigns (a cheap refuter attacks each candidate first; the verifier is only
called when the refuter comes up empty).
Data read: results/cegis_verify_*/<tag>/compiled.csv and
           results/cegis_improve_*/<tag>/compiled.csv (compiled on demand from
           runs/ if missing).

Two panels over matched (engine, env, seed) runs:
  left   total wall time per pipeline, split into verifier / refuter /
         training+overhead stacks — the headline "what did the refuter buy";
  right  sound-verifier calls per pipeline — the mechanism: the refuter
         replaces expensive verifier counterexample rounds with cheap ones.

    python -m visualisation.cegis_pipelines
    python -m visualisation.cegis_pipelines --envs linstoch2D --doc
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from visualisation.common import (PUB_RC, GREEN, VIOLET, MUT, INK,
                                  glob_campaigns, load_compiled, fnum,
                                  save, base_parser)

DOC_NAME = "cegis_pipelines.png"


def load_rows(tag: str | None):
    rows = []
    for title in glob_campaigns("cegis_verify_") + glob_campaigns("cegis_improve_"):
        rows.extend(load_compiled(title, tag))
    return [r for r in rows if r.get("status") in (None, "", "completed")]


def pipeline_of(row: dict) -> str:
    ref = (row.get("refuter") or "none").strip() or "none"
    return "verifier only" if ref in ("none", "None") else f"+ {ref} refuter"


def make_figure(rows: list[dict], envs: list[str] | None):
    if envs:
        rows = [r for r in rows if r.get("env") in envs]
    pipes = sorted({pipeline_of(r) for r in rows})
    engines = sorted({r.get("engine", "?") for r in rows})
    label_of = lambda r: (f"{r.get('engine')}\n{pipeline_of(r)}"
                          if len(engines) > 1 else pipeline_of(r))
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(label_of(r), []).append(r)
    labels = sorted(groups)

    with plt.rc_context(PUB_RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.3), constrained_layout=True)

        # left: stacked mean times
        vt = [np.nanmean([fnum(r, "verifier_time") for r in groups[k]]) for k in labels]
        rt = [np.nanmean([fnum(r, "refuter_time") for r in groups[k]]) for k in labels]
        tt = [np.nanmean([fnum(r, "total_time") for r in groups[k]]) for k in labels]
        other = [max(t - v - rr, 0.0) for t, v, rr in zip(tt, vt, rt)]
        x = np.arange(len(labels))
        ax1.bar(x, vt, 0.6, color=INK, label="sound verifier")
        ax1.bar(x, rt, 0.6, bottom=vt, color=GREEN, label="refuter")
        ax1.bar(x, other, 0.6, bottom=np.array(vt) + np.array(rt), color="#cfcdc6",
                label="training + overhead")
        for xi, t, n in zip(x, tt, [len(groups[k]) for k in labels]):
            ax1.text(xi, t, f" {t:.0f}s\n (n={n})", ha="center", va="bottom", fontsize=8)
        ax1.set_xticks(x, labels)
        ax1.set_ylabel("mean wall time per run (s)")
        ax1.set_title("where the time goes", fontsize=11)
        ax1.legend(frameon=False, fontsize=8.5)

        # right: verifier calls per run (the mechanism)
        data = [[fnum(r, "n_verifier_calls") for r in groups[k]
                 if not np.isnan(fnum(r, "n_verifier_calls"))] for k in labels]
        bp = ax2.boxplot(data, labels=labels, showfliers=True, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set(facecolor=VIOLET, alpha=0.35)
        ax2.set_ylabel("sound-verifier calls per run")
        ax2.set_title("what the refuter replaces", fontsize=11)

        for ax in (ax1, ax2):
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        env_str = ",".join(envs) if envs else "all envs"
        fig.suptitle(f"CEGIS pipelines compared ({env_str})", fontsize=13)
    return fig


def main():
    p = base_parser("Verifier-only CEGIS vs CEGIS + refuter comparison")
    p.add_argument("--tag", default=None, help="campaign tag (default: latest per campaign)")
    p.add_argument("--envs", default=None, help="comma list of envs to include")
    a = p.parse_args()

    rows = load_rows(a.tag)
    if not rows:
        raise SystemExit("no cegis_verify_*/cegis_improve_* compiled rows found")
    envs = a.envs.split(",") if a.envs else None
    fig = make_figure(rows, envs)
    save(fig, "cegis_pipelines", "cegis_pipelines", DOC_NAME if a.doc else None)


if __name__ == "__main__":
    main()
