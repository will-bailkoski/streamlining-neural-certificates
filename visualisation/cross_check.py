"""
Thesis figure: cross-verification agreement matrix (verification chapter).

Experiment: cross_check (experiments/cross_check.py) — every engine re-verifies
the certificates the other engines produced, one-shot. Data read:
results/cross_check/<tag>/compiled.csv (compiled on demand from runs/).

One heatmap over (source engine, check engine): cell colour is the fraction of
soundness CONFLICTS (a genuine counterexample against a verified cert — 0 is
the hoped-for result everywhere), annotated with agree / tighter+looser /
conflict counts. A second panel lists the conflicting and phantom cells by env,
because a conflict's location (which env, which pair) is the actual finding.

    python -m visualisation.cross_check
    python -m visualisation.cross_check --tag bigtest --doc
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from visualisation.common import (PUB_RC, INK, MUT,
                                  load_compiled, save, base_parser)

DOC_NAME = "cross_check.png"

# canonical engine display order (source rows / check columns)
ORDER = ["mab", "milp", "smt", "ibp", "crown", "alpha-crown", "mc"]


def load_rows(tag: str | None) -> list[dict]:
    rows = load_compiled("cross_check", tag)
    return [r for r in rows if r.get("status") in (None, "", "completed")
            and r.get("check_verdict") not in ("unsupported",)]


def make_figure(rows: list[dict]):
    sources = [e for e in ORDER if any(r.get("source_engine") == e for r in rows)]
    checks = [e for e in ORDER if any(r.get("check_engine") == e for r in rows)]

    def cell(s, c):
        return [r for r in rows
                if r.get("source_engine") == s and r.get("check_engine") == c]

    conflicts = [r for r in rows if r.get("agreement") == "conflict"]
    phantoms = [r for r in rows if str(r.get("recheck_genuine")).lower() == "false"]

    with plt.rc_context(PUB_RC):
        fig, (ax, ax2) = plt.subplots(
            1, 2, figsize=(10.2, 4.2), constrained_layout=True,
            gridspec_kw={"width_ratios": [1.6, 1.0]})

        frac = np.full((len(sources), len(checks)), np.nan)
        for i, s in enumerate(sources):
            for j, c in enumerate(checks):
                rs = cell(s, c)
                if rs:
                    frac[i, j] = np.mean([r.get("agreement") == "conflict" for r in rs])
        im = ax.imshow(frac, cmap="Reds", vmin=0.0, vmax=1.0, aspect="auto")
        for i, s in enumerate(sources):
            for j, c in enumerate(checks):
                rs = cell(s, c)
                if not rs:
                    ax.text(j, i, "–", ha="center", va="center", color=MUT)
                    continue
                n_ag = sum(r.get("agreement") == "agree" for r in rs)
                n_tl = sum(r.get("agreement") in ("tighter", "looser") for r in rs)
                n_cf = sum(r.get("agreement") == "conflict" for r in rs)
                col = "white" if (frac[i, j] or 0) > 0.5 else INK
                ax.text(j, i, f"{n_ag}✓ {n_tl}○"
                              + (f" {n_cf}✗" if n_cf else ""),
                        ha="center", va="center", fontsize=8.5, color=col)
        ax.set_xticks(range(len(checks)), checks, rotation=30, ha="right")
        ax.set_yticks(range(len(sources)), sources)
        ax.set_xlabel("check engine")
        ax.set_ylabel("source engine (cert producer)")
        ax.set_title("agreement per engine pair"
                     "  (✓ agree, ○ tighter/looser, ✗ conflict)",
                     fontsize=10.5)
        fig.colorbar(im, ax=ax, shrink=0.85, label="conflict fraction")

        # right: the actual findings — conflicts (and phantom CEs) by location
        ax2.axis("off")
        lines = ["conflicts (soundness red flags):"]
        if conflicts:
            lines += [f"  {r.get('source_engine')} vs {r.get('check_engine')}"
                      f"  on {r.get('env')} (seed {r.get('seed')}),"
                      f"  drift {r.get('recheck_drift')}"
                      for r in conflicts[:12]]
            if len(conflicts) > 12:
                lines.append(f"  ... +{len(conflicts) - 12} more")
        else:
            lines.append("  none — every verified cert survived")
        lines.append("")
        lines.append(f"phantom counterexamples (looseness): {len(phantoms)}")
        for env in sorted({r.get("env") for r in phantoms}):
            n = sum(r.get("env") == env for r in phantoms)
            engs = sorted({r.get("check_engine") for r in phantoms if r.get("env") == env})
            lines.append(f"  {env}: {n}  ({', '.join(engs)})")
        ax2.text(0.0, 0.98, "\n".join(lines), transform=ax2.transAxes,
                 va="top", ha="left", fontsize=9, family="monospace", color=INK)

        fig.suptitle("Do the verifiers agree with each other?", fontsize=13)
    return fig


def main():
    p = base_parser("Cross-verification agreement matrix")
    p.add_argument("--tag", default=None, help="campaign tag (default: latest)")
    a = p.parse_args()

    rows = load_rows(a.tag)
    if not rows:
        raise SystemExit("no cross_check compiled rows found — run "
                         "experiments/cross_check.py first")
    fig = make_figure(rows)
    save(fig, "cross_check", "cross_check_matrix", DOC_NAME if a.doc else None)


if __name__ == "__main__":
    main()
