"""
Thesis figures: the MAB verifier dashboard (verification chapter).

Paired experiment: any campaign whose runs hold a BOUNDED certificate the MAB
engine can attack (cegis_verify_mab, milp_ndcerts, ...).
Data read:         <run_dir>/objects/mab_trace.npz — the engine's `record`
trace. If the cache is missing, the engine is re-run ONCE on the stored
certificate with record=True and the trace is written back into the run dir;
every later plot tweak reads the cache instantly.

Two figures:
  A  mab_area      verified VOLUME (not box count — boxes shrink adaptively,
                    so counting them overstates frontier work) as a fraction
                    of the domain, plus the undecided remainder, over time.
  B  mab_bounds    the certification frontier: max UCB over live cells (must
                    be driven <= 0 to verify) and max LCB (> 0 would BE a
                    counterexample), over time.

    python -m visualisation.mab_dashboard                       # auto-pick cert
    python -m visualisation.mab_dashboard --run <dir> --refresh # re-trace
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from visualisation.common import (PUB_RC, GREEN, CORAL, SAND, MUT, INK,
                                  find_cert_run, load_cert, save, base_parser)

# campaigns searched (in order) for a bounded 2D certificate when --run is absent
CERT_CAMPAIGNS = ["cegis_verify_mab", "milp_ndcerts"]
CACHE = "mab_trace.npz"

DOC_NAME_AREA = "mab_area.png"
DOC_NAME_BOUNDS = "mab_bounds.png"


def pick_run(env: str | None) -> Path:
    for c in CERT_CAMPAIGNS:
        d = find_cert_run(c, env=env, verdict="verified")
        if d is None:
            d = find_cert_run(c, env=env, verdict=None)
        if d is not None:
            return d
    raise SystemExit(f"no run with a certificate found in {CERT_CAMPAIGNS} — pass --run")


def load_or_record(run_dir: Path, epsilon: float, max_samples: int, max_boxes: int,
                   refresh: bool) -> dict:
    cache = run_dir / "objects" / CACHE
    if cache.exists() and not refresh:
        z = np.load(cache)
        return {k: z[k] for k in z.files}

    env, spec, params, args = load_cert(run_dir)
    import jax.random as jrn
    from src.verifiers.base import get_engine
    import src.verifiers  # noqa: F401

    print(f"tracing mab on {args.get('env')} (cert from {run_dir.name}) ...")
    # record=<large int>: the dashboard needs the per-iteration trace rows, not
    # the box frames — sparse frames keep long runs from paying the snapshot
    # cost (use visualisation.verifier_grid for dense frames).
    r = get_engine("mab").find_counterexample(
        env, spec, params, epsilon, key=jrn.key(0),
        max_samples=max_samples, max_boxes=max_boxes, record=50_000,
    )
    trace = r.stats.get("trace", [])
    if not trace:
        raise SystemExit(f"engine returned no trace (stats: {list(r.stats)})")
    verdict = {True: "verified", False: "counterexample", None: "budget exhausted"}[r.verified]
    print(f"  -> {verdict} after {trace[-1]['t']:.1f}s, "
          f"{trace[-1]['samples']:,} samples, {len(trace)} iterations")

    cols = {k: np.array([row[k] for row in trace]) for k in trace[0]}
    cols["initial_volume"] = np.array(r.stats.get("initial_volume", np.nan))
    cols["verdict"] = np.array(verdict)
    np.savez_compressed(cache, **cols)
    print(f"  cached trace -> {cache.as_posix()}")
    return {k: v for k, v in cols.items()}


def figure_area(tr: dict, title: str):
    t = tr["t"]
    v0 = float(tr["initial_volume"])
    safe = tr["safe_volume"] / v0
    live = tr["live_volume"] / v0
    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
        ax.fill_between(t, 0, safe, color=GREEN, alpha=0.75, lw=0, label="verified volume")
        ax.fill_between(t, safe, safe + live, color=SAND, alpha=0.55, lw=0,
                        label="undecided volume")
        ax.plot(t, safe, color=GREEN, lw=1.6)
        ax.axhline(1.0, color=MUT, lw=0.8, ls=":")
        ax.set_xlabel("wall time (s)")
        ax.set_ylabel("fraction of domain volume")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(left=0)
        verdict = str(tr.get("verdict", ""))
        ax.set_title(f"MAB verification progress — {title}\n"
                     f"(final: {verdict}, {int(tr['samples'][-1]):,} samples)", fontsize=11)
        ax.legend(loc="center right", frameon=False, fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    return fig


def figure_bounds(tr: dict, title: str):
    t, ucb, lcb = tr["t"], tr["max_ucb"], tr["max_lcb"]
    finite = np.isfinite(ucb)
    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
        ax.plot(t[finite], ucb[finite], color=INK, lw=1.8,
                label=r"max UCB over live cells  (verify $\Leftrightarrow$ $\leq 0$)")
        ax.plot(t, lcb, color=CORAL, lw=1.8,
                label=r"max LCB  ($> 0$ $\Rightarrow$ counterexample)")
        ax.axhline(0.0, color=MUT, lw=1.0, ls="--")
        ax.fill_between(t, 0, ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else -1e-3,
                        color=GREEN, alpha=0.05)
        ax.set_xlabel("wall time (s)")
        ax.set_ylabel(r"drift bound  $\mathbb{E}[V(x')] - V(x) + \varepsilon$")
        ax.set_title(f"MAB confidence bounds — {title}", fontsize=11)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    return fig


def main():
    p = base_parser("MAB verifier dashboard (verified area + UCB/LCB over time)")
    p.add_argument("--run", default=None, help="run dir holding the certificate")
    p.add_argument("--env", default=None, help="env filter for the auto-picked run")
    p.add_argument("--epsilon", type=float, default=1e-3)
    p.add_argument("--max_samples", type=int, default=20_000_000)
    p.add_argument("--max_boxes", type=int, default=2**18)
    p.add_argument("--refresh", action="store_true", help="re-run the engine trace")
    a = p.parse_args()

    run_dir = Path(a.run) if a.run else pick_run(a.env)
    from visualisation.common import load_manifest
    env_name = load_manifest(run_dir).get("args", {}).get("env", run_dir.name)
    tr = load_or_record(run_dir, a.epsilon, a.max_samples, a.max_boxes, a.refresh)

    save(figure_area(tr, env_name), "mab_dashboard", f"mab_area_{env_name}",
         DOC_NAME_AREA if a.doc else None)
    save(figure_bounds(tr, env_name), "mab_dashboard", f"mab_bounds_{env_name}",
         DOC_NAME_BOUNDS if a.doc else None)


if __name__ == "__main__":
    main()
