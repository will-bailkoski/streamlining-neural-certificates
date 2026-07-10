"""
Thesis figure: HOW each refuter searches — evaluation patterns over time,
overlaid on the objective's heatmap (refutation chapter).

Two modes, one renderer:

  * BBOB mode (default): the refuters x functions grid on the analytic BBOB
    suite. Each cell replays the refuter with record=True and draws its
    mechanism faithfully — sampled points coloured by ITERATION (random / grid
    / adalip), per-agent trajectories (gradient / whale), or the final
    dividing-rectangles partition (direct). + marks the true optimum, * the
    incumbent best.
  * --env mode: the same view on the REAL refutation objective — the MC drift
    surface E[V(x')] - V(x) + eps of a trained certificate from a run dir —
    so the search pattern can be judged on the landscape CEGIS actually feeds
    the refuters (careful: drift surfaces are noisy near-flat ridges, nothing
    like BBOB bowls).

Replays are seeded and cost seconds (budget-bounded), so nothing is cached.

Paired experiments: refute_bbob_* campaigns (BBOB mode); any campaign with a
stored certificate (env mode).

    python -m visualisation.refuter_patterns
    python -m visualisation.refuter_patterns --refuters whale,direct --budget 4096
    python -m visualisation.refuter_patterns --env-run results/milp_ndcerts/runs/<id>
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from visualisation.common import PUB_RC, REFUTER_ORDER, save, base_parser
from src.plotting.refuter_search import plot_grid, draw_search

DOC_NAME = "refuter_search_patterns.png"


def bbob_grid(refuters, functions, budget, batch_size, seed):
    fig = plot_grid(refuters, functions, budget=budget, batch_size=batch_size, seed=seed)
    return fig


def env_grid(run_dir: Path, refuters, budget, batch_size, seed, mc_samples=64):
    """Search patterns on a trained certificate's MC drift surface."""
    from visualisation.common import load_cert
    import jax
    import jax.numpy as jnp
    import jax.random as jrn
    from experiments.bbob_functions import TestFunction

    env, spec, params, args = load_cert(run_dir)
    if env.dim != 2:
        raise SystemExit("env mode needs a 2D environment")
    eps = float(args.get("epsilon", 1e-3))
    V = lambda x: spec.forward(params, x)

    # deterministic MC drift surface: same fixed noise keys at every x, so the
    # refuters see a well-defined (still rugged) objective
    keys = jrn.split(jrn.key(seed + 7), mc_samples)

    def drift_one(x):
        vn = jax.vmap(lambda k: V(env.step(x, k)[0]))(keys)
        return jnp.mean(vn) - V(x) + eps

    drift = jax.jit(jax.vmap(drift_one))
    fn = TestFunction(name=f"drift({args.get('env')})", f=lambda xs: drift(xs),
                      domain=jnp.asarray(env.domain.bounds),
                      x_opt=jnp.zeros(env.dim))  # placeholder; optimum unknown

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(1, len(refuters), figsize=(3.1 * len(refuters), 3.5),
                                 constrained_layout=True, squeeze=False)
        for ax, name in zip(axes[0], refuters):
            draw_search(ax, name, fn, budget=budget, batch_size=batch_size, seed=seed)
            # the true optimum marker is meaningless here — remove it (last two
            # artists added by draw_search are the optimum + incumbent markers)
            ax.lines[-2].remove()
            ax.set_title(name, fontsize=11)
        fig.suptitle(f"Refuter search patterns on the certificate drift surface "
                     f"({args.get('env')}, * = incumbent max)", fontsize=12)
    return fig


def main():
    p = base_parser("Refuter evaluation patterns over the objective heatmap")
    p.add_argument("--refuters", default=",".join(REFUTER_ORDER))
    p.add_argument("--functions", default=None, help="BBOB mode: comma list (default all)")
    p.add_argument("--env-run", default=None,
                   help="run dir with a certificate: draw patterns on ITS drift surface")
    p.add_argument("--budget", type=int, default=8192)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    refuters = [r for r in a.refuters.split(",") if r]

    if a.env_run:
        fig = env_grid(Path(a.env_run), refuters, a.budget, a.batch_size, a.seed)
        env_name = Path(a.env_run).name.split("env-")[-1][:12]
        save(fig, "refuter_patterns", f"drift_patterns_{env_name}")
        return

    from experiments.bbob_functions import make_suite
    suite = make_suite(a.dim)
    pick = a.functions.split(",") if a.functions else None
    functions = [f for f in suite if pick is None or f.name in pick]
    fig = bbob_grid(refuters, functions, a.budget, a.batch_size, a.seed)
    save(fig, "refuter_patterns", "bbob_search_patterns", DOC_NAME if a.doc else None)


if __name__ == "__main__":
    main()
