"""Well-formedness checks + landscape profile for the BBOB refuter suite.

    python -m experiments.bbob_profile          # checks + profile table (stdout)

Checks (hard assertions — a failure means the suite is mis-specified):
  * f(x_opt) == f_opt (= 0) to float tolerance;
  * gradient at x_opt ~ 0 (interior stationary maximum);
  * NO sampled point beats f_opt (200k uniform + 20k near-optimum draws);
  * x_opt lies strictly inside the domain.

Profile stats (the columns that explain WHY refuters rank differently):
  * range of f over the domain, and the 99th-percentile gap to the optimum;
  * empirical Lipschitz (max |grad| over samples) — step-size scale;
  * basin size: fraction of the domain within 1% (of range) of the optimum —
    the "target size" a coverage-based refuter must hit;
  * plateau fraction: samples with |grad| < 1e-3 * L_emp — where gradient
    refuters receive no signal;
  * multimodality: local maxima among 4096 short gradient-ascent runs,
    clustered — how many traps exploitation can fall into.
"""

from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrn

from experiments.bbob_functions import make_suite
from experiments.common import config

N_UNIFORM = 200_000
N_NEAR = 20_000
N_ASCENT = 4096
ASCENT_STEPS = 200


def profile(fn, key):
    d = fn.domain.shape[0]
    lo, hi = fn.domain[:, 0], fn.domain[:, 1]
    width = hi - lo

    # -- well-formedness ------------------------------------------------
    f_at_opt = float(fn.f(fn.x_opt[None, :])[0])
    assert abs(f_at_opt - fn.f_opt) < 1e-4, f"{fn.name}: f(x_opt)={f_at_opt}"
    # local-max test by finite differences (NOT autodiff: ackley's sqrt term is
    # non-smooth exactly at the optimum, so grad(x_opt) is NaN by construction)
    for i in range(d):
        for s in (-1.0, 1.0):
            probe = fn.x_opt.at[i].add(s * 1e-3 * float(width[i]))
            assert float(fn.f(probe[None, :])[0]) <= fn.f_opt + 1e-6, \
                f"{fn.name}: not a local max along dim {i}"
    assert bool(jnp.all((fn.x_opt > lo) & (fn.x_opt < hi))), \
        f"{fn.name}: x_opt not strictly inside the domain"

    k1, k2, k3 = jrn.split(key, 3)
    xs = lo + jrn.uniform(k1, (N_UNIFORM, d)) * width
    near = fn.x_opt + 0.01 * width * (jrn.uniform(k2, (N_NEAR, d)) - 0.5)
    near = jnp.clip(near, lo, hi)
    fs = np.asarray(fn.f(xs))
    fs_near = np.asarray(fn.f(near))
    best_seen = max(fs.max(), fs_near.max())
    assert best_seen <= fn.f_opt + 1e-6, \
        f"{fn.name}: sampled {best_seen} > f_opt {fn.f_opt} (optimum mis-specified)"

    # -- profile ---------------------------------------------------------
    grad_f = jax.jit(jax.vmap(jax.grad(lambda x: fn.f(x[None, :])[0])))
    gnorms = np.linalg.norm(np.asarray(grad_f(xs[:20_000])), axis=1)
    l_emp = float(gnorms.max())
    f_range = float(fn.f_opt - fs.min())
    basin = float((fs >= fn.f_opt - 0.01 * f_range).mean())
    plateau = float((gnorms < 1e-3 * l_emp).mean())
    p99_gap = float(fn.f_opt - np.percentile(fs, 99))

    # multimodality: short gradient ascents from random starts, cluster endpoints
    x0 = lo + jrn.uniform(k3, (N_ASCENT, d)) * width
    step = 0.5 * float(width.max()) / max(l_emp, 1e-12)  # conservative ascent step

    @jax.jit
    def ascend(x):
        def body(x, _):
            g = grad_f(x)
            gn = jnp.linalg.norm(g, axis=1, keepdims=True)
            x = jnp.clip(x + step * g / jnp.maximum(gn, 1e-12) * 0.01 * width, lo, hi)
            return x, None
        x, _ = jax.lax.scan(body, x, None, length=ASCENT_STEPS)
        return x

    ends = np.asarray(ascend(jnp.asarray(x0)))
    # only CONVERGED ascents count as local maxima (unconverged endpoints are
    # crawl positions in a valley/plateau, not modes — rosenbrock's ascents
    # scatter along its valley and would read as thousands of fake maxima).
    end_g = np.linalg.norm(np.asarray(grad_f(jnp.asarray(ends))), axis=1)
    converged = end_g < 1e-2 * l_emp
    conv_frac = float(converged.mean())
    # a mode must also have IMPROVED during ascent: on a flat plateau (easom)
    # every start "converges" in place with zero gradient — those are not maxima
    improved = (np.asarray(fn.f(jnp.asarray(ends)))
                > np.asarray(fn.f(jnp.asarray(x0))) + 1e-6 * f_range)
    tol = 0.01 * float(width.max())
    centers: list[np.ndarray] = []
    for p in ends[converged & improved]:
        if not any(np.linalg.norm(p - c) < tol for c in centers):
            centers.append(p)
    return dict(
        dim=d, f_range=f_range, p99_gap=p99_gap, lipschitz_emp=l_emp,
        basin_frac=basin, plateau_frac=plateau, n_local_maxima=len(centers),
        ascent_conv_frac=conv_frac,
    )


def main():
    print(f"BBOB refuter suite — dim={config.BBOB_DIM}, "
          f"{N_UNIFORM:,} uniform + {N_NEAR:,} near-opt samples per function\n")
    hdr = (f"{'function':<11} {'range':>10} {'p99 gap':>9} {'L_emp':>10} "
           f"{'basin%':>8} {'plateau%':>9} {'#maxima':>8} {'conv%':>7}")
    print(hdr)
    print("-" * len(hdr))
    for i, fn in enumerate(make_suite(config.BBOB_DIM)):
        s = profile(fn, jrn.key(1000 + i))
        print(f"{fn.name:<11} {s['f_range']:>10.3g} {s['p99_gap']:>9.3g} "
              f"{s['lipschitz_emp']:>10.3g} {100*s['basin_frac']:>7.3f}% "
              f"{100*s['plateau_frac']:>8.2f}% {s['n_local_maxima']:>8d} "
              f"{100*s['ascent_conv_frac']:>6.1f}%")
    print("\nall well-formedness checks passed")


if __name__ == "__main__":
    main()
