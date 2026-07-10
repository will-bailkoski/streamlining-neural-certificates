"""
AdaLIPO — adaptive-Lipschitz optimisation (Malherbe & Vayatis, ICML 2017,
"Global optimization of Lipschitz functions", arXiv:1703.02628, Algorithm 2).

Faithful to the paper's decision rule:

  * LIPO acceptance test (their Lemma 8) — a fresh uniform candidate x is a
    "potential maximiser" and is accepted iff its tightest Lipschitz upper bound
    over the evaluated history is still at least the current best value:
        min_i [ y_i + k_hat * ||x - X_i|| ]  >=  max_i y_i.
    Candidates are drawn uniformly and REJECTED (free — no objective call) until
    one passes; only accepted points spend the evaluation budget.
  * Bernoulli(p) exploration/exploitation split (their Algorithm 2): with prob p
    accept an UNCONDITIONAL uniform sample (to gather unbiased slope data for the
    k estimate); with prob 1-p accept only a sample that passes the LIPO test
    with the current estimate k_hat. The paper fixes p = 0.1.
  * Lipschitz estimate on a geometric grid (their Remark 17): k_hat = (1+alpha)^i
    with i = ceil( ln(Lambda) / ln(1+alpha) ), where Lambda is the largest slope
    |y_i - y_j| / ||X_i - X_j|| seen over all evaluated pairs. The paper's rule of
    thumb is alpha = 0.01/d (k_i = (1 + 0.01/d)^i); k_hat starts at 0.

Adaptation to this harness: the paper is strictly sequential (one accepted point,
then update k_hat). To stay a single JIT-compiled `lax.scan` and be timed on the
same batched footing as the other refuters, each iteration accepts a *batch* of
`batch_size` points against the CURRENT (X_i, y_i, k_hat) and then updates the
history and k_hat once. The decision rule (LIPO gate, Bernoulli-p split, grid
k_hat) is exactly the paper's; only the update granularity is batched. Rejection
sampling per exploitation slot is truncated to `pool_mult` uniform attempts (then
falls back to the best-upper-bound attempt), the fixed-shape analogue of
"resample until accepted". Evaluations per iteration = `batch_size`.
"""

import jax.numpy as jnp
import jax.random as jrn
from jax import lax

from src.refuters.base import register, uniform_evals_trace


@register("adalip")
def adalip_refuter(
    objective, domain, key, *, budget, batch_size=64, pool_mult=8, alpha=None,
    p=0.1, record=False, **hp
):
    num_iters = max(budget // batch_size, 1)
    cap = num_iters * batch_size
    low, high = domain[:, 0], domain[:, 1]
    dim = domain.shape[0]
    if alpha is None:
        alpha = 0.01 / dim                 # paper's rule of thumb: k_i = (1 + 0.01/d)^i
    ln1p = jnp.log1p(alpha)
    K = max(int(pool_mult), 1)             # uniform attempts per exploitation slot

    Xh = jnp.zeros((cap, dim))
    Yh = jnp.full((cap,), -jnp.inf)
    hist_idx = jnp.arange(cap)

    def k_from_lambda(lam):
        # snap the max observed slope UP to the grid (1+alpha)^i; k_hat = 0 until a
        # finite positive slope exists (fewer than two distinct points).
        i = jnp.ceil(jnp.log(jnp.where(lam > 0, lam, 1.0)) / ln1p)
        return jnp.where(lam > 0, jnp.exp(i * ln1p), 0.0)

    def body(carry, _):
        Xh, Yh, cursor, lam, best_x, best_y, k = carry
        k, kb, kp, ke = jrn.split(k, 4)
        valid = hist_idx < cursor
        finite = valid & jnp.isfinite(Yh)                 # invalid-domain evals (-inf) are excluded
        k_hat = k_from_lambda(lam)
        best_hist = jnp.max(jnp.where(finite, Yh, -jnp.inf))   # -inf when history empty

        # -- propose: K uniform attempts per output slot ---------------------
        cand = jrn.uniform(kp, (batch_size, K, dim), minval=low, maxval=high)
        d = jnp.linalg.norm(cand[:, :, None, :] - Xh[None, None, :, :], axis=-1)  # (B,K,cap)
        # per-history-point upper bound y_i + k_hat*dist; excluded points -> +inf
        # so they never lower the min (an empty/effectively-empty history accepts).
        ub_i = jnp.where(finite[None, None, :], Yh[None, None, :] + k_hat * d, jnp.inf)
        ub = jnp.min(ub_i, axis=-1)                       # (B,K) tightest upper bound
        passes = ub >= best_hist                          # LIPO potential-maximiser test

        # first passing attempt, else the best-upper-bound attempt (fixed-shape
        # stand-in for "keep resampling until one is accepted").
        has_pass = jnp.any(passes, axis=1)
        first_pass = jnp.argmax(passes, axis=1)           # first True, or 0 if none
        best_ub = jnp.argmax(ub, axis=1)
        pick = jnp.where(has_pass, first_pass, best_ub)
        exploit = jnp.take_along_axis(cand, pick[:, None, None], axis=1)[:, 0, :]

        # Bernoulli(p): explore -> an unconditional uniform draw (attempt 0);
        # exploit -> the LIPO-accepted attempt above.
        explore = jrn.bernoulli(kb, p, (batch_size,))
        pts = jnp.where(explore[:, None], cand[:, 0, :], exploit)

        ys = objective(pts, ke)

        # -- append to history, update Lambda (grid k_hat is derived from it) --
        idx = cursor + jnp.arange(batch_size)
        safe = jnp.clip(idx, 0, cap - 1)
        can = idx < cap
        Xh = Xh.at[safe].set(jnp.where(can[:, None], pts, Xh[safe]))
        Yh = Yh.at[safe].set(jnp.where(can, ys, Yh[safe]))
        cursor = jnp.minimum(cursor + batch_size, cap)

        newvalid = hist_idx < cursor
        newfinite = newvalid & jnp.isfinite(Yh)
        dd = jnp.linalg.norm(pts[:, None, :] - Xh[None, :, :], axis=-1)          # (B,cap)
        dy = jnp.abs(ys[:, None] - Yh[None, :])
        good = newfinite[None, :] & jnp.isfinite(ys)[:, None] & (dd > 0)
        slope = jnp.where(good, dy / jnp.where(dd > 0, dd, 1.0), 0.0)
        lam = jnp.maximum(lam, jnp.max(slope))

        bi = jnp.argmax(ys)
        better = ys[bi] > best_y
        best_x = jnp.where(better, pts[bi], best_x)
        best_y = jnp.maximum(best_y, ys[bi])
        return (Xh, Yh, cursor, lam, best_x, best_y, k), ((best_y, pts) if record else best_y)

    init = (Xh, Yh, jnp.int32(0), jnp.float32(0.0), 0.5 * (low + high), -jnp.inf, key)
    final, out = lax.scan(body, init, None, length=num_iters)
    best_x, best_y = final[4], final[5]
    evals = uniform_evals_trace(num_iters, batch_size)
    if record:
        best_trace, positions = out
        return best_x, best_y, evals, best_trace, {"kind": "points", "positions": positions}
    return best_x, best_y, evals, out
