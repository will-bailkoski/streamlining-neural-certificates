"""
DIRECT — DIviding RECTangles (Jones, Perttunen & Stuckman 1993, "Lipschitzian
Optimization Without the Lipschitz Constant", JOTA 79(1):157-181; multivariate
identification per Gablonsky 2001 Lemma 2.4; scale-robust epsilon per Jones &
Martins 2021 Eq. 5).

The whole point of DIRECT is that it NEVER estimates a Lipschitz constant. Each
round it divides every "potentially optimal" (PO) hyper-rectangle: a box j that
would attain the best Lipschitzian bound for SOME rate constant K > 0. As K sweeps
0 -> inf that set is exactly the (upper-)right convex hull of the cloud of
(d_j, f(c_j)) points, so DIRECT considers ALL K at once instead of committing to
one. The size measure d_j is the l2 centre-to-vertex distance
    d_j = 1/2 * sqrt( sum_k side_k^2 )
(the ORIGINAL measure; DIRECT-l's l_inf half-longest-side is a different, more
local algorithm). The epsilon sufficient-improvement test uses the scale-robust
form f(c_j) + K d_j >= f_max + eps (f_max - f_median), which (unlike eps|f_max|)
does not collapse when the incumbent is 0 -- the BBOB optima are exactly 0.

Maximisation: the harness maximises, so every inequality is the max-mirror of the
minimising paper (upper-right hull, +K d bound, incumbent = max centre value).

Fixed-shape adaptation (one JIT-compiled lax.scan, timed like the other refuters):
  * Each box carries an integer level vector `lev` (trisections per axis; unit-cube
    side_k = 3^-lev_k), which buckets same-size boxes so the PO hull is an
    O(#buckets^2) masked reduction over per-size representatives, not an
    O(#boxes^2) all-pairs matrix.
  * A box is divided along its SINGLE longest side (the one-side-per-division
    variant of Jones & Martins 2021 / DIRECT-GL): uniform 2 samples per box, so no
    evaluation slot is ever wasted. The all-K PO SELECTION -- the defining feature
    the paper is named for -- is exact; only the split granularity is the variant.
  * Because a DIRECT round divides only its (small) PO set, one fixed batch cannot
    reach the budget. We run many cheap rounds dividing up to H PO boxes each and
    stop (lax.cond) once `budget` real evaluations are spent, so DIRECT gets the
    same evaluation budget as every other refuter.
"""

import jax.numpy as jnp
import jax.random as jrn
from jax import lax, nn as jnn
from jax.ops import segment_max

from src.refuters.base import register

_PROBE = False  # set True to also return per-round (#PO, evals) for calibration


@register("direct")
def direct_refuter(objective, domain, key, *, budget, batch_size=64, po_eps=1e-4, record=False, **hp):
    dim = domain.shape[0]
    low = domain[:, 0]
    width = domain[:, 1] - domain[:, 0]
    to_real = lambda cu: low + cu * width

    H = max(batch_size // 2, 1)               # PO boxes divided per round (whole PO set fits)
    slots = 2 * H                             # objective calls per active round
    num_iters = max(6 * budget // slots, 64)  # generous; early-stop caps real evals at budget
    max_rects = budget + slots + 2

    LMAX = max(2, int(round(1024 ** (1.0 / dim))))
    num_buckets = LMAX ** dim
    radix = LMAX ** jnp.arange(dim)
    bkt = jnp.arange(num_buckets)
    lev_of = (bkt[:, None] // radix[None, :]) % LMAX
    side_of = jnp.power(3.0, -lev_of.astype(jnp.float32))
    d_bucket = 0.5 * jnp.sqrt(jnp.sum(side_of ** 2, axis=1))
    eye_b = jnp.eye(num_buckets, dtype=bool)

    centers = jnp.zeros((max_rects, dim))
    lev = jnp.zeros((max_rects, dim), jnp.int32)
    f_vals = jnp.full((max_rects,), -jnp.inf)
    active = jnp.zeros((max_rects,), dtype=bool)

    key, k0 = jrn.split(key)
    c0 = jnp.full((dim,), 0.5)
    f0 = objective(to_real(c0)[None, :], k0)[0]
    centers = centers.at[0].set(c0)
    f_vals = f_vals.at[0].set(f0)
    active = active.at[0].set(True)

    def divide(carry):
        centers, lev, f_vals, active, cursor, best_x, best_y, evals, key = carry
        key, k_eval = jrn.split(key)

        finite = active & jnp.isfinite(f_vals)
        f_max = jnp.max(jnp.where(finite, f_vals, -jnp.inf))
        sorted_f = jnp.sort(jnp.where(finite, f_vals, jnp.inf))
        f_median = sorted_f[jnp.clip((jnp.sum(finite) - 1) // 2, 0, max_rects - 1)]

        # ---- potentially-optimal boxes: upper-right hull over size buckets ----
        bucket = jnp.clip(jnp.sum(jnp.clip(lev, 0, LMAX - 1) * radix, axis=1), 0, num_buckets - 1)
        bestf = segment_max(jnp.where(active, f_vals, -jnp.inf), bucket, num_segments=num_buckets)
        present = bestf > -jnp.inf

        dd = d_bucket[None, :] - d_bucket[:, None]     # d_b' - d_b
        df = bestf[None, :] - bestf[:, None]           # f_b' - f_b
        pres2 = present[None, :] & present[:, None]
        slope = df / jnp.where(dd != 0, dd, 1.0)       # (f_b' - f_b)/(d_b' - d_b)
        # admissible K interval (max-mirror of Gablonsky Lemma 2.4): the required
        # slopes are -slope, so K >= -min_{smaller} slope and K <= -max_{larger} slope.
        Klo = -jnp.min(jnp.where(pres2 & (dd < 0), slope, jnp.inf), axis=1)
        Khi = -jnp.max(jnp.where(pres2 & (dd > 0), slope, -jnp.inf), axis=1)
        hull_ok = (Klo <= Khi) & (Khi > 0)
        same = pres2 & (dd == 0) & ~eye_b
        size_ok = bestf >= jnp.max(jnp.where(same, bestf[None, :], -jnp.inf), axis=1)
        ub_Khi = jnp.where(jnp.isinf(Khi), jnp.inf, bestf + Khi * d_bucket)
        eps_ok = ub_Khi >= f_max + po_eps * (f_max - f_median)
        po_bucket = present & hull_ok & size_ok & eps_ok

        is_rep = active & po_bucket[bucket] & (f_vals == bestf[bucket])
        priority = jnp.where(is_rep, d_bucket[bucket], -jnp.inf)
        top_v, sel = lax.top_k(priority, H)            # up to H PO boxes, larger first
        sel_valid = top_v > -jnp.inf

        # ---- divide each along its single longest side ----
        sc = centers[sel]
        sl = lev[sel]
        longest = jnp.argmin(sl, axis=1)               # longest side = smallest level
        e_long = jnn.one_hot(longest, dim)
        cur = jnp.take_along_axis(sl, longest[:, None], axis=1)[:, 0]
        delta = jnp.power(3.0, -(cur.astype(jnp.float32) + 1.0))
        plus = sc + delta[:, None] * e_long
        minus = sc - delta[:, None] * e_long
        new_lev = jnp.clip(sl + e_long.astype(jnp.int32), 0, LMAX - 1)

        ys = objective(to_real(jnp.concatenate([plus, minus], axis=0)), k_eval)
        f_plus, f_minus = ys[:H], ys[H:]

        cc = jnp.concatenate([plus, minus], axis=0)
        clv = jnp.concatenate([new_lev, new_lev], axis=0)
        cf = jnp.concatenate([f_plus, f_minus], axis=0)
        cvalid = jnp.concatenate([sel_valid, sel_valid], axis=0)

        idx = cursor + jnp.arange(slots)
        safe = jnp.clip(idx, 0, max_rects - 1)
        can = (idx < max_rects) & cvalid
        centers = centers.at[safe].set(jnp.where(can[:, None], cc, centers[safe]))
        lev = lev.at[safe].set(jnp.where(can[:, None], clv, lev[safe]))
        f_vals = f_vals.at[safe].set(jnp.where(can, cf, f_vals[safe]))
        active = active.at[safe].set(active[safe] | can)
        lev = lev.at[sel].set(jnp.where(sel_valid[:, None], new_lev, lev[sel]))
        cursor = cursor + jnp.sum(can)

        bi = jnp.argmax(jnp.where(can, cf, -jnp.inf))
        better = can[bi] & (cf[bi] > best_y)
        best_x = jnp.where(better, to_real(cc[bi]), best_x)
        best_y = jnp.maximum(best_y, jnp.where(can[bi], cf[bi], -jnp.inf))
        evals = evals + jnp.sum(can)
        n_po = jnp.sum(is_rep)
        return (centers, lev, f_vals, active, cursor, best_x, best_y, evals, key), n_po

    def body(carry, _):
        evals = carry[7]
        newcarry, n_po = lax.cond(evals < budget, divide, lambda c: (c, jnp.int32(0)), carry)
        out = (newcarry[7], newcarry[6])
        return newcarry, ((out, n_po) if _PROBE else out)

    init = (centers, lev, f_vals, active, jnp.int32(1), to_real(c0), f0, jnp.int32(1), key)
    final, out = lax.scan(body, init, None, length=num_iters)
    if _PROBE:
        (evals_trace, best_trace), n_po_trace = out
    else:
        evals_trace, best_trace = out
    if record:
        fc, fl, ff, fa = final[0], final[1], final[2], final[3]
        history = {"kind": "boxes",
                   "centers": low + fc * width,
                   "sizes": 0.5 * width * jnp.power(3.0, -fl.astype(jnp.float32)),
                   "f": ff, "active": fa}
        return final[5], final[6], evals_trace, best_trace, history
    if _PROBE:
        return final[5], final[6], evals_trace, best_trace, n_po_trace
    return final[5], final[6], evals_trace, best_trace
