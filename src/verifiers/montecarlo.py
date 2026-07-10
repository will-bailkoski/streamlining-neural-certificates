"""
Monte-Carlo statistical verification engine — SOUND, via a global mean +
Lipschitz "mean-to-max" conversion (the non-adaptive counterpart of `mab`).

It checks the SAME canonical drift as every other engine (`src.verifiers.drift`),

    g(x) = E_w[ V(f(x, w)) ] - V(x) + epsilon,        f_e(g) := max(0, g),

over `domain \\ equilibrium`, and certifies an upper bound on  sup_x g(x).

Why the naive "mean is small => verified" is UNSOUND, and how Lipschitz fixes it
--------------------------------------------------------------------------------
The uniform-sample mean E_X[f_e(g(X))] (X ~ Uniform(domain\\eq)) is easy to bound
by Hoeffding, but max >= mean: a thin violating spike barely moves the mean. The
repair is that g is L_g-Lipschitz, so a violation of height eta cannot be a spike
— it persists over a ball of radius eta/(2 L_g), which forces the mean up:

    mu := E_X[f_e(g(X))]  >=  (kappa * V_d) / (2^{d+1} L_g^d Vol) * eta^{d+1},

with V_d the unit-ball volume, Vol an over-estimate of vol(domain\\eq), and
kappa a lower bound on the fraction of a violation ball that stays inside
domain\\eq. Inverting gives, with probability >= 1 - significance,

    sup_x g(x)  <=  Phi(mu_UB),
    Phi(mu) := ( 2^{d+1} L_g^d Vol / (kappa V_d) * mu )^{1/(d+1)},

where mu_UB is the one-sided Hoeffding upper bound on mu. This IS a sound
worst-case certificate — but Phi(mu) >= Phi(radius) > 0 for any finite sample
count, so the method verifies  sup g <= tau  only for a POSITIVE tolerance tau,
at a sample cost exploding like tau^{-2(d+1)}. It can never certify sup g <= 0.
That gap (a d-dimensional volume converted back to a height) is exactly the cost
that the adaptive box refinement of `mab` avoids by keeping spatial locality.

Nested estimation is safe: g(x_i) is itself estimated with `mc_samples` successor
draws, but f_e = max(0, .) is convex, so E[f_e(ghat)] >= f_e(E[ghat]) = f_e(g)
(Jensen) — the inner noise pushes the mean estimate UP, the sound direction for
an upper bound on mu.

Constants / assumptions
-----------------------
* `range_bound` (b): V in [0, b] (b = 1 for bounded_pwl / clip_pwl). f_e(g) then
  lies in [0, epsilon + b], the range used by the Hoeffding radius.
* L_g = L_V (L_f + 1): the same rigorous drift-Lipschitz bound `mab` uses
  (LipBaB on the raw net composed with the output-activation slope, times
  env.lip_f + 1).
* `kappa` (default 2^{-d}): SOUND when the equilibrium lies in the domain
  interior, away from the corners (true for every env in the suite — the eq is
  central, so a corner violation ball is eq-free, and an eq-boundary violation
  ball is domain-interior; the two worst cases never coincide). If an env placed
  eq into a domain corner this would need shrinking — flagged for the theory.
* `Vol`: the domain BOUNDING-BOX volume (>= vol(domain\\eq)); over-estimating Vol
  only loosens Phi, so it stays sound.

Verdicts
--------
    verified=False : a sampled state has a confident E_w-drift > 0 (a genuine
                     counterexample, re-checked by the caller).
    verified=True  : Phi(mu_UB) <= verify_tol (w.p. >= 1 - significance): the
                     worst-case drift is certified below the tolerance.
    verified=None  : neither — the certified ceiling Phi(mu_UB) exceeds verify_tol
                     (need more samples, or the cert is genuinely too weak).
`stats["certified_sup_drift"] = Phi(mu_UB)` is ALWAYS reported, so the sound
worst-case ceiling and its sample-count scaling are visible regardless of verdict.
"""

from __future__ import annotations
import math
import time
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrn

from src.verifiers.base import VerifierResult, register
from src.verifiers.drift import make_drift


def _hoeffding_radius(n: int, b: float, significance: float) -> float:
    return float(b * np.sqrt(np.log(1.0 / significance) / (2.0 * n)))


def _unit_ball_volume(d: int) -> float:
    return math.pi ** (d / 2.0) / math.gamma(d / 2.0 + 1.0)


def _mean_to_max(mu: float, d: int, lipschitz: float, vol: float, kappa: float) -> float:
    """Phi(mu): the Lipschitz mean-to-max conversion (a sound sup-drift ceiling)."""
    if mu <= 0.0:
        return 0.0
    V_d = _unit_ball_volume(d)
    coef = (2.0 ** (d + 1)) * (lipschitz ** d) * vol / (kappa * V_d)
    return float((coef * mu) ** (1.0 / (d + 1)))


def find_counterexample(
    env,
    spec,
    params,
    epsilon: float,
    key=None,
    n_states: int = 100_000,
    mc_samples: int = 64,
    batch_size: int = 8192,
    significance: float = 0.05,
    range_bound: float = 1.0,
    history_len: int = 1,
    verify_tol: float = 0.0,
    verify_samples: int = 2048,   # fresh draws to confirm a counterexample point
    kappa: float | None = None,
    lipschitz_method: str = "lipbab",
    lipbab_timeout: float = 20.0,
    **hp,
):
    if key is None:
        key = jrn.key(0)

    d = int(env.dim)
    if kappa is None:
        kappa = 2.0 ** (-d)  # sound for interior equilibria (see module docstring)

    # Rigorous drift-Lipschitz bound L_g = L_V (L_f + 1) — the SAME bound mab uses.
    from src.verifiers.lipbab import lipschitz_v_bound

    lip_v = float(lipschitz_v_bound(spec, params, env.domain.bounds,
                                    method=lipschitz_method, timeout=lipbab_timeout))
    L_g = lip_v * (float(env.lip_f) + 1.0)

    # Bounding-box volume of the domain (>= vol(domain\eq): over-estimate is sound).
    b = np.asarray(env.domain.bounds, dtype=float)
    vol = float(np.prod(b[:, 1] - b[:, 0]))

    # f_e(g) = max(0, g) lies in [0, epsilon + range_bound]; that is its Hoeffding range.
    fe_range = float(epsilon + range_bound)

    V = lambda z: spec.forward(params, z)
    _, _, mc_batched = make_drift(env, V, epsilon, n=mc_samples)
    mc_batched_j = jax.jit(mc_batched)
    _, _, mc_verify = make_drift(env, V, epsilon, n=verify_samples)
    mc_verify_j = jax.jit(mc_verify)

    n_batches = max(1, n_states // batch_size)
    total = n_batches * batch_size

    t0 = time.perf_counter()
    best_x = None
    best_d = -np.inf
    fe_sum = 0.0  # sum of max(0, g_hat) over all sampled states

    for _ in range(n_batches):
        key, ks, kr = jrn.split(key, 3)
        xs, _ = env.sample(ks, batch_size)  # excludes equilibrium
        ds = np.array(mc_batched_j(xs, kr))  # E_w drift estimate per state
        fe_sum += float(np.maximum(0.0, ds).sum())
        idx = int(np.argmax(ds))
        if ds[idx] > best_d:
            best_d = float(ds[idx])
            best_x = np.array(xs[idx])

    mean_exceedance = fe_sum / total
    radius = _hoeffding_radius(total, fe_range, significance)
    mu_ub = mean_exceedance + radius  # one-sided upper bound on mu, w.p. 1 - sig
    certified_sup = _mean_to_max(mu_ub, d, L_g, vol, kappa)
    elapsed = time.perf_counter() - t0

    stats = {
        "time": elapsed,
        "best_drift": best_d,
        "samples": total,
        "mc_samples": mc_samples,
        "mean_exceedance": mean_exceedance,
        "hoeffding_radius": radius,
        "mu_ub": mu_ub,
        "certified_sup_drift": certified_sup,  # sound Phi(mu_UB) ceiling on sup g
        "lipschitz": L_g,
        "lip_v": lip_v,
        "domain_volume": vol,
        "kappa": kappa,
        "significance": significance,
        "range_sound": bool(spec.bounded),
    }

    # Genuine counterexample: RE-ESTIMATE the drift at the worst sampled point with
    # fresh independent draws and a one-sided Hoeffding LCB. Selecting the point by
    # maximising noisy estimates biases WHICH point we test, but the fresh estimate
    # at that fixed point is unbiased — LCB > 0 soundly certifies g(x) > 0. (Testing
    # best_d directly would false-refute good certs: the max over ~1e5 noisy
    # estimates drifts positive by an extreme-value margin.)
    if best_x is not None:
        key, kv = jrn.split(key)
        gv = float(np.asarray(mc_verify_j(best_x[None, :], kv))[0])
        lcb = gv - _hoeffding_radius(verify_samples, fe_range, significance)
        stats["cex_drift"] = gv
        stats["cex_lcb"] = lcb
        if lcb > 0.0:
            return VerifierResult(
                verified=False, violation=best_x[None, :], stats=stats,
                engine="montecarlo",
            )

    # Sound verification: the certified worst-case drift clears the tolerance.
    # Only trust it when the reward range is a valid bound (V bounded to [0, b]);
    # on an unbounded cert the ceiling is informative but not a sound verdict.
    if spec.bounded and certified_sup <= verify_tol:
        return VerifierResult(
            verified=True, violation=None, stats=stats, engine="montecarlo"
        )

    return VerifierResult(verified=None, stats=stats, engine="montecarlo")


class _MonteCarloEngine:
    name = "montecarlo"
    find_counterexample = staticmethod(find_counterexample)


register("montecarlo", _MonteCarloEngine())
