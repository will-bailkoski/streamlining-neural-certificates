"""
The canonical drift functional — the one quantity every engine must agree on.

    drift(x) = E_w[ V(f(x, w)) ] - V(x) + epsilon

This module provides the jax/Monte-Carlo ground truth used to (a) re-check
counterexamples reported by the sound-but-over-approximating engines, and
(b) cross-check that all engines evaluate the same condition.

`V` here is `spec.forward(params, ·)` — the certificate. `f` and the noise are
the env's (`env.step`).
"""

from __future__ import annotations
import jax.numpy as jnp
import jax.random as jrn
from jax import vmap


def make_drift(env, V, epsilon: float, n: int = 256):
    """
    Returns (single, mc, mc_batched):
      single(x, key)      -> one-sample drift  V(step(x,key)) - V(x) + eps
      mc(x, key)          -> MC estimate of E_w[...] using n successor samples
      mc_batched(xs, key) -> mc over a batch of states
    """

    def single(x, key):
        x_next, _ = env.step(x, key)
        return V(x_next) - V(x) + epsilon

    def mc(x, key):
        keys = jrn.split(key, n)
        return jnp.mean(vmap(lambda k: single(x, k))(keys))

    def mc_batched(xs, key):
        keys = jrn.split(key, xs.shape[0])
        return vmap(mc)(xs, keys)

    return single, mc, mc_batched


def recheck_violation(env, spec, params, epsilon, x, key, n: int = 4096, tol: float = 0.0):
    """
    Re-evaluate a reported counterexample with the canonical jax MC drift.

    Returns (is_genuine, drift_value). A counterexample is genuine when the
    MC drift exceeds `tol` (default 0): the certificate truly fails there.
    Used to filter spurious counterexamples from over-approximating engines.
    """
    V = lambda z: spec.forward(params, z)
    _, mc, _ = make_drift(env, V, epsilon, n=n)
    d = float(mc(jnp.asarray(x, dtype=jnp.float32), key))
    return d > tol, d
