"""
Sampling verification engine (statistical refuter).

Searches (domain \\ equilibrium) for a state where the Monte-Carlo drift is
positive, via random search followed by gradient ascent on the drift. It uses
the SAME canonical drift as every other engine (`src.verifiers.drift`).

This engine is a *refuter*, not a prover: it can return a counterexample or
"inconclusive", but never "verified" — establishing verified=True requires the
sound mesh/Lipschitz method (a future `sampling_mesh` engine). Verdicts:

    verified=False -> a state with MC drift > 0 was found (`violation`)
    verified=None  -> budget exhausted without finding one (inconclusive)
"""

from __future__ import annotations
import time
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrn

from src.verifiers.base import VerifierResult, register
from src.verifiers.drift import make_drift


def find_counterexample(
    env,
    spec,
    params,
    epsilon: float,
    key=None,
    batch_size: int = 4096,
    rounds: int = 8,
    mc_samples: int = 64,
    refine_steps: int = 50,
    refine_lr: float = 1e-2,
    tol: float = 0.0,
    **hp,
):
    if key is None:
        key = jrn.key(0)

    V = lambda z: spec.forward(params, z)
    _, mc, mc_batched = make_drift(env, V, epsilon, n=mc_samples)

    # gradient of the MC drift wrt the state (for refinement)
    drift_grad = jax.jit(jax.grad(lambda x, k: mc(x, k)))
    mc_batched_j = jax.jit(mc_batched)

    lo = np.array(env.domain.bounds[:, 0])
    hi = np.array(env.domain.bounds[:, 1])

    t0 = time.perf_counter()
    best_x = None
    best_d = -np.inf

    for _ in range(rounds):
        key, ks, kb = jrn.split(key, 3)
        xs, _ = env.sample(ks, batch_size)  # already excludes equilibrium
        ds = np.array(mc_batched_j(xs, kb))
        idx = int(np.argmax(ds))
        if ds[idx] > best_d:
            best_d = float(ds[idx])
            best_x = np.array(xs[idx])

    # gradient-ascent refinement from the best random candidate
    if best_x is not None and refine_steps > 0:
        x = jnp.array(best_x)
        for _ in range(refine_steps):
            key, kg = jrn.split(key)
            g = drift_grad(x, kg)
            x = jnp.clip(x + refine_lr * g, jnp.array(lo), jnp.array(hi))
            if bool(env.equilibrium.contains(x)):
                break  # drifted into the equilibrium; keep the pre-step point
            key, kd = jrn.split(key)
            d = float(mc(x, kd))
            if d > best_d:
                best_d = d
                best_x = np.array(x)

    elapsed = time.perf_counter() - t0
    stats = {
        "time": elapsed,
        "best_drift": best_d,
        "samples": batch_size * rounds,
    }

    if best_x is not None and best_d > tol:
        return VerifierResult(
            verified=False, violation=best_x[None, :], stats=stats, engine="sampling"
        )
    return VerifierResult(verified=None, stats=stats, engine="sampling")


class _SamplingEngine:
    name = "sampling"
    find_counterexample = staticmethod(find_counterexample)


register("sampling", _SamplingEngine())
