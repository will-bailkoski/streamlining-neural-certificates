"""
Refuters as black-box maximisers.

A refuter searches a box `domain` for the point that MAXIMISES a (possibly
stochastic) objective, under a fixed evaluation budget. Decoupling them from the
CEGIS drift (their original use) lets us benchmark them as optimisers and reuse
them anywhere: the drift is just one objective you can pass in.

Common signature
----------------
    refuter(objective, domain, key, *, budget, batch_size, **hp)
        -> (best_x, best_y, evals_trace, best_trace)

    objective(xs, key) -> ys        # xs: (B, d) -> ys: (B,)   [MAXIMISED]
                                    # `key` supports stochastic objectives;
                                    # deterministic ones simply ignore it.
    domain : (d, 2) array of [lo, hi] per dimension
    budget : total objective evaluations
    batch_size : evaluations per iteration (num_iters = budget // batch_size)

    best_trace[i]  = best objective value found through iteration i
    evals_trace[i] = cumulative evaluations through iteration i

Every refuter is one `lax.scan` so the whole search is a single JIT-compiled
call: fast, and trivially precompiled before timing.
"""

from __future__ import annotations
import jax.numpy as jnp

DIRECTORY: dict[str, callable] = {}


def register(name: str):
    def deco(fn):
        DIRECTORY[name] = fn
        fn.refuter_name = name
        return fn

    return deco


def get_refuter(name: str):
    try:
        return DIRECTORY[name]
    except KeyError:
        raise ValueError(f"No such refuter '{name}'. Available: {sorted(DIRECTORY)}")


def uniform_evals_trace(num_iters: int, batch_size: int) -> jnp.ndarray:
    """Cumulative-evaluation checkpoints for a fixed batch_size-per-iter refuter."""
    return (jnp.arange(num_iters) + 1) * batch_size
