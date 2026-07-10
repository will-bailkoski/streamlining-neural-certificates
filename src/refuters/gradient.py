"""
Multi-start projected gradient ascent.

`batch_size` parallel chains start uniformly in the domain and take one
normalised, domain-scaled ascent step per iteration (gradient via autodiff of
the objective), clipped back into the box. Multi-start gives exploration; the
step is counted as `batch_size` evaluations per iteration.
"""

import jax.numpy as jnp
import jax.random as jrn
from jax import lax, grad

from src.refuters.base import register, uniform_evals_trace


@register("gradient")
def gradient_refuter(objective, domain, key, *, budget, batch_size=256, lr=0.1, decay=0.99, record=False, **hp):
    num_iters = max(budget // batch_size, 1)
    low, high = domain[:, 0], domain[:, 1]
    width = high - low
    dim = domain.shape[0]

    def body(carry, _):
        xs, best_x, best_y, step, k = carry
        k, ke = jrn.split(k)
        ys = objective(xs, ke)
        i = jnp.argmax(ys)
        better = ys[i] > best_y
        best_x = jnp.where(better, xs[i], best_x)
        best_y = jnp.maximum(best_y, ys[i])

        g = grad(lambda X: objective(X, ke).sum())(xs)  # (B, dim) per-chain gradients
        g = g / (jnp.linalg.norm(g, axis=1, keepdims=True) + 1e-9)
        xs_next = jnp.clip(xs + lr * (decay**step) * width * g, low, high)
        return (xs_next, best_x, best_y, step + 1, k), ((best_y, xs) if record else best_y)

    key, ksamp = jrn.split(key)
    xs0 = jrn.uniform(ksamp, (batch_size, dim), minval=low, maxval=high)
    init = (xs0, 0.5 * (low + high), -jnp.inf, 0, key)
    (_, best_x, best_y, _, _), out = lax.scan(body, init, None, length=num_iters)
    evals = uniform_evals_trace(num_iters, batch_size)
    if record:
        best_trace, positions = out
        return best_x, best_y, evals, best_trace, {"kind": "agents", "positions": positions}
    return best_x, best_y, evals, out
