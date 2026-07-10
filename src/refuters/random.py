"""Random search: sample a fresh uniform batch each iteration, keep the best."""

import jax.numpy as jnp
import jax.random as jrn
from jax import lax

from src.refuters.base import register, uniform_evals_trace


@register("random")
def random_refuter(objective, domain, key, *, budget, batch_size=256, record=False, **hp):
    num_iters = max(budget // batch_size, 1)
    low, high = domain[:, 0], domain[:, 1]
    dim = domain.shape[0]

    def body(carry, _):
        best_x, best_y, k = carry
        k, ks, ke = jrn.split(k, 3)
        xs = jrn.uniform(ks, (batch_size, dim), minval=low, maxval=high)
        ys = objective(xs, ke)
        i = jnp.argmax(ys)
        better = ys[i] > best_y
        best_x = jnp.where(better, xs[i], best_x)
        best_y = jnp.maximum(best_y, ys[i])
        return (best_x, best_y, k), ((best_y, xs) if record else best_y)

    init = (0.5 * (low + high), -jnp.inf, key)
    (best_x, best_y, _), out = lax.scan(body, init, None, length=num_iters)
    evals = uniform_evals_trace(num_iters, batch_size)
    if record:
        best_trace, positions = out
        return best_x, best_y, evals, best_trace, {"kind": "points", "positions": positions}
    return best_x, best_y, evals, out
