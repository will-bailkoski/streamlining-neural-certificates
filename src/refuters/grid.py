"""Grid search: evaluate a fixed regular lattice over the domain, batch by batch."""

import jax.numpy as jnp
import jax.random as jrn
from jax import lax

from src.refuters.base import register, uniform_evals_trace


@register("grid")
def grid_refuter(objective, domain, key, *, budget, batch_size=256, record=False, **hp):
    low, high = domain[:, 0], domain[:, 1]
    dim = domain.shape[0]

    # ~budget points: p per axis so p**dim <= budget.
    p = max(int(round(budget ** (1.0 / dim))), 2)
    axes = [jnp.linspace(low[i], high[i], p) for i in range(dim)]
    mesh = jnp.meshgrid(*axes, indexing="ij")
    grid = jnp.stack([m.ravel() for m in mesh], axis=-1)  # (p**dim, dim)

    # clamp batch to the number of grid points (handles budget < batch_size)
    bs = min(batch_size, grid.shape[0])
    num_iters = max(grid.shape[0] // bs, 1)
    grid = grid[: num_iters * bs].reshape(num_iters, bs, dim)

    def body(carry, batch):
        best_x, best_y, k = carry
        k, ke = jrn.split(k)  # grid points are fixed; key only feeds the objective
        ys = objective(batch, ke)
        i = jnp.argmax(ys)
        better = ys[i] > best_y
        best_x = jnp.where(better, batch[i], best_x)
        best_y = jnp.maximum(best_y, ys[i])
        return (best_x, best_y, k), ((best_y, batch) if record else best_y)

    init = (0.5 * (low + high), -jnp.inf, key)
    (best_x, best_y, _), out = lax.scan(body, init, grid)
    evals = uniform_evals_trace(num_iters, bs)
    if record:
        best_trace, positions = out
        return best_x, best_y, evals, best_trace, {"kind": "points", "positions": positions}
    return best_x, best_y, evals, out
