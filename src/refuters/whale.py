"""
Whale Optimisation Algorithm (WOA) — Mirjalili & Lewis, Advances in Engineering
Software 95 (2016) 51-67, faithful to the author's reference MATLAB code.

A population of `batch_size` whales moves each iteration relative to the incumbent
leader X* by one of three behaviours, chosen exactly as in the reference:

    prob < p        : spiral / bubble-net   X* + |X* - X| e^{b l} cos(2 pi l)
    prob >= p, |A|<1: shrink-encircle        X* - A |C X* - X|
    prob >= p, |A|>=1: explore (random peer)  X_rand - A |C X_rand - X|

with a = a_max annealed linearly to ~0, A = 2 a r1 - a, C = 2 r2 (independent
draws r1, r2), l ~ U[-1, 1], and one random coin `prob` per agent. Per the
reference, r1, r2, l and prob are drawn once PER AGENT and reused across all
dimensions (so |A| is a scalar switch), and the leader is the best-so-far, frozen
during the whole position-update sweep. One population evaluation per iteration =
`batch_size` evals.

The reference constants are a_max = 2, b = 1, p = 0.5; they are exposed as
hyperparameters (p = the spiral probability, so p = 0.5 is the canonical split)
but default to those values. Original minimises; we mirror only the leader
selection to maximise.
"""

import jax.numpy as jnp
import jax.random as jrn
from jax import lax

from src.refuters.base import register, uniform_evals_trace


@register("whale")
def whale_refuter(objective, domain, key, *, budget, batch_size=64, a_max=2.0, b=1.0, p=0.5, record=False, **hp):
    num_iters = max(budget // batch_size, 1)
    low, high = domain[:, 0], domain[:, 1]
    dim = domain.shape[0]

    key, pop_key, eval_key = jrn.split(key, 3)
    pop = jrn.uniform(pop_key, (batch_size, dim), minval=low, maxval=high)
    fitness = objective(pop, eval_key)  # initial eval (counts as iteration 0)
    bi = jnp.argmax(fitness)
    best_x, best_y = pop[bi], fitness[bi]

    a_sched = a_max * (1.0 - jnp.arange(num_iters) / num_iters)  # a: a_max -> ~0 (Eq. 2.3)

    def body(carry, a):
        pop, best_x, best_y, k = carry
        k, r1_key, r2_key, l_key, prob_key, idx_key, eval_key = jrn.split(k, 7)
        # scalar-per-agent coefficients (reference draws these outside the dim loop)
        r1 = jrn.uniform(r1_key, (batch_size, 1))
        r2 = jrn.uniform(r2_key, (batch_size, 1))
        A = 2.0 * a * r1 - a                              # Eq. (2.3), independent of C
        C = 2.0 * r2                                      # Eq. (2.4)
        l = jrn.uniform(l_key, (batch_size, 1), minval=-1.0, maxval=1.0)  # Eq. (2.5), U[-1,1]
        prob = jrn.uniform(prob_key, (batch_size, 1))     # Eq. (2.6)

        rand_idx = jrn.randint(idx_key, (batch_size,), 0, batch_size)
        x_rand = pop[rand_idx]
        encircle = best_x - A * jnp.abs(C * best_x - pop)                 # Eq. (2.1)-(2.2)
        explore = x_rand - A * jnp.abs(C * x_rand - pop)                  # Eq. (2.7)-(2.8)
        spiral = jnp.abs(best_x - pop) * jnp.exp(b * l) * jnp.cos(2.0 * jnp.pi * l) + best_x  # Eq. (2.5)

        a_branch = jnp.where(jnp.abs(A) < 1.0, encircle, explore)         # |A|<1 exploit else explore
        pop = jnp.clip(jnp.where(prob < p, spiral, a_branch), low, high)  # prob<p -> spiral

        fitness = objective(pop, eval_key)
        li = jnp.argmax(fitness)
        better = fitness[li] > best_y
        best_x = jnp.where(better, pop[li], best_x)
        best_y = jnp.maximum(best_y, fitness[li])
        return (pop, best_x, best_y, k), ((best_y, pop) if record else best_y)

    (_, best_x, best_y, _), out = lax.scan(body, (pop, best_x, best_y, key), a_sched)
    evals = uniform_evals_trace(num_iters, batch_size)
    if record:
        best_trace, positions = out
        return best_x, best_y, evals, best_trace, {"kind": "agents", "positions": positions}
    return best_x, best_y, evals, out
