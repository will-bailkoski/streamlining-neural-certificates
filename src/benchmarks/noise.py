"""
Additive noise sources.

One noise object serves every backend:
  - JAX:    `sample(key, shape)` draws true (continuous or discrete) samples,
            and `step` applies them additively for trajectory rollouts.
  - Symbolic (Z3 / Gurobi): `get_support()` partitions the noise support into
            cells `(lo, hi, weight)`; each cell is applied as a *box*
            `x_next ∈ x_old + [lo, hi]` (see `*_apply_box`).

The box encoding is the single mechanism behind sound expectation bounds:
  * discrete noise → lo == hi, so the box collapses to a point  ⇒ EXACT.
  * continuous noise → lo < hi, so the solver may pick the worst point in the
    cell ⇒ a sound OVER-APPROXIMATION of E[V] (never falsely "verified").

This is the fix for the old behaviour, which used the cell midpoint as a
deterministic representative — exact for discrete noise but unsound for
continuous noise.
"""

import abc
import jax.numpy as jnp
import jax.random as jrn
import numpy as np

import gurobipy as gp


class Noise(abc.ABC):
    """Base class for bounded additive noise."""

    def __init__(self, discretisation, lo, hi, dim):
        self.discretisation = discretisation
        self.lo = lo
        self.hi = hi
        self.dim = dim

    @abc.abstractmethod
    def sample(self, key, shape):
        """Returns 'true' continuous or discrete samples."""
        pass

    @abc.abstractmethod
    def _support(self, disc: int) -> tuple:
        """Returns (los, his, weights) for the support split into `disc` cells/dim."""
        pass

    def _cartesian_support(self, per_dim_los, per_dim_his, per_dim_weights):
        """
        Given per-dimension 1D supports, return the full Cartesian product.

        per_dim_los: list of D arrays, each shape (k_d,)
        -> los, his, weights each shape (prod(k_d), D) resp. (prod(k_d),)

        Blowup: n_cells = prod(k_d) — exponential in D. Documented limitation.
        """
        grids_lo = np.meshgrid(*per_dim_los, indexing="ij")
        grids_hi = np.meshgrid(*per_dim_his, indexing="ij")
        grids_w = np.meshgrid(*per_dim_weights, indexing="ij")

        los = np.stack([g.ravel() for g in grids_lo], axis=-1)
        his = np.stack([g.ravel() for g in grids_hi], axis=-1)
        # Joint weight = product of marginal weights (independence assumption).
        weights = np.prod(np.stack([g.ravel() for g in grids_w], axis=-1), axis=-1)
        return los, his, weights

    def get_support(self, disc: int | None = None):
        """
        Full (los, his, weights) over all D dimensions; weights sum to 1.

        `disc` overrides the construction-time discretisation per dimension — used
        by the lirpa/symbolic engines to bin continuous noise at a tunable
        resolution. Discrete noises ignore it.
        """
        return self._support(self.discretisation if disc is None else disc)

    # ------------------------------------------------------------------
    # JAX (true samples, additive)
    # ------------------------------------------------------------------
    def apply(self, x, noise):
        return x + noise

    def step(self, x, key):
        key, new = jrn.split(key)
        return self.apply(x, self.sample(key, x.shape)), new

    # ------------------------------------------------------------------
    # Symbolic: apply one noise CELL as a box  x_new ∈ x_old + [lo, hi]
    # ------------------------------------------------------------------
    def gurobi_apply_box(
        self, model: gp.Model, x_old: list, lo, hi, b_idx: int, s_idx: int, stage: str = ""
    ) -> list:
        D = len(x_old)
        lo = np.atleast_1d(lo)
        hi = np.atleast_1d(hi)
        assert len(lo) == D and len(hi) == D, "noise cell dim mismatch"
        x_new = model.addVars(D, lb=-gp.GRB.INFINITY, name=f"x_noise_{stage}_{b_idx}_{s_idx}")
        for i in range(D):
            model.addConstr(x_new[i] >= x_old[i] + float(lo[i]))
            model.addConstr(x_new[i] <= x_old[i] + float(hi[i]))
        return list(x_new.values())

    def z3_apply_box(
        self, solver, x_old: list, lo, hi, b_idx: int, s_idx: int, stage: str = ""
    ) -> list:
        import z3

        D = len(x_old)
        lo = np.atleast_1d(lo)
        hi = np.atleast_1d(hi)
        assert len(lo) == D and len(hi) == D, "noise cell dim mismatch"
        # `stage` keeps pre- and post-noise variables distinct: z3.Real aliases
        # by name, so a shared name would silently equate pre and post states.
        x_new = [z3.Real(f"x_noise_{stage}_{b_idx}_{s_idx}_{i}") for i in range(D)]
        for i in range(D):
            solver.add(x_new[i] >= x_old[i] + float(lo[i]))
            solver.add(x_new[i] <= x_old[i] + float(hi[i]))
        return x_new


class NullNoise(Noise):
    """No noise: a single degenerate cell at 0 with weight 1 (box collapses)."""

    def __init__(self, dim, discretisation=1):
        super().__init__(discretisation, 0.0, 0.0, dim)

    def sample(self, key, shape):
        return jnp.zeros(shape)

    def _support(self, disc):
        return self._cartesian_support(
            [jnp.array([0.0])] * self.dim,
            [jnp.array([0.0])] * self.dim,
            [jnp.array([1.0])] * self.dim,
        )


class Uniform(Noise):
    """Uniform(low, high) per dimension, split into `discretisation` equal-mass cells."""

    def __init__(self, low, high, dim, discretisation=1):
        super().__init__(discretisation, low, high, dim)

    def sample(self, key, shape):
        return jrn.uniform(key, shape, minval=self.lo, maxval=self.hi)

    def _support(self, disc):
        edges = jnp.linspace(self.lo, self.hi, disc + 1)
        los = edges[:-1]
        his = edges[1:]
        weights = jnp.ones(disc) / disc
        return self._cartesian_support(
            [los] * self.dim, [his] * self.dim, [weights] * self.dim
        )


class Rademacher(Noise):
    """±scale with probability 1/2 each (naturally discrete: lo == hi)."""

    def __init__(self, dim, scale=1.0):
        super().__init__(discretisation=2, lo=-scale, hi=scale, dim=dim)
        self.scale = scale

    def sample(self, key, shape):
        return self.scale * (2.0 * jrn.bernoulli(key, shape=shape).astype(jnp.float32) - 1.0)

    def _support(self, disc):
        vals = jnp.array([-self.scale, self.scale])  # naturally discrete; disc ignored
        weights = jnp.array([0.5, 0.5])
        return self._cartesian_support(
            [vals] * self.dim, [vals] * self.dim, [weights] * self.dim
        )


class Triangular(Noise):
    """Triangular(low, peak, high), split into `discretisation` equal-mass cells."""

    def __init__(self, low, peak, high, dim, discretisation=1):
        super().__init__(discretisation, low, high, dim)
        self.peak = peak

    def sample(self, key, shape):
        return jrn.triangular(key, left=self.lo, mode=self.peak, right=self.hi, shape=shape)

    def _icdf(self, p):
        fc = (self.peak - self.lo) / (self.hi - self.lo)
        return jnp.where(
            p < fc,
            self.lo + jnp.sqrt(p * (self.hi - self.lo) * (self.peak - self.lo)),
            self.hi - jnp.sqrt((1.0 - p) * (self.hi - self.lo) * (self.hi - self.peak)),
        )

    def _support(self, disc):
        quantiles = jnp.linspace(0.0, 1.0, disc + 1)
        edges = self._icdf(quantiles)
        los = edges[:-1]
        his = edges[1:]
        weights = jnp.ones(disc) / disc
        return self._cartesian_support(
            [los] * self.dim, [his] * self.dim, [weights] * self.dim
        )


class BernoulliSwitch(Noise):
    """
    Binary mode switch (sigma ∈ {0, 1}); purely discrete so lo == hi.

    Used by switched-linear-style systems where the switch selects between
    dynamics modes inside `_dynamics` rather than acting as additive noise.
    """

    def __init__(self, p: float, dim: int):
        super().__init__(discretisation=2, lo=0.0, hi=1.0, dim=dim)
        self.p = p

    def sample(self, key, shape):
        return jrn.bernoulli(key, p=self.p, shape=shape).astype(jnp.float32)

    def _support(self, disc):
        val = jnp.array([0.0, 1.0])  # naturally discrete; disc ignored
        weights = jnp.array([1.0 - self.p, self.p])
        # A single scalar switch shared across the state (dim-1 support).
        return self._cartesian_support([val], [val], [weights])
