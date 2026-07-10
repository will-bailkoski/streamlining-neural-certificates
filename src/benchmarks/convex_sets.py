"""
Convex sets used for environment domains and equilibria.

Each set exposes one definition that every backend consumes:
  - JAX:    `contains` (membership) and `sample` (uniform interior sample)
  - Gurobi: `encode_gurobi_inclusion` / `encode_gurobi_exclusion`
  - Z3:     `encode_z3_inclusion`   / `encode_z3_exclusion`

This is the single source of truth for "where does the system live" and
"what region counts as the goal/equilibrium" across the sampling, SMT, MILP
and autoLiRPA engines.
"""

from __future__ import annotations
import abc
import itertools

import numpy as np
import jax.numpy as jnp
import jax.random as jrn

import gurobipy as gp


class ConvexSet(abc.ABC):
    def __init__(self, center):
        self.center = jnp.array(center)

    @property
    @abc.abstractmethod
    def bounds(self) -> jnp.ndarray:
        """(D, 2) bounding box — used by the base Env for dim inference."""
        pass

    @abc.abstractmethod
    def sample(self, key, batch_size: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Uniform sample from the set interior. Returns (xs, new_key)."""
        pass

    @abc.abstractmethod
    def contains(self, x: jnp.ndarray) -> jnp.ndarray:
        """JAX-compatible membership test for a single point."""
        pass

    @abc.abstractmethod
    def encode_gurobi_inclusion(self, model: gp.Model, x_vars: list) -> None:
        """Add constraints: x in set."""
        pass

    @abc.abstractmethod
    def encode_gurobi_exclusion(self, model: gp.Model, x_vars: list) -> None:
        """Add constraints: x not in set."""
        pass

    @abc.abstractmethod
    def encode_z3_inclusion(self, solver, x_vars: list) -> None:
        """Add Z3 constraints: x in set."""
        pass

    @abc.abstractmethod
    def encode_z3_exclusion(self, solver, x_vars: list) -> None:
        """Add Z3 constraints: x not in set."""
        pass


class EmptySet(ConvexSet):
    """The empty set ∅ — a null equilibrium (nothing is ever excluded)."""

    def __init__(self, dim: int = 0):
        super().__init__(jnp.zeros(dim))
        self.dim = dim

    @property
    def bounds(self) -> jnp.ndarray:
        return jnp.stack(
            [jnp.full((self.dim,), jnp.inf), jnp.full((self.dim,), -jnp.inf)],
            axis=-1,
        )

    def sample(self, key, batch_size: int):
        raise ValueError("Cannot sample from the empty set.")

    def contains(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array(False)

    def encode_gurobi_inclusion(self, model: gp.Model, x_vars: list) -> None:
        model.addConstr(0 >= 1)  # infeasible

    def encode_gurobi_exclusion(self, model: gp.Model, x_vars: list) -> None:
        pass  # always true

    def encode_z3_inclusion(self, solver, x_vars: list) -> None:
        import z3

        solver.add(z3.BoolVal(False))  # infeasible

    def encode_z3_exclusion(self, solver, x_vars: list) -> None:
        pass  # always true


class HyperRectangle(ConvexSet):
    """Axis-aligned box: |x - center| <= half_widths (elementwise)."""

    def __init__(self, center, half_widths):
        super().__init__(center)
        self.half_widths = jnp.array(half_widths)

    @classmethod
    def from_bounds(cls, bound_intervals: jnp.ndarray) -> "HyperRectangle":
        bound_intervals = jnp.array(bound_intervals)
        center = jnp.mean(bound_intervals, axis=1)
        half_widths = (bound_intervals[:, 1] - bound_intervals[:, 0]) / 2.0
        return HyperRectangle(center, half_widths)

    @property
    def bounds(self) -> jnp.ndarray:
        lower = self.center - self.half_widths
        upper = self.center + self.half_widths
        return jnp.stack([lower, upper], axis=-1)

    @property
    def corners(self) -> jnp.ndarray:
        """All 2^D vertices of the box — handy for seeding a mesh triangulation."""
        lower = self.center - self.half_widths
        upper = self.center + self.half_widths
        per_dim = [jnp.array([lo, hi]) for lo, hi in zip(lower, upper)]
        return jnp.array(list(itertools.product(*per_dim)))

    def sample(self, key, batch_size: int):
        lower = self.center - self.half_widths
        upper = self.center + self.half_widths
        key, sub = jrn.split(key)
        xs = jrn.uniform(
            sub,
            shape=(batch_size, self.center.shape[0]),
            minval=lower,
            maxval=upper,
        )
        return xs, key

    def contains(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.all(jnp.abs(x - self.center) <= self.half_widths)

    def encode_gurobi_inclusion(self, model, x_vars):
        for i, xv in enumerate(x_vars):
            model.addConstr(xv >= float(self.center[i] - self.half_widths[i]))
            model.addConstr(xv <= float(self.center[i] + self.half_widths[i]))

    def encode_gurobi_exclusion(self, model, x_vars):
        # x outside the box: at least one coordinate beyond a face (disjunctive / big-M).
        D = len(x_vars)
        z = model.addVars(D, vtype=gp.GRB.BINARY)
        M = 1e6
        for i, xv in enumerate(x_vars):
            lo = float(self.center[i] - self.half_widths[i])
            hi = float(self.center[i] + self.half_widths[i])
            # If z[i]==1, force xv <= lo (below face) OR xv >= hi (above face) via a second binary.
            below = model.addVar(vtype=gp.GRB.BINARY)
            model.addConstr(xv <= lo + M * (1 - z[i]) + M * (1 - below))
            model.addConstr(xv >= hi - M * (1 - z[i]) - M * below)
        model.addConstr(gp.quicksum(z[i] for i in range(D)) >= 1)

    def encode_z3_inclusion(self, solver, x_vars):
        for i, xv in enumerate(x_vars):
            solver.add(xv >= float(self.center[i] - self.half_widths[i]))
            solver.add(xv <= float(self.center[i] + self.half_widths[i]))

    def encode_z3_exclusion(self, solver, x_vars):
        import z3

        clauses = []
        for i, xv in enumerate(x_vars):
            lo = float(self.center[i] - self.half_widths[i])
            hi = float(self.center[i] + self.half_widths[i])
            clauses.append(xv < lo)
            clauses.append(xv > hi)
        solver.add(z3.Or(clauses))


class Hypersphere(ConvexSet):
    """Euclidean ball: ||x - center||_2 <= radius."""

    def __init__(self, center, radius: float):
        super().__init__(center)
        self.radius = radius

    @property
    def bounds(self) -> jnp.ndarray:
        lower = self.center - self.radius
        upper = self.center + self.radius
        return jnp.stack([lower, upper], axis=-1)

    def sample(self, key, batch_size: int):
        dim = self.center.shape[0]
        key, sub = jrn.split(key)
        key_dirs, key_radii = jrn.split(sub)
        dirs = jrn.normal(key_dirs, (batch_size, dim))
        dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
        radii = jrn.uniform(key_radii, (batch_size,)) ** (1.0 / dim)
        xs = self.center + dirs * (radii[:, None] * self.radius)
        return xs, key

    def contains(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.norm(x - self.center) <= self.radius

    def encode_gurobi_inclusion(self, model, x_vars):
        expr = gp.quicksum(
            (x_vars[i] - float(self.center[i])) * (x_vars[i] - float(self.center[i]))
            for i in range(len(x_vars))
        )
        model.addQConstr(expr <= self.radius**2)

    def encode_gurobi_exclusion(self, model, x_vars):
        expr = gp.quicksum(
            (x_vars[i] - float(self.center[i])) * (x_vars[i] - float(self.center[i]))
            for i in range(len(x_vars))
        )
        model.addQConstr(expr >= self.radius**2 + 1e-6)

    def encode_z3_inclusion(self, solver, x_vars):
        import z3

        expr = z3.Sum(
            [
                (x_vars[i] - float(self.center[i]))
                * (x_vars[i] - float(self.center[i]))
                for i in range(len(x_vars))
            ]
        )
        solver.add(expr <= self.radius**2)

    def encode_z3_exclusion(self, solver, x_vars):
        import z3

        expr = z3.Sum(
            [
                (x_vars[i] - float(self.center[i]))
                * (x_vars[i] - float(self.center[i]))
                for i in range(len(x_vars))
            ]
        )
        solver.add(expr >= self.radius**2 + 1e-6)


class Ellipsoid(ConvexSet):
    """Quadratic sublevel set ``{x : (x-c)^T M (x-c) <= 1}``, M symmetric PD.

    The natural forward-invariant domain for a NON-NORMAL stable system: a box or
    ball need not be invariant under such dynamics (transient growth, ``||A||_2 >
    1`` even when Schur), but the ellipsoidal sublevel sets of the quadratic
    Lyapunov function ``V(x) = x^T P x`` always are. Use ``M = P / level`` for the
    sublevel set ``{x : V(x) <= level}``.
    """

    def __init__(self, center, M):
        super().__init__(center)
        self.M = jnp.asarray(M, dtype=float)
        # Tight axis-aligned bounding box: max |x_i - c_i| over the set is
        # sqrt((M^{-1})_ii).
        self._half = jnp.sqrt(jnp.diag(jnp.linalg.inv(self.M)))

    @property
    def bounds(self) -> jnp.ndarray:
        return jnp.stack([self.center - self._half, self.center + self._half], axis=-1)

    @property
    def corners(self) -> jnp.ndarray:
        """The 2^D vertices of the bounding box (for seeding a mesh)."""
        lower, upper = self.center - self._half, self.center + self._half
        per_dim = [jnp.array([lo, hi]) for lo, hi in zip(lower, upper)]
        return jnp.array(list(itertools.product(*per_dim)))

    def sample(self, key, batch_size: int):
        # Uniform-in-volume: unit ball, then map by M^{-1/2}.
        key, kd, kr = jrn.split(key, 3)
        dim = self.center.shape[0]
        dirs = jrn.normal(kd, (batch_size, dim))
        dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
        radii = jrn.uniform(kr, (batch_size,)) ** (1.0 / dim)
        ball = dirs * radii[:, None]
        L = jnp.linalg.cholesky(self.M)  # M = L L^T
        xs = self.center + ball @ jnp.linalg.inv(L)
        return xs, key

    def contains(self, x: jnp.ndarray) -> jnp.ndarray:
        d = x - self.center
        return d @ self.M @ d <= 1.0

    def _quad(self, x_vars, backend):
        """Sum_{i,j} M_ij (x_i - c_i)(x_j - c_j) as a backend expression."""
        D = len(x_vars)
        M = np.array(self.M)
        terms = []
        for i in range(D):
            di = x_vars[i] - float(self.center[i])
            for j in range(D):
                dj = x_vars[j] - float(self.center[j])
                terms.append(float(M[i, j]) * di * dj)
        if backend == "gurobi":
            return gp.quicksum(terms)
        import z3
        return z3.Sum(terms)

    def encode_gurobi_inclusion(self, model, x_vars):
        model.addQConstr(self._quad(x_vars, "gurobi") <= 1.0)

    def encode_gurobi_exclusion(self, model, x_vars):
        model.addQConstr(self._quad(x_vars, "gurobi") >= 1.0 + 1e-6)

    def encode_z3_inclusion(self, solver, x_vars):
        solver.add(self._quad(x_vars, "z3") <= 1.0)

    def encode_z3_exclusion(self, solver, x_vars):
        solver.add(self._quad(x_vars, "z3") >= 1.0 + 1e-6)
