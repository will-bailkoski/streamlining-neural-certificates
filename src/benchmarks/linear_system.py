"""
Discrete-time randomly switched linear stochastic system.

    x_{t+1} = A_sigma x_t,   sigma ~ Bernoulli(p)

The Bernoulli switch is *internal* (it selects the dynamics mode), so it is
encoded directly as exact finite branches — there is no additive noise. Works in
every engine (the dynamics are linear and the branches are exact).

Two ways to build one:

  * the default 2D system (Chatterjee & Pal 2009 matrices), which is contractive
    in the L1 norm, so V(x) = |x0| + |x1| is an exact, PWL supermartingale; or
  * `SwitchedLinearEnv.random_construction(ndims, lip)`, which draws two random
    matrices scaled to spectral norm `lip < 1` in any dimension, for which
    V(x) = ||x||_2 is a valid supermartingale.

`known_V` adapts to the construction (`cert_norm`: "l1" exact-PWL, or "l2").
"""

import numpy as np
import gurobipy as gp
from jax import numpy as jnp
from jax import random as jrn

from src.benchmarks.env import Env
from src.benchmarks.convex_sets import HyperRectangle, Hypersphere

# Default 2D matrices (both L1-contractive: ||A0||_1=0.8, ||A1||_1=0.9).
_DEFAULT_A0 = [[0.6, -0.1], [0.2, 0.5]]
_DEFAULT_A1 = [[0.4, 0.2], [-0.1, 0.7]]


class SwitchedLinearEnv(Env):
    def __init__(
        self,
        A0=None,
        A1=None,
        p: float = 0.5,
        eq_center=None,
        eq_radius: float = 0.1,
        domain_bounds=None,
        cert_norm: str = "l1",
    ):
        self.A0 = jnp.array(_DEFAULT_A0 if A0 is None else A0)
        self.A1 = jnp.array(_DEFAULT_A1 if A1 is None else A1)
        assert self.A0.shape == self.A1.shape and self.A0.shape[0] == self.A0.shape[1]
        dim = self.A0.shape[0]

        self.p = p
        self.cert_norm = cert_norm
        self.eq_center = jnp.zeros(dim) if eq_center is None else jnp.array(eq_center)
        self.eq_radius = eq_radius
        self.domain_bounds = (
            [[-5.0, 5.0]] * dim if domain_bounds is None else domain_bounds
        )

        a0_lip = jnp.linalg.norm(self.A0, ord=2).item()
        a1_lip = jnp.linalg.norm(self.A1, ord=2).item()
        if p == 1.0:
            lip = a1_lip
        elif p == 0.0:
            lip = a0_lip
        else:
            lip = max(a0_lip, a1_lip)

        super().__init__(
            lip_f=lip,
            lip_t=lip,
            domain=HyperRectangle.from_bounds(jnp.array(self.domain_bounds)),
            equilibrium=Hypersphere(self.eq_center, eq_radius),
            name=f"SwitchedLinear{dim}D(L={lip:.3f})",
        )

    @classmethod
    def random_construction(
        cls,
        ndims: int,
        lip: float,
        p: float = 0.5,
        key=None,
        eq_radius: float = 0.1,
        domain_half: float = 5.0,
    ):
        """
        Two random ndims x ndims matrices, each scaled to spectral norm `lip`.

        For lip < 1 both modes are L2-contractive, so V(x) = ||x||_2 is a valid
        (and strict, on domain \\ equilibrium) supermartingale. The draw is
        deterministic in `key`, so `make_env(name)` always rebuilds the same
        system.
        """
        if key is None:
            key = jrn.key(0)
        k0, k1 = jrn.split(key)

        def scaled(k):
            A = jrn.normal(k, (ndims, ndims))
            return A * (lip / jnp.linalg.norm(A, ord=2))

        return cls(
            A0=scaled(k0),
            A1=scaled(k1),
            p=p,
            eq_radius=eq_radius,
            domain_bounds=[[-domain_half, domain_half]] * ndims,
            cert_norm="l2",
        )

    # ------------------------------------------------------------------
    # JAX
    # ------------------------------------------------------------------
    def _dynamics(self, x, key):
        key, sub = jrn.split(key)
        sigma = jrn.bernoulli(sub, p=self.p)
        A = jnp.where(sigma, self.A1, self.A0)
        return A @ x, key

    def known_V(self, x: jnp.ndarray) -> jnp.ndarray:
        """A valid supermartingale: L1 norm (default matrices) or L2 (random)."""
        if self.cert_norm == "l2":
            return jnp.linalg.norm(x)
        return jnp.sum(jnp.abs(x))

    # ------------------------------------------------------------------
    # Symbolic: exact finite branches over the two modes (any dimension)
    # ------------------------------------------------------------------
    def _gurobi_dynamics(self, model: gp.Model, branches: list[tuple]):
        D = self.dim
        new_branches = []
        for b_idx, (w, x_vars) in enumerate(branches):
            for i, (prob, A) in enumerate([(1 - self.p, self.A0), (self.p, self.A1)]):
                x_next = model.addVars(D, lb=-gp.GRB.INFINITY, name=f"x_next_{b_idx}_{i}")
                A_np = np.array(A)
                for row in range(D):
                    model.addConstr(
                        x_next[row]
                        == gp.quicksum(A_np[row, col] * x_vars[col] for col in range(D))
                    )
                new_branches.append((w * prob, list(x_next.values())))
        return new_branches

    def _z3_dynamics(self, solver, branches: list[tuple]):
        import z3

        D = self.dim
        new_branches = []
        for b_idx, (w, x_vars) in enumerate(branches):
            for i, (prob, A) in enumerate([(1 - self.p, self.A0), (self.p, self.A1)]):
                x_next = [z3.Real(f"x_next_{b_idx}_{i}_{d}") for d in range(D)]
                A_np = np.array(A)
                for row in range(D):
                    expr = z3.Sum([A_np[row, col] * x_vars[col] for col in range(D)])
                    solver.add(x_next[row] == expr)
                new_branches.append((w * prob, x_next))
        return new_branches

    # ------------------------------------------------------------------
    # autoLiRPA: drift(x) = (1-p) V(A0 x) + p V(A1 x) - V(x) + eps
    # The switch is discrete (no continuous noise), so noise_disc is ignored
    # and there is no extra noise input (noise_box = None).
    # ------------------------------------------------------------------
    def lirpa_drift_module(self, V_module, epsilon: float, noise_disc: int = 1):
        import torch
        import torch.nn as nn

        modes = [(1.0 - self.p, np.array(self.A0)), (self.p, np.array(self.A1))]

        class _Drift(nn.Module):
            def __init__(self):
                super().__init__()
                self.V = V_module
                self.eps = float(epsilon)
                self.weights = [float(w) for w, _ in modes]
                self.mode_maps = nn.ModuleList()
                for _, A in modes:
                    lin = nn.Linear(A.shape[1], A.shape[0], bias=False)
                    with torch.no_grad():
                        lin.weight.copy_(torch.tensor(A, dtype=torch.float32))
                    self.mode_maps.append(lin)

            def forward(self, x):
                out = -self.V(x) + self.eps
                for w, lin in zip(self.weights, self.mode_maps):
                    out = out + w * self.V(lin(x))
                return out

        return _Drift(), None
