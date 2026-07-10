"""
Time-reversed Van der Pol oscillator with additive bounded noise.

The (forward) Van der Pol system has an unstable origin and a stable limit
cycle, so it is NOT stabilisable to the origin. The *time-reversed* system,

    x' = -y
    y' =  x - mu (1 - x^2) y

has an asymptotically stable origin (and an unstable limit cycle bounding its
region of attraction). This is the variant used as a Lyapunov / region-of-
attraction benchmark in FOSSIL (Abate, Giacobbe, Peruffo); here we take its
discrete-time Euler map plus small additive noise:

    x_{k+1} = x_k + dt (-y_k)            + w_x
    y_{k+1} = y_k + dt (x_k - mu (1 - x_k^2) y_k) + w_y

The domain must lie inside the region of attraction (inside the unstable limit
cycle, amplitude ~2 for mu=1), so it defaults to a conservative box. The
dynamics are polynomial, so Z3/Gurobi encode them exactly (as in the double
well) and autoLiRPA bounds the x^2 y term via its multiply relaxation. No
analytic `known_V` — the certificate is learned (the noise drives a.s. stability
to a small neighbourhood, not the exact origin).

`lip_f` is a *rigorous* upper bound on the one-step state Lipschitz: an
entrywise box bound on the discrete Jacobian Jg = I + dt J, then its spectral
norm (||A||_2 <= ||M||_2 whenever M >= |A| entrywise).
"""

import numpy as np
import gurobipy as gp
from jax import numpy as jnp

from src.benchmarks.env import Env
from src.benchmarks.noise import Uniform, Triangular
from src.benchmarks.convex_sets import Ellipsoid, Hypersphere
from src.benchmarks.linear_stochastic import _solve_discrete_lyapunov


class VanDerPolEnv(Env):
    def __init__(
        self,
        mu: float = 1.0,
        dt: float = 0.1,
        noise_scale: float = 0.02,
        noise_kind: str = "uniform",
        noise_disc: int = 1,
        level: float = 6.0,   # Lyapunov sublevel {x: x^T P x <= level}; must stay in the ROA
        eq_radius: float = 0.2,
    ):
        self.mu = mu
        self.dt = dt
        self.noise_scale = noise_scale
        self.noise_disc = noise_disc
        dim = 2

        if noise_kind == "uniform":
            post = Uniform(-noise_scale, noise_scale, dim=dim, discretisation=noise_disc)
        elif noise_kind == "triangular":
            post = Triangular(-noise_scale, 0.0, noise_scale, dim=dim, discretisation=noise_disc)
        else:
            raise ValueError(f"unknown noise_kind {noise_kind!r}")

        # Invariant domain: the reversed VdP is Schur-but-the-ROA-is-bounded, so a
        # box leaks. Use the Lyapunov sublevel ellipsoid of the linearisation at 0:
        # J(0) = [[0, -1], [1, -mu]], A_d = I + dt J(0), P from A_d^T P A_d - P = -I.
        A_d = np.eye(2) + dt * np.array([[0.0, -1.0], [1.0, -mu]])
        self.P = _solve_discrete_lyapunov(A_d, np.eye(2))
        domain = Ellipsoid(jnp.zeros(dim), jnp.array(self.P / level))
        self.level = level

        # Rigorous one-step Lipschitz over the ellipsoid's bounding box: entrywise
        # bound on Jg = I + dt*J, J = [[0,-1],[1 + 2 mu x y, -mu(1 - x^2)]].
        xm, ym = [float(h) for h in np.array(domain.bounds[:, 1])]
        M = np.array([
            [1.0, dt],
            [dt * (1.0 + 2.0 * mu * xm * ym), abs(1.0 - dt) + dt * mu * xm * xm],
        ])
        lip = float(np.linalg.norm(M, ord=2))

        super().__init__(
            lip_f=lip,
            lip_t=lip,
            domain=domain,
            equilibrium=Hypersphere(jnp.zeros(dim), eq_radius),
            post_noise=post,
            name=f"VanDerPol(mu={mu},L={lip:.3f})",
        )

    # ------------------------------------------------------------------
    # JAX (deterministic Euler step; additive noise added by post_noise)
    # ------------------------------------------------------------------
    def _dynamics(self, x, key):
        x0, x1 = x[0], x[1]
        nx = x0 + self.dt * (-x1)
        ny = x1 + self.dt * (x0 - self.mu * (1.0 - x0 * x0) * x1)
        return jnp.array([nx, ny]), key

    # ------------------------------------------------------------------
    # Symbolic (polynomial; noise boxes added by Env.*_step)
    # ------------------------------------------------------------------
    def _gurobi_dynamics(self, model: gp.Model, branches: list[tuple]):
        out = []
        for b_idx, (w, x_vars) in enumerate(branches):
            x0, x1 = x_vars[0], x_vars[1]
            x0_sq = model.addVar(lb=-gp.GRB.INFINITY, name=f"x0_sq_{b_idx}")
            model.addConstr(x0_sq == x0 * x0)
            # term = (1 - x0^2) * x1
            term = model.addVar(lb=-gp.GRB.INFINITY, name=f"vdp_term_{b_idx}")
            model.addConstr(term == (1.0 - x0_sq) * x1)
            x_next = [
                model.addVar(lb=-gp.GRB.INFINITY, name=f"x_next_0_{b_idx}"),
                model.addVar(lb=-gp.GRB.INFINITY, name=f"x_next_1_{b_idx}"),
            ]
            model.addConstr(x_next[0] == x0 + self.dt * (-x1))
            model.addConstr(x_next[1] == x1 + self.dt * (x0 - self.mu * term))
            out.append((w, x_next))
        return out

    def _z3_dynamics(self, solver, branches: list[tuple]):
        import z3

        out = []
        for b_idx, (w, x_vars) in enumerate(branches):
            x0, x1 = x_vars[0], x_vars[1]
            x_next = [z3.Real(f"x_next_0_{b_idx}"), z3.Real(f"x_next_1_{b_idx}")]
            solver.add(x_next[0] == x0 + self.dt * (-x1))
            solver.add(x_next[1] == x1 + self.dt * (x0 - self.mu * (1.0 - x0 * x0) * x1))
            out.append((w, x_next))
        return out

    # ------------------------------------------------------------------
    # autoLiRPA: additive noise binned; the x^2 y term uses the multiply relax.
    # ------------------------------------------------------------------
    def lirpa_drift_module(self, V_module, epsilon: float, noise_disc: int = 1):
        from src.benchmarks._lirpa import cell_drift_module

        los, his, weights = self._post.get_support(noise_disc)
        module = cell_drift_module(
            V_module, self._torch_step(), weights, noise_dim=self.dim, epsilon=epsilon
        )
        noise_box = (np.asarray(los).reshape(-1), np.asarray(his).reshape(-1))
        return module, noise_box

    def _torch_step(self):
        import torch
        import torch.nn as nn

        mu, dt = float(self.mu), float(self.dt)

        class _Step(nn.Module):
            def forward(self, x, w):  # x:(B,2) w:(B,2)
                x0 = x[:, 0:1]
                x1 = x[:, 1:2]
                nx = x0 + dt * (-x1)
                ny = x1 + dt * (x0 - mu * (1.0 - x0 * x0) * x1)
                return torch.cat([nx, ny], dim=1) + w

        return _Step()
