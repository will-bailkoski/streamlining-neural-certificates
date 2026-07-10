"""
2D stochastic gradient-flow system with a double-well potential.

    x_{k+1} = x_k - eta * grad U(x_k) + noise
    U(x, y) = 1/4 (x^2 - 1)^2 + 1/2 y^2 - 0.3 x
    grad U  = [x(x^2 - 1) - 0.3,  y]

The deterministic map  x - eta * grad U(x)  is `_dynamics`; the additive noise
is declared as `post_noise` so it is applied identically in the jax rollout and
in the symbolic (Z3 / Gurobi) encodings. The additive term reproduces the
original  sigma * Uniform(-sigma, sigma)  =  Uniform(-sigma^2, sigma^2)  exactly.
(The "double sigma" is preserved from the original model; raise `noise_disc`
for tighter symbolic over-approximation of the continuous noise.)

`known_V` is the potential itself, a ground-truth Lyapunov function.
"""

import numpy as np
import gurobipy as gp
from jax import numpy as jnp

from src.benchmarks.env import Env
from src.benchmarks.noise import Uniform
from src.benchmarks.convex_sets import HyperRectangle, Hypersphere


class DoubleWellEnv(Env):
    def __init__(
        self,
        eta: float = 0.05,
        sigma: float = 0.1,
        eq_center=(1.0, 0.0),
        eq_radius: float = 0.1,
        domain_bounds=((0.0, 3.0), (-1.5, 1.5)),
        noise_disc: int = 1,
    ):
        self.eta = eta
        self.sigma = sigma
        self.eq_center = jnp.array(eq_center)
        self.eq_radius = eq_radius
        self.domain_bounds = domain_bounds
        self.noise_disc = noise_disc

        domain = HyperRectangle.from_bounds(jnp.array(domain_bounds))
        dim = int(domain.bounds.shape[0])

        # Lipschitz of the deterministic map over the domain (worst eigenvalue).
        a, b = domain_bounds[0]
        candidates = [a, b]
        crit = 1.0 / np.sqrt(3)
        if a <= crit <= b:
            candidates.append(crit)
        if a <= -crit <= b:
            candidates.append(-crit)
        candidates = np.array(candidates)
        lam1 = np.max(np.abs(1 - eta * (3 * candidates**2 - 1)))
        lam2 = np.abs(1 - eta)
        lip = float(max(lam1, lam2))

        super().__init__(
            lip_f=lip,
            lip_t=lip,
            domain=domain,
            equilibrium=Hypersphere(self.eq_center, eq_radius),
            post_noise=Uniform(
                low=-sigma * sigma, high=sigma * sigma, dim=dim, discretisation=noise_disc
            ),
        )

    # ------------------------------------------------------------------
    # JAX (deterministic part; noise added by post_noise in Env.step)
    # ------------------------------------------------------------------
    def _dynamics(self, x, key):
        grad_u = jnp.array([x[0] * (x[0] ** 2 - 1) - 0.3, x[1]])
        return x - self.eta * grad_u, key

    @staticmethod
    def known_V(x: jnp.ndarray) -> jnp.ndarray:
        """Ground-truth Lyapunov: the potential function itself."""
        return 0.25 * (x[0] ** 2 - 1) ** 2 + 0.5 * x[1] ** 2 - 0.3 * x[0]

    # ------------------------------------------------------------------
    # Symbolic deterministic part (noise boxes added by Env.*_step)
    # ------------------------------------------------------------------
    def _gurobi_dynamics(self, model: gp.Model, branches: list[tuple]):
        new_branches = []
        for b_idx, (w, x_vars) in enumerate(branches):
            x0, x1 = x_vars[0], x_vars[1]
            x0_sq = model.addVar(lb=-gp.GRB.INFINITY, name=f"x0_sq_{b_idx}")
            model.addConstr(x0_sq == x0 * x0)
            grad_u0 = model.addVar(lb=-gp.GRB.INFINITY, name=f"grad_u0_{b_idx}")
            model.addConstr(grad_u0 == x0 * (x0_sq - 1) - 0.3)
            grad_u1 = x1
            x_next = [
                model.addVar(lb=-gp.GRB.INFINITY, name=f"x_next_0_{b_idx}"),
                model.addVar(lb=-gp.GRB.INFINITY, name=f"x_next_1_{b_idx}"),
            ]
            model.addConstr(x_next[0] == x0 - self.eta * grad_u0)
            model.addConstr(x_next[1] == x1 - self.eta * grad_u1)
            new_branches.append((w, x_next))
        return new_branches

    def _z3_dynamics(self, solver, branches: list[tuple]):
        import z3

        new_branches = []
        for b_idx, (w, x_vars) in enumerate(branches):
            x0, x1 = x_vars[0], x_vars[1]
            grad_u0 = x0 * (x0**2 - 1) - 0.3
            grad_u1 = x1
            x_next = [z3.Real(f"x_next_0_{b_idx}"), z3.Real(f"x_next_1_{b_idx}")]
            solver.add(x_next[0] == x0 - self.eta * grad_u0)
            solver.add(x_next[1] == x1 - self.eta * grad_u1)
            new_branches.append((w, x_next))
        return new_branches

    # ------------------------------------------------------------------
    # autoLiRPA: additive noise binned into cells (over-approximation)
    # ------------------------------------------------------------------
    def lirpa_drift_module(self, V_module, epsilon: float, noise_disc: int = 1):
        import numpy as np
        from src.benchmarks._lirpa import cell_drift_module

        los, his, weights = self._post.get_support(noise_disc)  # (K, D), (K, D), (K,)
        step = self._torch_step()
        module = cell_drift_module(
            V_module, step, weights, noise_dim=self.dim, epsilon=epsilon
        )
        noise_box = (np.asarray(los).reshape(-1), np.asarray(his).reshape(-1))
        return module, noise_box

    def _torch_step(self):
        """Torch next-state map: x_next = (x - eta grad U(x)) + w (additive)."""
        import torch
        import torch.nn as nn

        eta = float(self.eta)

        class _Step(nn.Module):
            def forward(self, x, w):  # x:(B,2), w:(B,2)
                x0 = x[:, 0:1]
                x1 = x[:, 1:2]
                g0 = x0 * (x0 * x0 - 1.0) - 0.3
                det0 = x0 - eta * g0
                det1 = x1 - eta * x1
                return torch.cat([det0, det1], dim=1) + w

        return _Step()
