"""
Discrete-time linear system with additive bounded noise.

    x_{t+1} = A x_t + w,    w ~ bounded zero-mean noise (uniform / triangular)

This is the additive-noise counterpart to the switched-linear system, and the
canonical "2D system" used across the stochastic-supermartingale literature
(Lechner, Žikelić, Chatterjee & Henzinger, AAAI 2022; Badings et al.). `A` is
Schur-stable (built at a target spectral norm < 1), so the deterministic part is
contractive and the only obstruction to stability is the noise floor.

Ground-truth certificate.  For `V(x) = x^T P x` with `P` the solution of the
discrete Lyapunov equation `A^T P A - P = -Q` (here `Q = I`),

    E[V(x')] - V(x) = -x^T Q x + tr(P Σ_w),

which is <= 0 exactly on the complement of a ball of radius
`r* = sqrt(tr(P Σ_w) / λ_min(Q))`. The equilibrium radius is therefore *derived
from the noise level* so that `known_V` is a genuine supermartingale on
`domain \\ equilibrium` — the textbook "a.s. stable to a neighbourhood" picture.

Linear dynamics + bounded additive noise ⇒ exact in Z3/Gurobi, sound (binned)
in autoLiRPA, and native in the sampling/MAB engines. Scales to any dimension.
"""

import numpy as np
import gurobipy as gp
from jax import numpy as jnp
from jax import random as jrn

from src.benchmarks.env import Env
from src.benchmarks.noise import Uniform, Triangular
from src.benchmarks.convex_sets import HyperRectangle, Hypersphere


def _solve_discrete_lyapunov(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Solve A^T P A - P = -Q for P via the Kronecker vectorisation."""
    n = A.shape[0]
    kron = np.kron(A.T, A.T)
    vecP = np.linalg.solve(np.eye(n * n) - kron, Q.reshape(-1))
    return vecP.reshape(n, n)


def _noise_covariance(noise) -> np.ndarray:
    """Diagonal covariance of a zero-mean bounded noise (per-dim variance)."""
    if isinstance(noise, Uniform):
        var = (float(noise.hi) - float(noise.lo)) ** 2 / 12.0
    elif isinstance(noise, Triangular):
        a, b, c = float(noise.lo), float(noise.peak), float(noise.hi)
        var = (a * a + b * b + c * c - a * b - a * c - b * c) / 18.0
    else:  # conservative fallback: treat support as a uniform box
        var = (float(noise.hi) - float(noise.lo)) ** 2 / 12.0
    return var * np.eye(noise.dim)


class LinearStochasticEnv(Env):
    def __init__(
        self,
        A=None,
        noise_scale: float = 0.1,
        noise_kind: str = "uniform",  # "uniform" | "triangular"
        noise_disc: int = 2,
        domain_half: float = 5.0,
        eq_radius: float | None = None,  # None -> derive from the noise floor
        eq_margin: float = 1.5,
    ):
        A = np.array([[0.5, 0.4], [-0.3, 0.5]]) if A is None else np.asarray(A, float)
        assert A.ndim == 2 and A.shape[0] == A.shape[1]
        self.A = jnp.array(A)
        dim = A.shape[0]
        self.noise_scale = noise_scale
        self.noise_kind = noise_kind
        self.noise_disc = noise_disc

        if noise_kind == "uniform":
            post = Uniform(
                -noise_scale, noise_scale, dim=dim, discretisation=noise_disc
            )
        elif noise_kind == "triangular":
            post = Triangular(
                -noise_scale, 0.0, noise_scale, dim=dim, discretisation=noise_disc
            )
        else:
            raise ValueError(f"unknown noise_kind {noise_kind!r}")

        # Ground-truth quadratic certificate and the noise-floor equilibrium.
        Q = np.eye(dim)
        self.P = _solve_discrete_lyapunov(A, Q)
        Sigma = _noise_covariance(post)
        noise_floor = float(np.trace(self.P @ Sigma) / np.min(np.linalg.eigvalsh(Q)))
        derived_r = float(np.sqrt(max(noise_floor, 0.0)))
        self.eq_radius = eq_margin * derived_r if eq_radius is None else eq_radius

        lip = float(np.linalg.norm(A, ord=2))  # exact one-step state Lipschitz

        super().__init__(
            lip_f=lip,
            lip_t=lip,
            domain=HyperRectangle.from_bounds(
                jnp.array([[-domain_half, domain_half]] * dim)
            ),
            equilibrium=Hypersphere(jnp.zeros(dim), self.eq_radius),
            post_noise=post,
            name=f"LinearStochastic{dim}D(L={lip:.3f})",
        )

    @classmethod
    def random_construction(
        cls, ndims: int, spectral_norm: float = 0.9, key=None, **kw
    ):
        """Random Schur-stable A scaled to `spectral_norm` < 1, in any dimension."""
        if key is None:
            key = jrn.key(0)
        M = jrn.normal(key, (ndims, ndims))
        A = np.array(M * (spectral_norm / jnp.linalg.norm(M, ord=2)))
        return cls(A=A, **kw)

    # ------------------------------------------------------------------
    # JAX (deterministic part; additive noise added by post_noise in Env.step)
    # ------------------------------------------------------------------
    def _dynamics(self, x, key):
        return self.A @ x, key

    def known_V(self, x: jnp.ndarray) -> jnp.ndarray:
        """Quadratic Lyapunov V(x) = x^T P x (discrete-Lyapunov ground truth)."""
        return x @ jnp.array(self.P) @ x

    # ------------------------------------------------------------------
    # Symbolic: x_next = A x (exact; noise boxes added by Env.*_step)
    # ------------------------------------------------------------------
    def _gurobi_dynamics(self, model: gp.Model, branches: list[tuple]):
        D = self.dim
        A_np = np.array(self.A)
        out = []
        for b_idx, (w, x_vars) in enumerate(branches):
            x_next = model.addVars(D, lb=-gp.GRB.INFINITY, name=f"x_next_{b_idx}")
            for row in range(D):
                model.addConstr(
                    x_next[row]
                    == gp.quicksum(A_np[row, col] * x_vars[col] for col in range(D))
                )
            out.append((w, list(x_next.values())))
        return out

    def _z3_dynamics(self, solver, branches: list[tuple]):
        import z3

        D = self.dim
        A_np = np.array(self.A)
        out = []
        for b_idx, (w, x_vars) in enumerate(branches):
            x_next = [z3.Real(f"x_next_{b_idx}_{d}") for d in range(D)]
            for row in range(D):
                solver.add(
                    x_next[row]
                    == z3.Sum([A_np[row, col] * x_vars[col] for col in range(D)])
                )
            out.append((w, x_next))
        return out

    # ------------------------------------------------------------------
    # autoLiRPA: additive noise binned into cells (sound over-approximation)
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
        """Torch next-state map: x_next = A x + w (additive)."""
        import torch
        import torch.nn as nn

        A_np = np.array(self.A)

        class _Step(nn.Module):
            def __init__(self):
                super().__init__()
                # buffer (not a closure) so .to(device) moves it with the module
                self.register_buffer("A_t", torch.tensor(A_np, dtype=torch.float32))

            def forward(self, x, w):  # x:(B,d) w:(B,d)
                return x @ self.A_t.T + w

        return _Step()
