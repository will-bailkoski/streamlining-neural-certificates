"""
Building-automation thermal regulation (BAS-family) — affine linear, stochastic.

A faithful single-zone RC-network reduction of the Cauchi & Abate building
automation benchmark (github.com/natchi92/BASBenchmarks; ARCH-COMP "Stochastic
Models" BAS), using their authentic thermal parameters. Two coupled thermal
masses — the zone air `T_z` and the radiator water `T_w` — exchange heat through
thermal resistances, with a constant boiler/ambient load (the affine term) and
additive temperature noise:

    [T_z]      [T_z]            [ambient + radiator load]
    [T_w]  =  A[T_w]  +  b  +  w,     w ~ bounded (sigma_z = 0.02 in the source).

`A = I + dt·M` where M is the RC conductance matrix divided by the heat
capacities; it is Schur-stable (dissipative), so the system settles to the
setpoint  x* = (I - A)^{-1} b. Working in shifted coordinates z = x - x*, the
dynamics are `z' = A z + w`, so the discrete-Lyapunov certificate
`V(x) = (x - x*)^T P (x - x*)`  (with `A^T P A - P = -I`) is an exact ground-truth
supermartingale on the complement of the noise-floor ball around the setpoint
(equilibrium radius derived from the noise level, as in `LinearStochasticEnv`).

Affine-linear ⇒ exact in Z3/Gurobi, sound (binned) in autoLiRPA, native in
sampling/MAB. To use the *exact* ARCH-COMP 4-state BAS matrices instead, pass
`A=`, `b=` directly (evaluate them from BASParameters.m / createModel.m).
"""

import numpy as np
import gurobipy as gp
from jax import numpy as jnp

from src.benchmarks.env import Env
from src.benchmarks.noise import Uniform, Triangular
from src.benchmarks.convex_sets import HyperRectangle, Hypersphere
from src.benchmarks.linear_stochastic import _solve_discrete_lyapunov, _noise_covariance

# Authentic Zone-1 parameters from natchi92/BASBenchmarks (BASParameters.m).
_BAS = dict(Cz=51.5203, Cw=52.3759, Rout=7.20546, Rrad=5.4176, Tout=9.0, Prad=0.8, sigma=0.02)


def _bas_thermal_AB(dt, Cz, Cw, Rout, Rrad, Tout, Prad):
    """Continuous RC network -> discrete (A, b) via forward Euler."""
    # dT_z/dt = [(Tout - T_z)/Rout + (T_w - T_z)/Rrad] / Cz
    # dT_w/dt = [(T_z - T_w)/Rrad + Prad] / Cw
    M = np.array([
        [-(1.0 / Rout + 1.0 / Rrad) / Cz, (1.0 / Rrad) / Cz],
        [(1.0 / Rrad) / Cw, -(1.0 / Rrad) / Cw],
    ])
    c = np.array([(Tout / Rout) / Cz, Prad / Cw])  # affine load (ambient + radiator)
    A = np.eye(2) + dt * M
    b = dt * c
    return A, b


class BuildingThermalEnv(Env):
    def __init__(
        self,
        A=None,
        b=None,
        dt: float = 15.0,           # sampling time (minutes), as in the BAS benchmark
        noise_scale: float = 0.06,  # bounded approx of the source's sigma_z = 0.02
        noise_kind: str = "uniform",
        noise_disc: int = 1,
        domain_half: float = 2.0,   # +/- temperature band around the setpoint (deg C)
        # eq_margin scales the MEAN-SQUARE noise floor sqrt(tr(P Sigma)/lmin(Q))
        # into the eq radius. The old default 1.5 gave r=0.31 — but the SUP of
        # the invariant set (what the eq must contain for any cert to exist)
        # empirically reaches 0.80 from the setpoint (512 chains x 400 steps):
        # the slow RC contraction lets bounded noise accumulate far beyond the
        # mean-square floor, so with 1.5 NO valid certificate existed and every
        # engine refuted forever. 6.0 (r=1.23) covers the reach with the cushion
        # a bounded cert needs to hold a trainable margin (2026-07-03 probes:
        # seed loss 0 at train_eps=2e-3, bounded_pwl 8,8, centered init).
        eq_margin: float = 6.0,
        params: dict | None = None,
    ):
        p = {**_BAS, **(params or {})}
        if A is None or b is None:
            A, b = _bas_thermal_AB(dt, p["Cz"], p["Cw"], p["Rout"], p["Rrad"], p["Tout"], p["Prad"])
        self.A = jnp.array(np.asarray(A, float))
        self.b = jnp.array(np.asarray(b, float))
        dim = self.A.shape[0]
        self.dt = dt
        self.noise_scale = noise_scale
        self.noise_disc = noise_disc

        if noise_kind == "uniform":
            post = Uniform(-noise_scale, noise_scale, dim=dim, discretisation=noise_disc)
        elif noise_kind == "triangular":
            post = Triangular(-noise_scale, 0.0, noise_scale, dim=dim, discretisation=noise_disc)
        else:
            raise ValueError(f"unknown noise_kind {noise_kind!r}")

        A_np, b_np = np.array(self.A), np.array(self.b)
        self.setpoint = np.linalg.solve(np.eye(dim) - A_np, b_np)  # fixed point x*

        Q = np.eye(dim)
        self.P = _solve_discrete_lyapunov(A_np, Q)
        Sigma = _noise_covariance(post)
        noise_floor = float(np.trace(self.P @ Sigma) / np.min(np.linalg.eigvalsh(Q)))
        eq_radius = eq_margin * float(np.sqrt(max(noise_floor, 0.0)))

        lip = float(np.linalg.norm(A_np, ord=2))
        center = jnp.array(self.setpoint)
        bounds = jnp.stack([center - domain_half, center + domain_half], axis=-1)

        super().__init__(
            lip_f=lip,
            lip_t=lip,
            domain=HyperRectangle.from_bounds(bounds),
            equilibrium=Hypersphere(center, eq_radius),
            post_noise=post,
            name=f"BuildingThermal{dim}D(L={lip:.3f})",
        )

    # ------------------------------------------------------------------
    # JAX (affine deterministic part; additive noise added by post_noise)
    # ------------------------------------------------------------------
    def _dynamics(self, x, key):
        return self.A @ x + self.b, key

    def known_V(self, x: jnp.ndarray) -> jnp.ndarray:
        """Quadratic Lyapunov about the setpoint: V(x) = (x-x*)^T P (x-x*)."""
        z = x - jnp.array(self.setpoint)
        return z @ jnp.array(self.P) @ z

    # ------------------------------------------------------------------
    # Symbolic: x_next = A x + b (exact; noise boxes added by Env.*_step)
    # ------------------------------------------------------------------
    def _gurobi_dynamics(self, model: gp.Model, branches: list[tuple]):
        D = self.dim
        A_np, b_np = np.array(self.A), np.array(self.b)
        out = []
        for b_idx, (w, x_vars) in enumerate(branches):
            x_next = model.addVars(D, lb=-gp.GRB.INFINITY, name=f"x_next_{b_idx}")
            for row in range(D):
                model.addConstr(
                    x_next[row]
                    == gp.quicksum(A_np[row, col] * x_vars[col] for col in range(D)) + float(b_np[row])
                )
            out.append((w, list(x_next.values())))
        return out

    def _z3_dynamics(self, solver, branches: list[tuple]):
        import z3

        D = self.dim
        A_np, b_np = np.array(self.A), np.array(self.b)
        out = []
        for b_idx, (w, x_vars) in enumerate(branches):
            x_next = [z3.Real(f"x_next_{b_idx}_{d}") for d in range(D)]
            for row in range(D):
                solver.add(
                    x_next[row]
                    == z3.Sum([A_np[row, col] * x_vars[col] for col in range(D)]) + float(b_np[row])
                )
            out.append((w, x_next))
        return out

    # ------------------------------------------------------------------
    # autoLiRPA: additive noise binned (sound over-approximation)
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

        A_np = np.array(self.A)
        b_np = np.array(self.b)

        class _Step(nn.Module):
            def __init__(self):
                super().__init__()
                # buffers (not closures) so .to(device) moves them with the module
                self.register_buffer("A_t", torch.tensor(A_np, dtype=torch.float32))
                self.register_buffer("b_t", torch.tensor(b_np, dtype=torch.float32))

            def forward(self, x, w):  # x:(B,d) w:(B,d)
                return x @ self.A_t.T + self.b_t + w

        return _Step()
