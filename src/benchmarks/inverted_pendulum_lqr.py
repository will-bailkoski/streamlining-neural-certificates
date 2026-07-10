"""
Inverted pendulum (FOSSIL ``SineModelLQR``) with additive noise.

Ported from the deterministic project's `InvertedPendulumLQR`, itself from
oxford-oxcav/fossil. FOSSIL synthesises an LQR feedback `K = [[7.21, 1.34],
[1.34, 0.33]]` for the `SineModel` pendulum; we fold that linear controller into
autonomous closed-loop dynamics (no controller needed at verification time) and
add bounded triangular process noise to make the stochastic variant.

Closed loop (state `x = [theta, theta_dot]`, `u = -K x`), explicit Euler:

    theta'     = theta + h (theta_dot + u1)
    theta_dot' = theta_dot + h ( u2 + (m g L sin(theta) - b theta_dot)/(m L^2) )
    x_{k+1}    = [theta', theta_dot'] + w,   w ~ Triangular (additive)

Why an ellipsoidal domain.  The closed loop is Schur but NON-NORMAL
(`||A_d||_2 > 1`: transient growth), so neither a box nor a ball is forward
invariant — exactly the failure mode that made the NN pendulum and Van der Pol
leak. The invariant domain is the Lyapunov sublevel ellipsoid `{x : x^T P x <=
level}` with `P` from `A_d^T P A_d - P = -I` (the discretised linearisation).
`known_V(x) = x^T P x` is then a ground-truth quadratic supermartingale on
`domain \\ equilibrium` (validated against the nonlinear MC drift); the
equilibrium radius is sized from the noise floor.

`sin(theta)` is nonlinear, so SMT/MILP cannot encode it (they raise); verified
with the sampling/MAB engines and autoLiRPA (which relaxes `sin`).
"""

import numpy as np
from jax import numpy as jnp

from src.benchmarks.env import Env
from src.benchmarks.convex_sets import Ellipsoid, Hypersphere
from src.benchmarks.linear_stochastic import _solve_discrete_lyapunov, _noise_covariance
from src.benchmarks.noise import Triangular, Uniform


class InvertedPendulumLQREnv(Env):
    K = np.array([[7.21, 1.34], [1.34, 0.33]])  # FOSSIL SineModelLQR verified gain
    G, L, M, B = 9.81, 0.5, 0.15, 0.1           # FOSSIL SineModel physical parameters

    def __init__(
        self,
        h: float = 0.05,
        level: float = 0.5,
        noise_scale: float = 0.01,
        noise_kind: str = "triangular",
        noise_disc: int = 1,
        eq_radius: float | None = None,
        eq_margin: float = 2.0,
    ):
        self.h = float(h)
        G, L, m, b = self.G, self.L, self.M, self.B
        inv_mL2 = 1.0 / (m * L**2)
        K = self.K

        # Linearised closed loop at the origin and its Euler discretisation.
        A_cl = np.array([
            [-K[0, 0], 1.0 - K[0, 1]],
            [-K[1, 0] + m * G * L * inv_mL2, -K[1, 1] - b * inv_mL2],
        ])
        A_d = np.eye(2) + self.h * A_cl
        self.P = _solve_discrete_lyapunov(A_d, np.eye(2))  # A_d^T P A_d - P = -I
        self.level = float(level)
        domain = Ellipsoid(jnp.zeros(2), jnp.array(self.P / self.level))

        if noise_kind == "triangular":
            post = Triangular(-noise_scale, 0.0, noise_scale, dim=2, discretisation=noise_disc)
        elif noise_kind == "uniform":
            post = Uniform(-noise_scale, noise_scale, dim=2, discretisation=noise_disc)
        else:
            raise ValueError(f"unknown noise_kind {noise_kind!r}")
        self.noise_scale = noise_scale

        # Equilibrium sized from the noise floor (drift ~ -x^T x + tr(P Sigma)).
        Sigma = _noise_covariance(post)
        noise_floor = float(np.trace(self.P @ Sigma))
        derived_r = float(np.sqrt(max(noise_floor, 0.0)))
        self.eq_radius = eq_margin * derived_r if eq_radius is None else eq_radius

        lip = self._max_step_lipschitz(domain)

        super().__init__(
            lip_f=lip,
            lip_t=lip,
            domain=domain,
            equilibrium=Hypersphere(jnp.zeros(2), self.eq_radius),
            post_noise=post,
            name=f"InvertedPendulumLQR(h={h})",
        )

    def _max_step_lipschitz(self, domain) -> float:
        """sup ||I + h J(x)||_2 over the domain bbox (J varies only in cos(theta))."""
        G, L, m, b = self.G, self.L, self.M, self.B
        inv_mL2 = 1.0 / (m * L**2)
        half = np.array(domain.bounds[:, 1])
        worst = 0.0
        for th in np.linspace(-half[0], half[0], 21):
            J = np.array([
                [-self.K[0, 0], 1.0 - self.K[0, 1]],
                [-self.K[1, 0] + m * G * L * np.cos(th) * inv_mL2, -self.K[1, 1] - b * inv_mL2],
            ])
            worst = max(worst, float(np.linalg.norm(np.eye(2) + self.h * J, 2)))
        return worst

    # ------------------------------------------------------------------
    # JAX (deterministic closed loop; additive noise added by post_noise)
    # ------------------------------------------------------------------
    def _dynamics(self, x, key):
        theta, omega = x[0], x[1]
        u1 = -self.K[0, 0] * theta - self.K[0, 1] * omega
        u2 = -self.K[1, 0] * theta - self.K[1, 1] * omega
        m, G, L, b = self.M, self.G, self.L, self.B
        dtheta = omega + u1
        domega = u2 + (m * G * L * jnp.sin(theta) - b * omega) / (m * L**2)
        return jnp.array([x[0] + self.h * dtheta, x[1] + self.h * domega]), key

    def known_V(self, x: jnp.ndarray) -> jnp.ndarray:
        """Analytic quadratic Lyapunov x^T P x (P from the Schur linearisation)."""
        return x @ jnp.array(self.P) @ x

    @property
    def symbolic_encodable(self) -> bool:
        return False  # sin(theta) -> SMT/MILP cannot encode

    # ------------------------------------------------------------------
    # Symbolic engines cannot encode the nonlinear sin term
    # ------------------------------------------------------------------
    def _gurobi_dynamics(self, model, branches):
        raise NotImplementedError(
            "InvertedPendulumLQR is nonlinear (sin); MILP cannot encode it. Use sampling."
        )

    def _z3_dynamics(self, solver, branches):
        raise NotImplementedError(
            "InvertedPendulumLQR is nonlinear (sin); SMT cannot encode it. Use sampling."
        )

    # ------------------------------------------------------------------
    # autoLiRPA: additive noise binned; sin handled by its trig relaxation.
    # ------------------------------------------------------------------
    def lirpa_drift_module(self, V_module, epsilon: float, noise_disc: int = 1):
        from src.benchmarks._lirpa import cell_drift_module

        los, his, weights = self._post.get_support(noise_disc)
        module = cell_drift_module(
            V_module, self._torch_step(), weights, noise_dim=2, epsilon=epsilon
        )
        noise_box = (np.asarray(los).reshape(-1), np.asarray(his).reshape(-1))
        return module, noise_box

    def _torch_step(self):
        import torch
        import torch.nn as nn

        K_np = np.array(self.K)
        h, m, G, L, b = self.h, self.M, self.G, self.L, self.B

        class _Step(nn.Module):
            def __init__(self):
                super().__init__()
                # buffer (not a closure) so .to(device) moves the gain with the module
                self.register_buffer("K", torch.tensor(K_np, dtype=torch.float32))

            def forward(self, x, w):  # x:(B,2) w:(B,2)
                theta = x[:, 0:1]
                omega = x[:, 1:2]
                K = self.K
                u1 = -K[0, 0] * theta - K[0, 1] * omega
                u2 = -K[1, 0] * theta - K[1, 1] * omega
                dtheta = omega + u1
                domega = u2 + (m * G * L * torch.sin(theta) - b * omega) / (m * L * L)
                nxt = torch.cat([theta + h * dtheta, omega + h * domega], dim=1)
                return nxt + w

        return _Step()
