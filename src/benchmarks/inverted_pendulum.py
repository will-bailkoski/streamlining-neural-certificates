"""
Inverted pendulum with a neural-network controller.

    state  x = [theta, theta_dot]
    action u = pi(x)  (scalar torque, tanh-MLP, clipped to [-1, 1])

    theta_dot' = (1-b) theta_dot + delta(-1.5 G sin(theta+pi)/(2l) + 6u/(m l^2)) + 0.002 xi_0
    theta'     = theta + delta theta_dot' + 0.005 xi_1
    xi ~ Triangular(-1, 0, 1)

This system is NONLINEAR (the sin term), so the SMT and MILP engines cannot
encode it exactly and raise; it is verified with the sampling engine (and, once
the continuous-noise autoLiRPA path lands, with `lirpa`). The noise is internal
(it enters the dynamics with per-component scales), so it lives inside
`_dynamics` rather than as additive `_pre`/`_post` noise.

The controller is trained by policy gradient with a fixed seed, so
`make_env("pendulum")` rebuilds the same env every time (no pickle, no asset
files). There is no analytic `known_V` — the certificate is learned.
"""

import numpy as np
from gurobipy import Model
from jax import numpy as jnp
from jax import random as jrn
from jax import vmap, lax, value_and_grad
import optax

from src.benchmarks.env import Env
from src.benchmarks.convex_sets import Hypersphere, EmptySet
from src.training.utils import (
    tanh_forward as _forward,
    glorot_init as _init_mlp,
    get_spectral_norm_product as _spectral_norm,
)


def _triangular_sample(key, shape, left=-1.0, mode=0.0, right=1.0):
    u = jrn.uniform(key, shape=shape)
    c = (mode - left) / (right - left)
    return jnp.where(
        u < c,
        left + jnp.sqrt(u * (right - left) * (mode - left)),
        right - jnp.sqrt((1 - u) * (right - left) * (right - mode)),
    )


def _control(params, x):
    """Controller policy: tanh-MLP clipped to [-1, 1]."""
    return jnp.clip(_forward(params, x), -1.0, 1.0)


class InvertedPendulumEnv(Env):
    def __init__(
        self,
        domain_center=(0.0, 0.0),
        domain_radius: float = 1.0,
        eq_center=(0.0, 0.0),
        eq_radius: float = 0.1,
        G: float = 10.0,
        m: float = 0.15,
        l: float = 0.5,
        b: float = 0.1,
        delta: float = 0.05,
        controller_params=None,
        layer_sizes=(2, 16, 16, 1),
        train_epochs: int = 40,
        domain_level: float = 2.0,
    ):
        self.G, self.m, self.l, self.b, self.delta = G, m, l, b, delta
        self.domain_level = domain_level
        self.domain_center = jnp.array(domain_center)
        self.domain_radius = domain_radius
        self.eq_center = jnp.array(eq_center)
        self.eq_radius = eq_radius
        self.layer_sizes = list(layer_sizes)

        # Minimal attrs so self.sample works during controller training,
        # before the full base __init__ runs.
        self.domain = Hypersphere(self.domain_center, domain_radius)
        self.equilibrium = Hypersphere(self.eq_center, eq_radius)
        self.dim = int(self.domain.center.shape[0])

        if controller_params is None:
            self.controller_params, self.training_log = self._train_controller(
                self.layer_sizes, jrn.PRNGKey(10), epochs=train_epochs
            )
        else:
            self.controller_params = [
                (jnp.asarray(w), jnp.asarray(bb)) for w, bb in controller_params
            ]
            self.training_log = None
            # Infer architecture from the injected params (W shape is (out, in)).
            self.layer_sizes = [int(self.controller_params[0][0].shape[1])] + [
                int(W.shape[0]) for W, _ in self.controller_params
            ]

        # Heuristic step Lipschitz (used by the mesh verifier).
        Lu = float(_spectral_norm(self.controller_params))
        gravity = (3.0 * G) / (4.0 * l)
        control_gain = 6.0 / (m * l * l)
        base = abs(1 - b) + delta * gravity + delta * control_gain * Lu
        step_lip = float(max(1 + delta * base, base))

        super().__init__(
            lip_f=step_lip,
            lip_t=step_lip,
            domain=self._invariant_domain(domain_level),
            equilibrium=Hypersphere(self.eq_center, eq_radius),
            name="InvertedPendulum",
        )

    @property
    def symbolic_encodable(self) -> bool:
        return False  # sin(theta) -> SMT/MILP cannot encode

    def _invariant_domain(self, level: float):
        """Lyapunov-ellipsoid domain {x : x^T P x <= level} of the linearised closed
        loop (the box/ball is not forward-invariant under the non-normal dynamics).
        Falls back to the original ball if the closed loop is not Schur — e.g. an
        injected, untrained controller in the tests."""
        from src.benchmarks.convex_sets import Ellipsoid
        from src.benchmarks.linear_stochastic import _solve_discrete_lyapunov
        import numpy as np
        from jax import jacobian

        def det_step(x):  # noise-free closed-loop step
            return self._step_dynamics(x, _control(self.controller_params, x), jnp.zeros(2))

        try:
            A_d = np.array(jacobian(det_step)(self.eq_center.astype(jnp.float32)))
            if max(abs(np.linalg.eigvals(A_d))) >= 1.0:
                raise ValueError("closed loop not Schur")
            P = _solve_discrete_lyapunov(A_d, np.eye(2))
            np.linalg.cholesky(P)  # confirm P is positive definite
            self.P = P
            return Ellipsoid(self.eq_center, jnp.array(P / level))
        except Exception:
            self.P = None
            return Hypersphere(self.domain_center, self.domain_radius)

    # ------------------------------------------------------------------
    # Dynamics (JAX) — noise is internal
    # ------------------------------------------------------------------
    def _step_dynamics(self, x, u, noise):
        xdot = (
            (1.0 - self.b) * x[..., 1]
            + self.delta
            * (
                -1.5 * self.G * jnp.sin(x[..., 0] + jnp.pi) / (2.0 * self.l)
                + 6.0 * u / (self.m * self.l * self.l)
            )
            + 0.002 * noise[..., 0]
        )
        th_next = x[..., 0] + self.delta * xdot + 0.005 * noise[..., 1]
        return jnp.stack([th_next, xdot], axis=-1)

    def _dynamics(self, x, key):
        key, sub = jrn.split(key)
        noise = _triangular_sample(sub, shape=x.shape)
        return self._step_dynamics(x, _control(self.controller_params, x), noise), key

    def controller_fn(self, x):
        return _control(self.controller_params, x)

    # ------------------------------------------------------------------
    # Symbolic engines cannot handle the nonlinear sin term
    # ------------------------------------------------------------------
    def _gurobi_dynamics(self, model: Model, branches: list[tuple]):
        raise NotImplementedError(
            "InvertedPendulum is nonlinear (sin); MILP cannot encode it. "
            "Use the sampling engine."
        )

    def _z3_dynamics(self, solver, branches: list[tuple]):
        raise NotImplementedError(
            "InvertedPendulum is nonlinear (sin); SMT cannot encode it. "
            "Use the sampling engine."
        )

    # ------------------------------------------------------------------
    # Torch pieces (controller + next-state module) for autoLiRPA.
    # The drift-over-noise BaB wiring is the continuous-noise lirpa follow-up;
    # these are validated against the jax dynamics in the tests.
    # ------------------------------------------------------------------
    def controller_torch(self):
        return _TanhMLP(self.layer_sizes, self.controller_params)

    def torch_step_module(self):
        return _PendulumStep(
            self.controller_torch(), self.G, self.m, self.l, self.b, self.delta
        )

    @staticmethod
    def build_noise_tensor(n: int):
        import torch
        from auto_LiRPA import BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm

        nominal = torch.zeros((n, 2))
        ptb = PerturbationLpNorm(x_L=torch.full((n, 2), -1.0), x_U=torch.full((n, 2), 1.0))
        return BoundedTensor(nominal, ptb)

    def lirpa_drift_module(self, V_module, epsilon: float, noise_disc: int = 1):
        """
        autoLiRPA drift with the internal triangular noise binned into cells.
        The raw noise xi in [-1, 1]^2 is partitioned; torch_step_module applies
        the 0.002 / 0.005 scaling internally (matching the jax dynamics).
        """
        import numpy as np
        from src.benchmarks.noise import Triangular
        from src.benchmarks._lirpa import cell_drift_module

        noise = Triangular(-1.0, 0.0, 1.0, dim=2, discretisation=noise_disc)
        los, his, weights = noise.get_support(noise_disc)
        module = cell_drift_module(
            V_module, self.torch_step_module(), weights, noise_dim=2, epsilon=epsilon
        )
        return module, (np.asarray(los).reshape(-1), np.asarray(his).reshape(-1))

    # ------------------------------------------------------------------
    # Controller training (policy gradient, fixed seed -> reproducible)
    # ------------------------------------------------------------------
    def _train_controller(
        self,
        layer_sizes,
        key,
        lr: float = 1e-3,
        epochs: int = 40,
        rollout_len: int = 15,
        batch_size: int = 96,
        gamma: float = 0.999,
        lipschitz_weight: float = 2.5,
        domain_pen: float = 10.0,
        action_pen: float = 0.0,
    ):
        params = _init_mlp(list(layer_sizes), key)
        bias_mask = [(False, True) for _ in params]
        opt = optax.chain(optax.masked(optax.set_to_zero(), bias_mask), optax.adam(lr))
        opt_state = opt.init(params)
        log = []

        def reward_fn(x, u):
            return (
                -x[0] ** 2
                - x[1] ** 2
                - action_pen * jnp.clip(u, -1.0, 1.0) ** 2
                - domain_pen
                * jnp.maximum(0.0, jnp.max(jnp.abs(x)) - self.domain_radius) ** 2
            )

        def sim_step(params, x, rng):
            rng, sub = jrn.split(rng)
            noise = _triangular_sample(sub, shape=x.shape)
            return self._step_dynamics(x, _control(params, x), noise), rng

        def loss_fn(params, key):
            key, sub = jrn.split(key)
            all_states, _ = self.sample(sub, batch_size)
            rollout_keys = jrn.split(key, batch_size)

            def single_rollout(x0, rng):
                def step_fn(carry, _):
                    x, total, disc, rng = carry
                    u = _control(params, x)
                    x_next, rng = sim_step(params, x, rng)
                    r = reward_fn(x, u)
                    return (x_next, total + disc * r, disc * gamma, rng), None

                (_, total, _, _), _ = lax.scan(
                    step_fn, (x0, 0.0, 1.0, rng), jnp.arange(rollout_len)
                )
                return total

            returns = vmap(single_rollout)(all_states, rollout_keys)
            lip = lipschitz_weight * sum(_spectral_norm(W) for W, _ in params)
            return -jnp.mean(returns) + lip

        for ep in range(epochs):
            key, sub = jrn.split(key)
            loss, grads = value_and_grad(loss_fn)(params, sub)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            log.append({"epoch": ep, "loss": float(loss)})

        return params, log


# ----------------------------------------------------------------------
# Torch modules (kept local to this env)
# ----------------------------------------------------------------------
class _TanhMLP:
    """Lazy torch tanh-MLP controller (clipped to [-1, 1] via Hardtanh)."""

    def __new__(cls, layer_sizes, jax_params):
        import torch.nn as nn
        import torch

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.Tanh())
        layers.append(nn.Hardtanh())
        net = nn.Sequential(*layers)
        linear = [mod for mod in net if isinstance(mod, nn.Linear)]
        with torch.no_grad():
            for layer, (W, b) in zip(linear, jax_params):
                layer.weight.copy_(torch.tensor(np.array(W), dtype=torch.float32))
                layer.bias.copy_(torch.tensor(np.array(b), dtype=torch.float32))
        return net


def _PendulumStep(controller, G, m, l, b, delta):
    import torch
    import torch.nn as nn

    class _Step(nn.Module):
        def __init__(self):
            super().__init__()
            self.controller = controller

        def forward(self, x, noise):
            th, thdot = x[:, 0], x[:, 1]
            u = self.controller(x).squeeze(-1)
            inner = (-1.5 * G / (2.0 * l)) * torch.sin(th + torch.pi) + (
                6.0 / (m * l * l)
            ) * u
            xdot = (1.0 - b) * thdot + delta * inner + 0.002 * noise[:, 0]
            th_next = th + delta * xdot + 0.005 * noise[:, 1]
            return torch.stack([th_next, xdot], dim=1)

    return _Step()
