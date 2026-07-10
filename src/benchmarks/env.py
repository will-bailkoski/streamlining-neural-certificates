"""
The single abstract environment definition.

An `Env` is the one source of truth for a dynamical system. It owns the
dynamics, the additive noise, the domain and the equilibrium, and exposes them
to every verification engine through a shared interface:

  - JAX (training + sampling engine):   `step(x, key)`  and `sample`
  - Z3  (SMT engine):                    `z3_step(solver, branches)`
  - Gurobi (MILP engine):                `gurobi_step(model, branches)`
  - autoLiRPA (bound-propagation engine): `torch_dynamics()`  (subclass opt-in)

Convention (so every engine checks the SAME drift):
  * `_dynamics(x, key)` is the CORE dynamics: deterministic, or internally
    branching over discrete modes (e.g. switched-linear). It must NOT add
    additive process/observation noise.
  * Additive noise lives in `_pre` / `_post` (`Noise` objects) and is applied
    identically in the jax `step` and in the symbolic `*_step` encodings.

A full step is:   x --[pre noise]--> --[dynamics]--> --[post noise]--> x'
"""

import abc
import inspect

from jax import vmap, lax
import jax.numpy as jnp

from src.benchmarks.noise import NullNoise, Noise
from src.benchmarks.convex_sets import ConvexSet, EmptySet


class Env(abc.ABC):
    def __init__(
        self,
        lip_f: float,  # Lipschitz constant of one step wrt the state
        lip_t: float,  # Lipschitz constant wrt successors (used by the mesh)
        domain: ConvexSet,
        equilibrium: ConvexSet | None = None,
        pre_noise: Noise | None = None,
        post_noise: Noise | None = None,
        name: str | None = None,
    ):
        self.domain = domain
        self.dim = int(domain.bounds.shape[0])
        self.name = name if name is not None else self.__class__.__name__
        self.lip_f = lip_f
        self.lip_t = lip_t
        self.equilibrium = equilibrium if equilibrium is not None else EmptySet(self.dim)

        self._pre = pre_noise if pre_noise is not None else NullNoise(dim=self.dim)
        self._post = post_noise if post_noise is not None else NullNoise(dim=self.dim)

    # ------------------------------------------------------------------
    # Abstract interface — subclasses implement these
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def _dynamics(self, x, key):
        """Core jax dynamics. Returns (x_next, key). No additive noise here."""
        raise NotImplementedError

    @abc.abstractmethod
    def _gurobi_dynamics(self, model, branches: list[tuple]) -> list[tuple]:
        """Encode the core dynamics in Gurobi. branches: [(weight, x_vars), ...]."""
        raise NotImplementedError

    @abc.abstractmethod
    def _z3_dynamics(self, solver, branches: list[tuple]) -> list[tuple]:
        """Encode the core dynamics in Z3. branches: [(weight, x_vars), ...]."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # JAX step + sampling
    # ------------------------------------------------------------------
    def step(self, x, key):
        x, key = self._pre.step(x, key)
        x, key = self._dynamics(x, key)
        return self._post.step(x, key)

    def is_valid_domain(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.domain.contains(x) & ~self.equilibrium.contains(x)

    def is_valid_polytope(self, vertices: jnp.ndarray):
        return jnp.any(vmap(self.is_valid_domain)(vertices))

    def sample(self, key, batch_size: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample from domain minus equilibrium via rejection sampling."""
        xs, key = self.domain.sample(key, batch_size)
        mask = vmap(self.equilibrium.contains)(xs)

        def body(carry):
            xs, mask, key = carry
            new_xs, key = self.domain.sample(key, batch_size)
            new_mask = vmap(self.equilibrium.contains)(new_xs)
            xs = jnp.where(mask[:, None], new_xs, xs)
            mask = jnp.where(mask, new_mask, mask)
            return xs, mask, key

        def cond(carry):
            _, mask, _ = carry
            return jnp.any(mask)

        xs, _, key = lax.while_loop(cond, body, (xs, mask, key))
        return xs, key

    # ------------------------------------------------------------------
    # Noise configuration
    # ------------------------------------------------------------------
    def set_process_noise(self, noise: Noise):
        self._pre = noise

    def set_observation_noise(self, noise: Noise):
        self._post = noise

    def reset_noise(self):
        self._pre = NullNoise(dim=self.dim)
        self._post = NullNoise(dim=self.dim)

    # ------------------------------------------------------------------
    # Capability flags (used by the sweep launcher to skip unsupported combos)
    # ------------------------------------------------------------------
    @property
    def symbolic_encodable(self) -> bool:
        """Whether the dynamics can be encoded EXACTLY in Z3/Gurobi. False for
        transcendental dynamics (e.g. sin), where smt/milp raise NotImplementedError."""
        return True

    # ------------------------------------------------------------------
    # Symbolic steps (shared noise handling; subclass only writes _dynamics)
    # ------------------------------------------------------------------
    @staticmethod
    def _fan_out_noise_gurobi(noise, model, branches, stage, disc=None):
        los, his, weights = noise.get_support(disc)
        out = []
        for s in range(len(weights)):
            for b_idx, (w_old, x_old) in enumerate(branches):
                x_new = noise.gurobi_apply_box(model, x_old, los[s], his[s], b_idx, s, stage)
                out.append((w_old * float(weights[s]), x_new))
        return out

    @staticmethod
    def _fan_out_noise_z3(noise, solver, branches, stage, disc=None):
        los, his, weights = noise.get_support(disc)
        out = []
        for s in range(len(weights)):
            for b_idx, (w_old, x_old) in enumerate(branches):
                x_new = noise.z3_apply_box(solver, x_old, los[s], his[s], b_idx, s, stage)
                out.append((w_old * float(weights[s]), x_new))
        return out

    def gurobi_step(self, model, x_branches: list[tuple], noise_disc=None) -> list[tuple]:
        """Encode one stochastic step. `noise_disc` bins continuous noise into that
        many equal-mass cells per dimension (higher = tighter E[V] over-approx, more
        branches); None keeps each Noise's construction-time discretisation. Discrete
        noises ignore it (exact)."""
        pre = self._fan_out_noise_gurobi(self._pre, model, x_branches, "pre", noise_disc)
        dyn = self._gurobi_dynamics(model, pre)
        return self._fan_out_noise_gurobi(self._post, model, dyn, "post", noise_disc)

    def z3_step(self, solver, x_branches: list[tuple], noise_disc=None) -> list[tuple]:
        """As `gurobi_step`, for Z3. `noise_disc` tunes the continuous-noise binning."""
        pre = self._fan_out_noise_z3(self._pre, solver, x_branches, "pre", noise_disc)
        dyn = self._z3_dynamics(solver, pre)
        return self._fan_out_noise_z3(self._post, solver, dyn, "post", noise_disc)

    # ------------------------------------------------------------------
    # autoLiRPA hook — subclasses opt in by returning (module, noise_box):
    #   module    : nn.Module for the drift d(x) = E_w[V(f(x,w))] - V(x) + eps
    #   noise_box : None for discrete/no-noise dynamics (module.forward(x)), or
    #               (w_lo, w_hi) flat arrays of length K*noise_dim bounding a
    #               packed noise input (module.forward(x, w)). Continuous noise
    #               is binned into `noise_disc` cells per dimension.
    # ------------------------------------------------------------------
    def lirpa_drift_module(self, V_module, epsilon: float, noise_disc: int = 1):
        raise NotImplementedError(
            f"{self.name} does not expose an autoLiRPA drift module; "
            f"use the sampling engine (or implement lirpa_drift_module)."
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def get_config_params(self) -> dict:
        """Map constructor argument names to instance attributes (JSON-friendly)."""
        sig = inspect.signature(self.__init__)
        out = {}
        for k in sig.parameters.keys():
            if k == "self":
                continue
            if hasattr(self, k):
                val = getattr(self, k)
                if hasattr(val, "tolist"):
                    val = val.tolist()
                out[k] = val
        return out
