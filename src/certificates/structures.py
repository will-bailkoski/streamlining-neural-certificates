"""
Neural certificate structures — the single source of truth for V's parametric form.

A `CertificateSpec` fixes the hidden activation, the output activation, and the
jax forward function. Every backend encoder (z3, gurobi, torch) is derived from
the SAME spec (see `encoders.py`), so training and verification can never end up
checking different functions.

Parameters are always `[(W0, b0), (W1, b1), ...]` with W shape (out, in).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrn
from jax import tree_util


# ----------------------------------------------------------------------
# Activations (names are shared with the symbolic encoders)
# ----------------------------------------------------------------------
HIDDEN_ACTIVATIONS = {
    "relu": jnn.relu,
    "tanh": jnn.tanh,
    "sigmoid": jnn.sigmoid,
}

# hard_sigmoid(z) = relu6(z + 3) / 6 = clip(z/6 + 0.5, 0, 1)
OUTPUT_ACTIVATIONS = {
    "linear": lambda z: z,
    "clip01": lambda z: jnp.clip(z, 0.0, 1.0),
    "hardsigmoid": jnn.hard_sigmoid,
    "relu": jnn.relu,
}

# Max |slope| of each output activation — its Lipschitz constant. Composed with
# the raw-net Lipschitz bound to get L_V (hard-sigmoid contracts by 1/6, which a
# spectral/LipBaB bound on the raw net alone misses).
OUTPUT_LIPSCHITZ = {
    "linear": 1.0,
    "clip01": 1.0,
    "hardsigmoid": 1.0 / 6.0,
    "relu": 1.0,
}


def _make_forward(hidden: str, output: str) -> Callable:
    h_act = HIDDEN_ACTIVATIONS[hidden]
    o_act = OUTPUT_ACTIVATIONS[output]

    def forward(params, x):
        a = x
        for w, b in params[:-1]:
            a = h_act(a @ w.T + b)
        w, b = params[-1]
        raw = (a @ w.T + b).squeeze(-1)
        return o_act(raw)

    return forward


@dataclass(frozen=True)
class CertificateSpec:
    name: str
    hidden_activation: str  # key into HIDDEN_ACTIVATIONS
    output_activation: str  # key into OUTPUT_ACTIVATIONS

    @property
    def forward(self) -> Callable:
        return _make_forward(self.hidden_activation, self.output_activation)

    @property
    def symbolic_friendly(self) -> bool:
        """Whether z3/gurobi can encode this exactly (piecewise-linear only)."""
        return self.hidden_activation == "relu" and self.output_activation in (
            "linear",
            "clip01",
            "hardsigmoid",
            "relu",
        )

    @property
    def output_lipschitz(self) -> float:
        """Lipschitz constant of the output activation (1, except hard-sigmoid = 1/6)."""
        return OUTPUT_LIPSCHITZ[self.output_activation]

    @property
    def bounded(self) -> bool:
        """True if V is bounded to [0, 1] (clip01 / hard-sigmoid) — required for the
        statistical (Hoeffding) soundness of the mab / montecarlo engines."""
        return self.output_activation in ("clip01", "hardsigmoid")


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------
DIRECTORY = {
    # Main supermartingale: relu hidden, hard-sigmoid output -> V in [0, 1].
    "bounded_pwl": CertificateSpec("bounded_pwl", "relu", "hardsigmoid"),
    # relu hidden, clip(raw, 0, 1) output.
    "clip_pwl": CertificateSpec("clip_pwl", "relu", "clip01"),
    # relu hidden, unbounded linear output.
    "relu_pwl": CertificateSpec("relu_pwl", "relu", "linear"),
    # relu hidden, relu output -> V >= 0 (bottom-bounded).
    "bottom_pwl": CertificateSpec("bottom_pwl", "relu", "relu"),
    # smooth (tanh) — sampling / autoLiRPA engines only (not exact in z3/gurobi).
    "tanh": CertificateSpec("tanh", "tanh", "linear"),
}


def get_spec(name: str) -> CertificateSpec:
    try:
        return DIRECTORY[name]
    except KeyError:
        raise ValueError(f"No such certificate '{name}'. Available: {sorted(DIRECTORY)}")


# ----------------------------------------------------------------------
# Parameter init + Lipschitz estimate (shared by training and verification)
# ----------------------------------------------------------------------
def glorot_init(sizes, key=jrn.PRNGKey(0), input_center=None):
    """sizes: layer widths; returns [(W, b), ...] with W shape (out, in).

    `input_center` (optional, e.g. the domain midpoint): fold b1 = -W1 @ center
    into the first layer so it sees zero-mean inputs. Purely an INITIALISATION
    reparameterisation — the network class is unchanged, so every engine encodes
    the trained weights as-is. For origin-centered domains this IS plain glorot
    (b1 = 0). Off-origin domains (e.g. doublewell, centred near x = 1.12) train
    better with it: otherwise the mean input direction dominates every gradient.
    """
    keys = jrn.split(key, len(sizes) - 1)
    params = []
    for m, n, k in zip(sizes[:-1], sizes[1:], keys):
        w = jrn.normal(k, (n, m)) * jnp.sqrt(2.0 / (m + n))
        b = jnp.zeros((n,))
        params.append((w, b))
    if input_center is not None:
        W1, b1 = params[0]
        params[0] = (W1, b1 - W1 @ jnp.asarray(input_center))
    return params


def get_spectral_norm_product(params):
    """Product of spectral norms of weight matrices — an upper Lipschitz bound for V."""
    leaves = tree_util.tree_leaves(params)
    log_product = 0.0
    for p in leaves:
        if p.ndim >= 2:
            matrix = p.reshape(p.shape[0], -1)
            s = jnp.linalg.svd(matrix, compute_uv=False)
            log_product += jnp.log(s[0] + 1e-12)
    return jnp.exp(log_product)
