from jax import lax, jit, tree_util
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrn
import numpy as np

# Neural architecture


def _clip_action(x, l=-1.0, u=1.0):
    return jnp.clip(x, l, u)


def sigmoid_forward(params, x):
    a = x
    for w, b in params[:-1]:
        a = jnn.sigmoid(a @ w.T + b)
    w, b = params[-1]
    return (a @ w.T + b).squeeze(-1)


def relu_forward(params, x):
    a = x
    for w, b in params[:-1]:
        a = jnn.relu(a @ w.T + b)
    w, b = params[-1]
    return (a @ w.T + b).squeeze(-1)


def tanh_forward(params, x):
    a = x
    for w, b in params[:-1]:
        a = jnn.tanh(a @ w.T + b)
    w, b = params[-1]
    return (a @ w.T + b).squeeze(-1)


def put_activation_on_output(forward_method, activation):
    return lambda p, x: activation(forward_method(p, x))


bottom_bounded_pwl_martingale = put_activation_on_output(relu_forward, jnn.relu)
bounded_pwl_martingale = put_activation_on_output(relu_forward, jnn.hard_sigmoid)


def get_spectral_norm_product(params):
    """
    Calculates the product of spectral norms of weight matrices in a JAX model.
    Handles parameters provided as a PyTree (e.g., from Equinox or Haiku).
    """
    # Flatten the PyTree to get a list of all arrays
    leaves = tree_util.tree_leaves(params)

    log_product = 0.0

    for p in leaves:
        # Filter for weight matrices (usually 2D or 4D for convs)
        # We skip 1D arrays (biases)
        if p.ndim >= 2:
            # Flatten to 2D: [out_dim, everything_else]
            matrix = p.reshape(p.shape[0], -1)

            # Compute only singular values for efficiency
            # s[0] is the largest singular value (the spectral norm)
            s = jnp.linalg.svd(matrix, compute_uv=False)
            spectral_norm = s[0]

            # Use log-sum to prevent numerical overflow/underflow
            log_product += jnp.log(spectral_norm + 1e-12)

    return jnp.exp(log_product)


def glorot_init(sizes, key=jrn.PRNGKey(0)):
    """sizes: sequence of ints (layer widths); returns list[(W,b)] where W shape (out, in)
    using glorot init"""
    keys = jrn.split(key, len(sizes) - 1)
    params = []
    for m, n, k in zip(sizes[:-1], sizes[1:], keys):
        w = jrn.normal(k, (n, m)) * jnp.sqrt(2.0 / (m + n))
        b = jnp.zeros((n,))
        params.append((w, b))
    return params
