"""
Model faithfulness — the full DRIFT functional.

The drift is the one quantity every engine must agree on:

    d(x) = E_w[ V(f(x, w)) ] - V(x) + epsilon

This suite checks that the Z3, Gurobi and autoLiRPA encodings of d compose the
network V and the dynamics f into the SAME functional as the jax Monte-Carlo
ground truth (`src.verifiers.drift`).

Two regimes:
  * DISCRETE dynamics (switched-linear, no additive noise): the symbolic E_w is
    EXACT, so every backend's drift must equal the analytic expectation drift
    (1-p) V(A0 x) + p V(A1 x) - V(x) + eps, and all backends must agree.
  * CONTINUOUS additive noise: the symbolic E_w is a sound OVER-approximation,
    so the maximised symbolic drift (and the autoLiRPA upper bound) must lie at
    or above the true Monte-Carlo drift -- never below (that would be unsound).
    Separately, the torch drift module must reproduce a jax quadrature drift
    exactly when fed the same cell noises (validates the composition arithmetic).

Run:
    ./venv/Scripts/python.exe -m pytest tests/test_drift_models.py -q
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrn
import pytest

from src.benchmarks.utils import make_env
from src.certificates.structures import get_spec, glorot_init
from src.certificates.encoders import build_torch
from tests._model_harness import (
    HAS_GUROBI,
    gurobi_required,
    gurobi_drift_max,
    z3_drift_max,
    jax_mc_drift,
)

EPS = 1e-3
SPEC = get_spec("relu_pwl")  # relu hidden, linear (unbounded) output


def _net(dim, key, scale=1.0):
    params = glorot_init([dim, 6, 1], key)
    return [(w * scale, b) for w, b in params]


def _states(env, n=5, seed=0):
    xs, _ = env.sample(jrn.key(seed), n)
    return np.array(xs)


def _exact_discrete_drift(env, params, x):
    """Analytic E_w[V(x')]-V(x)+eps for a two-mode switched-linear env (exact)."""
    V = lambda z: float(SPEC.forward(params, jnp.asarray(z, jnp.float32)))
    A0, A1, p = np.array(env.A0), np.array(env.A1), float(env.p)
    return (1 - p) * V(A0 @ x) + p * V(A1 @ x) - V(x) + EPS


# ======================================================================
# DISCRETE: every backend reproduces the exact expectation drift
# ======================================================================
@gurobi_required
def test_discrete_drift_exact_all_backends_agree():
    env = make_env("linear2D")
    params = _net(env.dim, jrn.key(1), scale=1.5)
    dom = (np.array(env.domain.bounds[:, 0]), np.array(env.domain.bounds[:, 1]))

    import torch

    module, noise_box = env.lirpa_drift_module(build_torch(SPEC, params), EPS)
    assert noise_box is None  # discrete modes carry no noise input
    module.eval()

    for x in _states(env, n=6):
        exact = _exact_discrete_drift(env, params, x)
        d_z3 = z3_drift_max(env, SPEC, params, EPS, x)
        d_milp = gurobi_drift_max(env, SPEC, params, EPS, x, dom, dom)
        with torch.no_grad():
            d_torch = float(module(torch.tensor(x, dtype=torch.float32).reshape(1, -1)).reshape(-1)[0])
        assert abs(d_z3 - exact) < 1e-4, f"z3 {d_z3} vs {exact}"
        assert abs(d_milp - exact) < 1e-4, f"milp {d_milp} vs {exact}"
        assert abs(d_torch - exact) < 1e-4, f"torch {d_torch} vs {exact}"


def test_discrete_drift_z3_torch_agree_no_gurobi():
    # Same as above but the part that needs no gurobi license, so CI without a
    # license still pins z3 <-> torch <-> analytic agreement.
    import torch

    env = make_env("linear2D")
    params = _net(env.dim, jrn.key(2), scale=1.0)
    module, _ = env.lirpa_drift_module(build_torch(SPEC, params), EPS)
    module.eval()
    for x in _states(env, n=6, seed=3):
        exact = _exact_discrete_drift(env, params, x)
        assert abs(z3_drift_max(env, SPEC, params, EPS, x) - exact) < 1e-4
        with torch.no_grad():
            d_torch = float(module(torch.tensor(x, dtype=torch.float32).reshape(1, -1)).reshape(-1)[0])
        assert abs(d_torch - exact) < 1e-4


# ======================================================================
# CONTINUOUS: symbolic maxima are SOUND upper bounds on the MC drift
# ======================================================================
@pytest.mark.parametrize("env_name", ["linstoch2D", "doublewell"])
def test_z3_drift_is_sound_upper_bound(env_name):
    env = make_env(env_name)
    params = _net(env.dim, jrn.key(4))
    V = lambda z: SPEC.forward(params, z)
    for i, x in enumerate(_states(env, n=5)):
        d_mc = jax_mc_drift(env, V, EPS, x, jrn.fold_in(jrn.key(0), i))
        d_sym = z3_drift_max(env, SPEC, params, EPS, x)
        assert d_sym >= d_mc - 5e-3, f"z3 unsound: {d_sym} < MC {d_mc}"


@gurobi_required
@pytest.mark.parametrize("env_name", ["linstoch2D", "doublewell"])
def test_gurobi_drift_is_sound_upper_bound(env_name):
    from src.verifiers.milp import _successor_bounds

    env = make_env(env_name)
    params = _net(env.dim, jrn.key(4))
    V = lambda z: SPEC.forward(params, z)
    dom = (np.array(env.domain.bounds[:, 0]), np.array(env.domain.bounds[:, 1]))
    succ = _successor_bounds(env)
    for i, x in enumerate(_states(env, n=5)):
        d_mc = jax_mc_drift(env, V, EPS, x, jrn.fold_in(jrn.key(0), i))
        d_sym = gurobi_drift_max(env, SPEC, params, EPS, x, dom, succ)
        assert d_sym >= d_mc - 5e-3, f"milp unsound: {d_sym} < MC {d_mc}"


@pytest.mark.parametrize("env_name", ["linstoch2D", "doublewell"])
def test_lirpa_drift_is_sound_upper_bound(env_name):
    import torch
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

    env = make_env(env_name)
    params = _net(env.dim, jrn.key(4))
    V = lambda z: SPEC.forward(params, z)

    module, nb = env.lirpa_drift_module(build_torch(SPEC, params), EPS, noise_disc=2)
    module.eval()
    w_lo, w_hi = nb
    bm = BoundedModule(module, (torch.empty((1, env.dim)), torch.empty((1, w_lo.size))), device="cpu")

    for i, x in enumerate(_states(env, n=5)):
        d_mc = jax_mc_drift(env, V, EPS, x, jrn.fold_in(jrn.key(0), i))
        xt = torch.tensor(x, dtype=torch.float32).reshape(1, -1)
        x_bt = BoundedTensor(xt, PerturbationLpNorm(x_L=xt, x_U=xt))  # degenerate: the point x
        wl = torch.tensor(w_lo, dtype=torch.float32).reshape(1, -1)
        wu = torch.tensor(w_hi, dtype=torch.float32).reshape(1, -1)
        w_bt = BoundedTensor((wl + wu) / 2, PerturbationLpNorm(x_L=wl, x_U=wu))
        _, ub = bm.compute_bounds(x=(x_bt, w_bt), method="CROWN")
        ub_val = float(ub.detach().reshape(-1)[0])
        assert ub_val >= d_mc - 5e-3, f"lirpa unsound: {ub_val} < MC {d_mc}"


# ======================================================================
# The torch drift module composes V and f correctly: feeding the cell-midpoint
# noises reproduces a jax quadrature drift exactly (no bound propagation here).
# ======================================================================
@pytest.mark.parametrize("env_name", ["linstoch2D", "doublewell"])
@pytest.mark.parametrize("disc", [1, 2])
def test_torch_drift_module_matches_jax_quadrature(env_name, disc):
    import torch

    env = make_env(env_name)
    params = _net(env.dim, jrn.key(6))
    V = lambda z: SPEC.forward(params, jnp.asarray(z, jnp.float32))

    los, his, weights = env._post.get_support(disc)
    los = np.asarray(los, float)
    his = np.asarray(his, float)
    weights = np.asarray(weights, float)
    mids = (los + his) / 2  # (K, nd) cell-representative noises

    module, _ = env.lirpa_drift_module(build_torch(SPEC, params), EPS, noise_disc=disc)
    module.eval()

    for x in _states(env, n=4, seed=7):
        # jax quadrature: sum_k w_k V(core(x) + mid_k) - V(x) + eps
        core = np.array(env._dynamics(jnp.asarray(x, jnp.float32), jrn.key(0))[0])
        quad = sum(
            float(weights[k]) * float(V(core + mids[k])) for k in range(len(weights))
        ) - float(V(x)) + EPS

        w_packed = torch.tensor(mids.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            d_torch = float(
                module(torch.tensor(x, dtype=torch.float32).reshape(1, -1), w_packed).reshape(-1)[0]
            )
        assert abs(d_torch - quad) < 1e-4, f"{env_name} disc={disc}: torch {d_torch} vs jax {quad}"
