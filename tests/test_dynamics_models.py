"""
Model faithfulness — DYNAMICS.

Goal: prove that the Z3, Gurobi and autoLiRPA encodings of the one-step map
x -> x' represent the SAME stochastic step as the jax `env.step`, which is the
ground truth used for training and for re-checking counterexamples.

What "represent correctly" means per backend:
  * Z3 / Gurobi encode the step as a set of weighted successor BOXES. For
    discrete / noise-free dynamics the boxes collapse to points and must match
    the jax successors EXACTLY; for continuous additive noise the boxes must be
    SOUND (contain every jax successor) and the box midpoints must reproduce the
    true successor expectation.
  * autoLiRPA uses a torch next-state module; it must equal the jax core
    dynamics plus the same additive noise.

Run:
    ./venv/Scripts/python.exe -m pytest tests/test_dynamics_models.py -q
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrn
import pytest

from src.benchmarks.utils import make_env
from tests._model_harness import (
    HAS_GUROBI,
    gurobi_required,
    gurobi_succ_boxes,
    z3_succ_boxes,
)

# Envs whose step is exactly symbolic-encodable (z3/gurobi).
DISCRETE_ENVS = ["linear2D", "linear2D_p0", "linear2D_p1"]  # switched modes, no add. noise
CONTINUOUS_ENVS = ["linstoch2D", "doublewell"]  # additive bounded noise -> boxes
SYMBOLIC_ENVS = DISCRETE_ENVS + CONTINUOUS_ENVS

# Envs that expose a torch next-state module (autoLiRPA path).
TORCH_STEP_ENVS = ["linstoch2D", "doublewell", "thermal", "vanderpol", "pendulum_lqr"]


def _states(env, n=6, seed=0):
    """A spread of valid states (inside the domain, outside the equilibrium)."""
    xs, _ = env.sample(jrn.key(seed), n)
    return np.array(xs)


# ----------------------------------------------------------------------
# 1. Z3 and Gurobi agree with each other on the successor boxes
# ----------------------------------------------------------------------
@gurobi_required
@pytest.mark.parametrize("env_name", SYMBOLIC_ENVS)
def test_z3_and_gurobi_boxes_agree(env_name):
    env = make_env(env_name)
    for x in _states(env):
        gb = sorted(gurobi_succ_boxes(env, x), key=lambda t: (round(t[0], 6), t[1].tolist()))
        zb = sorted(z3_succ_boxes(env, x), key=lambda t: (round(t[0], 6), t[1].tolist()))
        assert len(gb) == len(zb)
        for (wg, lg, hg), (wz, lz, hz) in zip(gb, zb):
            assert abs(wg - wz) < 1e-9
            assert np.allclose(lg, lz, atol=1e-6)
            assert np.allclose(hg, hz, atol=1e-6)


# ----------------------------------------------------------------------
# 2. Box weights are a probability distribution
# ----------------------------------------------------------------------
@pytest.mark.parametrize("env_name", SYMBOLIC_ENVS)
def test_box_weights_sum_to_one(env_name):
    env = make_env(env_name)
    for x in _states(env):
        boxes = z3_succ_boxes(env, x)
        assert abs(sum(w for w, _, _ in boxes) - 1.0) < 1e-9
        if HAS_GUROBI:
            gboxes = gurobi_succ_boxes(env, x)
            assert abs(sum(w for w, _, _ in gboxes) - 1.0) < 1e-9


# ----------------------------------------------------------------------
# 3. Soundness: every sampled jax successor lies inside some symbolic box
# ----------------------------------------------------------------------
@pytest.mark.parametrize("env_name", SYMBOLIC_ENVS)
@pytest.mark.parametrize("backend", ["z3", "gurobi"])
def test_jax_successors_contained_in_boxes(env_name, backend):
    if backend == "gurobi" and not HAS_GUROBI:
        pytest.skip("no gurobi license")
    env = make_env(env_name)
    extract = gurobi_succ_boxes if backend == "gurobi" else z3_succ_boxes
    for x in _states(env):
        boxes = extract(env, x)
        ks = jrn.split(jrn.key(11), 4000)
        succ = np.array(jax.vmap(lambda k: env.step(jnp.asarray(x, jnp.float32), k)[0])(ks))
        inside = np.zeros(len(succ), bool)
        for _, lo, hi in boxes:
            inside |= np.all((succ >= lo - 1e-5) & (succ <= hi + 1e-5), axis=1)
        assert inside.all(), f"{(~inside).sum()} jax successors fell outside the boxes"


# ----------------------------------------------------------------------
# 4. Discrete dynamics are EXACT: boxes collapse to points that match the
#    deterministic mode successors of the switched-linear system.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("env_name", DISCRETE_ENVS)
def test_discrete_boxes_are_points_matching_modes(env_name):
    env = make_env(env_name)
    # float64 so the comparison is against the same exact arithmetic z3 does, not
    # a float32 matmul that rounds differently in the last digit.
    A0 = np.array(env.A0, dtype=np.float64)
    A1 = np.array(env.A1, dtype=np.float64)
    for x in _states(env):
        xx = np.asarray(x, dtype=np.float64)
        boxes = z3_succ_boxes(env, x)
        # boxes collapse to points (no additive noise -> lo == hi)
        for _, lo, hi in boxes:
            assert np.allclose(lo, hi, atol=1e-6)
        pts = [(lo + hi) / 2 for _, lo, hi in boxes]
        expect = [A0 @ xx, A1 @ xx]
        # every symbolic successor point equals one of the two mode successors
        for p in pts:
            assert min(np.max(np.abs(p - e)) for e in expect) < 1e-5
        # and both modes are represented
        for e in expect:
            assert min(np.max(np.abs(p - e)) for p in pts) < 1e-5


# ----------------------------------------------------------------------
# 5. Successor EXPECTATION matches a high-sample Monte-Carlo estimate.
#    Sum_i w_i * midpoint_i  ==  E_w[x']  (exact for discrete, tight for
#    symmetric continuous noise whose cell midpoints are unbiased).
# ----------------------------------------------------------------------
@pytest.mark.parametrize("env_name", SYMBOLIC_ENVS)
def test_box_expectation_matches_mc(env_name):
    env = make_env(env_name)
    for x in _states(env, n=4):
        boxes = z3_succ_boxes(env, x)
        e_sym = sum(w * (lo + hi) / 2 for w, lo, hi in boxes)
        ks = jrn.split(jrn.key(5), 80000)
        e_mc = np.array(
            jax.vmap(lambda k: env.step(jnp.asarray(x, jnp.float32), k)[0])(ks)
        ).mean(axis=0)
        assert np.allclose(e_sym, e_mc, atol=3e-2)


# ----------------------------------------------------------------------
# 6. autoLiRPA torch step == jax core dynamics + same additive noise.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("env_name", TORCH_STEP_ENVS)
def test_torch_step_matches_jax_core(env_name):
    import torch

    env = make_env(env_name)
    step = env._torch_step()
    xs = _states(env, n=8, seed=2).astype(np.float32)
    rng = np.random.default_rng(0)
    w = rng.normal(scale=0.05, size=xs.shape).astype(np.float32)

    # jax core dynamics (deterministic part); additive noise added explicitly
    jax_core = np.array(
        jax.vmap(lambda z: env._dynamics(z, jrn.key(0))[0])(jnp.asarray(xs))
    )
    jax_next = jax_core + w

    with torch.no_grad():
        torch_next = step(torch.tensor(xs), torch.tensor(w)).cpu().numpy()

    assert np.max(np.abs(jax_next - torch_next)) < 1e-4
