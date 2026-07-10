"""
Model faithfulness — NETWORK (the certificate V).

Goal: prove that the Z3, Gurobi and torch encoders of a certificate network all
compute the SAME function as the jax `spec.forward` they are derived from. If any
encoder drifts, the verifier would be checking a different V than was trained.

Coverage:
  * every piecewise-linear spec, several random nets, many probe points;
  * probe points chosen ON the ReLU kinks and in the saturating regions of the
    output activations (clip01 / hardsigmoid), where encodings are most likely
    to disagree;
  * smooth (tanh) spec: torch must still match jax, and the symbolic encoders
    must REFUSE (they cannot represent it exactly).

Run:
    ./venv/Scripts/python.exe -m pytest tests/test_network_models.py -q
"""

import numpy as np
import jax.numpy as jnp
import jax.random as jrn
import pytest

from src.certificates.structures import DIRECTORY as CERTS, glorot_init
from src.certificates.encoders import encode_z3, encode_gurobi
from tests._model_harness import (
    HAS_GUROBI,
    z3_network_value,
    gurobi_network_value,
    torch_network_value,
)

TOL = 1e-4

PWL_SPECS = [n for n, s in CERTS.items() if s.symbolic_friendly]
SMOOTH_SPECS = [n for n, s in CERTS.items() if not s.symbolic_friendly]

# A few architectures (incl. deep + multi-output hidden) and random scalings.
ARCHES = [[2, 5, 1], [2, 6, 6, 1], [3, 8, 4, 1], [4, 5, 5, 1]]


def _random_params(arch, key, scale):
    params = glorot_init(arch, key)
    return [(w * scale, b + 0.1 * scale) for w, b in params]


def _probe_points(dim, key, n=12):
    """Random points spanning a wide box (drives activations across breakpoints)."""
    return np.array(jrn.uniform(key, (n, dim), minval=-3.0, maxval=3.0))


# ----------------------------------------------------------------------
# 1. torch encoder matches jax forward (all specs, incl. smooth)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("spec_name", list(CERTS))
def test_torch_matches_jax(spec_name):
    spec = CERTS[spec_name]
    for ai, arch in enumerate(ARCHES):
        params = _random_params(arch, jrn.key(ai), scale=2.0)
        for x in _probe_points(arch[0], jrn.fold_in(jrn.key(99), ai)):
            v_jax = float(spec.forward(params, jnp.asarray(x)))
            v_torch = torch_network_value(spec, params, x)
            assert abs(v_jax - v_torch) < TOL, f"{spec_name} arch={arch} x={x}"


# ----------------------------------------------------------------------
# 2. Z3 encoder matches jax forward (PWL specs)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("spec_name", PWL_SPECS)
def test_z3_matches_jax(spec_name):
    spec = CERTS[spec_name]
    for ai, arch in enumerate(ARCHES):
        params = _random_params(arch, jrn.key(ai), scale=2.0)
        for x in _probe_points(arch[0], jrn.fold_in(jrn.key(7), ai)):
            v_jax = float(spec.forward(params, jnp.asarray(x)))
            v_z3 = z3_network_value(spec, params, x)
            assert abs(v_jax - v_z3) < TOL, f"{spec_name} arch={arch} x={x}"


# ----------------------------------------------------------------------
# 3. Gurobi encoder matches jax forward (PWL specs)
# ----------------------------------------------------------------------
@pytest.mark.skipif(not HAS_GUROBI, reason="no gurobi license")
@pytest.mark.parametrize("spec_name", PWL_SPECS)
def test_gurobi_matches_jax(spec_name):
    spec = CERTS[spec_name]
    for ai, arch in enumerate(ARCHES):
        params = _random_params(arch, jrn.key(ai), scale=2.0)
        # sound big-M bounds for the probe box [-3, 3]^d
        lo = -3.0 * np.ones(arch[0])
        hi = 3.0 * np.ones(arch[0])
        for x in _probe_points(arch[0], jrn.fold_in(jrn.key(7), ai)):
            v_jax = float(spec.forward(params, jnp.asarray(x)))
            v_g = gurobi_network_value(spec, params, x, input_bounds=(lo, hi))
            assert abs(v_jax - v_g) < TOL, f"{spec_name} arch={arch} x={x}"


# ----------------------------------------------------------------------
# 4. Breakpoint stress: points placed exactly on activation breakpoints, where
#    relu(0)=0, clip01 saturates at {0,1}, hardsigmoid kinks at z=+-3.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("spec_name", PWL_SPECS)
def test_breakpoints_all_backends_agree(spec_name):
    spec = CERTS[spec_name]
    # Hand-built net so we can land the raw output on activation breakpoints.
    # hidden relu makes h = relu([x0, -x0, x1, -x1]); output weights sum the four.
    W0 = jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    params = [(W0, jnp.zeros(4)), (jnp.array([[1.0, 1.0, 1.0, 1.0]]), jnp.zeros(1))]
    probes = np.array(
        [
            [0.0, 0.0],     # everything at the kink, raw output 0
            [3.0, 0.0],     # raw 3  -> hardsigmoid kink, clip01 mid
            [-3.0, 0.0],    # raw 3
            [0.5, -0.5],    # raw 1  -> clip01 saturates at 1
            [6.0, 6.0],     # raw 12 -> deep in saturation
            [-1.7, 2.3],    # generic
        ]
    )
    for x in probes:
        v_jax = float(spec.forward(params, jnp.asarray(x)))
        assert abs(v_jax - z3_network_value(spec, params, x)) < TOL
        assert abs(v_jax - torch_network_value(spec, params, x)) < TOL
        if HAS_GUROBI:
            v_g = gurobi_network_value(spec, params, x, input_bounds=(-7 * np.ones(2), 7 * np.ones(2)))
            assert abs(v_jax - v_g) < TOL


# ----------------------------------------------------------------------
# 5. Smooth specs: torch matches jax, symbolic encoders correctly REFUSE.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("spec_name", SMOOTH_SPECS)
def test_smooth_spec_torch_ok_symbolic_rejected(spec_name):
    import z3

    spec = CERTS[spec_name]
    params = _random_params([2, 6, 1], jrn.key(0), scale=1.5)
    for x in _probe_points(2, jrn.key(3), n=8):
        v_jax = float(spec.forward(params, jnp.asarray(x)))
        assert abs(v_jax - torch_network_value(spec, params, x)) < 1e-3

    s = z3.Solver()
    xv = [z3.Real("x_0"), z3.Real("x_1")]
    with pytest.raises(NotImplementedError):
        encode_z3(spec, s, params, xv)

    if HAS_GUROBI:
        import gurobipy as gp

        m = gp.Model()
        m.Params.OutputFlag = 0
        xg = m.addVars(2, lb=-gp.GRB.INFINITY)
        with pytest.raises(NotImplementedError):
            encode_gurobi(spec, m, params, list(xg.values()), input_bounds=(-np.ones(2), np.ones(2)))
