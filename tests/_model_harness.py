"""
Shared helpers for the model-faithfulness test scripts.

The three suites
    test_dynamics_models.py   -- does each backend encode x -> x' ?
    test_network_models.py    -- does each backend encode V(x) ?
    test_drift_models.py      -- does each backend encode the full drift ?

all need the same primitive: "evaluate the symbolic encoding at a concrete
input and read a number back out". This module collects those extractors so the
suites read declaratively. It is NOT a test module (leading underscore -> pytest
does not collect it).

Backends:
  * Z3 / Gurobi  encode V and the step EXACTLY for piecewise-linear specs and
    linear/discrete dynamics; continuous additive noise is encoded as a BOX, so
    a maximised query returns a sound OVER-approximation of an expectation.
  * autoLiRPA (torch) builds the same V and step as torch modules.
"""

from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrn
import pytest

from src.certificates.encoders import encode_z3, encode_gurobi, build_torch


# ----------------------------------------------------------------------
# Backend availability
# ----------------------------------------------------------------------
def _has_gurobi() -> bool:
    try:
        import gurobipy as gp

        m = gp.Model()
        m.Params.OutputFlag = 0
        return True
    except Exception:
        return False


HAS_GUROBI = _has_gurobi()
gurobi_required = pytest.mark.skipif(not HAS_GUROBI, reason="no gurobi license")


# ----------------------------------------------------------------------
# Numeric helpers
# ----------------------------------------------------------------------
def z3_num(r) -> float:
    """Robustly turn a Z3 numeral / optimisation bound into a python float."""
    import z3

    if isinstance(r, z3.IntNumRef):
        return float(r.as_long())
    try:
        return float(r.as_fraction())
    except Exception:
        # algebraic or epsilon-infinitesimal bound: drop the trailing '?' marker
        return float(r.as_decimal(25).rstrip("?"))


def jax_mc_drift(env, V, eps, x, key, n=20000) -> float:
    """Monte-Carlo estimate of E_w[V(x')] - V(x) + eps at a single state x."""
    ks = jrn.split(key, n)
    x = jnp.asarray(x, dtype=jnp.float32)
    ev = jnp.mean(jax.vmap(lambda k: V(env.step(x, k)[0]))(ks))
    return float(ev - V(x) + eps)


# ----------------------------------------------------------------------
# Network V(x): evaluate each backend's encoding at a concrete x
# ----------------------------------------------------------------------
def z3_network_value(spec, params, x) -> float:
    import z3

    x = np.asarray(x, dtype=float)
    s = z3.Solver()
    xv = [z3.Real(f"x_{i}") for i in range(x.shape[0])]
    for i in range(x.shape[0]):
        s.add(xv[i] == float(x[i]))
    out = z3.Real("V_out")
    s.add(out == encode_z3(spec, s, params, xv))
    assert s.check() == z3.sat
    return float(s.model()[out].as_fraction())


def gurobi_network_value(spec, params, x, input_bounds=None) -> float:
    import gurobipy as gp

    x = np.asarray(x, dtype=float)
    m = gp.Model()
    m.Params.OutputFlag = 0
    xg = m.addVars(x.shape[0], lb=-gp.GRB.INFINITY)
    for i in range(x.shape[0]):
        m.addConstr(xg[i] == float(x[i]))
    ib = (x, x) if input_bounds is None else input_bounds
    vg = encode_gurobi(spec, m, params, list(xg.values()), input_bounds=ib)
    m.optimize()
    return float(vg.X)


def torch_network_value(spec, params, x) -> float:
    import torch

    x = np.asarray(x, dtype=np.float32)
    tmod = build_torch(spec, params)
    with torch.no_grad():
        return float(tmod(torch.tensor(x)).reshape(-1)[0].item())


# ----------------------------------------------------------------------
# Dynamics x -> x': extract the symbolic successor boxes at a concrete x
# Returns a list of (weight, lo, hi); discrete/no-noise cells have lo == hi.
# ----------------------------------------------------------------------
def gurobi_succ_boxes(env, x_point):
    import gurobipy as gp

    x_point = np.asarray(x_point, dtype=float)
    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.NonConvex = 2
    xv = m.addVars(env.dim, lb=-gp.GRB.INFINITY)
    for i in range(env.dim):
        m.addConstr(xv[i] == float(x_point[i]))
    branches = env.gurobi_step(m, [(1.0, list(xv.values()))])
    out = []
    for w, xn in branches:
        lo = np.empty(env.dim)
        hi = np.empty(env.dim)
        for i in range(env.dim):
            m.setObjective(xn[i], gp.GRB.MINIMIZE)
            m.optimize()
            lo[i] = xn[i].X
            m.setObjective(xn[i], gp.GRB.MAXIMIZE)
            m.optimize()
            hi[i] = xn[i].X
        out.append((float(w), lo, hi))
    return out


def z3_succ_boxes(env, x_point):
    import z3

    x_point = np.asarray(x_point, dtype=float)

    # First, a plain solve to discover branch count + weights.
    s = z3.Solver()
    xv = [z3.Real(f"x_{i}") for i in range(env.dim)]
    for i in range(env.dim):
        s.add(xv[i] == float(x_point[i]))
    weights = [float(w) for w, _ in env.z3_step(s, [(1.0, xv)])]

    out = []
    for b, w in enumerate(weights):
        lo = np.empty(env.dim)
        hi = np.empty(env.dim)
        for c in range(env.dim):
            for maximize in (False, True):
                o = z3.Optimize()
                xo = [z3.Real(f"x_{i}") for i in range(env.dim)]
                for i in range(env.dim):
                    o.add(xo[i] == float(x_point[i]))
                _, xn = env.z3_step(o, [(1.0, xo)])[b]
                h = o.maximize(xn[c]) if maximize else o.minimize(xn[c])
                assert o.check() == z3.sat
                val = z3_num(o.upper(h) if maximize else o.lower(h))
                if maximize:
                    hi[c] = val
                else:
                    lo[c] = val
        out.append((w, lo, hi))
    return out


# ----------------------------------------------------------------------
# Drift: symbolic MAX of  E_w[V(x')] - V(x) + eps  at a concrete x.
# For discrete noise this is the exact expectation drift; for continuous
# noise it is a sound upper bound (worst-case noise per cell).
# ----------------------------------------------------------------------
def gurobi_drift_max(env, spec, params, eps, x_point, domain_bounds, succ_bounds):
    import gurobipy as gp

    x_point = np.asarray(x_point, dtype=float)
    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.NonConvex = 2
    xv = m.addVars(env.dim, lb=-gp.GRB.INFINITY)
    xs = list(xv.values())
    for i in range(env.dim):
        m.addConstr(xs[i] == float(x_point[i]))
    Vx = encode_gurobi(spec, m, params, xs, input_bounds=domain_bounds, name="V", tag="t")
    branches = env.gurobi_step(m, [(1.0, xs)])
    EV = gp.quicksum(
        w * encode_gurobi(spec, m, params, xn, input_bounds=succ_bounds, name="V", tag=f"n{i}")
        for i, (w, xn) in enumerate(branches)
    )
    m.setObjective(EV - Vx + eps, gp.GRB.MAXIMIZE)
    m.optimize()
    return float(m.ObjVal)


def z3_drift_max(env, spec, params, eps, x_point):
    import z3

    x_point = np.asarray(x_point, dtype=float)
    o = z3.Optimize()
    xv = [z3.Real(f"x_{i}") for i in range(env.dim)]
    for i in range(env.dim):
        o.add(xv[i] == float(x_point[i]))
    Vx = encode_z3(spec, o, params, xv, name="V", tag="t")
    branches = env.z3_step(o, [(1.0, xv)])
    EV = z3.Sum(
        [
            z3.RealVal(w) * encode_z3(spec, o, params, xn, name="V", tag=f"n{i}")
            for i, (w, xn) in enumerate(branches)
        ]
    )
    h = o.maximize(EV - Vx + z3.RealVal(eps))
    assert o.check() == z3.sat
    return z3_num(o.upper(h))
