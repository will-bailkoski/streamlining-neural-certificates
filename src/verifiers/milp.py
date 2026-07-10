"""
MILP verification engine (Gurobi).

Encodes V and the env's stochastic step as a mixed-integer program, then
MAXIMISES the (over-approximated) drift over (domain \\ equilibrium):

    max_x  E_w[V(x')] - V(x) + epsilon

max <= 0  -> verified (sound: continuous-noise cells are encoded as boxes, so
             the maximiser sees an upper bound on the true expectation)
max  > 0  -> the maximiser x is a counterexample (re-checked by the caller)

ReLU big-M constants are derived from sound interval bounds: the input domain
for V(x), and a reachable-set pre-pass for V(x').
"""

from __future__ import annotations
import time
import numpy as np

import gurobipy as gp
from gurobipy import GRB

from src.verifiers.base import VerifierResult, register
from src.certificates.encoders import encode_gurobi


def _successor_bounds(env):
    """Sound box (lo, hi) bounding every successor reachable from the domain.

    Built by min/maxing each successor coordinate over the domain with the same
    symbolic step the verifier uses, so it is valid for nonlinear dynamics too.
    """
    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.NonConvex = 2
    x = m.addVars(env.dim, lb=-GRB.INFINITY)
    env.domain.encode_gurobi_inclusion(m, list(x.values()))
    branches = env.gurobi_step(m, [(1.0, list(x.values()))])

    lo = np.full(env.dim, np.inf)
    hi = np.full(env.dim, -np.inf)
    for _, x_next in branches:
        for i in range(env.dim):
            m.setObjective(x_next[i], GRB.MINIMIZE)
            m.optimize()
            lo[i] = min(lo[i], x_next[i].X)
            m.setObjective(x_next[i], GRB.MAXIMIZE)
            m.optimize()
            hi[i] = max(hi[i], x_next[i].X)
    return lo, hi


def find_counterexample(env, spec, params, epsilon: float, key=None, time_limit=None,
                        noise_disc=None, **hp):
    domain_bounds = (env.domain.bounds[:, 0], env.domain.bounds[:, 1])
    succ_bounds = _successor_bounds(env)

    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.NonConvex = 2
    if time_limit is not None:
        m.Params.TimeLimit = float(time_limit)

    x = m.addVars(env.dim, lb=-GRB.INFINITY)
    xs = list(x.values())
    env.domain.encode_gurobi_inclusion(m, xs)
    env.equilibrium.encode_gurobi_exclusion(m, xs)

    Vx = encode_gurobi(spec, m, params, xs, input_bounds=domain_bounds, name="V", tag="t")
    branches = env.gurobi_step(m, [(1.0, xs)], noise_disc=noise_disc)
    EV = gp.quicksum(
        w * encode_gurobi(spec, m, params, x_next, input_bounds=succ_bounds, name="V", tag=f"n{i}")
        for i, (w, x_next) in enumerate(branches)
    )

    m.setObjective(EV - Vx + epsilon, GRB.MAXIMIZE)

    # Verification needs only the SIGN of the maximum drift, not its exact value,
    # so we stop as soon as Gurobi settles that sign — far sooner than OPTIMAL:
    #   * MIPSOL with objective > 0  -> a point violates the drift condition; that
    #     point is a counterexample, no need to prove it is the worst one.
    #   * MIP dual bound <= 0        -> the best (upper) bound on the max is <= 0,
    #     which SOUNDLY proves max drift <= 0; no need to close the optimality gap.
    # Without these, Gurobi maximises all the way to OPTIMAL (or the TimeLimit),
    # proving the exact worst-case drift — the wasted work that caused the timeouts.
    m._x = xs
    m._viol = None
    m._verified_by_bound = False

    def _cb(model, where):
        if where == GRB.Callback.MIPSOL:
            if model.cbGet(GRB.Callback.MIPSOL_OBJ) > 0.0:
                model._viol = model.cbGetSolution(model._x)
                model.terminate()
        elif where == GRB.Callback.MIP:
            if model.cbGet(GRB.Callback.MIP_OBJBND) <= 0.0:
                model._verified_by_bound = True
                model.terminate()

    t0 = time.perf_counter()
    m.optimize(_cb)
    elapsed = time.perf_counter() - t0
    stats = {"time": elapsed, "n_branches": len(branches),
             "noise_disc": noise_disc, "status": int(m.Status),
             "n_vars": int(m.NumVars), "n_bin_vars": int(m.NumBinVars),
             "n_constrs": int(m.NumConstrs)}

    if m.Status == GRB.INFEASIBLE:
        # No state in domain\equilibrium at all — vacuously verified.
        return VerifierResult(verified=True, stats=stats, engine="milp")

    try:
        stats["obj_bound"] = float(m.ObjBound)
    except Exception:
        stats["obj_bound"] = float("inf")

    # a violating point: caught by the callback, or a proven maximiser with drift > 0
    viol = None
    if m._viol is not None:
        viol = np.array(m._viol, dtype=float)
    elif m.SolCount > 0 and m.ObjVal > 0.0:
        viol = np.array([x[i].X for i in range(env.dim)], dtype=float)
    if viol is not None:
        stats["max_drift"] = float(m.ObjVal) if m.SolCount > 0 else None
        return VerifierResult(verified=False, violation=viol[None, :], stats=stats, engine="milp")

    # no violation exists: a nonpositive upper bound proves it (sound even without OPTIMAL)
    if m._verified_by_bound or stats["obj_bound"] <= 0.0:
        return VerifierResult(verified=True, stats=stats, engine="milp")

    # ran out of budget with the bound still positive and no violation found
    return VerifierResult(verified=None, stats=stats, engine="milp")


class _MILPEngine:
    name = "milp"
    find_counterexample = staticmethod(find_counterexample)


register("milp", _MILPEngine())
