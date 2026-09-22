"""
Backend encoders for certificate structures.

Each encoder takes a `CertificateSpec` (from `structures.py`) plus the trained
params and emits the SAME function V in the target backend:

  - `encode_z3`     -> a Z3 arithmetic expression for V(x_vars)
  - `encode_gurobi` -> a Gurobi variable equal to V(x_vars)
  - `build_torch`   -> an nn.Module computing V (for autoLiRPA)

Only piecewise-linear specs (`spec.symbolic_friendly`) can be encoded in z3 /
gurobi; smooth specs (e.g. tanh) are restricted to the sampling / autoLiRPA
engines and raise here.
"""

from __future__ import annotations
import numpy as np

import gurobipy as gp
from gurobipy import GRB

from src.certificates.structures import CertificateSpec


# ----------------------------------------------------------------------
# Z3
# ----------------------------------------------------------------------
def encode_z3(spec: CertificateSpec, solver, params, x_vars, name="V", tag=""):
    import z3
    from z3 import Real, If, Sum

    if not spec.symbolic_friendly:
        raise NotImplementedError(
            f"Certificate '{spec.name}' (hidden={spec.hidden_activation}) is not "
            f"exactly encodable in Z3; use the autoLiRPA, Monte Carlo or MAB engine."
        )

    current = list(x_vars)
    for layer_idx, (W, b) in enumerate(params):
        W_np = np.array(W)
        b_np = np.array(b)
        out_dim = W_np.shape[0]
        is_last = layer_idx == len(params) - 1

        z_vars = []
        for i in range(out_dim):
            z_i = Real(f"{name}_z_{layer_idx}_{i}_{tag}")
            solver.add(
                z_i == Sum([W_np[i, j] * current[j] for j in range(len(current))]) + b_np[i]
            )
            z_vars.append(z_i)

        if is_last:
            assert out_dim == 1, "certificate output must be scalar"
            return _z3_output(spec.output_activation, z_vars[0])

        h_vars = []
        for i in range(out_dim):
            h_i = Real(f"{name}_h_{layer_idx}_{i}_{tag}")
            solver.add(h_i == If(z_vars[i] >= 0, z_vars[i], 0))
            h_vars.append(h_i)
        current = h_vars

    raise ValueError("Empty params: no layers to encode.")


def _z3_output(activation: str, raw):
    from z3 import If

    if activation == "linear":
        return raw
    if activation == "relu":
        return If(raw > 0, raw, 0)
    if activation == "clip01":
        return If(raw > 1, 1.0, If(raw < 0, 0.0, raw))
    if activation == "hardsigmoid":
        a = raw / 6.0 + 0.5
        return If(a > 1, 1.0, If(a < 0, 0.0, a))
    raise ValueError(f"Unknown output activation '{activation}'")


# ----------------------------------------------------------------------
# Gurobi
# ----------------------------------------------------------------------
def _affine_interval(W, b, lo, hi):
    """Interval image of [lo, hi] under z = W @ a + b (sound IBP)."""
    Wp = np.maximum(W, 0.0)
    Wn = np.minimum(W, 0.0)
    z_lo = Wp @ lo + Wn @ hi + b
    z_hi = Wp @ hi + Wn @ lo + b
    return z_lo, z_hi


def encode_gurobi(
    spec: CertificateSpec, model: gp.Model, params, x_vars, input_bounds, name="V", tag=""
):
    """
    input_bounds: (lo, hi) arrays bounding x_vars. Required — used to derive
    sound, tight big-M constants for the ReLU encoding via interval propagation.
    A loose/invalid big-M would make verification unsound, so it is not optional.
    """
    if not spec.symbolic_friendly:
        raise NotImplementedError(
            f"Certificate '{spec.name}' (hidden={spec.hidden_activation}) is not "
            f"exactly encodable in Gurobi; use the autoLiRPA, Monte Carlo or MAB engine."
        )

    lo = np.array(input_bounds[0], dtype=float)
    hi = np.array(input_bounds[1], dtype=float)

    current = list(x_vars)
    for layer_idx, (W, b) in enumerate(params):
        W_np = np.array(W)
        b_np = np.array(b)
        out_dim = W_np.shape[0]
        is_last = layer_idx == len(params) - 1

        z_lo, z_hi = _affine_interval(W_np, b_np, lo, hi)

        z = model.addVars(out_dim, lb=-GRB.INFINITY, name=f"{name}_z_{layer_idx}_{tag}")
        for i in range(out_dim):
            model.addConstr(
                z[i]
                == gp.quicksum(W_np[i, j] * current[j] for j in range(len(current))) + b_np[i]
            )

        if is_last:
            assert out_dim == 1, "certificate output must be scalar"
            return _gurobi_output(model, spec.output_activation, z[0], name, tag)

        # Tight ReLU MILP (Tjeng et al.) using sound per-neuron bounds l <= z <= u.
        h = model.addVars(out_dim, lb=0.0, name=f"{name}_h_{layer_idx}_{tag}")
        d = model.addVars(out_dim, vtype=GRB.BINARY, name=f"{name}_d_{layer_idx}_{tag}")
        for i in range(out_dim):
            l, u = float(z_lo[i]), float(z_hi[i])
            if u <= 0:
                model.addConstr(h[i] == 0.0)
            elif l >= 0:
                model.addConstr(h[i] == z[i])
            else:
                model.addConstr(h[i] >= z[i])
                model.addConstr(h[i] <= z[i] - l * (1 - d[i]))
                model.addConstr(h[i] <= u * d[i])
        current = list(h.values())

        # propagate interval through the ReLU for the next layer
        lo = np.maximum(z_lo, 0.0)
        hi = np.maximum(z_hi, 0.0)

    raise ValueError("Empty params: no layers to encode.")


def _gurobi_output(model, activation: str, raw, name, tag):
    if activation == "linear":
        out = model.addVar(lb=-GRB.INFINITY, name=f"{name}_out_{tag}")
        model.addConstr(out == raw)
        return out
    if activation == "relu":
        out = model.addVar(lb=0.0, name=f"{name}_out_{tag}")
        model.addGenConstrMax(out, [raw], constant=0.0)
        return out
    if activation == "clip01":
        tmp = model.addVar(lb=-GRB.INFINITY)
        model.addGenConstrMax(tmp, [raw], constant=0.0)
        out = model.addVar(lb=0.0, ub=1.0, name=f"{name}_out_{tag}")
        model.addGenConstrMin(out, [tmp], constant=1.0)
        return out
    if activation == "hardsigmoid":
        a = model.addVar(lb=-GRB.INFINITY, name=f"{name}_hs_affine_{tag}")
        model.addConstr(a == (1.0 / 6.0) * raw + 0.5)
        relu = model.addVar(lb=0.0, name=f"{name}_hs_relu_{tag}")
        model.addGenConstrMax(relu, [a], constant=0.0)
        out = model.addVar(lb=0.0, ub=1.0, name=f"{name}_hs_out_{tag}")
        model.addGenConstrMin(out, [relu], constant=1.0)
        return out
    raise ValueError(f"Unknown output activation '{activation}'")


# ----------------------------------------------------------------------
# Torch (for autoLiRPA) — full integration happens in the LiRPA engine
# ----------------------------------------------------------------------
def build_torch(spec: CertificateSpec, params):
    """
    Return an nn.Module computing V, matching `spec`, with `params` loaded.

    Uses only standard nn layers so the module is traceable by auto_LiRPA, while
    staying numerically identical to the jax forward:
      clip01      -> nn.Hardtanh(0, 1) = clamp(z, 0, 1)
      hardsigmoid -> scale-shift + nn.Hardtanh(0, 1) = clamp(z/6 + 0.5, 0, 1).
                     NOT nn.Hardsigmoid(): auto_LiRPA has no bound rule for the
                     onnx::HardSigmoid op and raises NotImplementedError; the
                     decomposition is the same function through supported ops.
    """
    import torch
    import torch.nn as nn

    class _HardSigmoidAffine(nn.Module):
        """z -> z/6 + 0.5 (the affine part of hard-sigmoid, LiRPA-traceable)."""

        def forward(self, z):
            return z * (1.0 / 6.0) + 0.5

    hidden_act = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }[spec.hidden_activation]

    output_layers = {
        "linear": [],
        "relu": [nn.ReLU()],
        "clip01": [nn.Hardtanh(0.0, 1.0)],
        "hardsigmoid": [_HardSigmoidAffine(), nn.Hardtanh(0.0, 1.0)],
    }[spec.output_activation]

    layers = []
    n = len(params)
    for i, (W, b) in enumerate(params):
        W_np = np.array(W)
        lin = nn.Linear(W_np.shape[1], W_np.shape[0])
        with torch.no_grad():
            lin.weight.copy_(torch.tensor(W_np, dtype=torch.float32))
            lin.bias.copy_(torch.tensor(np.array(b), dtype=torch.float32))
        layers.append(lin)
        if i < n - 1:
            layers.append(hidden_act())
    layers.extend(output_layers)
    return nn.Sequential(*layers)
