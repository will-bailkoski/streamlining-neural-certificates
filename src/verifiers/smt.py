"""
SMT verification engine (Z3).

Encodes the certificate V and the env's stochastic step symbolically, then asks
Z3 whether any state in (domain \\ equilibrium) violates the drift condition:

    E_w[V(x')] - V(x) + epsilon > 0

unsat  -> verified (sound; the symbolic E_w uses the noise-box over-approximation)
sat    -> counterexample state (re-checked by the caller for continuous noise)
unknown-> inconclusive
"""

from __future__ import annotations
import time
import numpy as np

from src.verifiers.base import VerifierResult, register
from src.certificates.encoders import encode_z3


def find_counterexample(env, spec, params, epsilon: float, key=None, timeout_ms=None,
                        noise_disc=None, **hp):
    import z3

    s = z3.Solver()
    if timeout_ms is not None:
        s.set("timeout", int(timeout_ms))

    x = [z3.Real(f"x_{i}") for i in range(env.dim)]

    # x in domain, x not in equilibrium
    env.domain.encode_z3_inclusion(s, x)
    env.equilibrium.encode_z3_exclusion(s, x)

    # V(x) and the expected successor value over the symbolic branches
    Vx = encode_z3(spec, s, params, x, name="V", tag="t")
    branches = env.z3_step(s, [(1.0, x)], noise_disc=noise_disc)
    EV = z3.Sum(
        [
            z3.RealVal(w) * encode_z3(spec, s, params, x_next, name="V", tag=f"n{i}")
            for i, (w, x_next) in enumerate(branches)
        ]
    )

    # violation: E[V'] - V + eps > 0
    s.add(EV - Vx + z3.RealVal(epsilon) > 0)

    t0 = time.perf_counter()
    result = s.check()
    elapsed = time.perf_counter() - t0
    stats = {"time": elapsed, "n_branches": len(branches), "noise_disc": noise_disc}

    if result == z3.unsat:
        return VerifierResult(verified=True, stats=stats, engine="smt")
    if result == z3.unknown:
        stats["reason"] = s.reason_unknown()
        return VerifierResult(verified=None, stats=stats, engine="smt")

    m = s.model()
    viol = np.array(
        [float(m[x[i]].as_fraction()) for i in range(env.dim)], dtype=float
    )
    return VerifierResult(
        verified=False, violation=viol[None, :], stats=stats, engine="smt"
    )


class _SMTEngine:
    name = "smt"
    find_counterexample = staticmethod(find_counterexample)


register("smt", _SMTEngine())
