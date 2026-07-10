"""
Backend capability filter — which (backend, env, cert-spec) combos are runnable.

Constraints are otherwise implicit (backends raise NotImplementedError). The
sweep expander calls `supports(...)` to drop unsupported combos cleanly instead
of generating tasks that crash.

Rules:
  * smt / milp : need a piecewise-linear cert (`spec.symbolic_friendly`) AND
                 symbolically-encodable dynamics (`env.symbolic_encodable`).
  * lirpa      : needs `env.lirpa_drift_module` (all current envs implement it).
  * sampling / montecarlo / mab : unrestricted.
  * any refuter: unrestricted (acts on an objective, not the env directly).
"""

from __future__ import annotations

from src.verifiers.base import DIRECTORY as VERIFIERS
from src.refuters.base import DIRECTORY as REFUTERS
from src.verifiers.lirpa import LIRPA_METHODS

SYMBOLIC = {"smt", "milp"}
# autoLiRPA exposes one engine per bound method (crown, ibp, ...); all need only
# env.lirpa_drift_module, so they are unrestricted like the other sampling engines.
UNRESTRICTED_VERIFIERS = {"sampling", "montecarlo", "mab", "lirpa", *LIRPA_METHODS}


def is_verifier(backend: str) -> bool:
    return backend in VERIFIERS


def is_refuter(backend: str) -> bool:
    return backend in REFUTERS


def supports(backend: str, env, spec) -> bool:
    """True if `backend` can run on this env with this certificate spec."""
    if is_refuter(backend):
        return True
    if backend in SYMBOLIC:
        return bool(spec.symbolic_friendly) and bool(env.symbolic_encodable)
    if backend in UNRESTRICTED_VERIFIERS:
        return True
    raise ValueError(f"unknown backend {backend!r} (not a registered verifier or refuter)")


def unsupported_reason(backend: str, env, spec) -> str:
    """Human-readable reason a combo is skipped (for logging)."""
    if backend in SYMBOLIC:
        if not spec.symbolic_friendly:
            return f"{backend}: cert '{spec.name}' is not piecewise-linear"
        if not env.symbolic_encodable:
            return f"{backend}: env '{env.name}' has non-encodable (e.g. transcendental) dynamics"
    return f"{backend}: unsupported"
