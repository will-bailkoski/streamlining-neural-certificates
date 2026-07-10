"""
The single verifier interface.

Every engine (sampling mesh, SMT, MILP, autoLiRPA) is a callable with the same
signature and returns the same `VerifierResult`, so an experiment can swap
engines without touching anything else. Each engine asks exactly one question
about the SAME drift condition:

    Is there x in (domain \\ equilibrium) with  E_w[V(f(x,w))] - V(x) + epsilon > 0 ?

Verdicts:
    verified=True   -> proven: no such x exists (certificate holds)
    verified=False  -> a counterexample x was found (`violation`)
    verified=None   -> inconclusive (budget/timeout/unknown)

Sound engines (SMT, MILP, autoLiRPA) over-approximate, so a reported
counterexample may be spurious for continuous noise; callers re-check it with
the canonical jax drift (`src.verifiers.drift`).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Protocol, Any
import numpy as np


@dataclass
class VerifierResult:
    verified: Optional[bool]
    violation: Optional[np.ndarray] = None
    stats: dict = field(default_factory=dict)
    # The region a counterexample was localised to, as a (d, 2) [lo, hi] box, when
    # the engine knows one (box engines: mab, lirpa). Used by the CEGIS fairness
    # control to draw multiple counterexamples from inside the offending region;
    # point engines leave this None (callers fall back to a ball around violation).
    region: Optional[np.ndarray] = None
    engine: str = ""

    @property
    def found_counterexample(self) -> bool:
        return self.violation is not None


class Verifier(Protocol):
    name: str

    def find_counterexample(
        self, env, spec, params, epsilon: float, key: Any = None, **hp
    ) -> VerifierResult:
        ...


# Engines register themselves here (name -> callable returning VerifierResult).
DIRECTORY: dict[str, "Verifier"] = {}


def register(name: str, engine: "Verifier") -> None:
    DIRECTORY[name] = engine


def get_engine(name: str) -> "Verifier":
    try:
        return DIRECTORY[name]
    except KeyError:
        raise ValueError(f"No such engine '{name}'. Available: {sorted(DIRECTORY)}")
