"""Result dataclasses shared by the runner and the recorder."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class RoundSummary:
    """One CEGIS round: train V, then query the verifier."""

    round: int
    verdict: str  # "counterexample" | "verified" | "inconclusive"
    n_counterexamples: int
    train_samples: int
    loss: float
    lipschitz: float
    verifier_time: float
    counterexample_genuine: Optional[bool] = None
    counterexample_drift: Optional[float] = None
    engine: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunManifest:
    run_id: str
    config: dict
    status: str = "running"  # running | completed | inconclusive | failed
    verified: Optional[bool] = None
    rounds: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
