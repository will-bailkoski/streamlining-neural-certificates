"""Verification engines behind one interface (see base.VerifierResult / DIRECTORY)."""

from src.verifiers.base import VerifierResult, Verifier, DIRECTORY, get_engine, register
from src.verifiers import drift

# Importing an engine module registers it in DIRECTORY.
from src.verifiers import smt  # noqa: F401  (registers "smt")
from src.verifiers import milp  # noqa: F401  (registers "milp")
from src.verifiers import lirpa  # noqa: F401  (registers "lirpa")
from src.verifiers import montecarlo  # noqa: F401  (registers "montecarlo")
from src.verifiers import mab  # noqa: F401  (registers "mab")

__all__ = [
    "VerifierResult",
    "Verifier",
    "DIRECTORY",
    "get_engine",
    "register",
    "drift",
]
