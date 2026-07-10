"""Refuters: black-box maximisers with a common interface (see base.py)."""

from src.refuters.base import DIRECTORY, register, get_refuter

# importing each module registers it
from src.refuters import random  # noqa: F401
from src.refuters import grid  # noqa: F401
from src.refuters import gradient  # noqa: F401
from src.refuters import whale  # noqa: F401
from src.refuters import adalip  # noqa: F401
from src.refuters import direct  # noqa: F401

__all__ = ["DIRECTORY", "register", "get_refuter"]
