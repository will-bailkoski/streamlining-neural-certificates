"""Cartesian expansion of a sweep's args: list-valued keys are the axes."""

from __future__ import annotations
import itertools


def expand_args(args: dict) -> list[dict]:
    """{k: v | [v, ...]} -> list of concrete {k: v} (product over list-valued keys)."""
    names = list(args)
    value_lists = [v if isinstance(v, list) else [v] for v in (args[n] for n in names)]
    return [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]
