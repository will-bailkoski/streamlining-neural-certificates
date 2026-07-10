"""Run result persistence (Recorder) and result dataclasses."""

from src.results.recorder import Recorder, load_params
from src.results.loader import Run, load_run, list_runs
from src.results.types import RoundSummary, RunManifest

__all__ = [
    "Recorder",
    "load_params",
    "Run",
    "load_run",
    "list_runs",
    "RoundSummary",
    "RunManifest",
]
