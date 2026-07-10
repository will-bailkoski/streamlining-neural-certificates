"""
Recorder — owns all run I/O. The runner never touches file formats directly.

Per-run layout under  results/<run_id>/ :
    run.json                 manifest: config + status + per-round summaries
    metrics.csv              one row per CEGIS round (header from RoundSummary)
    round_<k>/params.npz     trained certificate params for round k (columnar)
    round_<k>/counterexamples.npy   counterexample(s) found in round k

No pickle anywhere — params are stored as flat npz columns (layer{i}_W / _b).
"""

from __future__ import annotations
import csv
import dataclasses
import json
from pathlib import Path

import numpy as np

from src.results.types import RoundSummary, RunManifest


def _params_to_npz(params) -> dict:
    out = {}
    for i, (W, b) in enumerate(params):
        out[f"layer{i}_W"] = np.asarray(W)
        out[f"layer{i}_b"] = np.asarray(b)
    return out


def load_params(path: Path):
    """Inverse of the npz layout -> [(W0, b0), ...] as numpy arrays."""
    arrays = np.load(path)
    n = sum(1 for k in arrays.files if k.endswith("_W"))
    return [(arrays[f"layer{i}_W"], arrays[f"layer{i}_b"]) for i in range(n)]


class Recorder:
    def __init__(self, results_root, run_id: str, config: dict):
        self.run_dir = Path(results_root) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = RunManifest(run_id=run_id, config=config)
        self._csv_path = self.run_dir / "metrics.csv"
        self._csv_fields = [f.name for f in dataclasses.fields(RoundSummary)]
        self._write_manifest()
        with open(self._csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self._csv_fields).writeheader()

    def round_dir(self, k: int) -> Path:
        d = self.run_dir / f"round_{k}"
        d.mkdir(exist_ok=True)
        return d

    def log_round(self, summary: RoundSummary, params=None, counterexamples=None) -> None:
        self.manifest.rounds.append(summary.to_dict())
        with open(self._csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._csv_fields).writerow(summary.to_dict())
        if params is not None:
            np.savez(self.round_dir(summary.round) / "params.npz", **_params_to_npz(params))
        if counterexamples is not None:
            np.save(self.round_dir(summary.round) / "counterexamples.npy", np.asarray(counterexamples))
        self._write_manifest()

    def finish(self, status: str, verified) -> None:
        self.manifest.status = status
        self.manifest.verified = verified
        self._write_manifest()

    def _write_manifest(self) -> None:
        with open(self.run_dir / "run.json", "w") as f:
            json.dump(self.manifest.to_dict(), f, indent=2, default=str)
