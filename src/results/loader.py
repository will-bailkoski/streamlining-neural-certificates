"""
Loading runs back from `results/`.

Every run is self-describing: `run.json` records the config (env name, certificate
structure, engine, epsilon, ...), and each `round_<k>/params.npz` holds that
round's trained certificate. Loading therefore needs NO pickle — the env is
rebuilt from its registry name and the certificate from its spec name, then the
params are attached.

    from src.results.loader import list_runs, load_run

    for r in list_runs():
        print(r["run_id"], r["status"], r["verified"])

    run = load_run("linear2D_smt_2026-06-14_16-20-16")
    env  = run.env()             # reconstructed Env
    V, p = run.certificate()     # callable V(x) + raw params (final round)
"""

from __future__ import annotations
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.results.recorder import load_params


@dataclass
class Run:
    run_id: str
    run_dir: Path
    config: dict
    manifest: dict

    # -- manifest convenience --------------------------------------------
    @property
    def status(self) -> str:
        return self.manifest.get("status", "unknown")

    @property
    def verified(self):
        return self.manifest.get("verified")

    @property
    def rounds(self) -> list:
        return self.manifest.get("rounds", [])

    # -- reconstruction --------------------------------------------------
    def env(self):
        from src.benchmarks.utils import make_env

        return make_env(self.config["env"])

    def spec(self):
        from src.certificates.structures import get_spec

        return get_spec(self.config["cert_structure"])

    def available_rounds(self) -> list[int]:
        """Round indices that have a saved params.npz, ascending."""
        ks = []
        for d in self.run_dir.glob("round_*"):
            if (d / "params.npz").exists():
                ks.append(int(d.name.split("_")[1]))
        return sorted(ks)

    def params(self, round: int = -1):
        """Load params from a round (-1 = last saved)."""
        ks = self.available_rounds()
        if not ks:
            raise FileNotFoundError(f"run '{self.run_id}' has no saved params")
        k = ks[round] if round < 0 else round
        return load_params(self.run_dir / f"round_{k}" / "params.npz")

    def certificate(self, round: int = -1):
        """Return (V_callable, params) for a round; V(x) = spec.forward(params, x)."""
        spec = self.spec()
        params = self.params(round)
        return (lambda x: spec.forward(params, x)), params

    def counterexamples(self, round: int):
        import numpy as np

        path = self.run_dir / f"round_{round}" / "counterexamples.npy"
        return np.load(path) if path.exists() else None

    def metrics(self) -> list[dict]:
        path = self.run_dir / "metrics.csv"
        if not path.exists():
            return []
        with open(path, newline="") as f:
            return list(csv.DictReader(f))


def _results_root(results_dir) -> Path:
    return Path(results_dir)


def load_run(run_id: str, results_dir="results") -> Run:
    run_dir = _results_root(results_dir) / run_id
    manifest_path = run_dir / "run.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no run.json under {run_dir}")
    manifest = json.loads(manifest_path.read_text())
    return Run(
        run_id=run_id, run_dir=run_dir, config=manifest.get("config", {}), manifest=manifest
    )


def list_runs(results_dir="results") -> list[dict]:
    """Summarise every run under `results_dir` (those with a run.json)."""
    root = _results_root(results_dir)
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir()):
        manifest_path = d / "run.json"
        if not manifest_path.exists():
            continue
        try:
            m = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            continue
        cfg = m.get("config", {})
        out.append(
            {
                "run_id": d.name,
                "status": m.get("status"),
                "verified": m.get("verified"),
                "env": cfg.get("env"),
                "engine": cfg.get("engine"),
                "cert": cfg.get("cert_structure"),
                "rounds": len(m.get("rounds", [])),
            }
        )
    return out
