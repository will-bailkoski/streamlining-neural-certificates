"""
One results directory per experiment run, handling every result shape uniformly.

    results/<campaign>/runs/<run_id>/
        manifest.json   experiment + args + status + index of outputs
        stats.json      a single stat set
        iterations.csv  a per-iteration series
        figures/<name>.{pdf,png}
        objects/<name>.npz   networks / artefacts (params, recorder npz layout)

`run_id` is a deterministic slug of the experiment args, so SLURM array tasks
never collide and a rerun resumes by skipping an already-completed run. Every
experiment script opens a `Run`, writes whichever of stats / iterations /
figures / objects it produces, and calls `finish()`. `aggregate()` collates all
runs' `stats.json` (joined with their args) into one `results.csv`.
"""

from __future__ import annotations
from pathlib import Path
import csv
import hashlib
import json
import re

import numpy as np

from src.results.recorder import _params_to_npz, load_params

# args that locate I/O rather than identify the run — excluded from the slug
_META_KEYS = {"campaign", "results_dir", "overwrite", "plots", "cert_root", "root"}


def make_run_id(args: dict) -> str:
    """Readable + collision-resistant slug from the experiment args (unset
    optional args, i.e. None, are dropped so the slug stays clean)."""
    core = {k: v for k, v in args.items() if k not in _META_KEYS and v is not None}
    slug = "_".join(f"{k}-{core[k]}" for k in sorted(core))
    slug = re.sub(r"[^A-Za-z0-9._-]+", "", slug)[:90]
    digest = hashlib.md5(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:6]
    return f"{slug}__{digest}" if slug else digest


class Run:
    def __init__(self, experiment, args: dict, *, campaign=None, results_dir="results",
                 overwrite=False):
        self.experiment = experiment
        self.args = {k: v for k, v in args.items() if k not in _META_KEYS and v is not None}
        self.campaign = campaign or experiment
        self.run_id = make_run_id(args)
        self.dir = Path(results_dir) / self.campaign / "runs" / self.run_id
        self._iter_rows: list[dict] = []
        self._manifest = dict(experiment=experiment, campaign=self.campaign,
                              run_id=self.run_id, args=self.args, status="running",
                              outputs={})
        # resume: an already-completed run is skipped unless overwrite
        self.done = (not overwrite and (self.dir / "manifest.json").exists()
                     and self._existing_status() == "completed")
        if not self.done:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._write_manifest()

    def _existing_status(self):
        try:
            return json.loads((self.dir / "manifest.json").read_text()).get("status")
        except Exception:
            return None

    def _write_manifest(self):
        (self.dir / "manifest.json").write_text(json.dumps(self._manifest, indent=2, default=str))

    # ------------------------------------------------------------------
    def stats(self, d: dict) -> None:
        (self.dir / "stats.json").write_text(json.dumps(d, indent=2, default=str))
        self._manifest["outputs"]["stats"] = True
        self._write_manifest()

    def log_iter(self, d: dict) -> None:
        self._iter_rows.append(d)

    def iterations(self, rows: list[dict]) -> None:
        self._iter_rows.extend(rows)

    def _flush_iterations(self) -> None:
        if not self._iter_rows:
            return
        fields = list(dict.fromkeys(k for r in self._iter_rows for k in r))
        with open(self.dir / "iterations.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(self._iter_rows)
        self._manifest["outputs"]["iterations"] = len(self._iter_rows)

    def flush(self) -> None:
        """Persist iterations + manifest right now, so partial results survive a
        kill (e.g. a SLURM timeout mid-run)."""
        self._flush_iterations()
        self._write_manifest()

    def figure(self, fig, name: str) -> None:
        fdir = self.dir / "figures"
        fdir.mkdir(exist_ok=True)
        fig.savefig(fdir / f"{name}.pdf")
        fig.savefig(fdir / f"{name}.png")
        self._manifest["outputs"].setdefault("figures", []).append(name)
        self._write_manifest()

    def object(self, name: str, params, meta: dict | None = None) -> None:
        odir = self.dir / "objects"
        odir.mkdir(exist_ok=True)
        np.savez(odir / f"{name}.npz", **_params_to_npz(params))
        self._manifest["outputs"].setdefault("objects", {})[name] = (meta or {})
        self._write_manifest()

    def load_object(self, name: str):
        return load_params(self.dir / "objects" / f"{name}.npz")

    def snapshot(self, name: str, params, meta: dict | None = None) -> None:
        """Save an intermediate cert under objects/invalid/<name>.npz with meta.

        Used by the CEGIS runner to persist every cert a verifier refuted (with
        the counterexample and the verifier's wall-time), so the refute-certs
        experiment can replay those exact refutation tasks. Indexed in the
        manifest under outputs['invalid'][name]."""
        sdir = self.dir / "objects" / "invalid"
        sdir.mkdir(parents=True, exist_ok=True)
        np.savez(sdir / f"{name}.npz", **_params_to_npz(params))
        self._manifest["outputs"].setdefault("invalid", {})[name] = (meta or {})
        self._write_manifest()

    def load_snapshot(self, name: str):
        return load_params(self.dir / "objects" / "invalid" / f"{name}.npz")

    def finish(self, status: str = "completed") -> None:
        self._flush_iterations()
        self._manifest["status"] = status
        self._write_manifest()


def aggregate(campaign: str, results_dir="results") -> Path:
    """Collate every run's stats.json (joined with its args) into results.csv."""
    runs = Path(results_dir) / campaign / "runs"
    rows = []
    for d in sorted(runs.glob("*")):
        man = d / "manifest.json"
        if not man.exists():
            continue
        m = json.loads(man.read_text())
        row = {"run_id": m["run_id"], "experiment": m["experiment"], **m["args"]}
        sp = d / "stats.json"
        if sp.exists():
            row.update(json.loads(sp.read_text()))
        rows.append(row)
    out = Path(results_dir) / campaign / "results.csv"
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(f"aggregated {len(rows)} runs -> {out}")
    return out
