"""
Shared plumbing for every thesis figure script in visualisation/.

Design contract (one figure = one file):
  * every script in this package is tied to ONE experiment campaign and knows
    where that campaign writes its data (it imports the experiment module's
    constants), so `python -m visualisation.<name>` with no arguments finds the
    right data by itself;
  * scripts only READ results (results/<campaign>/... produced by experiments/);
    they never run experiments — except opt-in `record` replays that cache their
    trace back into the run directory they came from;
  * output goes to results/figures/<script>/ as PDF + PNG, and `--doc` copies
    the PNG into doc/Figures/ under the exact name the thesis \\includegraphics
    uses, so "tune → rebuild thesis" is one flag.

Styling: PUB_RC (serif + CM math) everywhere, plus one palette so refuters,
engines and verification statuses keep the same colour in every figure.
"""

from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.plotting.systems import PUB_RC  # noqa: F401  (re-export)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FIG_ROOT = RESULTS / "figures"
DOC_FIGURES = REPO / "doc" / "Figures"

# ----------------------------------------------------------------------
# One palette for the whole thesis
# ----------------------------------------------------------------------
# accents (match the lipschitz_budget figures already in the document)
VIOLET, CORAL, GREEN, SAND = "#5D50C6", "#D85A30", "#1D9E75", "#E3B448"
MUT, INK = "#888780", "#2C2C2A"

# a fixed colour per refuter, in the canonical comparison order
REFUTER_ORDER = ["random", "grid", "gradient", "whale", "adalip", "direct"]
REFUTER_COLOR = {
    "random": "#b0b7bf",
    "grid": "#5c6670",
    "gradient": "#D85A30",
    "whale": "#1D9E75",
    "adalip": "#3557d4",
    "direct": "#E3B448",
}

# a fixed colour per bound-propagation / verification engine
ENGINE_COLOR = {
    "ibp": "#c05621",
    "crown": "#2b6cb0",
    "alpha-crown": "#d734d2",
    "milp": "#2C2C2A",
    "mab": "#1D9E75",
    "smt": "#888780",
    "mc": "#E3B448",
}

# cell verification status (shared by the lirpa and mab grid views)
STATUS_COLOR = {1: "#2a9d8f", 0: "#e9c46a", 2: "#e76f51"}  # verified/unknown/violating
STATUS_NAME = {1: "verified", 0: "inconclusive", 2: "violating"}

FUNCTION_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]  # per BBOB function


# ----------------------------------------------------------------------
# Figure output
# ----------------------------------------------------------------------
def save(fig, script: str, stem: str, doc_name: str | None = None):
    """Save PDF+PNG under results/figures/<script>/; optionally copy the PNG to
    doc/Figures/<doc_name> (the name the thesis includes)."""
    out = FIG_ROOT / script
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf")
    png = out / f"{stem}.png"
    fig.savefig(png)
    plt.close(fig)
    print(f"  -> {png.relative_to(REPO).as_posix()}  (+ .pdf)")
    if doc_name:
        DOC_FIGURES.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(png, DOC_FIGURES / doc_name)
        print(f"  -> doc/Figures/{doc_name}")


def base_parser(
    description: str, doc_help: str | None = None
) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--doc",
        action="store_true",
        help=doc_help
        or "also copy the PNG(s) into doc/Figures/ under the " "thesis figure name(s)",
    )
    return p


# ----------------------------------------------------------------------
# Campaign data discovery
# ----------------------------------------------------------------------
def campaign_dir(title: str) -> Path:
    return RESULTS / title


def latest_tag(title: str) -> Path | None:
    """Newest tag dir under results/<title>/ that actually holds runs (or a
    compiled.csv). Flat campaigns (results/<title>/runs) return the title dir."""
    root = campaign_dir(title)
    if (root / "runs").is_dir():
        return root
    tags = (
        [
            d
            for d in root.iterdir()
            if d.is_dir() and ((d / "runs").is_dir() or (d / "compiled.csv").exists())
        ]
        if root.is_dir()
        else []
    )
    return max(tags, key=lambda d: d.stat().st_mtime) if tags else None


def load_compiled(
    title: str, tag: str | None = None, compile_if_missing: bool = True
) -> list[dict]:
    """Rows of results/<title>/<tag>/compiled.csv (latest tag by default),
    compiling the campaign's runs first if the csv is missing/stale."""
    import csv as _csv

    tdir = (campaign_dir(title) / tag) if tag else latest_tag(title)
    if tdir is None:
        return []
    out = tdir / "compiled.csv"
    if not out.exists() and compile_if_missing and (tdir / "runs").is_dir():
        from experiments.common.array import compile_campaign

        compile_campaign(tdir)
    if not out.exists():
        return []
    with open(out, newline="") as f:
        return list(_csv.DictReader(f))


def glob_campaigns(prefix: str) -> list[str]:
    """Campaign titles under results/ starting with `prefix` (e.g. refute_bbob_*)."""
    if not RESULTS.is_dir():
        return []
    return sorted(
        d.name for d in RESULTS.iterdir() if d.is_dir() and d.name.startswith(prefix)
    )


def run_dirs(title: str, tag: str | None = None) -> list[Path]:
    tdir = (campaign_dir(title) / tag) if tag else latest_tag(title)
    if tdir is None:
        return []
    runs = tdir / "runs"
    return (
        sorted(d for d in runs.iterdir() if (d / "manifest.json").exists())
        if runs.is_dir()
        else []
    )


def load_manifest(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "manifest.json").read_text())


def load_stats(run_dir: Path) -> dict:
    p = Path(run_dir) / "stats.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_cert(run_dir: Path):
    """(env, spec, params, manifest_args) for the certificate stored in a run dir."""
    from src.benchmarks.utils import make_env
    from src.certificates.structures import get_spec
    from src.results.params import load_params

    args = load_manifest(run_dir).get("args", {})
    env = make_env(args["env"])
    spec = get_spec(args.get("cert_structure", "relu_pwl"))
    params = load_params(Path(run_dir) / "objects" / "certificate.npz")
    return env, spec, params, args


def find_cert_run(
    title: str,
    tag: str | None = None,
    *,
    env: str | None = None,
    verdict: str | None = "verified",
    seed: int | None = None,
) -> Path | None:
    """First run dir in a campaign holding a certificate matching the filters —
    the standard way a figure script picks 'a representative trained cert'."""
    for d in run_dirs(title, tag):
        if not (d / "objects" / "certificate.npz").exists():
            continue
        a = load_manifest(d).get("args", {})
        if env is not None and a.get("env") != env:
            continue
        if seed is not None and int(a.get("seed", -1)) != seed:
            continue
        if verdict is not None and load_stats(d).get("verdict") != verdict:
            continue
        return d
    return None


def fnum(row: dict, key: str, default=np.nan) -> float:
    try:
        return float(row.get(key))
    except (TypeError, ValueError):
        return default
