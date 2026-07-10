"""
Cross-check campaign: every engine re-verifies the certificates the OTHER
engines produced (one launch-array task per cert x check-engine cell; the unit
task is experiments/runners/cross_check.py).

Sources are the cegis campaigns' finished runs: every run whose stats verdict is
in --verdicts (default: the "verified" ones — the soundness-critical claims) and
that saved objects/certificate.npz. For each such cert, one combo per check
engine (skipping the engine that produced it — that is just the source run).

Engine split, mirroring the campaign constraints:
  * CLUSTER (GPU array; smt/mc ride along on the GPU node):
        python -m experiments.cross_check --tag bigtest
    (default --engines mab,smt,ibp,crown,alpha-crown,mc — milp excluded: the
    Gurobi token server is login-node-only, experiments/common/slurm.py)
  * LOCAL laptop (WLS license) for the milp column:
        python -m experiments.cross_check --engines milp --local --tag bigtest
  * collate: python -m experiments.cross_check --aggregate --tag bigtest

mab is only emitted for bounded_pwl certs (its empirical-Bernstein bound needs
V in [0,1]; on an unbounded relu_pwl cert the check would be unsound noise).
"""

from __future__ import annotations
import json
from pathlib import Path

from experiments.common.launch import launch
from experiments.common.slurm import SLURM_GPU_LONG

CAMPAIGN_TITLE = "cross_check"

DEFAULT_SOURCES = ",".join(
    f"results/cegis_verify_{e}"
    for e in ("mab", "milp", "smt", "ibp", "crown", "alpha_crown", "mc")
)
DEFAULT_ENGINES = "mab,smt,ibp,crown,alpha-crown,mc"  # milp runs --local (license)

# One-shot check budgets: smaller than the campaign CEGIS budgets (a cell is a
# single verify call, and the matrix has many cells), still big enough that an
# inconclusive cell is a meaningful "looser" datapoint. timeout keeps every cell
# comfortably inside the 12h SLURM_GPU_LONG wall.
ENGINE_HP = {
    "mab": dict(grid_per_dim=32, significance=0.05, range_bound=1.0,
                k=1024, sample_per_select=128,
                max_boxes=10_000_000, max_samples=1_000_000_000, timeout=10_800),
    "milp": dict(time_limit=600.0, noise_disc=8),
    "smt": dict(timeout_ms=600_000, noise_disc=4),
    "ibp": dict(noise_disc=4, split="longest", max_depth=40, max_boxes=2_000_000,
                timeout=10_800),
    "crown": dict(noise_disc=4, split="longest", max_depth=40, max_boxes=2_000_000,
                  timeout=10_800),
    "alpha-crown": dict(noise_disc=4, split="longest", max_depth=40,
                        max_boxes=2_000_000, timeout=10_800),
    "mc": dict(n_states=400_000, significance=0.05, range_bound=1.0),
}

# campaign engine flag (manifest args) -> check-engine name, for self-check skip
_ALIASES = {"montecarlo": "mc"}


def _latest_tag(root: Path) -> Path | None:
    """Newest tag dir holding runs/ (flat campaigns return the title dir) — the
    same resolution visualisation.common.latest_tag uses."""
    if (root / "runs").is_dir():
        return root
    if not root.is_dir():
        return None
    tags = [d for d in root.iterdir() if d.is_dir() and (d / "runs").is_dir()]
    return max(tags, key=lambda d: d.stat().st_mtime) if tags else None


def build_args(a) -> list[dict]:
    engines = [e.strip() for e in a.engines.split(",") if e.strip()]
    verdicts = {v.strip() for v in a.verdicts.split(",") if v.strip()}
    combos: list[dict] = []
    n_certs = 0
    for src in a.source.split(","):
        tdir = (Path(src) / a.source_tag) if a.source_tag else _latest_tag(Path(src))
        if tdir is None or not (tdir / "runs").is_dir():
            print(f"[cross_check] no runs under {src}"
                  + (f"/{a.source_tag}" if a.source_tag else "") + " — skipped")
            continue
        for d in sorted((tdir / "runs").iterdir()):
            mf, sp = d / "manifest.json", d / "stats.json"
            if not (mf.exists() and sp.exists()
                    and (d / "objects" / "certificate.npz").exists()):
                continue
            src_args = json.loads(mf.read_text()).get("args", {})
            if json.loads(sp.read_text()).get("verdict") not in verdicts:
                continue
            n_certs += 1
            src_engine = _ALIASES.get(src_args.get("engine"), src_args.get("engine"))
            bounded = src_args.get("cert_structure") == "bounded_pwl"
            for eng in engines:
                if eng == src_engine:
                    continue          # self-check is just the source run again
                if eng == "mab" and not bounded:
                    continue          # mab is unsound on unbounded certs
                combos.append(dict(cert_run=d.as_posix(), engine=eng,
                                   seed=a.check_seed, **ENGINE_HP[eng]))
    print(f"[cross_check] {n_certs} source cert(s) -> {len(combos)} check cell(s)")
    return combos


if __name__ == "__main__":
    launch(
        campaign_title=CAMPAIGN_TITLE,
        runner="cross_check",
        slurm=SLURM_GPU_LONG,
        build_args=build_args,
        extra_cli=[
            ("--source", dict(default=DEFAULT_SOURCES,
                              help="comma list of source campaign dirs")),
            ("--source_tag", dict(default=None,
                                  help="tag under each source (default: newest with runs/)")),
            ("--engines", dict(default=DEFAULT_ENGINES,
                               help="check engines (run milp separately with --local)")),
            ("--verdicts", dict(default="verified",
                                help="source verdicts to re-check, comma list "
                                     "(e.g. verified,inconclusive)")),
            ("--check_seed", dict(type=int, default=0,
                                  help="seed for the check engines' keys")),
        ],
    )
