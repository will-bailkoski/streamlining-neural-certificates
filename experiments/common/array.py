"""
Shared SLURM-array machinery for the experiment launchers.

A launcher describes WHAT to run (a dict of axes where every list is a sweep
axis); this module turns that into the array artefacts and runs / collates them:

    results/<title>/<tag>/
        combos.txt        one runner command per array task (also the local command)
        submit.sbatch     the SLURM array (logs -> logs/%A_%a.{out,err})
        logs/             SLURM stdout/stderr per task
        runs/<run_id>/    each task's result dir (written by src/results/run_dir.Run)
        compiled.csv      collated stats (written by --aggregate)

The combo lines carry hyperparameters as PLAIN flags, so a line is exactly the
command you'd run locally — no opaque param files. Stdlib only, so a launcher is
safe to run on a thin login node.
"""

from __future__ import annotations
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# project root (…/masters_thesis), so the sbatch can `cd` there
ROOT = Path(__file__).resolve().parents[2]


def now_tag() -> str:
    """Filesystem-safe launch timestamp, used as the default run tag."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def to_argv(combo: dict) -> list[str]:
    """{k: v} -> ['--k', 'v', ...]; bool True -> bare flag, False -> omitted."""
    argv: list[str] = []
    for k, v in combo.items():
        if isinstance(v, bool):
            if v:
                argv.append(f"--{k}")
        else:
            argv += [f"--{k}", str(v)]
    return argv


def write_combos(campaign_dir: Path, lines: list[str]) -> Path:
    combos = campaign_dir / "combos.txt"
    combos.write_text("\n".join(lines) + "\n")
    return combos


def _resource_lines(slurm: dict) -> list[str]:
    """The `#SBATCH` resource directives shared by the array and the single-job sbatch."""
    lines = [
        f"#SBATCH --mem={slurm.get('mem', '8G')}",
        f"#SBATCH --time={slurm.get('time', '02:00:00')}",
        f"#SBATCH --cpus-per-task={slurm.get('cpus', 1)}",
    ]
    for key in ("partition", "qos", "account", "gres"):
        if slurm.get(key):
            lines.append(f"#SBATCH --{key}={slurm[key]}")
    return lines


def _preamble(slurm: dict) -> list[str]:
    """cd into the repo, run the cluster setup (module loads / venv activation / env),
    then unbuffer stdout.

    Unbuffered: a timed-out job's .out otherwise loses everything still sitting in
    Python's block buffer (stdout is not a tty on compute nodes).
    """
    return ["", f'cd "{ROOT.as_posix()}"', *(slurm.get("setup") or []),
            "", "export PYTHONUNBUFFERED=1"]


def write_sbatch(campaign_dir: Path, n: int, slurm: dict, runner: str) -> Path:
    """Write submit.sbatch: a 0..n-1 array running experiments.runners.<runner> per line."""
    combos = (campaign_dir / "combos.txt").as_posix()
    logs = (campaign_dir / "logs").as_posix()
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={slurm.get('job_name', runner)}",
        f"#SBATCH --array=0-{n - 1}%{slurm.get('throttle', 16)}",
        f"#SBATCH --output={logs}/%A_%a.out",
        f"#SBATCH --error={logs}/%A_%a.err",
        *_resource_lines(slurm),
        *_preamble(slurm),
        f'LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "{combos}")',
        f"python -m experiments.runners.{runner} $LINE",
        "",
    ]
    sbatch = campaign_dir / "submit.sbatch"
    sbatch.write_text("\n".join(lines))
    return sbatch


def write_job_sbatch(campaign_dir: Path, slurm: dict, command: str,
                     job_name: str | None = None) -> Path:
    """Write submit.sbatch for a SINGLE (non-array) task that runs `command`.

    For the monolithic experiments — one process that sweeps internally (e.g.
    pendulum_anytime's trace step) — that still want a dedicated GPU/CPU allocation
    and the same repo-root + venv preamble as the array launchers. Logs -> logs/%j.
    """
    logs = (campaign_dir / "logs").as_posix()
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name or slurm.get('job_name', 'job')}",
        f"#SBATCH --output={logs}/%j.out",
        f"#SBATCH --error={logs}/%j.err",
        *_resource_lines(slurm),
        *_preamble(slurm),
        command,
        "",
    ]
    sbatch = campaign_dir / "submit.sbatch"
    sbatch.write_text("\n".join(lines))
    return sbatch


def find_invalid_snapshots(source) -> list[str]:
    """Every invalid-cert snapshot under a cegis_verify campaign dir (or parent).

    Scans for runs/<run_id>/objects/invalid/round_<k>.npz — the certs that a
    verifier refuted during experiment 1, which experiment 3 replays. Returns
    posix paths (no spaces), safe to use as plain `--snapshot` combo flags.
    """
    return sorted(p.as_posix() for p in Path(source).glob("**/objects/invalid/*.npz"))


def run_local(lines: list[str], runner: str, n: int | None = None) -> None:
    """Run combos locally in series (optionally just the first n) — for smoke tests."""
    todo = lines if (n is None or n < 0) else lines[:n]
    print(f"running {len(todo)}/{len(lines)} combos locally via experiments.runners.{runner}\n")
    for i, line in enumerate(todo, 1):
        print(f"  [{i}/{len(todo)}] {runner} {line}")
        subprocess.run(
            [sys.executable, "-m", f"experiments.runners.{runner}", *line.split()],
            check=False,
        )


# ----------------------------------------------------------------------
# Collation: every run's stats joined with its args -> compiled.csv
# ----------------------------------------------------------------------
# Readable leading columns; everything else follows in first-seen order.
_LEAD = [
    "experiment", "run_id", "status",
    "env", "dim", "function", "engine", "refuter", "strategy",
    "cert_structure", "hidden_layers", "epsilon", "seed", "history_len",
    "verdict", "rounds", "final_value",
    "n_counterexamples", "n_refuter_ces", "n_verifier_ces", "n_verifier_calls",
    "total_time", "verifier_time", "refuter_time", "time",
    "method", "split", "n_params", "device",
    "total_boxes", "peak_boxes", "n_bound_calls", "boxes_per_sec", "depth",
    "lip_f", "lip_v_spectral", "lip_d_spectral", "lip_v_lipbab", "lip_d_lipbab",
]


def _run_rows(campaign_dir: Path, iters: bool = False) -> list[dict]:
    """One row per run (or per iterations.csv entry if iters), args joined with stats."""
    rows: list[dict] = []
    for man in sorted((campaign_dir / "runs").glob("*/manifest.json")):
        try:
            m = json.loads(man.read_text())
        except Exception:
            continue
        base = {"experiment": m.get("experiment"), "run_id": m.get("run_id"),
                "status": m.get("status"), **m.get("args", {})}
        stats = man.parent / "stats.json"
        if stats.exists():
            try:
                base.update(json.loads(stats.read_text()))
            except Exception:
                pass
        if iters:
            ic = man.parent / "iterations.csv"
            if not ic.exists():
                continue
            with open(ic, newline="") as f:
                for ir in csv.DictReader(f):
                    rows.append({**base, **ir})
        else:
            rows.append(base)
    return rows


def compile_campaign(campaign_dir: Path, iters: bool = False,
                     lead: list[str] | None = None) -> Path:
    """Collate a campaign's runs into <campaign_dir>/compiled.csv and print a summary.

    Idempotent: overwrites compiled.csv, so re-run as results stream in. `lead`
    overrides the leading column order (campaign-aware callers pass their own).
    """
    campaign_dir = Path(campaign_dir)
    rows = _run_rows(campaign_dir, iters=iters)
    out = campaign_dir / "compiled.csv"
    if not rows:
        print(f"no runs found under {campaign_dir}")
        return out

    lead = lead or _LEAD
    seen = list(dict.fromkeys(k for r in rows for k in r))
    cols = [c for c in lead if c in seen] + [c for c in seen if c not in lead]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"compiled {len(rows)} runs -> {out}")
    verdicts: dict[str, int] = {}
    for r in rows:
        v = r.get("verdict")
        if v is not None:
            verdicts[v] = verdicts.get(v, 0) + 1
    if verdicts:
        print("  verdicts: " + "  ".join(f"{k}={v}" for k, v in sorted(verdicts.items())))
    return out
