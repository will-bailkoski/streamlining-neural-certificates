"""
Anytime speed-vs-tightness of the three bound-propagation methods on VERIFIED
pendulum_lqr certificates, with the noise discretisation fixed at 8.

A verified certificate is a valid supermartingale, so when a bound-prop method
"attempts to refute" it (bound the max drift over the domain) the SOUND upper bound
tightens toward <= 0 as it splits more boxes. For IBP, CROWN and alpha-CROWN on the
SAME certificate we record the anytime trace -- (cumulative boxes, wall time, drift
upper bound) at every branch-and-bound depth -- and plot the upper bound against
compute (boxes) and against time. The three lines expose the classic trade-off:
IBP cheap-but-loose, alpha-CROWN tight-but-slow, CROWN in between.

Reproducible: the verified certificates are generated ONCE via CEGIS (crown, nd=8,
one per seed) and SCRAPED on every later run; only the (cheap) anytime traces are
recomputed. Pass --regen to force fresh certs.

    python -m experiments.pendulum_anytime --plot
    python -m experiments.pendulum_anytime --seeds 0,1,2 --max-boxes 500000 --plot

CLUSTER (GPU) — this is GPU work (CROWN/alpha-CROWN propagate on torch), so run it via
SLURM to get a DEDICATED GPU rather than contending on a shared node (which OOMs). Two
phases, chained by a SLURM dependency:

    # phase 1: verified certs, one seed per GPU, in parallel
    python -m experiments.pendulum_certs            # then sbatch the printed submit.sbatch
    # phase 2: the anytime traces + figure, on one GPU, after the certs land
    python -m experiments.pendulum_anytime --emit-sbatch
    sbatch --dependency=afterok:<phase-1 jobid> results/pendulum_anytime/submit.sbatch
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

# JAX shares the GPU with torch/autoLiRPA (CROWN/alpha-CROWN below); stop it
# preallocating ~75% of VRAM on import (starves torch -> OOM, or collides with a
# co-tenant on a shared GPU). setdefault so an explicit export wins; MUST precede any
# import that pulls in jax — the src.* imports below do. The cegis subprocess spawned
# by ensure_cert inherits it via ENVV (a copy of os.environ made after this runs).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

from src.benchmarks.utils import make_env
from src.certificates.structures import get_spec
from src.results.params import load_params
from src.verifiers.base import get_engine
import src.verifiers  # noqa: F401  (register engines)

ENV = "pendulum_lqr"
CERTSPEC = "relu_pwl"          # lirpa-encodable (linear output); bounded_pwl's hard-sigmoid isn't
WIDTH = "8,8"
NOISE_DISC = 8
EPS = 1e-3
TRAIN_EPS = "2e-2"
CERT_CAMPAIGN = "pendulum_certs"
CAMPAIGN = "pendulum_anytime"
METHODS = ["ibp", "crown", "alpha-crown"]      # loosest -> tightest
ENVV = dict(os.environ, PYTHONPATH=".", PYTHONUNBUFFERED="1")


# --------------------------------------------------------------------------- #
# verified certificates: scrape, or generate once via the standard runner
# --------------------------------------------------------------------------- #
def _find_cert(seed, rdir, campaign):
    """A verified pendulum cert for this seed in `campaign`, with the structure it
    was trained as (so this works on whatever certs you saved on the cluster).

    The scan is RECURSIVE over manifests, so it finds certs whether they were written
    flat (results/<campaign>/runs/*, inline generation) or under a launch tag
    (results/<campaign>/<tag>/runs/*, the pendulum_certs SLURM array)."""
    for mf in sorted(glob.glob(f"{rdir}/{campaign}/**/manifest.json", recursive=True)):
        d = os.path.dirname(mf)
        cert = os.path.join(d, "objects", "certificate.npz")
        st = os.path.join(d, "stats.json")
        if not os.path.exists(cert):
            continue
        a = json.load(open(mf)).get("args", {})
        ver = os.path.exists(st) and json.load(open(st)).get("verdict") == "verified"
        if int(a.get("seed", -1)) == seed and a.get("env") == ENV and ver:
            return cert, a.get("cert_structure", CERTSPEC)
    return None, None


def ensure_cert(seed, rdir, campaign, regen=False):
    if not regen:
        p, s = _find_cert(seed, rdir, campaign)
        if p:
            return p, s
    cmd = [sys.executable, "-m", "experiments.runners.cegis", "--env", ENV,
           "--engine", "crown", "--refuter", "none", "--seed", str(seed),
           "--cert_structure", CERTSPEC, "--hidden_layers", WIDTH, "--epsilon", str(EPS),
           "--train_epsilon", TRAIN_EPS, "--max_rounds", "40", "--noise_disc", str(NOISE_DISC),
           "--split", "longest", "--max_depth", "40", "--max_boxes", "200000",
           "--campaign", campaign, "--results_dir", rdir] + (["--overwrite"] if regen else [])
    print(f"  generating verified cert seed={seed} (CEGIS crown, nd={NOISE_DISC}) ...", flush=True)
    subprocess.run(cmd, env=ENVV, check=False)
    p, s = _find_cert(seed, rdir, campaign)
    if not p:
        raise RuntimeError(
            f"no verified pendulum cert for seed {seed} under results/{campaign}/ "
            f"(scrape found none, and inline generation produced none — likely a GPU OOM "
            f"if this ran on a shared node). Generate the certs on the GPU partition first:\n"
            f"    python -m experiments.pendulum_certs   # then sbatch the printed submit.sbatch\n"
            f"then re-run the traces with --cert-campaign {campaign} (the default).")
    return p, s


# --------------------------------------------------------------------------- #
# anytime trace: one bound method, one cert
# --------------------------------------------------------------------------- #
def anytime_trace(env, spec, params, method, max_boxes, max_depth):
    import jax.random as jrn
    r = get_engine(method).find_counterexample(
        env, spec, params, EPS, key=jrn.key(0), noise_disc=NOISE_DISC,
        max_boxes=max_boxes, max_depth=max_depth, trace=True)
    trace = [(int(b), float(t), float(g)) for (b, t, g) in (r.stats or {}).get("trace", [])]
    return trace, r.verified


def run(seeds, max_boxes, max_depth, rdir, regen, campaign):
    env = make_env(ENV)
    results = {"meta": dict(env=ENV, noise_disc=NOISE_DISC, eps=EPS, cert_campaign=campaign,
                            seeds=seeds, max_boxes=max_boxes), "runs": []}
    for seed in seeds:
        cert, structure = ensure_cert(seed, rdir, campaign, regen)
        spec = get_spec(structure)
        params = load_params(cert)
        print(f"\n=== seed {seed}  cert={os.path.relpath(cert, rdir)}  ({structure}) ===", flush=True)
        for method in METHODS:
            trace, ver = anytime_trace(env, spec, params, method, max_boxes, max_depth)
            if trace:
                b, t, g = trace[-1]
                print(f"  {method:<11} verified={str(ver):<5} depths={len(trace):>2}  "
                      f"final: {b:>7} boxes  {t:>6.1f}s  bound {g:+.4f}", flush=True)
            else:
                print(f"  {method:<11} verified={ver}  (no trace -- decided at depth 0)", flush=True)
            results["runs"].append(dict(seed=seed, method=method, verified=bool(ver), trace=trace))
    return results


# --------------------------------------------------------------------------- #
# plot: drift upper bound vs boxes, and vs time -- one colour per method
# --------------------------------------------------------------------------- #
def plot(results, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"ibp": "#c05621", "crown": "#2b6cb0", "alpha-crown": "#6b46c1"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.7))
    seen = set()
    for r in results["runs"]:
        tr = r["trace"]
        if len(tr) < 1:
            continue
        boxes = [max(p[0], 1) for p in tr]
        times = [max(p[1], 1e-3) for p in tr]
        bound = [p[2] for p in tr]
        c = colors.get(r["method"], "#444")
        lbl = r["method"] if r["method"] not in seen else None
        seen.add(r["method"])
        ax1.plot(boxes, bound, "-o", color=c, ms=3.5, lw=1.4, alpha=0.8, label=lbl)
        ax2.plot(times, bound, "-o", color=c, ms=3.5, lw=1.4, alpha=0.8, label=lbl)
    for ax, xl in ((ax1, "boxes bounded  (compute)"), (ax2, "wall time (s)")):
        ax.axhline(0.0, ls="--", color="k", lw=1, alpha=0.6)
        ax.text(ax.get_xlim()[0], 0, " verified (bound $\\leq$ 0)", va="bottom", ha="left", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel("drift upper bound found")
        ax.grid(True, which="both", alpha=0.22)
        ax.legend(title="bound method", fontsize=9)
    fig.suptitle("Refuting verified pendulum_lqr certificates: tightness vs compute and vs time "
                 f"(noise\\_disc={NOISE_DISC})")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_png.with_suffix(".pdf"))
    print(f"\nwrote {out_png}", flush=True)


def emit_sbatch(args):
    """Write a dedicated-GPU submit.sbatch for the trace+plot step (phase 2).

    The certs (phase 1) come from the pendulum_certs launcher; this job just scrapes
    them and computes the cheap bound-prop traces on ONE dedicated GPU (so JAX + torch
    don't fight for VRAM). Chain it after the cert array with
    --dependency=afterok:<array_jobid>."""
    from experiments.common import array
    from experiments.common.slurm import SLURM_GPU

    campaign_dir = Path(args.results_dir) / CAMPAIGN
    (campaign_dir / "logs").mkdir(parents=True, exist_ok=True)
    cmd = ("python -m experiments.pendulum_anytime"
           f" --seeds {args.seeds} --max-boxes {args.max_boxes}"
           f" --max-depth {args.max_depth} --cert-campaign {args.cert_campaign}"
           f" --results-dir {args.results_dir} --plot")
    sb = array.write_job_sbatch(campaign_dir, SLURM_GPU, cmd, job_name="pend_anytime")
    print(f"[pendulum_anytime]  wrote {sb.as_posix()}")
    print(f"  trace command : {cmd}")
    print(f"  submit        : sbatch {sb.as_posix()}")
    print(f"  after certs   : sbatch --dependency=afterok:<pendulum_certs jobid> {sb.as_posix()}")


def main():
    p = argparse.ArgumentParser(description="Anytime speed/tightness of bound-prop on pendulum certs")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--max-boxes", type=int, default=500_000)
    p.add_argument("--max-depth", type=int, default=60)
    p.add_argument("--regen", action="store_true", help="regenerate certs even if present")
    p.add_argument("--cert-campaign", default=CERT_CAMPAIGN,
                   help="results campaign to scrape verified pendulum certs from "
                        "(point this at your saved cluster certs)")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--emit-sbatch", action="store_true",
                   help="write results/pendulum_anytime/submit.sbatch (a dedicated-GPU job "
                        "for the trace+plot step) and exit; run experiments.pendulum_certs first")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s]

    if args.emit_sbatch:
        emit_sbatch(args)
        return

    results = run(seeds, args.max_boxes, args.max_depth, args.results_dir, args.regen,
                  args.cert_campaign)
    outdir = Path(args.results_dir) / CAMPAIGN
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {outdir / 'results.json'}", flush=True)
    if args.plot:
        plot(results, Path("doc/Figures/pendulum_anytime.png"))


if __name__ == "__main__":
    main()
