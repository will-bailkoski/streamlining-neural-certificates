"""
MILP noise-discretisation sweet spot on linstoch{2,3,4}D  (verification / MILP section).

The MILP verifier over-approximates the stochastic drift's expectation E_w[V(x')]
by binning the noise into nd^d cells, EACH a copy of the V(x') network. The single
knob `noise_disc` therefore trades TIGHTNESS against MODEL SIZE, and verification of
a genuinely-valid certificate has a two-sided failure mode:

  * nd too SMALL  -> over-approximation too loose -> a PHANTOM counterexample: the
                     over-approx drift exceeds 0 at a point whose TRUE drift is < 0.
  * nd too LARGE  -> nd^d network copies -> Gurobi cannot drive the dual bound <= 0
                     within the budget -> TIMEOUT (inconclusive).
Only a narrow WINDOW of nd both certifies and fits the budget, and because the cell
count is nd^d it sits at lower nd (and narrows) as the dimension grows -- the concrete
reason exact MILP verification of a stochastic certificate does not scale past a few D.

The certificate itself is produced by the STANDARD CEGIS-MILP pipeline (this reuses
`experiments.runners.cegis`, which trains + hardens + MILP-verifies + SAVES the cert
and records rounds/time -- a valid end-to-end datapoint in its own right). This script
then loads that frozen, MILP-verified cert and sweeps nd on it -- isolating the
verifier so a low-nd counterexample is provably PHANTOM (the cert is verified at the
hardening nd) and a high-nd inconclusive is provably a TIMEOUT.

    python -m experiments.milp_noise_disc               # tables + results.json
    python -m experiments.milp_noise_disc --plot        # + doc/Figures/milp_noise_disc.png
    python -m experiments.milp_noise_disc --dims 2 --seeds 0,1,2      # linstoch2D robustness
    python -m experiments.milp_noise_disc --dims 2,3,4 --time-limit 3600

Cert generation is resume-friendly: the runner skips an already-completed (env,nd,seed),
so re-running only repeats the (fast) nd sweep.
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrn
from jax import vmap, jit

from src.benchmarks.utils import make_env
from src.certificates.structures import get_spec
from src.results.params import load_params
import src.verifiers  # noqa: F401  (register engines)
from src.verifiers import milp

CAMPAIGN = "milp_noise_disc"
CERT_CAMPAIGN = "milp_ndcerts"  # where the runner saves generated certs
CERT = "bounded_pwl"
EPS = 1e-3
GRB_STATUS = {2: "OPTIMAL", 9: "TIMELIMIT", 11: "INTERRUPT", 3: "INFEASIBLE"}

# Per-dimension knobs. width grows with D; train_eps is the TRAINING margin (> EPS);
# harden_nd is the noise_disc the cert is CEGIS-MILP-verified at (the presumed sweet
# spot); the sweep spans [nd_min, nd_max] and self-stops at the first timeout.
DIM_CFG = {
    2: dict(
        env="linstoch2D", width="4,4", train_eps="1e-2", harden_nd=4, nd_min=2, nd_max=8
    ),
    3: dict(
        env="linstoch3D", width="4,4", train_eps="1e-2", harden_nd=4, nd_min=2, nd_max=8
    ),
    4: dict(
        env="linstoch4D", width="4,4", train_eps="1e-2", harden_nd=4, nd_min=2, nd_max=8
    ),
}
ENVV = dict(os.environ, PYTHONPATH=".", PYTHONUNBUFFERED="1")


# --------------------------------------------------------------------------- #
# certificate: generate with the standard runner (saves it), then load
# --------------------------------------------------------------------------- #
def _find_run(env, nd, seed, results_dir):
    for d in sorted(glob.glob(f"{results_dir}/{CERT_CAMPAIGN}/runs/*")):
        mf = os.path.join(d, "manifest.json")
        if not os.path.exists(mf):
            continue
        a = json.load(open(mf)).get("args", {})
        if (
            a.get("env") == env
            and int(a.get("noise_disc", -1)) == nd
            and int(a.get("seed", -1)) == seed
            and os.path.exists(os.path.join(d, "objects", "certificate.npz"))
        ):
            return d
    return None


def ensure_cert(cfg, seed, results_dir, harden_tl, max_rounds):
    """Generate (or reuse) a CEGIS-MILP cert via the standard runner; return
    (params, runner_stats). runner_stats is the end-to-end datapoint at harden_nd."""
    env, nd = cfg["env"], cfg["harden_nd"]
    d = _find_run(env, nd, seed, results_dir)
    if d is None:
        cmd = [
            sys.executable,
            "-m",
            "experiments.runners.cegis",
            "--env",
            env,
            "--engine",
            "milp",
            "--refuter",
            "none",
            "--seed",
            str(seed),
            "--cert_structure",
            CERT,
            "--hidden_layers",
            cfg["width"],
            "--epsilon",
            "1e-3",
            "--train_epsilon",
            cfg["train_eps"],
            "--max_rounds",
            str(max_rounds),
            "--noise_disc",
            str(nd),
            "--time_limit",
            str(harden_tl),
            "--campaign",
            CERT_CAMPAIGN,
            "--results_dir",
            results_dir,
            "--sample_count",
            "10000",
        ]
        print(
            f"  generating cert: CEGIS-MILP {env} bounded_pwl {cfg['width']} "
            f"nd={nd} seed={seed} (runner) ...",
            flush=True,
        )
        subprocess.run(cmd, env=ENVV, check=False)
        d = _find_run(env, nd, seed, results_dir)
    if d is None:
        raise RuntimeError(f"cert generation failed for {env} nd={nd} seed={seed}")
    params = load_params(os.path.join(d, "objects", "certificate.npz"))
    sp = os.path.join(d, "stats.json")
    stats = json.load(open(sp)) if os.path.exists(sp) else {}
    return params, stats


# --------------------------------------------------------------------------- #
# independent validation + nd sweep
# --------------------------------------------------------------------------- #
def validate(env, spec, params, seed, n_val=20000, m_noise=4096):
    """Independent dense-sampling of the TRUE drift max over the domain\\eq
    (corroborates the runner's sound MILP verification)."""
    V = spec.forward
    dom = np.asarray(env.domain.bounds)
    key = jrn.fold_in(jrn.key(seed), 123)
    key, ks = jrn.split(key)
    xs = jrn.uniform(
        ks,
        (n_val, env.dim),
        minval=jnp.asarray(dom[:, 0]),
        maxval=jnp.asarray(dom[:, 1]),
    )
    xs = xs[np.asarray(vmap(env.is_valid_domain)(xs))]
    step = env.step

    @jit
    def drift_batch(xb, key):
        def one(x, k):
            ks = jrn.split(k, m_noise)
            vn = vmap(lambda kk: V(params, step(x, kk)[0]))(ks).mean()
            return vn - V(params, x) + EPS

        return vmap(one)(xb, jrn.split(key, xb.shape[0]))

    maxd = -np.inf
    for i in range(0, xs.shape[0], 256):
        key, sk = jrn.split(key)
        maxd = max(maxd, float(np.asarray(drift_batch(xs[i : i + 256], sk)).max()))
    return maxd, 0.5 / np.sqrt(m_noise)


def sweep_nd(env, spec, params, nd_min, nd_max, time_limit, cert_valid):
    rows = []
    for nd in range(nd_min, nd_max + 1):
        r = milp.find_counterexample(
            env, spec, params, EPS, key=jrn.key(1), time_limit=time_limit, noise_disc=nd
        )
        st = r.stats.get("status")
        if r.verified is True:
            verdict = "verified"
        elif r.verified is False:
            verdict = "phantom_ce" if cert_valid else "counterexample"
        else:
            verdict = "timeout" if st == 9 else "inconclusive"
        rows.append(
            dict(
                nd=nd,
                cells=int(r.stats.get("n_branches", 0)),
                verdict=verdict,
                time=round(float(r.stats.get("time", 0.0)), 2),
                obj_bound=r.stats.get("obj_bound"),
                status=GRB_STATUS.get(st, st),
            )
        )
        print(
            f"    nd={nd:>2} cells={rows[-1]['cells']:>4}  {verdict:>12}  "
            f"{rows[-1]['time']:>7.1f}s",
            flush=True,
        )
        if verdict == "timeout":
            break
    return rows


def _checkpoint(results, results_dir, do_plot):
    """Persist results (and re-plot) after each dimension, so a slow/aborted higher
    dimension never loses the lower ones."""
    outdir = Path(results_dir) / CAMPAIGN
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    if do_plot and results["runs"]:
        plot(results, Path("doc/Figures/milp_noise_disc.png"))


def run(dims, seeds, time_limit, harden_tl, max_rounds, results_dir, do_plot):
    results = dict(
        meta=dict(cert=CERT, eps=EPS, time_limit=time_limit, dims=dims, seeds=seeds),
        runs=[],
    )
    # resume: reuse any (D, seed) already saved so we never recompute lower dimensions
    done = set()
    outp = Path(results_dir) / CAMPAIGN / "results.json"
    if outp.exists():
        try:
            results["runs"] = json.load(open(outp)).get("runs", [])
            done = {(r["D"], r["seed"]) for r in results["runs"]}
        except Exception:
            pass
    for D in dims:
        cfg = DIM_CFG[D]
        env = make_env(cfg["env"])
        spec = get_spec(CERT)
        for seed in seeds:
            if (D, seed) in done:
                print(f"\n=== D={D} seed={seed}: reuse cached result ===", flush=True)
                continue
            print(f"\n=== D={D} ({cfg['env']}) seed={seed} ===", flush=True)
            params, gstats = ensure_cert(cfg, seed, results_dir, harden_tl, max_rounds)
            vmax, verr = validate(env, spec, params, seed)
            cert_ver = gstats.get("verdict") == "verified"
            # only a SOUND MILP verification licenses the "phantom" label; if the
            # cert-gen did not verify (e.g. the verify timed out at high dimension),
            # its counterexamples may be real, so we do not call them phantom.
            valid = cert_ver
            print(
                f"  cert (runner @nd={cfg['harden_nd']}): verdict={gstats.get('verdict')} "
                f"rounds={gstats.get('rounds')} verifier_time={gstats.get('verifier_time')}s"
                f"  | validation max TRUE drift={vmax:+.5f} (err~{verr:.4f}) -> "
                f"{'VALID' if valid else 'NOT VALID (interpret with care)'}",
                flush=True,
            )
            rows = sweep_nd(
                env, spec, params, cfg["nd_min"], cfg["nd_max"], time_limit, valid
            )
            win = [r["nd"] for r in rows if r["verdict"] == "verified"]
            print(f"  verified window: nd in {win or 'EMPTY'}", flush=True)
            results["runs"].append(
                dict(
                    D=D,
                    env=cfg["env"],
                    seed=seed,
                    harden_nd=cfg["harden_nd"],
                    cert_verdict=gstats.get("verdict"),
                    cert_rounds=gstats.get("rounds"),
                    cert_verifier_time=gstats.get("verifier_time"),
                    cert_valid=bool(valid),
                    cert_valid_max_drift=round(vmax, 6),
                    sweep=rows,
                    window=win,
                )
            )
        _checkpoint(results, results_dir, do_plot)  # persist after each dimension
    return results


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def plot(results, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tl = results["meta"]["time_limit"]
    dims = sorted({r["D"] for r in results["runs"]})
    colors = {2: "#2b6cb0", 3: "#c05621", 4: "#6b46c1"}
    mk = {
        "verified": "o",
        "phantom_ce": "X",
        "counterexample": "X",
        "timeout": "^",
        "inconclusive": "s",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for D in dims:
        runs = [r for r in results["runs"] if r["D"] == D]
        c = colors.get(D, "#444")
        base = runs[0]["sweep"]
        vx = [r["nd"] for r in base if r["verdict"] == "verified"]
        vy = [max(r["time"], 0.05) for r in base if r["verdict"] == "verified"]
        ax.plot(
            vx, vy, "-", color=c, lw=1.6, alpha=0.85, label=f"D={D} ({runs[0]['env']})"
        )
        for r in runs:
            for row in r["sweep"]:
                y = tl if row["verdict"] == "timeout" else max(row["time"], 0.05)
                ax.scatter(
                    row["nd"],
                    y,
                    marker=mk.get(row["verdict"], "o"),
                    s=66,
                    color=c,
                    edgecolor="k",
                    linewidth=0.5,
                    zorder=3,
                )
    ax.axhline(tl, ls="--", color="k", lw=1, alpha=0.6)
    ax.text(
        ax.get_xlim()[1], tl, f"  timeout ({tl:g}s)", va="center", ha="left", fontsize=8
    )
    ax.set_yscale("log")
    ax.set_xlabel("noise bins per dimension  $b$   (cells $= b^{\\,d}$)")
    ax.set_ylabel("MILP solve time  (s, log)")
    ax.set_title(
        "MILP noise-discretisation sweet spot on linstoch$d$D\n"
        "○ verified    ✕ phantom counterexample (too loose)    △ timeout (too big)"
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xticks(sorted({row["nd"] for r in results["runs"] for row in r["sweep"]}))
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_png.with_suffix(".pdf"))
    print(f"\nwrote {out_png}", flush=True)


def main():
    p = argparse.ArgumentParser(description="MILP noise-disc sweet spot vs dimension")
    p.add_argument("--dims", default="2,3,4")
    p.add_argument("--seeds", default="0")
    p.add_argument("--time-limit", type=float, default=3600.0, help="sweep per-call cap")
    p.add_argument(
        "--harden-tl", type=float, default=300.0, help="cert-gen per-call cap"
    )
    p.add_argument("--max-rounds", type=int, default=60, help="cert-gen CEGIS rounds")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()
    dims = [int(x) for x in args.dims.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]

    print(f"jax backend: {jax.default_backend()}")
    run(
        dims,
        seeds,
        args.time_limit,
        args.harden_tl,
        args.max_rounds,
        args.results_dir,
        args.plot,
    )  # checkpoints (json + figure) after each dimension
    print(
        f"\nwrote {Path(args.results_dir) / CAMPAIGN / 'results.json'}"
        + ("  + doc/Figures/milp_noise_disc.png" if args.plot else ""),
        flush=True,
    )


if __name__ == "__main__":
    main()
