"""
Visualise how an autoLiRPA verifier explored a 2D environment.

Trains (or loads) a candidate certificate, runs a bound-propagation engine with
record=True, and renders the branch-and-bound box exploration depth by depth,
each cell coloured by its status:

    green  = verified  (upper bound <= 0 on this box)
    amber  = undecided (will be split)
    red    = counterexample (lower bound > 0)

Outputs to results/figures/verifier_exploration/:
  * <env>_montage   selected depths side by side (static, for the paper)
  * <env>_evolution.gif   the full depth-by-depth time series as an animation

    python -m visualisation.verifier_exploration --env linstoch2D
    python -m visualisation.verifier_exploration --env vanderpol --method ibp --max_depth 6
    python -m visualisation.verifier_exploration --cert results/cegis_verify_crown/<tag>/runs/<id>

Recording is opt-in (the engine's `record` flag), so the verification hot path is
untouched. autoLiRPA runs on CPU here for portability.
"""

from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import jax.random as jrn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation

from src.benchmarks.utils import make_env
from src.certificates.structures import get_spec, glorot_init
from src.training.training_cycles import train_fixed_dataset
from src.verifiers.base import get_engine
import src.verifiers  # noqa: F401  (register engines)
from src.plotting.systems import PUB_RC
from src.plotting.utils import add_domain_border, add_equilibrium
from src.results.recorder import load_params

_STATUS_COLOR = {0: "#ff7f0e", 1: "#2ca02c", 2: "#d62728"}  # undecided / verified / cex
_STATUS_NAME = {0: "undecided", 1: "verified", 2: "counterexample"}


def _train_cert(env, spec, hidden, seed, epsilon, sample_count):
    init = glorot_init([env.dim] + hidden + [1], jrn.fold_in(jrn.key(seed), 99))
    params, _, _, _ = train_fixed_dataset(
        env, epsilon * 10.0, init, N=sample_count,
        dataset_key=jrn.fold_in(jrn.key(seed), 7),
        batch_size=256, lr=1e-2, n=16, decay=0.9, max_epochs=200,
        threshold=0.0, certificate=spec.forward, printing=False,
    )
    return params


def _draw_frame(ax, env, frame, bounds, depth, n_depths):
    ax.clear()
    los, his, status = frame["los"], frame["his"], frame["status"]
    rects = [Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1])
             for lo, hi in zip(los, his)]
    colors = [_STATUS_COLOR[int(s)] for s in status]
    ax.add_collection(PatchCollection(rects, facecolor=colors, edgecolor="white",
                                      linewidths=0.15, alpha=0.75))
    add_domain_border(env, ax)
    add_equilibrium(env, ax, zorder=10)
    counts = {k: int(np.sum(status == k)) for k in (1, 0, 2)}
    ax.set_title(f"depth {depth + 1}/{n_depths}   "
                 f"({counts[1]} verified, {counts[0]} undecided, {counts[2]} cex)")
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal", adjustable="box")


def draw_montage(env, name, frames, bounds, out: Path, n_show=6):
    idx = sorted(set(np.linspace(0, len(frames) - 1, min(n_show, len(frames))).astype(int)))
    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(1, len(idx), figsize=(3.0 * len(idx), 3.2),
                                 constrained_layout=True, squeeze=False)
        for ax, i in zip(axes[0], idx):
            _draw_frame(ax, env, frames[i], bounds, i, len(frames))
        fig.suptitle(f"{env.name} — autoLiRPA exploration", fontsize=13)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{name}_montage.pdf")
        fig.savefig(out / f"{name}_montage.png")
        plt.close(fig)
        print(f"  -> {name}_montage.pdf / .png")


def draw_gif(env, name, frames, bounds, out: Path, fps=2):
    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(4.0, 4.0), constrained_layout=True)

        def update(i):
            _draw_frame(ax, env, frames[i], bounds, i, len(frames))
            return []

        anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{name}_evolution.gif"
        anim.save(path, writer=animation.PillowWriter(fps=fps))
        plt.close(fig)
        print(f"  -> {name}_evolution.gif  ({len(frames)} frames)")


def main():
    p = argparse.ArgumentParser(description="autoLiRPA exploration visualisation")
    p.add_argument("--env", default="linstoch2D")
    p.add_argument("--cert", default=None, help="load a trained cert from a cegis run dir")
    p.add_argument("--cert_structure", default="relu_pwl")
    p.add_argument("--hidden", default="8,8")
    p.add_argument("--method", default="crown", help="ibp | crown | alpha-crown")
    p.add_argument("--split", default="all", help="all | longest")
    p.add_argument("--noise_disc", type=int, default=1)
    p.add_argument("--max_depth", type=int, default=7)
    p.add_argument("--epsilon", type=float, default=1e-3)
    p.add_argument("--sample_count", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--out", default="results/figures/verifier_exploration")
    a = p.parse_args()

    if a.cert:
        run_dir = Path(a.cert)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        env_name = manifest["args"].get("env", a.env)
        spec = get_spec(manifest["args"].get("cert_structure", a.cert_structure))
        env = make_env(env_name)
        params = load_params(run_dir / "objects" / "certificate.npz")
    else:
        env_name = a.env
        env = make_env(env_name)
        spec = get_spec(a.cert_structure)
        hidden = [int(x) for x in a.hidden.split(",") if x]
        print(f"training a {a.hidden} cert on {env_name} ...")
        params = _train_cert(env, spec, hidden, a.seed, a.epsilon, a.sample_count)

    if env.dim != 2:
        raise SystemExit(f"verifier_exploration only supports 2D envs (got dim={env.dim})")

    r = get_engine(a.method).find_counterexample(
        env, spec, params, a.epsilon, key=jrn.fold_in(jrn.key(a.seed), 3),
        split=a.split, noise_disc=a.noise_disc, max_depth=a.max_depth,
        device="cpu", record=True,
    )
    frames = r.stats.get("frames", [])
    verdict = {True: "verified", False: "counterexample", None: "inconclusive"}[r.verified]
    print(f"{env_name}: {a.method} -> {verdict}  ({len(frames)} depths recorded)")
    if not frames:
        raise SystemExit("no frames recorded (engine returned before bounding any box)")

    bounds = [(float(lo), float(hi)) for lo, hi in np.array(env.domain.bounds)]
    out = Path(a.out)
    draw_montage(env, env_name, frames, bounds, out)
    draw_gif(env, env_name, frames, bounds, out, fps=a.fps)
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
