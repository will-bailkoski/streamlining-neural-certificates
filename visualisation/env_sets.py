"""
Publication figures of the 2D benchmark environments.

Views, written as vector PDF + PNG to results/figures/env_sets/:

  * <env>_phase        just the phase portrait — a faint expected-drift field with the
                       stochastic sample trajectories in the foreground (--doc copies
                       this one into doc/Figures/)
  * <env>_sets         the convex sets only (domain + equilibrium) — a clean schematic
  * <env>_portrait     the 3-panel view: phase portrait + V + drift panels where the
                       env has a ground-truth known_V
  * <env>_cert         a candidate certificate V overlaid as a heatmap, optionally
                       with known counterexamples marked (needs --cert)

    # the sets + portrait views for one env (or 'all' for every 2D env)
    python -m visualisation.env_sets --env linstoch2D
    # overlay a TRAINED candidate cert from a cegis run, with its counterexamples
    python -m visualisation.env_sets --cert results/cegis_verify_mc/<tag>/runs/<run_id> --cex

Reuses the shared renderers in src/plotting (visualise_env, the set/equilibrium
annotators) so styling matches the rest of the project.
"""

from __future__ import annotations
import argparse
import json
import shutil
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.benchmarks.utils import make_env, DIRECTORY
from src.plotting.systems import PUB_RC, visualise_env, _phase_portrait, _rollouts, _grid
from src.plotting.utils import add_domain_border, add_equilibrium
from src.certificates.structures import get_spec
from src.results.params import load_params

DOC_FIGURES = Path(__file__).resolve().parents[1] / "doc" / "Figures"

_LABELS = {
    "pendulum_lqr": (r"$\theta$", r"$\dot\theta$"),
}


def _labels(name):
    return _LABELS.get(name, (r"$x_1$", r"$x_2$"))


def _finish(ax, env, bounds, name):
    add_domain_border(env, ax)
    add_equilibrium(env, ax, zorder=10)
    xl, yl = _labels(name)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_aspect("equal", adjustable="box")


def draw_sets(env, name, out_dir, seed=0):
    """Clean schematic: domain + equilibrium only."""
    bounds = [(float(lo), float(hi)) for lo, hi in np.array(env.domain.bounds)]
    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(3.6, 3.6), constrained_layout=True)
        _finish(ax, env, bounds, name)
        ax.set_title(f"{env.name} — sets")
        _save(fig, out_dir, f"{name}_sets")


def draw_portrait(env, name, out_dir, seed=0):
    """3-panel view (phase portrait + V + drift panels if known_V) via visualise_env."""
    xl, yl = _labels(name)
    fig = visualise_env(env, xlabel=xl, ylabel=yl, seed=seed)
    _save(fig, out_dir, f"{name}_portrait")


def draw_phase(env, name, out_dir, seed=0, doc=False):
    """Just the phase portrait (single panel): the faint expected-drift field with
    the stochastic sample trajectories in the foreground."""
    bounds = [(float(lo), float(hi)) for lo, hi in np.array(env.domain.bounds)]
    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(4.4, 4.4), constrained_layout=True)
        _phase_portrait(env, ax, bounds, n_noise=64, stream_res=26,
                        n_traj=10, traj_steps=30, key=jrn.fold_in(jrn.key(seed), 1))
        xl, yl = _labels(name)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(env.name)
        _save(fig, out_dir, f"{name}_phase", doc=doc)


def draw_cert(env, name, V, out_dir, cex=None, res=140, seed=0):
    """Candidate certificate V as a filled-contour heatmap, with the sets and any
    known counterexamples (red x) overlaid."""
    bounds = [(float(lo), float(hi)) for lo, hi in np.array(env.domain.bounds)]
    X, Y, pts = _grid(bounds, res)
    Z = np.array(jax.vmap(lambda p: V(jnp.asarray(p)))(jnp.asarray(pts))).reshape(X.shape)
    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(4.0, 3.6), constrained_layout=True)
        cf = ax.contourf(X, Y, Z, levels=20, cmap="viridis")
        ax.contour(X, Y, Z, levels=8, colors="white", linewidths=0.4, alpha=0.6)
        fig.colorbar(cf, ax=ax, shrink=0.85, label=r"$V(x)$")
        if cex is not None and len(cex):
            c = np.atleast_2d(np.asarray(cex))
            ax.scatter(c[:, 0], c[:, 1], marker="x", c="red", s=60, linewidths=1.8,
                       zorder=12, label="counterexample")
            ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        _finish(ax, env, bounds, name)
        ax.set_title(f"{env.name} — candidate $V$")
        _save(fig, out_dir, f"{name}_cert")


def _save(fig, out_dir, stem, doc=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    png = out_dir / f"{stem}.png"
    fig.savefig(png)
    plt.close(fig)
    print(f"  -> {stem}.pdf / .png")
    if doc:
        DOC_FIGURES.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(png, DOC_FIGURES / f"{stem}.png")
        print(f"  -> doc/Figures/{stem}.png")


def _load_cert(run_dir: Path):
    """(env_name, spec, params, cex_list) from a cegis run directory."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    args = manifest.get("args", {})
    env_name = args.get("env")
    spec = get_spec(args.get("cert_structure", "relu_pwl"))
    params = load_params(run_dir / "objects" / "certificate.npz")
    cex = [m["cex"] for m in manifest.get("outputs", {}).get("invalid", {}).values()
           if "cex" in m]
    return env_name, spec, params, cex


def main():
    p = argparse.ArgumentParser(description="2D environment / certificate figures")
    p.add_argument("--env", default=None, help="env key, or 'all' for every 2D env")
    p.add_argument("--cert", default=None,
                   help="a cegis run dir (…/runs/<run_id>) to overlay its trained cert")
    p.add_argument("--cex", action="store_true", help="overlay the run's counterexamples")
    p.add_argument("--out", default="results/figures/env_sets")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--doc", action="store_true",
                   help="also copy each <env>_portrait.png into doc/Figures/")
    a = p.parse_args()
    out = Path(a.out)

    if a.cert:
        run_dir = Path(a.cert)
        env_name, spec, params, cex = _load_cert(run_dir)
        env = make_env(a.env or env_name)
        V = lambda x: spec.forward(params, x)
        print(f"{env_name} (cert from {run_dir.name}):")
        draw_cert(env, env_name, V, out, cex=(cex if a.cex else None), seed=a.seed)
        return

    names = sorted(DIRECTORY) if a.env == "all" else [a.env]
    for name in names:
        try:
            env = make_env(name)
        except Exception as e:
            print(f"  {name:16s} SKIP ({type(e).__name__}: {e})")
            continue
        if env.dim != 2:
            print(f"  {name:16s} skip (dim={env.dim})")
            continue
        print(f"{name}:")
        draw_phase(env, name, out, seed=a.seed, doc=a.doc)
        draw_sets(env, name, out, seed=a.seed)
        draw_portrait(env, name, out, seed=a.seed)

    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
