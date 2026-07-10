"""
Appendix figure: a trained certificate and its Monte-Carlo drift surface
(verification / benchmarking appendix).

Paired experiment: any campaign whose runs store a certificate (milp_ndcerts,
cegis_verify_*, ...). Data read: <run_dir>/objects/certificate.npz + manifest.

Two panels for one 2D env + certificate:
  left   V(x) as filled contours — the shape the trainer produced;
  right  the MC-estimated drift E_w[V(x')] - V(x) (+eps optionally), a
         diverging map centred at 0 with the zero contour drawn — the
         supermartingale boundary the verifiers reason about. Red pockets are
         exactly what refuters must find and verifiers must bound.

    python -m visualisation.drift_heatmaps                    # auto-pick a cert
    python -m visualisation.drift_heatmaps --run <run_dir> --eps
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from visualisation.common import (PUB_RC, find_cert_run, load_cert, save,
                                  base_parser)
from src.plotting.systems import _grid, _expected_drift_V, _gaussian_smooth
from src.plotting.utils import add_domain_border, add_equilibrium

CERT_CAMPAIGNS = ["milp_ndcerts", "cegis_verify_mab", "pendulum_certs"]


def pick_run(env: str | None) -> Path:
    for c in CERT_CAMPAIGNS:
        d = find_cert_run(c, env=env, verdict="verified") or find_cert_run(c, env=env, verdict=None)
        if d is not None:
            return d
    raise SystemExit(f"no certificate run found in {CERT_CAMPAIGNS} — pass --run")


def make_figure(env, V, env_name: str, eps: float, res: int, n_noise: int,
                smooth: float, seed: int):
    import jax
    import jax.numpy as jnp
    import jax.random as jrn

    bounds = [(float(lo), float(hi)) for lo, hi in np.array(env.domain.bounds)]
    X, Y, pts = _grid(bounds, res)

    Zv = np.array(jax.vmap(lambda p: V(jnp.asarray(p)))(jnp.asarray(pts))).reshape(X.shape)
    Zd = np.array(_expected_drift_V(env, V, jnp.asarray(pts), n_noise,
                                    jrn.key(seed))).reshape(X.shape) + eps
    Zd = _gaussian_smooth(Zd, smooth)

    with plt.rc_context(PUB_RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.1), constrained_layout=True)

        cf = ax1.contourf(X, Y, Zv, levels=18, cmap="viridis")
        ax1.contour(X, Y, Zv, levels=8, colors="white", linewidths=0.4, alpha=0.6)
        fig.colorbar(cf, ax=ax1, shrink=0.85, label=r"$V(x)$")
        ax1.set_title("certificate $V$")

        vmax = float(np.nanmax(np.abs(Zd))) or 1e-6
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cf = ax2.contourf(X, Y, Zd, levels=24, cmap="RdBu_r", norm=norm)
        ax2.contour(X, Y, Zd, levels=[0.0], colors="black", linewidths=1.2)
        lbl = r"$\mathbb{E}[V(x')]-V(x)$" + (r"$+\varepsilon$" if eps else "")
        fig.colorbar(cf, ax=ax2, shrink=0.85, label=lbl)
        ax2.set_title("MC drift of $V$  (red $=$ violation candidates)")

        for ax in (ax1, ax2):
            add_domain_border(env, ax)
            add_equilibrium(env, ax, zorder=10)
            ax.set_xlim(bounds[0])
            ax.set_ylim(bounds[1])
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")
        fig.suptitle(f"{env_name} — trained certificate and its drift surface", fontsize=13)
    return fig


def main():
    p = base_parser("Certificate V + MC drift heatmaps for a stored run")
    p.add_argument("--run", default=None, help="run dir holding the certificate")
    p.add_argument("--env", default=None, help="env filter for the auto-picked run")
    p.add_argument("--eps", action="store_true",
                   help="add the run's epsilon to the drift (show the demanded margin)")
    p.add_argument("--res", type=int, default=130)
    p.add_argument("--n_noise", type=int, default=128)
    p.add_argument("--smooth", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    run_dir = Path(a.run) if a.run else pick_run(a.env)
    env, spec, params, args = load_cert(run_dir)
    if env.dim != 2:
        raise SystemExit(f"drift heatmaps support 2D envs (got dim={env.dim})")
    eps = float(args.get("epsilon", 0.0)) if a.eps else 0.0
    env_name = args.get("env", "env")

    V = lambda x: spec.forward(params, x)
    fig = make_figure(env, V, env_name, eps, a.res, a.n_noise, a.smooth, a.seed)
    save(fig, "drift_heatmaps", f"drift_{env_name}")


if __name__ == "__main__":
    main()
