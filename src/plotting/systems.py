"""
Publication-ready visualisation of a 2D benchmark environment.

`visualise_env(env)` builds a single figure that adapts to what the env exposes:

  * Phase portrait (always): streamlines of the *expected* one-step drift
    `E_w[f(x,w)] - x` (Monte-Carlo, so it works uniformly for additive, switched
    or internal noise) plus a few stochastic sample trajectories, with the domain
    and equilibrium drawn on top.
  * Certificate `V` (if the env defines `known_V`): filled contours + level sets.
  * Expected drift `E_w[V(x')] - V(x)` (if `known_V`): a diverging heatmap centred
    at 0 with the zero contour highlighted — the supermartingale boundary, which
    should enclose the equilibrium.

Everything is vectorised in jax and drawn through the shared `add_domain_border`
/ `add_equilibrium` helpers so the styling matches the rest of the project.
Figures are saved as vector PDF (for the paper) and PNG (for previews).
"""

from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrn

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.plotting.utils import add_domain_border, add_equilibrium

# Publication styling — serif + Computer-Modern math to match a LaTeX document.
PUB_RC = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

_TRAJ_CMAP = "viridis"
_V_CMAP = "viridis"
_DRIFT_CMAP = "RdBu_r"  # red = positive drift (bad), blue = negative (good)


# ----------------------------------------------------------------------
# Monte-Carlo field estimators (vectorised; work for any noise model)
# ----------------------------------------------------------------------
def _expected_next(env, pts: jnp.ndarray, n: int, key) -> jnp.ndarray:
    """E_w[ f(x, w) ] at each grid point. pts:(P,2) -> (P,2)."""
    keys = jrn.split(key, pts.shape[0])

    def per_point(x, k):
        ks = jrn.split(k, n)
        nxt = jax.vmap(lambda kk: env.step(x, kk)[0])(ks)
        return jnp.mean(nxt, axis=0)

    return jax.vmap(per_point)(pts, keys)


def _expected_drift_V(env, V, pts: jnp.ndarray, n: int, key) -> jnp.ndarray:
    """E_w[ V(f(x,w)) ] - V(x) at each grid point. pts:(P,2) -> (P,)."""
    keys = jrn.split(key, pts.shape[0])

    def per_point(x, k):
        ks = jrn.split(k, n)
        v_next = jax.vmap(lambda kk: V(env.step(x, kk)[0]))(ks)
        return jnp.mean(v_next) - V(x)

    return jax.vmap(per_point)(pts, keys)


def _rollouts(env, starts: jnp.ndarray, steps: int, key) -> np.ndarray:
    """Stochastic trajectories from each start. starts:(S,2) -> (S, steps+1, 2)."""
    keys = jrn.split(key, starts.shape[0])

    def one(x0, k):
        def step(carry, _):
            x, k = carry
            x, k = env.step(x, k)
            return (x, k), x
        (_, _), xs = jax.lax.scan(step, (x0, k), None, length=steps)
        return jnp.concatenate([x0[None], xs], axis=0)

    return np.array(jax.vmap(one)(starts, keys))


# ----------------------------------------------------------------------
# Panels
# ----------------------------------------------------------------------
def _grid(bounds, res):
    (xlo, xhi), (ylo, yhi) = bounds
    xs = np.linspace(xlo, xhi, res)
    ys = np.linspace(ylo, yhi, res)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    return X, Y, pts


def _gaussian_smooth(Z: np.ndarray, sigma: float) -> np.ndarray:
    """Light separable Gaussian blur (numpy only) to de-noise an MC-estimated field."""
    if sigma <= 0:
        return Z
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    conv = lambda a: np.convolve(np.pad(a, (r, r), mode="edge"), k, mode="valid")
    Z = np.apply_along_axis(conv, 0, Z)
    Z = np.apply_along_axis(conv, 1, Z)
    return Z


def _phase_portrait(env, ax, bounds, *, n_noise, stream_res, n_traj, traj_steps, key):
    X, Y, pts = _grid(bounds, stream_res)
    nxt = np.array(_expected_next(env, jnp.array(pts), n_noise, key))
    U = (nxt[:, 0] - pts[:, 0]).reshape(X.shape)
    V = (nxt[:, 1] - pts[:, 1]).reshape(X.shape)
    speed = np.sqrt(U**2 + V**2)
    lw = 0.5 + 1.0 * speed / (speed.max() + 1e-12)
    # faint expected-drift field so the stochastic sample trajectories read as the
    # foreground (same slate hue as before, just lightened and made translucent)
    ax.streamplot(X, Y, U, V, color=(0.42, 0.48, 0.56, 0.42), density=1.1,
                  linewidth=lw, arrowsize=0.7, zorder=1)

    # stochastic sample trajectories from *inside the domain* (respects ellipsoids)
    tkey, skey = jrn.split(key)
    starts, _ = env.domain.sample(skey, n_traj)
    trajs = _rollouts(env, starts, traj_steps, tkey)
    cmap = plt.get_cmap(_TRAJ_CMAP)
    for tr in trajs:
        # a white halo under the whole path makes it pop against the faint field
        ax.plot(tr[:, 0], tr[:, 1], color="white", lw=3.4, alpha=0.9,
                solid_capstyle="round", zorder=4)
        for i in range(len(tr) - 1):
            ax.plot(tr[i:i+2, 0], tr[i:i+2, 1], color=cmap(i / len(tr)), lw=2.0,
                    alpha=0.98, solid_capstyle="round", zorder=5)
        ax.plot(tr[0, 0], tr[0, 1], "o", color=cmap(0.0), ms=4.5, zorder=6,
                markeredgecolor="white", markeredgewidth=0.6)

    add_domain_border(env, ax)
    add_equilibrium(env, ax, zorder=10)
    eqc = np.array(env.equilibrium.center) if hasattr(env.equilibrium, "center") else None
    if eqc is not None and eqc.shape[0] == 2:
        ax.plot(eqc[0], eqc[1], "*", color="red", ms=9, zorder=11)
    ax.set_title("Phase portrait")


def _certificate_panel(env, ax, bounds, *, res):
    X, Y, pts = _grid(bounds, res)
    V = env.known_V
    Z = np.array(jax.vmap(lambda p: V(jnp.array(p)))(jnp.array(pts))).reshape(X.shape)
    cf = ax.contourf(X, Y, Z, levels=18, cmap=_V_CMAP)
    ax.contour(X, Y, Z, levels=8, colors="white", linewidths=0.4, alpha=0.6)
    ax.get_figure().colorbar(cf, ax=ax, shrink=0.85, label=r"$V(x)$")
    add_domain_border(env, ax)
    add_equilibrium(env, ax, zorder=10)
    ax.set_title("Certificate $V$")


def _drift_panel(env, ax, bounds, *, res, n_noise, key, smooth):
    X, Y, pts = _grid(bounds, res)
    V = env.known_V
    Z = np.array(_expected_drift_V(env, V, jnp.array(pts), n_noise, key)).reshape(X.shape)
    Z = _gaussian_smooth(Z, smooth)  # de-noise the MC estimate for a clean figure
    vmax = float(np.nanmax(np.abs(Z)))
    vmax = vmax if vmax > 0 else 1e-6
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cf = ax.contourf(X, Y, Z, levels=24, cmap=_DRIFT_CMAP, norm=norm)
    # the supermartingale boundary E[V']-V = 0
    ax.contour(X, Y, Z, levels=[0.0], colors="black", linewidths=1.2)
    ax.get_figure().colorbar(cf, ax=ax, shrink=0.85, label=r"$\mathbb{E}[V(x')]-V(x)$")
    add_domain_border(env, ax)
    add_equilibrium(env, ax, zorder=10)
    ax.set_title("Expected drift of $V$")


# ----------------------------------------------------------------------
# Top-level entry point
# ----------------------------------------------------------------------
def visualise_env(
    env,
    *,
    xlabel: str = r"$x_1$",
    ylabel: str = r"$x_2$",
    n_noise: int = 64,
    drift_n_noise: int = 160,
    drift_smooth: float = 1.0,
    stream_res: int = 26,
    heat_res: int = 120,
    n_traj: int = 10,
    traj_steps: int = 30,
    seed: int = 0,
    out_path: str | None = None,
    panel_size: float = 4.2,
):
    """Build (and optionally save) a publication-ready figure for a 2D env.

    Returns the matplotlib Figure. Envs with `known_V` get three panels
    (portrait, certificate, drift); others get the phase portrait only.
    """
    if env.dim != 2:
        raise ValueError(f"visualise_env only supports 2D envs (got dim={env.dim}).")

    bounds = [(float(lo), float(hi)) for lo, hi in np.array(env.domain.bounds)]
    has_V = hasattr(env, "known_V")
    key = jrn.key(seed)

    with plt.rc_context(PUB_RC):
        n_panels = 3 if has_V else 1
        fig, axes = plt.subplots(
            1, n_panels, figsize=(panel_size * n_panels, panel_size),
            constrained_layout=True, squeeze=False,
        )
        axes = axes[0]

        _phase_portrait(
            env, axes[0], bounds, n_noise=n_noise, stream_res=stream_res,
            n_traj=n_traj, traj_steps=traj_steps, key=jrn.fold_in(key, 1),
        )
        if has_V:
            _certificate_panel(env, axes[1], bounds, res=heat_res)
            _drift_panel(env, axes[2], bounds, res=heat_res, n_noise=drift_n_noise,
                         key=jrn.fold_in(key, 2), smooth=drift_smooth)

        for ax in axes:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xlim(bounds[0])
            ax.set_ylim(bounds[1])
            ax.set_aspect("equal", adjustable="box")

        if n_panels == 1:
            # avoid a redundant suptitle + subtitle on a single-panel figure
            axes[0].set_title(env.name)
        else:
            fig.suptitle(env.name, fontsize=13)

        if out_path is not None:
            fig.savefig(out_path)
    return fig
