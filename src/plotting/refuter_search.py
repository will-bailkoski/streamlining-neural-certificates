"""
Visualise a refuter's search pattern on a 2D test function.

`draw_search(ax, name, fn, ...)` draws the function as a heatmap and overlays the
refuter's evaluation history, faithfully to its mechanism (the history kind comes
from running the refuter with `record=True`):

  * points  (random / grid / adalip) : every evaluated point, coloured by iteration.
  * agents  (gradient / whale)        : each chain / whale's trajectory.
  * boxes   (direct)                  : the final dividing-rectangles partition.

`plot_grid(...)` tiles refuters (rows) x functions (cols) into one publication
figure. The incumbent best is marked with a star, the true optimum with a +.
"""

from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrn

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

from src.refuters.base import get_refuter
import src.refuters  # noqa: F401  (register refuters)
from src.plotting.systems import PUB_RC

_BG_CMAP = "plasma"  # "Greys"        # function landscape (muted, so overlays pop)
_TIME_CMAP = "plasma"  # evaluated points coloured by iteration
_AGENT_CMAP = "tab20"  # per-agent trajectories


def _function_heatmap(ax, fn, res=160):
    (xlo, xhi), (ylo, yhi) = [(float(a), float(b)) for a, b in np.array(fn.domain)]
    xs = np.linspace(xlo, xhi, res)
    ys = np.linspace(ylo, yhi, res)
    X, Y = np.meshgrid(xs, ys)
    pts = jnp.asarray(np.stack([X.ravel(), Y.ravel()], axis=1))
    Z = np.array(fn.f(pts)).reshape(X.shape)
    ax.contourf(X, Y, Z, levels=24, cmap=_BG_CMAP, alpha=0.85)
    # dark contour lines stay visible even under dense point overlays
    ax.contour(X, Y, Z, levels=10, colors="0.25", linewidths=0.4, alpha=0.7)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    return (xlo, xhi, ylo, yhi)


def _overlay_points(ax, positions):
    T, B, _ = positions.shape
    P = np.asarray(positions).reshape(T * B, 2)
    iters = np.repeat(np.arange(T), B)
    ax.scatter(
        P[:, 0],
        P[:, 1],
        c=iters,
        cmap=_TIME_CMAP,
        s=4,
        alpha=0.4,
        linewidths=0,
        rasterized=True,
    )


def _overlay_agents(ax, positions, n_show=24):
    T, B, _ = positions.shape
    P = np.asarray(positions)
    cmap = plt.get_cmap(_AGENT_CMAP)
    idx = np.linspace(0, B - 1, min(n_show, B)).astype(int)
    for j, b in enumerate(idx):
        ax.plot(P[:, b, 0], P[:, b, 1], "-", color=cmap(j % 20), lw=0.8, alpha=0.8)
        ax.plot(P[0, b, 0], P[0, b, 1], "o", color=cmap(j % 20), ms=2.5)  # start


def _overlay_boxes(ax, hist):
    centers = np.asarray(hist["centers"])
    sizes = np.asarray(hist["sizes"])
    active = np.asarray(hist["active"])
    fvals = np.asarray(hist["f"])
    sel = np.where(active)[0]
    rects = [
        Rectangle(
            (centers[i, 0] - sizes[i, 0], centers[i, 1] - sizes[i, 1]),
            2 * sizes[i, 0],
            2 * sizes[i, 1],
        )
        for i in sel
    ]
    ax.add_collection(
        PatchCollection(
            rects, facecolor="none", edgecolor="#1f4e79", linewidths=0.3, alpha=0.7
        )
    )
    ax.scatter(
        centers[sel, 0],
        centers[sel, 1],
        c=fvals[sel],
        cmap=_TIME_CMAP,
        s=5,
        alpha=0.8,
        linewidths=0,
    )


def draw_search(ax, name, fn, *, budget=8192, batch_size=64, seed=0, hp=None):
    bounds = _function_heatmap(ax, fn)
    # hp (e.g. a refuter that tunes batch_size) OVERRIDES the defaults; merge so a
    # kwarg is never passed twice.
    call = dict(budget=budget, batch_size=batch_size, record=True)
    call.update(dict(hp or {}))
    out = get_refuter(name)(
        lambda xs, k: fn.f(xs),
        fn.domain,
        jrn.key(seed),
        **call,
    )
    best_x, _, _, _, hist = out
    if hist["kind"] == "points":
        _overlay_points(ax, hist["positions"])
    elif hist["kind"] == "agents":
        _overlay_agents(ax, hist["positions"])
    elif hist["kind"] == "boxes":
        _overlay_boxes(ax, hist)
    xo = np.asarray(fn.x_opt)
    ax.plot(xo[0], xo[1], "+", color="lime", ms=10, mew=2, zorder=6)  # true optimum
    bx = np.asarray(best_x)
    ax.plot(
        bx[0], bx[1], "*", color="red", ms=11, mec="white", mew=0.6, zorder=7
    )  # incumbent
    ax.set_xticks([])
    ax.set_yticks([])


def plot_grid(
    refuters,
    functions,
    *,
    budget=8192,
    batch_size=64,
    seed=0,
    hp_by_refuter=None,
    out_path=None,
    cell=2.4,
):
    """Grid of search patterns: rows = refuters, cols = functions."""
    hp_by_refuter = hp_by_refuter or {}
    with plt.rc_context(PUB_RC):
        nr, nc = len(refuters), len(functions)
        fig, axes = plt.subplots(
            nr,
            nc,
            figsize=(cell * nc, cell * nr),
            constrained_layout=True,
            squeeze=False,
        )
        for i, name in enumerate(refuters):
            for j, fn in enumerate(functions):
                try:
                    draw_search(
                        axes[i][j],
                        name,
                        fn,
                        budget=budget,
                        batch_size=batch_size,
                        seed=seed,
                        hp=hp_by_refuter.get(name),
                    )
                except Exception as e:  # one bad cell shouldn't kill the figure
                    axes[i][j].text(0.5, 0.5, f"{type(e).__name__}", ha="center",
                                    va="center", transform=axes[i][j].transAxes, fontsize=7)
                    axes[i][j].set_xticks([]); axes[i][j].set_yticks([])
                if i == 0:
                    axes[i][j].set_title(fn.name)
                if j == 0:
                    axes[i][j].set_ylabel(name, rotation=90, labelpad=8, fontsize=11)
        fig.suptitle(
            "Refuter search patterns (+ true optimum,  * incumbent)", fontsize=13
        )
        if out_path is not None:
            fig.savefig(out_path)
    return fig
