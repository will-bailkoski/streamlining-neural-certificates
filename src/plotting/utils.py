"""Drawing helpers shared by the figure scripts: domain borders and equilibrium sets."""

import numpy as np
import matplotlib.axes
import matplotlib.patches as patches


def _add_ellipsoid(ellip, ax, *, edgecolor, linewidth=2, linestyle="-", alpha=1.0, zorder=None):
    """Draw the boundary of a 2D Ellipsoid {(x-c)^T M (x-c) = 1} as an Ellipse patch."""
    c = np.array(ellip.center)
    M = np.array(ellip.M)
    # Semi-axes / orientation from M^{-1} = V diag(w) V^T; axis lengths = 2*sqrt(w).
    w, V = np.linalg.eigh(np.linalg.inv(M))
    angle = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
    kw = dict(edgecolor=edgecolor, facecolor="none", linewidth=linewidth,
              linestyle=linestyle, alpha=alpha)
    if zorder is not None:
        kw["zorder"] = zorder
    ax.add_patch(patches.Ellipse((float(c[0]), float(c[1])),
                                 2 * np.sqrt(w[0]), 2 * np.sqrt(w[1]), angle=angle, **kw))

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def add_domain_border(env, ax: matplotlib.axes.Axes, linewidth=2, linestyle="-"):
    """
    Draw the environment's domain boundary on *ax*.

    Supports ``HyperRectangle`` (rectangle) and ``Hypersphere`` (circle).
    2-D domains only.

    Parameters
    ----------
    env : Env
        Environment with a ``domain`` attribute (a ``ConvexSet``).
    ax : matplotlib.axes.Axes
        Axes to annotate.  Must already belong to a figure.
    linewidth : float
    linestyle : str

    Returns
    -------
    ax : matplotlib.axes.Axes
        The same axes, for optional chaining.
    """
    from src.benchmarks.convex_sets import HyperRectangle, Hypersphere, Ellipsoid

    domain = env.domain

    if isinstance(domain, Ellipsoid):
        _add_ellipsoid(domain, ax, edgecolor="black", linewidth=linewidth, linestyle=linestyle)

    elif isinstance(domain, HyperRectangle):
        center, hw = domain.center, domain.half_widths
        if len(center) >= 2:
            ax.add_patch(
                patches.Rectangle(
                    (float(center[0] - hw[0]), float(center[1] - hw[1])),
                    float(2 * hw[0]),
                    float(2 * hw[1]),
                    linewidth=linewidth,
                    edgecolor="black",
                    facecolor="none",
                    linestyle=linestyle,
                )
            )

    elif isinstance(domain, Hypersphere):
        center, radius = domain.center, domain.radius
        if len(center) >= 2:
            ax.add_patch(
                patches.Circle(
                    (float(center[0]), float(center[1])),
                    float(radius),
                    linewidth=linewidth,
                    edgecolor="black",
                    facecolor="none",
                    linestyle=linestyle,
                )
            )

    return ax


def add_equilibrium(
    env, ax: matplotlib.axes.Axes, linewidth=2.5, linestyle="--", alpha=0.8, zorder=None
):
    """
    Draw the environment's equilibrium region boundary on *ax* in red.

    Supports ``HyperRectangle`` and ``Hypersphere``.  Does nothing if the
    equilibrium is ``None`` or an ``EmptySet``.

    Parameters
    ----------
    env : Env
        Environment with an ``equilibrium`` attribute (``ConvexSet`` or ``None``).
    ax : matplotlib.axes.Axes
        Axes to annotate.  Must already belong to a figure.
    linewidth : float
    linestyle : str
    alpha : float
    zorder : float, optional
        Z-order for layering (higher values on top).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The same axes, for optional chaining.
    """
    from src.benchmarks.convex_sets import HyperRectangle, Hypersphere, EmptySet

    eq = env.equilibrium
    if eq is None or isinstance(eq, EmptySet):
        return ax

    if isinstance(eq, HyperRectangle):
        center, hw = eq.center, eq.half_widths
        if len(center) >= 2:
            patch_kwargs = {
                "linewidth": linewidth,
                "edgecolor": "red",
                "facecolor": "none",
                "linestyle": linestyle,
                "alpha": alpha,
            }
            if zorder is not None:
                patch_kwargs["zorder"] = zorder
            ax.add_patch(
                patches.Rectangle(
                    (float(center[0] - hw[0]), float(center[1] - hw[1])),
                    float(2 * hw[0]),
                    float(2 * hw[1]),
                    **patch_kwargs,
                )
            )

    elif isinstance(eq, Hypersphere):
        center, radius = eq.center, eq.radius
        if len(center) >= 2:
            patch_kwargs = {
                "linewidth": linewidth,
                "edgecolor": "red",
                "facecolor": "none",
                "linestyle": linestyle,
                "alpha": alpha,
            }
            if zorder is not None:
                patch_kwargs["zorder"] = zorder
            ax.add_patch(
                patches.Circle(
                    (float(center[0]), float(center[1])),
                    float(radius),
                    **patch_kwargs,
                )
            )

    return ax
