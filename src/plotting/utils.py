"""
utils.py  –  figure layout helpers
===================================

Design contract
---------------
**One figure, one creation point.**

``create_figure_layout`` is the *only* place a ``plt.Figure`` is created.
Every downstream plotting function (``plot_heatmap``, ``plot_mesh_ucb``,
``plot_mesh_lcb``, ``add_domain_border``, ``add_equilibrium``) accepts the
``ax`` objects that come out of here and draws directly into them.
No figure is ever created inside a plotter, so the "artist already belongs
to another figure" error cannot occur.

Typical usage
-------------
::

    from utils   import create_figure_layout, add_domain_border, add_equilibrium
    from heatmaps import plot_heatmap
    from tilings  import plot_mesh_ucb, plot_mesh_lcb

    # 1. Create the layout – this is the ONLY place plt.Figure appears.
    fig, [[ax_ucb], [ax_lcb, ax_heat]] = create_figure_layout(
        [[None], [None, None]],
        figsize=(14, 10),
    )

    # 2. Pass axes into plotters.
    plot_mesh_ucb(mesh,  ax=ax_ucb)
    plot_mesh_lcb(mesh,  ax=ax_lcb)
    plot_heatmap(f, bounds, ax=ax_heat)
    add_domain_border(env, ax=ax_ucb)
    add_equilibrium(env,   ax=ax_ucb)

    # 3. Show or save through the figure you already have.
    fig.savefig("result.pdf")
    plt.show()

Object ownership summary
------------------------
=========================================  =========================================
Object                                     Who creates it / who owns it
=========================================  =========================================
``plt.Figure``                             ``create_figure_layout`` – **only here**
``matplotlib.axes.Axes``                   ``create_figure_layout`` – returned to caller
artists (images, collections, patches …)   each ``plot_*`` / ``add_*`` function
=========================================  =========================================

``show_figure_layout`` has been **removed**.  It tried to transplant artists
between figures, which matplotlib forbids.  Use ``create_figure_layout``
before plotting instead.
"""

import numpy as np
import matplotlib.pyplot as plt
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


def create_figure_layout(fig_grid_structure, figsize=(10, 8), bg_color="white"):
    """
    Create a figure with axes arranged in the requested layout.

    Parameters
    ----------
    fig_grid_structure : list
        Nested list defining the layout.  Each inner list is one row; each
        element is one subplot (use ``None`` as a placeholder).

        Examples::

            [None, None]               # 1 row, 2 columns
            [[None], [None, None]]     # 1 top  +  2 bottom  (triangle)
            [[None, None], [None]]     # 2 top  +  1 bottom  (inverted triangle)

    figsize : tuple[float, float]
        Figure size in inches.
    bg_color : str
        Figure background colour.

    Returns
    -------
    fig : plt.Figure
        The figure.  Keep a reference to it; you need it to call
        ``fig.savefig()``, ``fig.tight_layout()``, etc.
    axes_grid : list[list[matplotlib.axes.Axes]]
        Nested list of axes mirroring *fig_grid_structure*.
        Unpack with the same shape you passed in::

            fig, [[ax1], [ax2, ax3]] = create_figure_layout([[None], [None, None]])
    """
    # Normalise flat list → one row
    if not isinstance(fig_grid_structure[0], (list, tuple)):
        fig_grid_structure = [fig_grid_structure]

    nrows = len(fig_grid_structure)
    max_cols = max(len(row) for row in fig_grid_structure)

    fig = plt.figure(figsize=figsize, facecolor=bg_color, constrained_layout=True)
    gs = fig.add_gridspec(nrows, max_cols)

    axes_grid = []
    for r, row_spec in enumerate(fig_grid_structure):
        ncols = len(row_spec)
        start_col = (max_cols - ncols) // 2  # centre sparse rows
        axes_row = [fig.add_subplot(gs[r, start_col + c]) for c in range(ncols)]
        axes_grid.append(axes_row)

    # fig.tight_layout()
    return fig, axes_grid


# ---------------------------------------------------------------------------
# Annotation helpers  (draw into an existing ax – no figure creation)
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
