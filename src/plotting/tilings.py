import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.colors as mcolors
import numpy as np
from typing import Optional

from src.mesh.triangles import TrackingAdaptiveMesh


def _draw_mesh(
    mesh: TrackingAdaptiveMesh,
    value_attr: str,  # "ucb" or "lcb"
    label: str,  # colorbar label
    ax: matplotlib.axes.Axes,
    cmap: str,
    alpha: float,
    show_vertices: bool,
    show_edges: bool,
    vmin: Optional[float],
    vmax: Optional[float],
) -> None:
    """
    Internal helper – draws a mesh coloured by *value_attr* into *ax*.
    Not part of the public API; call ``plot_mesh_ucb`` or ``plot_mesh_lcb``.
    """
    if mesh.dim != 2:
        raise ValueError(f"Mesh plots only support 2-D meshes (got dim={mesh.dim}).")

    active_keys = mesh.active_simplices
    if not active_keys:
        raise ValueError("Mesh has no active simplices to plot.")

    # ── Geometry ──────────────────────────────────────────────────────────
    coord_to_idx: dict = {}
    vertices: list = []

    def get_idx(coord: tuple) -> int:
        if coord not in coord_to_idx:
            coord_to_idx[coord] = len(vertices)
            vertices.append(coord)
        return coord_to_idx[coord]

    triangles: list = []
    values: list = []

    for key in active_keys:
        verts = mesh.vertices_of(key)  # (3, 2)
        triangles.append([get_idx(tuple(v)) for v in verts])
        values.append(getattr(mesh.stats(key), value_attr))

    verts_arr = np.array(vertices, dtype=float)  # (V, 2)
    tri_arr = np.array(triangles, dtype=int)  # (T, 3)
    val_arr = np.array(values, dtype=float)  # (T,)

    # Replace ±inf with finite sentinels
    finite = np.isfinite(val_arr)
    _min = val_arr[finite].min() if finite.any() else -1.0
    _max = val_arr[finite].max() if finite.any() else 1.0
    val_arr = np.where(
        np.isposinf(val_arr), _max, np.where(np.isneginf(val_arr), _min, val_arr)
    )

    _vmin = vmin if vmin is not None else _min
    _vmax = vmax if vmax is not None else _max

    # ── Colormap norm ─────────────────────────────────────────────────────
    if _vmin < 0 < _vmax:
        norm = mcolors.TwoSlopeNorm(
            vmin=_vmin,
            vcenter=min(max(0.0, _vmin), _vmax),
            vmax=_vmax,
        )
    else:
        norm = mcolors.Normalize(vmin=_vmin, vmax=_vmax)

    # ── Drawing ───────────────────────────────────────────────────────────
    triang = mtri.Triangulation(verts_arr[:, 0], verts_arr[:, 1], tri_arr)

    tpc = ax.tripcolor(triang, facecolors=val_arr, cmap=cmap, norm=norm, alpha=alpha)

    if show_edges:
        ax.triplot(triang, color="k", linewidth=0.5, alpha=0.4)

    if show_vertices:
        ax.scatter(
            verts_arr[:, 0], verts_arr[:, 1], s=18, color="k", zorder=5, linewidths=0
        )

    cbar = ax.get_figure().colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, fontsize=11)
    cbar.ax.axhline(y=0, color="k", linewidth=1.0, linestyle="--")

    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$", fontsize=11)
    ax.set_ylabel("$x_2$", fontsize=11)
    ax.set_title(
        f"Mesh {label}  —  {len(active_keys)} active simplices, "
        f"{len(mesh.supersamples)} supersamples",
        fontsize=12,
    )


def plot_mesh_ucb(
    mesh: TrackingAdaptiveMesh,
    ax: matplotlib.axes.Axes,
    cmap: str = "RdYlGn_r",
    alpha: float = 0.85,
    show_vertices: bool = False,
    show_edges: bool = False,
    ucb_min: Optional[float] = 0.0,
    ucb_max: Optional[float] = None,
) -> None:
    """
    Draw a 2-D TrackingAdaptiveMesh coloured by UCB into *ax*.

    Parameters
    ----------
    mesh : TrackingAdaptiveMesh
        The mesh to visualise.
    ax : matplotlib.axes.Axes
        **Required.**  Obtain from ``utils.create_figure_layout``.
    cmap : str
        Matplotlib colormap name.
    alpha : float
        Triangle fill opacity.
    show_vertices : bool
        Scatter vertex points on top.
    show_edges : bool
        Draw triangle edges.
    ucb_min, ucb_max : float, optional
        Manual colour-scale limits; auto-detected if ``None``.

    Returns
    -------
    None
        All output is written into *ax*.

    Raises
    ------
    ValueError
        If *ax* is ``None``.
    """
    if ax is None:
        raise ValueError(
            "plot_mesh_ucb requires an explicit `ax` argument.\n"
            "Create one with utils.create_figure_layout() and pass it here."
        )
    _draw_mesh(
        mesh, "ucb", "UCB", ax, cmap, alpha, show_vertices, show_edges, ucb_min, ucb_max
    )


def plot_mesh_lcb(
    mesh: TrackingAdaptiveMesh,
    ax: matplotlib.axes.Axes,
    cmap: str = "RdYlGn_r",
    alpha: float = 0.85,
    show_vertices: bool = False,
    show_edges: bool = False,
    lcb_min: Optional[float] = None,
    lcb_max: Optional[float] = 0.0,
) -> None:
    """
    Draw a 2-D TrackingAdaptiveMesh coloured by LCB into *ax*.

    Parameters
    ----------
    mesh : TrackingAdaptiveMesh
        The mesh to visualise.
    ax : matplotlib.axes.Axes
        **Required.**  Obtain from ``utils.create_figure_layout``.
    cmap : str
        Matplotlib colormap name.
    alpha : float
        Triangle fill opacity.
    show_vertices : bool
        Scatter vertex points on top.
    show_edges : bool
        Draw triangle edges.
    ucb_min, ucb_max : float, optional
        Manual colour-scale limits for the LCB scale; auto-detected if ``None``.

    Returns
    -------
    None
        All output is written into *ax*.

    Raises
    ------
    ValueError
        If *ax* is ``None``.
    """
    if ax is None:
        raise ValueError(
            "plot_mesh_lcb requires an explicit `ax` argument.\n"
            "Create one with utils.create_figure_layout() and pass it here."
        )
    _draw_mesh(
        mesh, "lcb", "LCB", ax, cmap, alpha, show_vertices, show_edges, lcb_min, lcb_max
    )


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def plot_result_boxes(
    result,
    *,
    dims=(0, 1),
    ax=None,
    show_safe=True,
    show_unsafe=True,
    show_unknown=True,
    colors=None,
    alpha=0.4,
    linewidth=1.0,
):
    """
    Plot 2D projections of boxes stored in a Result object.

    Parameters
    ----------
    result : Result
        Output from verify_boxes().
    dims : tuple[int, int]
        Which dimensions to plot.
    ax : matplotlib axis, optional
        Existing axis to draw on.
    show_safe, show_unsafe, show_unknown : bool
        Toggle categories.
    colors : dict, optional
        Mapping {'safe': ..., 'unsafe': ..., 'unknown': ...}
    alpha : float
        Rectangle transparency.
    linewidth : float
        Rectangle edge width.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    if colors is None:
        colors = {
            "safe": "green",
            "unsafe": "red",
            "unknown": "orange",
        }

    d0, d1 = dims

    def draw_boxes(boxes, color, label):
        first = True
        for box in boxes:
            x0, x1 = box[d0]
            y0, y1 = box[d1]

            rect = Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor=color,
                edgecolor="black",
                linewidth=linewidth,
                alpha=alpha,
                label=label if first else None,
            )
            ax.add_patch(rect)
            first = False

    if show_safe:
        draw_boxes(result.safe, colors["safe"], "safe")

    if show_unsafe:
        draw_boxes(result.unsafe, colors["unsafe"], "unsafe")

    if show_unknown:
        draw_boxes(result.unknown, colors["unknown"], "unknown")

    ax.set_xlabel(f"dim {d0}")
    ax.set_ylabel(f"dim {d1}")
    ax.set_aspect("equal")

    ax.legend()
    ax.autoscale_view()

    return ax
