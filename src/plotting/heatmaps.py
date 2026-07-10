import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
from src.benchmarks.env import Env
from src.plotting.utils import add_equilibrium, add_domain_border


def plot_heatmap(
    func,
    bounds,
    resolution: int = 100,
    cmap: str = "viridis",
    ax: matplotlib.axes.Axes = None,
    show: bool = False,
) -> None:
    """
    Plot a 2D heatmap of a scalar function directly into *ax*.

    Parameters
    ----------
    func : callable
        f(point) where ``point`` is shape (2,) and the return value is a scalar.
    bounds : list[tuple[float, float]]
        ``[(xmin, xmax), (ymin, ymax)]``
    resolution : int
        Number of grid points along each axis.
    cmap : str
        Matplotlib colormap name.
    ax : matplotlib.axes.Axes
        **Required.**  The axes to draw into.  Always obtain this from
        ``create_figure_layout`` (see ``utils.py``) so the axes already belong
        to the figure you intend to display.

    Returns
    -------
    None
        All output is written into *ax*.  The caller owns the figure.

    Raises
    ------
    ValueError
        If *ax* is None.
    """
    if ax is None:
        raise ValueError(
            "plot_heatmap requires an explicit `ax` argument.\n"
            "Create one with utils.create_figure_layout() and pass it here."
        )

    (xmin, xmax), (ymin, ymax) = bounds

    x = np.linspace(xmin, xmax, resolution)
    y = np.linspace(ymin, ymax, resolution)
    X, Y = np.meshgrid(x, y)

    points = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z = np.array([func(p) for p in points], dtype=np.float32).reshape(X.shape)

    im = ax.imshow(
        Z,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        cmap=cmap,
        aspect="auto",
    )
    ax.get_figure().colorbar(im, ax=ax)

    if show:
        plt.show()


import jax.random as jrn


def plot_env(
    env: Env,
    n_trajectories: int = 100,
    steps_per_trajectory: int = 20,
    ax: matplotlib.axes.Axes = None,
    show: bool = False,
) -> None:
    """
    Plot trajectories of the environment in 2D phase space.

    Parameters
    ----------
    env : Env
        The environment to plot.
    n_trajectories : int
        Number of random trajectories to sample and plot.
    steps_per_trajectory : int
        Number of simulation steps per trajectory.
    ax : matplotlib.axes.Axes
        **Required.** The axes to draw into. Obtain from ``create_figure_layout``.
    show : bool
        Whether to display the plot immediately.

    Raises
    ------
    ValueError
        If *ax* is None or environment dimension is not 2.
    """
    if ax is None:
        raise ValueError(
            "plot_env requires an explicit `ax` argument.\n"
            "Create one with utils.create_figure_layout() and pass it here."
        )

    if env.dim != 2:
        raise ValueError(
            "Cannot plot trajectories for domain with more than 2 dimensions."
        )

    # Sample and plot trajectories
    key = jrn.key(0)
    (xmin, xmax), (ymin, ymax) = env.domain.bounds
    cmap = plt.cm.viridis

    for _ in range(n_trajectories):
        key, subkey = jrn.split(key)

        # Random initial condition
        x0 = np.array(
            [
                np.random.uniform(xmin, xmax),
                np.random.uniform(ymin, ymax),
            ]
        )

        # Simulate trajectory
        trajectory = [x0]
        x = x0
        for _ in range(steps_per_trajectory):
            key, subkey = jrn.split(key)
            x = env.step(x, subkey)[0]
            trajectory.append(x)

        trajectory = np.array(trajectory)

        # Plot trajectory segments with color gradient to show progression
        for i in range(len(trajectory) - 1):
            color = cmap(i / len(trajectory))
            ax.plot(
                trajectory[i : i + 2, 0],
                trajectory[i : i + 2, 1],
                color=color,
                linewidth=1.2,
                alpha=0.7,
            )

        # Add arrow at final position to show direction
        if len(trajectory) > 1:
            dx = trajectory[-1, 0] - trajectory[-2, 0]
            dy = trajectory[-1, 1] - trajectory[-2, 1]
            ax.arrow(
                trajectory[-2, 0],
                trajectory[-2, 1],
                dx,
                dy,
                head_width=0.02 * (xmax - xmin),
                head_length=0.02 * (xmax - xmin),
                fc=cmap(0.9),
                ec=cmap(0.9),
                alpha=0.8,
                zorder=5,
            )

    add_domain_border(env, ax)
    add_equilibrium(env, ax, zorder=10)

    if show:
        plt.show()
