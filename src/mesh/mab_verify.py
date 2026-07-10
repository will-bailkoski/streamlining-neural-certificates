"""
MAB verification algorithm using AdaptiveMesh.

The input space is covered by a Delaunay triangulation instead of a KD-tree.
Each iteration:

  1. Pick the simplex with the highest UCB.
  2. Sample one point uniformly from it and evaluate the reward.
  3. Record the joint observation (reward + certificate value).
  4. Return False (unsafe) if LCB > 0, or True (safe) if all UCBs <= 0.
  5. Prune the simplex if UCB <= 0 (proven safe).
  6. Refine (add a vertex at the sample point) if stat_err < disc_err,
     redistributing stored samples via Bowyer-Watson.

The reward Lipschitz constant is derived per-simplex from the certificate:
    lipschitz = ||∇V(simplex)|| * dynamics_lipschitz
so no global reward_lipschitz parameter is needed.
"""

from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, Callable, List, Tuple

import numpy as np
import jax.numpy as jnp

from src.mesh.triangles import AdaptiveMesh, SimplexStats, SimplexKey

# ---------------------------------------------------------------------------
# Simplex geometry helpers
# ---------------------------------------------------------------------------


def _sample_simplex(vertices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Sample a point uniformly from the interior of a simplex.

    Uses the Dirichlet(1, …, 1) distribution to produce uniform barycentric
    coordinates, then converts to Cartesian.
    """
    lam = rng.dirichlet(np.ones(len(vertices)))
    return lam @ vertices


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------


def mab_verify(
    initial_vertices,  # (M, dim) array — seed vertices for the initial mesh
    true_certificate,  # V(x: ndarray) -> float — neural-network certificate
    reward_fn,  # x (d,) -> float
    checker,  # bounds (d, 2) -> bool   True = feasible, prune if False
    dynamics_lipschitz,  # Lipschitz constant of the dynamics (global)
    significance,  # confidence level, e.g. 0.05
    M=1.0,  # sub-Gaussian parameter of reward noise
    rng=None,
    printing=False,
    print_interval=1000,
):
    """
    Returns
    -------
    decision         : True (safe) | False (unsafe)
    counterexample   : np.ndarray (dim+1, dim) simplex vertices | None
    n_iterations     : int
    n_samples        : int
    """

    if rng is None:
        rng = np.random.default_rng()

    mesh = AdaptiveMesh(
        initial_vertices=np.asarray(initial_vertices, dtype=float),
        true_certificate=true_certificate,
        dynamics_lipschitz=dynamics_lipschitz,
        M=M,
        sig=significance,
    )

    # Prune any initial simplex whose bounding box is infeasible
    for key in mesh.simplices:
        verts = mesh.vertices_of(key)
        if not checker(jnp.array([verts])):
            mesh.mark_inactive(key)

    n_iterations = 0
    n_samples = 0

    while True:

        active = mesh.active_simplices

        # -- Termination: every simplex proven safe --
        if not active or all(mesh.stats(k).ucb <= 0 for k in active):
            if printing:
                print(f"[mab_verify] SAFE — {n_iterations} iters, {n_samples} samples")
            return True, None, n_iterations, n_samples

        # -- Select simplex with the highest UCB --
        best = mesh.least_safe()
        best_stats = mesh.stats(best)
        best_verts = mesh.vertices_of(best)

        # -- Sample uniformly from the simplex --
        x = _sample_simplex(best_verts, rng)
        r = float(reward_fn(x, 0))
        # Pre-compute certificate value once to avoid a second network call
        # inside add_sample (add_vertex will make its own call — unavoidable,
        # as the vertex needs its cert value for future gradient computations).
        cert_val = float(mesh.true_certificate(x))
        n_samples += 1

        # -- Refine geometry if discretisation error dominates --
        # stat_err == inf means no samples yet — never split before first sample
        if (
            np.isfinite(best_stats.stat_error)
            and best_stats.stat_error < best_stats.disc_error
        ):
            deleted, added = mesh.add_vertex(x)

            # Prune any new simplices that are infeasible
            for key in added:
                verts_new = mesh.vertices_of(key)
                if not checker(jnp.array(verts_new)):
                    mesh.mark_inactive(key)

        # -- Record the joint observation --
        # cert_val is passed explicitly; no extra network call here
        current_key = mesh.add_sample(x, r, cert_value=cert_val)

        if current_key is None:
            # Point ended up outside the active hull (e.g. a boundary vertex
            # insertion shifted the geometry) — skip stats checks this round
            n_iterations += 1
            continue

        current_stats = mesh.stats(current_key)

        # -- Termination: counterexample (LCB > 0 => reward provably positive) --
        if current_stats.lcb > 0:
            witness = mesh.vertices_of(current_key)
            if printing:
                print(
                    f"[mab_verify] COUNTEREXAMPLE at iter {n_iterations}, "
                    f"samples {n_samples}.\n  Simplex vertices:\n{witness}"
                )
            return False, witness, n_iterations, n_samples

        # -- Prune if now proven safe --
        if current_stats.ucb <= 0:
            mesh.mark_inactive(current_key)

        # -- Periodic progress --
        if printing and n_iterations % print_interval == 0:
            all_stats = list(mesh.all_stats().values())
            finite_ucbs = [s.ucb for s in all_stats if np.isfinite(s.ucb)]
            finite_lcbs = [s.lcb for s in all_stats if np.isfinite(s.lcb)]
            max_ucb_str = f"{max(finite_ucbs):.4f}" if finite_ucbs else "inf"
            max_lcb_str = f"{max(finite_lcbs):.4f}" if finite_lcbs else "inf"
            print(
                f"[mab_verify] iter={n_iterations:6d} | samples={n_samples:8d} | "
                f"live simplices={len(active):5d} | "
                f"max_ucb={max_ucb_str} | max_lcb={max_lcb_str}"
            )

        n_iterations += 1


# ---------------------------------------------------------------------------
# Box verification (standalone utility — unchanged)
# ---------------------------------------------------------------------------


class Label(Enum):
    SAFE = auto()
    UNSAFE = auto()
    UNKNOWN = auto()


@dataclass
class Result:
    safe: List[np.ndarray] = field(default_factory=list)
    unsafe: List[np.ndarray] = field(default_factory=list)
    unknown: List[np.ndarray] = field(default_factory=list)

    def summary(self) -> str:
        return f"Result — safe: {len(self.safe)}, unsafe: {len(self.unsafe)}, unknown: {len(self.unknown)}"


Box = np.ndarray  # shape (d, 2)
import matplotlib.pyplot as plt
from src.plotting.tilings import plot_result_boxes


def verify_boxes(
    boxes: Iterable[Box],
    reward_fn: Callable[[np.ndarray], float],
    lipschitz: float,
    checker: Callable[[np.ndarray], bool],
    *,
    max_depth: int = 20,
    tol: float = 1e-9,
) -> Result:
    def centre(box: Box) -> np.ndarray:
        return 0.5 * (box[:, 0] + box[:, 1])

    def diameter(box: Box) -> float:
        return float(np.linalg.norm(box[:, 1] - box[:, 0]))

    def vertices(box: Box) -> np.ndarray:
        return np.array(list(product(*[[lo, hi] for lo, hi in box])))

    def split(box: Box) -> Tuple[Box, Box]:
        axis = int(np.argmax(box[:, 1] - box[:, 0]))
        mid = 0.5 * (box[axis, 0] + box[axis, 1])
        left, right = box.copy(), box.copy()
        left[axis, 1] = mid
        right[axis, 0] = mid
        return left, right

    result = Result()
    stack: List[Tuple[Box, int]] = [(np.asarray(b, dtype=float), 0) for b in boxes]

    while stack:
        box, depth = stack.pop()

        if not checker(vertices(box)):
            continue

        c = centre(box)
        diam = diameter(box)
        f_c = reward_fn(c)
        error = lipschitz * diam / 2.0
        lower, upper = f_c - error, f_c + error

        if upper <= 0.0:
            result.safe.append(box)
        elif lower > 0.0:
            result.unsafe.append(box)
            print("unsafe at", box)
            return result
        elif depth >= max_depth or diam < tol:
            result.unknown.append(box)
        else:
            for child in split(box):
                stack.append((child, depth + 1))

    print(len(result.safe))
    return result


def verify_single_box(
    box: Box,
    reward_fn: Callable[[np.ndarray], float],
    lipschitz: float,
    checker: Callable[[np.ndarray], bool],
    **kwargs,
) -> Result:
    return verify_boxes([box], reward_fn, lipschitz, checker, **kwargs)
