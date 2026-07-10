"""
adaptive_mesh.py
================
Adaptive Delaunay mesh for joint Lyapunov certificate approximation and
MAB-style reward exploration.

The mesh approximates a certificate V piecewise-linearly from vertex values.
"Supersamples" estimate the Lyapunov decrease condition:

    reward(x) ≈ mean_over_kernel(V(x')) - V(x) + ε

where x' are next-states drawn from the transition kernel.  This is negative
iff V decreases in expectation at x — the certificate condition.

Key structural decisions
------------------------
- Vertices live only in _vertex_cert, keyed by coordinate tuple.
- SimplexKeys are sorted tuples of integer vertex indices.  Sorting gives a
  canonical form that is stable across all call sites and matches exactly what
  _tri.get_vertices() expects.  frozenset is NOT used — it destroys ordering,
  which causes get_vertices() to return [] for unrecognised permutations and
  breaks barycentric coordinate consistency.
- The inverse index _simplex_to_supersamples makes mesh-update propagation
  O(removed simplices × local fan-out), not O(all supersamples).
- Supersample.reward_estimate is a @property; no explicit "pass 2" needed.
- on_mesh_updated rebuilds only the sub-points whose simplex was destroyed.
  Sub-points in surviving simplices keep their cached values exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple
from itertools import combinations

import numpy as np
import jax.numpy as jnp
import jax.random as jrn
from adaptive_triangulation import Triangulation

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

SimplexKey = tuple  # sorted tuple of integer vertex indices
SupersampleID = str
Point = np.ndarray


# ---------------------------------------------------------------------------
# Subsample — a single draw from the transition kernel
# ---------------------------------------------------------------------------


@dataclass
class Subsample:
    """
    One next-state sample drawn from the transition kernel at some x.

    position           : x' in state space
    containing_simplex : simplex that currently contains x'
    value              : mesh.certificate(x') — cached; refreshed on refinement
    """

    position: Point
    containing_simplex: Optional[SimplexKey]
    value: float


# ---------------------------------------------------------------------------
# Supersample — a reward observation at one query point
# ---------------------------------------------------------------------------


@dataclass
class Supersample:
    """
    Lyapunov decrease estimate at a single point x inside a simplex.

        reward_estimate = mean(sub.value for sub in subsamples) - own_value + ε

    own_value      : V_mesh(x)              — mesh interpolation at x
    true_cert_value: V_true(x)              — neural-network value; for approx_error
    subsamples     : {x'_i} drawn from T(·|x)

    The reward_estimate is a @property so no explicit propagation is needed
    after a mesh update — just refresh own_value and sub.value in place.
    """

    id: SupersampleID
    position: Point
    own_simplex: Optional[SimplexKey]
    own_value: float
    true_cert_value: float
    subsamples: List[Subsample]
    epsilon: float = 0.0

    @property
    def reward_estimate(self) -> float:
        finite = [s.value for s in self.subsamples if np.isfinite(s.value)]
        if not finite:
            return float("inf")
        return float(np.mean(finite)) - self.own_value + self.epsilon


# ---------------------------------------------------------------------------
# SimplexStats — per-simplex certificate + MAB statistics
# ---------------------------------------------------------------------------


@dataclass
class SimplexStats:
    """
    Certificate side
    ----------------
    gradient_norm : ||∇V||  — exact from vertex values
    approx_error  : max |V_true(x) - V_mesh(x)| over supersamples in simplex

    MAB side
    --------
    lipschitz  : gradient_norm × dynamics_lipschitz
    disc_error : diameter × lipschitz
    stat_error : MAB confidence width from sample count
    ucb / lcb  : mean ± (stat_error + disc_error)
    """

    count: int = 0
    mean: float = float("inf")
    stat_error: float = float("inf")
    gradient_norm: float = float("inf")
    lipschitz: float = float("inf")
    disc_error: float = float("inf")
    ucb: float = float("inf")
    lcb: float = float("-inf")
    approx_error: float = float("inf")

    @classmethod
    def from_simplex(
        cls,
        reward_values: np.ndarray,  # (N,)
        cert_true: np.ndarray,  # (N,)  V_true at supersample positions
        cert_interp: np.ndarray,  # (N,)  V_mesh at supersample positions
        gradient_norm: float,
        dynamics_lipschitz: float,
        M: float,
        sig: float,
        diameter: float,
    ) -> SimplexStats:

        valid = np.isfinite(cert_interp) & np.isfinite(cert_true)
        approx_error = (
            float(np.max(np.abs(cert_true[valid] - cert_interp[valid])))
            if valid.any()
            else float("inf")
        )

        lipschitz = (dynamics_lipschitz + 1) * gradient_norm
        disc_error = diameter * lipschitz

        if len(reward_values) == 0:
            return cls(
                gradient_norm=gradient_norm,
                lipschitz=lipschitz,
                disc_error=disc_error,
                approx_error=approx_error,
            )

        count = len(reward_values)
        mean = float(np.mean(reward_values))
        stat_error = float(
            np.sqrt(6.6 * M * np.log(np.log(max(np.e, M * count))) + np.log(4.0 / sig))
            / count
        )

        return cls(
            count=count,
            mean=mean,
            stat_error=stat_error,
            gradient_norm=gradient_norm,
            lipschitz=lipschitz,
            disc_error=disc_error,
            ucb=mean + stat_error + disc_error,
            lcb=mean - stat_error - disc_error,
            approx_error=approx_error,
        )

    def __repr__(self) -> str:
        if self.count == 0:
            return f"SimplexStats(empty, approx_error={self.approx_error:.4g})"
        return (
            f"SimplexStats(n={self.count}, ucb={self.ucb:.4g}, "
            f"lcb={self.lcb:.4g}, approx_error={self.approx_error:.4g})"
        )


# ---------------------------------------------------------------------------
# AdaptiveMesh — geometry, certificate interpolation, base stats
# ---------------------------------------------------------------------------


class AdaptiveMesh:
    """
    Piecewise-linear Lyapunov certificate mesh.

    Manages vertex certificate values, simplex geometry, gradient computation,
    and SimplexStats.  Does not know about supersamples — that is handled by
    TrackingAdaptiveMesh.

    SimplexKeys are sorted tuples of integer vertex indices.  All coordinate
    lookups go through _tri.get_vertices(key) so there is no parallel vertex
    list to drift out of sync, and key ordering is always canonical.

    Parameters
    ----------
    initial_vertices   : (N, dim) seed vertex array
    true_certificate   : V(x) — the neural-network certificate callable
    dynamics_lipschitz : global Lipschitz constant of the dynamics
    M, sig             : MAB confidence-bound hyperparameters
    """

    def __init__(self, initial_vertices, true_certificate, dynamics_lipschitz, M, sig):
        pts = np.asarray(initial_vertices, dtype=float)
        if pts.ndim != 2:
            raise ValueError("initial_vertices must be shape (N, dim)")

        self._dim = pts.shape[1]
        self._tri = Triangulation([tuple(p) for p in pts])
        self.true_certificate = true_certificate
        self.dynamics_lipschitz = dynamics_lipschitz
        self.M = M
        self.sig = sig

        # Certificate values keyed by coordinate tuple.
        # This is the only vertex store — no parallel index list.
        self._vertex_cert: Dict[tuple, float] = {}
        self._stats: Dict[SimplexKey, SimplexStats] = {}
        self._inactive: Set[SimplexKey] = set()

        for pt in pts:
            self._vertex_cert[tuple(pt)] = float(true_certificate(pt))

        for raw in self._tri.simplices:
            self._stats[tuple(sorted(raw))] = SimplexStats()

    # --- Geometry ----------------------------------------------------------

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def simplices(self) -> List[SimplexKey]:
        return [tuple(sorted(s)) for s in self._tri.simplices]

    @property
    def active_simplices(self) -> List[SimplexKey]:
        return [k for k in self.simplices if k not in self._inactive]

    def vertices_of(self, key: SimplexKey) -> np.ndarray:
        """
        Return shape (dim+1, dim) vertex coordinate matrix for a simplex key.

        Delegates entirely to _tri.get_vertices — the library is the sole
        source of truth for the mapping from integer index to coordinate.
        Key must be a sorted tuple of integer indices as produced by _locate
        and the simplices property.
        """
        return np.array(self._tri.get_vertices(key), dtype=float)

    def centroid(self, key: SimplexKey) -> np.ndarray:
        return self.vertices_of(key).mean(axis=0)

    def diameter(self, key: SimplexKey) -> float:
        verts = self.vertices_of(key)
        n = len(verts)
        return max(
            float(np.linalg.norm(verts[i] - verts[j]))
            for i in range(n)
            for j in range(i + 1, n)
        )

    def _locate(self, coords: np.ndarray) -> Optional[SimplexKey]:
        raw = self._tri.locate_point(tuple(np.asarray(coords, dtype=float)))
        if not raw:
            return None
        return tuple(sorted(raw))

    def mark_inactive(self, key: SimplexKey) -> None:
        self._inactive.add(key)

    def mark_active(self, key: SimplexKey) -> None:
        self._inactive.discard(key)

    def is_active(self, key: SimplexKey) -> bool:
        return key not in self._inactive

    # --- Certificate -------------------------------------------------------

    def _cert_values_for(self, key: SimplexKey) -> Optional[List[float]]:
        """
        Return the certificate values at each vertex of key, in the same order
        as vertices_of(key).  Returns None if any vertex is missing from the
        cache (should never happen in normal operation).
        """
        verts = self.vertices_of(key)
        vals = [self._vertex_cert.get(tuple(v)) for v in verts]
        if any(v is None for v in vals):
            return None
        return vals

    def _barycentric_coords(
        self, key: SimplexKey, point: np.ndarray
    ) -> Optional[np.ndarray]:
        verts = self.vertices_of(key)
        T = (verts[1:] - verts[0]).T
        try:
            lam = np.linalg.solve(T, np.asarray(point, dtype=float) - verts[0])
        except np.linalg.LinAlgError:
            return None
        return np.concatenate([[1.0 - lam.sum()], lam])

    def _interpolate_in(self, key: SimplexKey, coords: np.ndarray) -> Optional[float]:
        lam = self._barycentric_coords(key, coords)
        if lam is None:
            return None
        vals = self._cert_values_for(key)
        if vals is None:
            return None
        return float(np.dot(lam, vals))

    def gradient(self, key: SimplexKey) -> Optional[np.ndarray]:
        verts = self.vertices_of(key)
        vals = self._cert_values_for(key)
        if vals is None:
            return None
        values = np.array(vals, dtype=float)
        edges = verts[1:] - verts[0]
        try:
            return np.linalg.solve(edges, values[1:] - values[0])
        except np.linalg.LinAlgError:
            return None

    def certificate(self, coords: np.ndarray) -> Optional[float]:
        """Piecewise-linear V interpolation. Returns None if outside hull."""
        key = self._locate(coords)
        return None if key is None else self._interpolate_in(key, coords)

    def gradient_norm(self, key: SimplexKey) -> float:
        g = self.gradient(key)
        return float("inf") if g is None else float(np.linalg.norm(g))

    # --- Vertex insertion --------------------------------------------------

    def add_vertex(self, point: np.ndarray) -> Tuple[Set[SimplexKey], Set[SimplexKey]]:
        """
        Insert a new vertex, update certificate cache and stats dicts.

        Returns (deleted, added) as sets of sorted-tuple keys matching the
        format used throughout the rest of the class.
        """
        pt = np.asarray(point, dtype=float)
        coord = tuple(pt)

        # Register certificate value before mutating the triangulation so
        # that any immediate get_vertices call on new simplices can resolve it.
        self._vertex_cert[coord] = float(self.true_certificate(pt))

        raw_deleted, raw_added = self._tri.add_point(coord)
        deleted = {tuple(sorted(s)) for s in raw_deleted}
        added = {tuple(sorted(s)) for s in raw_added}

        for key in deleted:
            self._stats.pop(key, None)
            self._inactive.discard(key)
        for key in added:
            self._stats[key] = SimplexStats()

        return deleted, added

    # --- Stats (called by TrackingAdaptiveMesh) ----------------------------

    def _compute_stats(
        self, key: SimplexKey, supersamples_in_simplex: List[Supersample]
    ) -> SimplexStats:
        gnorm = self.gradient_norm(key)
        lipschitz = (gnorm + 1) * self.dynamics_lipschitz
        disc_error = self.diameter(key) * lipschitz

        if not supersamples_in_simplex:
            return SimplexStats(
                gradient_norm=gnorm,
                lipschitz=lipschitz,
                disc_error=disc_error,
            )

        reward_values = np.array(
            [ss.reward_estimate for ss in supersamples_in_simplex], dtype=float
        )
        cert_true = np.array([ss.true_cert_value for ss in supersamples_in_simplex])
        cert_interp = np.array([ss.own_value for ss in supersamples_in_simplex])

        return SimplexStats.from_simplex(
            reward_values=reward_values,
            cert_true=cert_true,
            cert_interp=cert_interp,
            gradient_norm=gnorm,
            dynamics_lipschitz=self.dynamics_lipschitz,
            M=self.M,
            sig=self.sig,
            diameter=self.diameter(key),
        )

    def stats(self, key: SimplexKey) -> SimplexStats:
        return self._stats.get(key, SimplexStats())

    def all_stats(self) -> Dict[SimplexKey, SimplexStats]:
        return {k: v for k, v in self._stats.items() if k not in self._inactive}

    def least_safe(self) -> Optional[SimplexKey]:
        """Highest-UCB active simplex."""
        active = self.active_simplices
        if not active:
            return None
        return max(active, key=lambda k: self._stats.get(k, SimplexStats()).ucb)

    def __repr__(self) -> str:
        return (
            f"AdaptiveMesh(dim={self._dim}, "
            f"simplices={len(self.active_simplices)} active "
            f"/ {len(self._inactive)} inactive)"
        )


# ---------------------------------------------------------------------------
# TrackingAdaptiveMesh — adds supersample dependency tracking
# ---------------------------------------------------------------------------


class TrackingAdaptiveMesh(AdaptiveMesh):
    """
    Extends AdaptiveMesh with a supersample inverse index.

    _simplex_to_supersamples[key] is the set of SupersampleIDs that have
    *any* point (own position or any subsample) inside `key`.  This makes
    mesh-update propagation O(removed simplices × fan-out).

    A supersample counts as a reward observation only for the simplex
    containing its *own* position (own_simplex).  The inverse index also
    tracks subsample simplices so that sub-point values are refreshed when
    their simplex is split — but those supersamples don't pollute the stats
    of the simplex they happen to land in.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.supersamples: Dict[SupersampleID, Supersample] = {}
        self._simplex_to_supersamples: Dict[SimplexKey, Set[SupersampleID]] = {}

    # --- Registration helpers ----------------------------------------------

    def _all_simplex_keys(self, ss: Supersample) -> Set[Optional[SimplexKey]]:
        return {ss.own_simplex} | {sub.containing_simplex for sub in ss.subsamples}

    def _register(self, ss: Supersample) -> None:
        for key in self._all_simplex_keys(ss):
            if key is not None:
                self._simplex_to_supersamples.setdefault(key, set()).add(ss.id)

    def _unregister(self, ss: Supersample) -> None:
        for key in self._all_simplex_keys(ss):
            if key is not None:
                bucket = self._simplex_to_supersamples.get(key)
                if bucket:
                    bucket.discard(ss.id)

    # --- Adding supersamples -----------------------------------------------

    def add_supersample(
        self,
        ss_id: SupersampleID,
        position: Point,
        subsample_positions: List[Point],
        epsilon: float = 0.0,
        true_cert_value: Optional[float] = None,
    ) -> Optional[Supersample]:
        """
        Register a supersample at `position` with draws from the transition kernel.

        true_cert_value : V_true(position).  If omitted, queried automatically.
                          Pass it when you are already evaluating the network at
                          this point to avoid a redundant call.
        """
        own_key = self._locate(position)
        if own_key is None:
            return None
        own_value = self._interpolate_in(own_key, position)
        if own_value is None:
            return None
        if true_cert_value is None:
            true_cert_value = float(self.true_certificate(position))

        subsamples = []
        for pt in subsample_positions:
            pt = np.asarray(pt, dtype=float)
            key = self._locate(pt)
            if key is None:
                continue
            val = self._interpolate_in(key, pt)
            subsamples.append(
                Subsample(
                    position=pt,
                    containing_simplex=key,
                    value=val if val is not None else float("inf"),
                )
            )

        ss = Supersample(
            id=ss_id,
            position=np.asarray(position, dtype=float),
            own_simplex=own_key,
            own_value=own_value,
            true_cert_value=true_cert_value,
            subsamples=subsamples,
            epsilon=epsilon,
        )
        self.supersamples[ss_id] = ss
        self._register(ss)
        return ss

    def add_supersamples_batch(
        self,
        entries: List[Tuple[SupersampleID, Point, List[Point]]],
        epsilon: float = 0.0,
    ) -> List[Supersample]:
        """
        Add many supersamples, then refresh stats once at the end.

        entries : list of (ss_id, position, subsample_positions)
        """
        added = []
        dirty_own: Set[Optional[SimplexKey]] = set()
        for ss_id, pos, sub_pts in entries:
            ss = self.add_supersample(ss_id, pos, sub_pts, epsilon=epsilon)
            if ss is not None:
                added.append(ss)
                dirty_own.add(ss.own_simplex)
        self._refresh_stats(dirty_own)
        return added

    # --- Mesh update propagation -------------------------------------------

    def on_mesh_updated(self, removed: Set[SimplexKey], added: Set[SimplexKey]) -> None:
        """
        Called after every structural mesh change.

        Pass 1 — structural: for each affected supersample, rebuild only the
        points whose containing simplex was destroyed.  Points in surviving
        simplices keep their cached values untouched.

        Pass 2 — stats: recompute SimplexStats for all newly-created simplices.
        (The @property reward_estimate means no explicit value propagation step
        is needed beyond refreshing own_value / sub.value in place.)
        """
        # Collect affected supersamples in one sweep over removed simplices
        affected_ids: Set[SupersampleID] = set()
        for key in removed:
            affected_ids |= self._simplex_to_supersamples.pop(key, set())

        for ss_id in affected_ids:
            ss = self.supersamples[ss_id]
            self._unregister(ss)

            # Own position displaced?
            if ss.own_simplex in removed:
                new_key = self._locate(ss.position)
                ss.own_simplex = new_key
                if new_key is not None:
                    val = self._interpolate_in(new_key, ss.position)
                    ss.own_value = val if val is not None else float("inf")
                else:
                    ss.own_value = float("inf")

            # Individual subsamples displaced?
            for sub in ss.subsamples:
                if sub.containing_simplex in removed:
                    new_key = self._locate(sub.position)
                    sub.containing_simplex = new_key
                    if new_key is not None:
                        val = self._interpolate_in(new_key, sub.position)
                        sub.value = val if val is not None else float("inf")
                    else:
                        sub.value = float("inf")

            self._register(ss)

        # Recompute stats for all new simplices
        self._refresh_stats(added)

    def _refresh_stats(self, keys: Set[Optional[SimplexKey]]) -> None:
        """Recompute SimplexStats for the given simplex keys."""
        for key in keys:
            if key is None:
                continue
            # Only supersamples *owned by* this simplex count as reward observations
            ss_in = [
                self.supersamples[sid]
                for sid in self._simplex_to_supersamples.get(key, set())
                if self.supersamples[sid].own_simplex == key
            ]
            self._stats[key] = self._compute_stats(key, ss_in)

    def refresh_all_stats(self) -> None:
        """Recompute stats for every active simplex.  Use after bulk changes."""
        self._refresh_stats(set(self.active_simplices))  # type: ignore[arg-type]

    # --- Override add_vertex -----------------------------------------------

    def add_vertex(self, point: np.ndarray) -> Tuple[Set[SimplexKey], Set[SimplexKey]]:
        deleted, added = super().add_vertex(point)
        self.on_mesh_updated(deleted, added)
        return deleted, added

    # --- Convenience -------------------------------------------------------

    def supersamples_in(self, key: SimplexKey) -> List[Supersample]:
        """All supersamples whose *own* simplex is `key`."""
        return [
            self.supersamples[sid]
            for sid in self._simplex_to_supersamples.get(key, set())
            if self.supersamples[sid].own_simplex == key
        ]

    def __repr__(self) -> str:
        return (
            f"TrackingAdaptiveMesh(dim={self._dim}, "
            f"simplices={len(self.active_simplices)} active "
            f"/ {len(self._inactive)} inactive, "
            f"supersamples={len(self.supersamples)})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_in_simplex(vertices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform sample from a simplex via Dirichlet weights."""
    weights = rng.dirichlet(np.ones(len(vertices)))
    return weights @ vertices


def _top_k_by_ucb(mesh: TrackingAdaptiveMesh, k: int) -> List[SimplexKey]:
    """Return up to k active simplices with the highest UCB."""
    active = mesh.active_simplices
    if not active:
        return []
    scored = sorted(
        active,
        key=lambda key: mesh.stats(key).ucb,
        reverse=True,
    )
    return scored[:k]


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------


@dataclass
class AlgorithmResult:
    mesh: TrackingAdaptiveMesh
    status: Optional[bool] = None
    simplex: Optional[SimplexKey] = None
    violation: Optional[np.ndarray] = None
    iteration: int = 0


def sample_efficient_method(
    initial_vertices: np.ndarray,
    true_certificate: Callable[[np.ndarray], float],
    transition_kernel: Callable[[jnp.ndarray, jnp.ndarray], np.ndarray],
    checker: Callable[[np.ndarray], np.ndarray],
    refuter: Callable[[Callable], Optional[np.ndarray]],
    dynamics_lipschitz: float,
    M: float,
    sig: float,
    epsilon: float = 0.0,
    K: int = 5,  # simplices to explore per round
    P: int = 10,  # supersamples per simplex per round
    N: int = 20,  # subsamples (kernel draws) per supersample
    max_iterations: int = 200,
    seed: int = 0,
    printing: bool = False,
    print_every: int = 100,
    plot_every: int = 10000,
) -> AlgorithmResult:
    """
    Certify or falsify a Lyapunov certificate V using an adaptive mesh.

    Algorithm outline
    -----------------
    Each round:
      1. Select the K most uncertain active simplices (highest UCB).
      2. If disc_error > stat_error for any candidate, refine those simplices
         by centroid insertion before sampling — discretisation error dominates
         and more samples won't help until the mesh is finer.
      3. For each surviving candidate, draw P supersamples uniformly from the
         simplex interior.  For each supersample x, draw N next-states
         x' ~ T(·|x) and record
             reward_estimate = mean(V_mesh(x')) - V_mesh(x) + ε
      4. Recompute stats.  Check termination:
           - all UCB ≤ 0  →  certified  (V decreases everywhere)
           - any LCB > 0  →  try to refute on true V
               * violation found  →  return it (trigger retraining)
               * no violation     →  refine those simplices and continue

    Parameters
    ----------
    initial_vertices   : (N, dim) seed array
    true_certificate   : V(x) — neural-network callable
    transition_kernel  : T(x, key) → x' — one next-state sample from the dynamics
    checker            : vertices → bool
                         Return False to mark a simplex inactive from the start.
                         Typically checks V(vertices) ≤ 0 or is outside ROI.
    refuter            : sampler → Point | None
                         Try to find x in the simplex where V(f(x)) - V(x) > 0.
                         Return the violating point, or None if none found.
    dynamics_lipschitz : L_f  (global Lipschitz constant of the dynamics)
    M, sig             : MAB confidence hyperparameters
    epsilon            : slack in the decrease condition
    K                  : exploration width (simplices per round)
    P                  : supersamples per simplex per round
    N                  : transition-kernel draws per supersample
    max_iterations     : safety cap on rounds
    seed               : RNG seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    mesh = TrackingAdaptiveMesh(
        initial_vertices=initial_vertices,
        true_certificate=true_certificate,
        dynamics_lipschitz=dynamics_lipschitz,
        M=M,
        sig=sig,
    )

    if printing:
        import matplotlib.pyplot as plt
        from src.plotting.tilings import plot_mesh_ucb, plot_mesh_lcb  # noqa: F401
        from src.plotting.heatmaps import plot_heatmap  # noqa: F401
        from src.plotting.utils import create_figure_layout  # noqa: F401

        print("[VERIFY] Mesh built.")

    # --- Initial validity check -------------------------------------------
    # Mark simplices that the checker deems out of scope (outside ROI, already
    # trivially safe by geometry, etc.)
    for key in mesh.simplices:
        if not checker(mesh.vertices_of(key)).item():
            mesh.mark_inactive(key)

    ss_counter = 0  # global supersample ID counter

    for iteration in range(max_iterations):

        # ------------------------------------------------------------------
        # Step 1 — Discretisation check: if disc_error dominates stat_error
        #          for any top-K candidate, refine those simplices first.
        #          Centroid insertion reduces disc_error geometrically; more
        #          samples would not help until the mesh is finer.
        #
        #          Centroids are snapshotted before any insertion so that
        #          keys cannot go stale mid-loop due to Delaunay neighbour
        #          flips triggered by an earlier insertion in the same batch.
        # ------------------------------------------------------------------
        candidates = _top_k_by_ucb(mesh, K)
        if not candidates:
            break

        to_refine = [
            key
            for key in candidates
            if mesh.stats(key).disc_error > mesh.stats(key).stat_error
        ]

        if to_refine:
            # Snapshot all centroids before mutating the mesh.
            centroids = {key: mesh.centroid(key) for key in to_refine}
            for key, centroid in centroids.items():
                # Key may have been consumed as a Delaunay neighbour of an
                # earlier insertion in this same batch — skip if so.
                if key not in mesh.active_simplices:
                    continue
                deleted, added = mesh.add_vertex(centroid)
                vertices = mesh.vertices_of(key)
                edges = list(combinations(range(vertices.shape[0]), 2))
                midpoints = np.array(
                    [(vertices[i] + vertices[j]) / 2 for i, j in edges]
                )
                for point in midpoints:
                    deleted, added = mesh.add_vertex(point)

                # print(float(len(added)) / float(len(mesh.simplices)))

            # Mark any new out-of-scope simplices produced by the refinement.
            active_before = set(candidates)
            for key in mesh.active_simplices:
                if (
                    key not in active_before
                    and not checker(mesh.vertices_of(key)).item()
                ):
                    mesh.mark_inactive(key)

        # Re-select candidates from the refined mesh.
        candidates = _top_k_by_ucb(mesh, K)
        if not candidates:
            break

        # ------------------------------------------------------------------
        # Step 2 — Exploration: draw P supersamples for each candidate.
        # ------------------------------------------------------------------
        batch: List[Tuple[SupersampleID, Point, List[Point]]] = []

        for key in candidates:
            # print(mesh.stats(key).lipschitz)
            if key not in mesh.active_simplices:
                continue
            verts = mesh.vertices_of(key)
            for _ in range(P):
                x = _sample_in_simplex(verts, rng)
                sub_pts = [transition_kernel(x, jrn.key(0)) for _ in range(N)]
                ss_id = f"ss_{ss_counter}"
                ss_counter += 1
                batch.append((ss_id, x, sub_pts))

        mesh.add_supersamples_batch(batch, epsilon=epsilon)

        # ------------------------------------------------------------------
        # Step 3 — Termination checks
        # ------------------------------------------------------------------
        active_stats = mesh.all_stats()

        all_safe = all(s.ucb <= 0 for s in active_stats.values())
        any_unsafe = any(s.lcb > 0 for s in active_stats.values())

        # --- Certified ---
        if all_safe:
            return AlgorithmResult(
                status=True,
                mesh=mesh,
                iteration=iteration,
            )

        # --- Potential violation — try to refute on the true certificate ---
        if any_unsafe:
            unsafe_keys = [k for k, s in active_stats.items() if s.lcb > 0]

            for key in unsafe_keys:
                verts = mesh.vertices_of(key)
                violation = refuter(
                    lambda _, batch: (
                        jnp.array(
                            [_sample_in_simplex(verts, rng) for _ in range(batch)]
                        ),
                        None,
                    )
                )

                if violation is not None:
                    # True counterexample found — caller should retrain V and restart
                    fig, [[ax_ucb], [ax_lcb, ax_heat]] = create_figure_layout(
                        [[None], [None, None]],
                        figsize=(14, 10),
                    )

                    # 2. Pass axes into every plotter.
                    plot_mesh_ucb(mesh, ax=ax_ucb)
                    plot_mesh_lcb(mesh, ax=ax_lcb)
                    plot_heatmap(
                        mesh.certificate,
                        ((-5.0, 5.0), (-5.0, 5.0)),
                        ax=ax_heat,
                    )
                    plt.show()
                    return AlgorithmResult(
                        status=False,
                        mesh=mesh,
                        simplex=key,
                        violation=violation,
                        iteration=iteration,
                    )

            # Refuter found nothing: mesh is too coarse in unsafe simplices.
            # Snapshot centroids, then refine each one.
            unsafe_centroids = {key: mesh.centroid(key) for key in unsafe_keys}
            for key, centroid in unsafe_centroids.items():
                if key not in mesh.active_simplices:
                    continue
                mesh.add_vertex(centroid)

            # Re-run the checker on any new simplices produced by refinement.
            active_before = set(active_stats.keys())
            for key in mesh.active_simplices:
                if (
                    key not in active_before
                    and not checker(mesh.vertices_of(key)).item()
                ):
                    mesh.mark_inactive(key)

        if printing and iteration % print_every == 0:
            print(f"Iteration {iteration}")

        if printing and (iteration + 1) % plot_every == 0:
            fig, [[ax_ucb], [ax_lcb, ax_heat]] = create_figure_layout(
                [[None], [None, None]],
                figsize=(14, 10),
            )

            # 2. Pass axes into every plotter.
            plot_mesh_ucb(mesh, ax=ax_ucb)
            plot_mesh_lcb(mesh, ax=ax_lcb)
            plot_heatmap(
                mesh.certificate,
                ((-5.0, 5.0), (-5.0, 5.0)),
                ax=ax_heat,
            )
            plt.show()

    # Exhausted iteration budget without a conclusive answer
    return AlgorithmResult(
        mesh=mesh,
        iteration=max_iterations,
    )
