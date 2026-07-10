"""
adaptive_mesh.py
================
Adaptive Delaunay mesh for joint Lyapunov certificate approximation.

The mesh approximates a certificate V piecewise-linearly from vertex values.
Reward observations are evaluated at each simplex centroid:

    reward(x) = V(f(x)) - V(x) + ε

which is negative iff V strictly decreases at x — the Lyapunov condition.

Design notes
------------
- All seeds must be set via environment variable before interpreter start:
      PYTHONHASHSEED=0 python adaptive_mesh.py
  Setting os.environ["PYTHONHASHSEED"] at runtime has no effect.

- SimplexKeys are sorted tuples of integer vertex indices. Sorting gives a
  canonical form stable across all call sites. frozenset is NOT used — it
  destroys ordering, breaking get_vertices() lookups.

- _simplex_to_supersamples uses sets of IDs; all iteration over these sets
  is explicitly sorted to eliminate hash-map order dependence.

- _refresh_stats sorts its key iterable for the same reason.

- The adaptive.learner.triangulation.Triangulation class is used for
  Delaunay triangulation. Its output (simplices, add_point results) is
  sorted at every boundary crossing to make the Python-side logic
  deterministic given the same triangulation topology. Identical topology
  across runs requires deterministic insertion order, which this code
  guarantees.

- LearnerND hard-codes random.Random(1) for its internal _random, so the
  warm-start pass is already deterministic from the library side.

- SupersampleIDs use a global counter rather than id() (memory addresses
  change between runs and risk collisions after GC). Each simplex holds at
  most one supersample (its centroid evaluation).

Three mesh-building strategies are exposed as top-level functions:

    build_mesh_under_approx(...)   — fast heuristic using local gradient norm
                                     as an *under-approximation* of the
                                     Lipschitz constant (not sound).

    build_mesh_sound_local(...)    — sound bounds computed from the reachable
                                     simplex set for each centroid evaluation.
                                     More expensive than under-approx but
                                     guarantees no false positives.

    build_mesh_sound_global(...)   — sound bounds using the global (maximum)
                                     gradient norm across all active simplices
                                     as the Lipschitz constant.  Conservative
                                     and cheap to compute.

All three accept an optional ``warm_start_mesh`` argument: pass an existing
``TrackingAdaptiveMesh`` to continue refinement from a previous run instead
of rebuilding from scratch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import combinations, count, product
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import jax
import jax.numpy as jnp
import jax.random as jrn
import numpy as np

USE_RUST_BINDING = True

if USE_RUST_BINDING:
    import adaptive_triangulation as at
    from adaptive_triangulation import Triangulation
    from adaptive.learner import learnerND as lnd_mod

    # monkey patch
    lnd_mod.Triangulation = at.Triangulation
    lnd_mod.circumsphere = at.circumsphere
    lnd_mod.simplex_volume_in_embedding = at.simplex_volume_in_embedding
    lnd_mod.point_in_simplex = at.point_in_simplex


else:
    from adaptive.learner.triangulation import Triangulation

import adaptive
from adaptive.learner.learnerND import LearnerND
from scipy.spatial import ConvexHull, KDTree
from adaptive.learner.learnerND import circumsphere

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

SimplexKey = tuple  # sorted tuple of integer vertex indices
SupersampleID = str
Point = np.ndarray

# Global counter for stable, unique SupersampleIDs
_ss_counter = count()


def _next_ss_id() -> SupersampleID:
    return f"ss_{next(_ss_counter)}"


# ---------------------------------------------------------------------------
# Supersample
# ---------------------------------------------------------------------------


@dataclass
class Supersample:
    """
    Lyapunov decrease estimate at a single query point x.

        reward_estimate = V(f(x)) - V(x) + ε

    where f(x) is the deterministic next state. own_value and next_value are
    updated in-place after mesh changes so reward_estimate stays current.
    """

    id: SupersampleID
    position: Point
    own_simplex: Optional[SimplexKey]
    own_value: float
    true_cert_value: float
    next_position: Point
    next_simplex: Optional[SimplexKey]
    next_value: float
    epsilon: float = 0.0

    @property
    def reward_estimate(self) -> float:
        if not np.isfinite(self.next_value) or not np.isfinite(self.own_value):
            return float("inf")
        return self.next_value - self.own_value + self.epsilon


# ---------------------------------------------------------------------------
# SimplexStats
# ---------------------------------------------------------------------------


@dataclass
class SimplexStats:
    """
    Per-simplex certificate and bound statistics.

    Uncertainty comes only from discretization error (mesh fineness).
    The reward is evaluated at the centroid, so the bound uses the simplex
    radius (diameter / 2) rather than the full diameter:

        disc_error = radius × (gradient_norm + 1) × dynamics_lipschitz
        ucb / lcb  = reward ± disc_error

    When sound=True, lipschitz is computed from the reachable simplex set,
    not just the local gradient norm.
    """

    count: int = 0
    mean: float = float("inf")
    gradient_norm: float = float("inf")
    lipschitz: float = float("inf")
    disc_error: float = float("inf")
    ucb: float = float("inf")
    lcb: float = float("-inf")
    approx_error: float = float("inf")
    sound: bool = False

    @classmethod
    def from_simplex(
        cls,
        reward_value: float,
        cert_true: float,
        cert_interp: float,
        gradient_norm: float,
        dynamics_lipschitz: float,
        radius: float,
    ) -> SimplexStats:
        approx_error = (
            abs(cert_true - cert_interp)
            if np.isfinite(cert_interp) and np.isfinite(cert_true)
            else float("inf")
        )
        lipschitz = (dynamics_lipschitz + 1) * gradient_norm
        disc_error = radius * lipschitz
        mean = float(reward_value) if np.isfinite(reward_value) else float("inf")
        return cls(
            count=1,
            mean=mean,
            gradient_norm=gradient_norm,
            lipschitz=lipschitz,
            disc_error=disc_error,
            ucb=mean + disc_error,
            lcb=mean - disc_error,
            approx_error=approx_error,
            sound=False,
        )

    @classmethod
    def from_sound(
        cls,
        reward_value: float,
        cert_true: float,
        cert_interp: float,
        gradient_norm: float,
        reachable_gradient_norm: float,
        dynamics_lipschitz: float,
        radius: float,
    ) -> SimplexStats:
        """Create sound stats using reachable gradient norm."""
        approx_error = (
            abs(cert_true - cert_interp)
            if np.isfinite(cert_interp) and np.isfinite(cert_true)
            else float("inf")
        )
        lipschitz = dynamics_lipschitz * reachable_gradient_norm + gradient_norm
        disc_error = radius * lipschitz
        mean = float(reward_value) if np.isfinite(reward_value) else float("inf")
        return cls(
            count=1,
            mean=mean,
            gradient_norm=gradient_norm,
            lipschitz=lipschitz,
            disc_error=disc_error,
            ucb=mean + disc_error,
            lcb=mean - disc_error,
            approx_error=approx_error,
            sound=True,
        )

    def __repr__(self) -> str:
        if self.count == 0:
            return f"SimplexStats(empty, approx_error={self.approx_error:.4g})"
        sound_marker = " [SOUND]" if self.sound else ""
        return (
            f"SimplexStats(n={self.count}, ucb={self.ucb:.4g}, "
            f"lcb={self.lcb:.4g}, approx_error={self.approx_error:.4g}){sound_marker}"
        )


# ---------------------------------------------------------------------------
# AdaptiveMesh
# ---------------------------------------------------------------------------


class AdaptiveMesh:
    """
    Piecewise-linear Lyapunov certificate mesh.

    Manages vertex certificate values, simplex geometry, gradient computation,
    and SimplexStats. Does not know about supersamples — see TrackingAdaptiveMesh.

    Parameters
    ----------
    initial_vertices   : (N, dim) seed vertex array
    true_certificate   : V(x) — the neural-network certificate callable
    dynamics_lipschitz : global Lipschitz constant of the dynamics
    """

    def __init__(
        self,
        true_certificate: Callable,
        dynamics_lipschitz: float,
        hull: Optional[ConvexHull] = None,
        triangulation: Optional[Triangulation] = None,
        points=None,
    ) -> None:

        if triangulation is None:
            self.hull = ConvexHull(points)
            self._tri = Triangulation(points)
            self.pts = self._tri.vertices
        else:
            self._hull = hull
            self._tri = triangulation
            self.pts = triangulation.vertices

        self._dim = self._tri.dim
        self.true_certificate = true_certificate
        self.dynamics_lipschitz = dynamics_lipschitz

        self._vertex_cert: Dict[tuple, float] = {}
        self._stats: Dict[SimplexKey, SimplexStats] = {}
        self._inactive: Set[SimplexKey] = set()

        for pt in self.pts:
            self._vertex_cert[tuple(pt)] = true_certificate(pt)

        for raw in sorted(tuple(sorted(s)) for s in self._tri.simplices):
            self._stats[raw] = SimplexStats()

    # --- Geometry ----------------------------------------------------------

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def simplices(self) -> List[SimplexKey]:
        return sorted(tuple(sorted(s)) for s in self._tri.simplices)

    @property
    def active_simplices(self) -> List[SimplexKey]:
        return [k for k in self.simplices if k not in self._inactive]

    def vertices_of(self, key: SimplexKey) -> np.ndarray:
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

    def radius(self, key: SimplexKey) -> float:
        """Half the diameter — max deviation from centroid is bounded by this."""
        return self.diameter(key) / 2.0

    def _locate(self, coords: np.ndarray) -> Optional[SimplexKey]:
        raw = self._tri.locate_point(tuple(np.asarray(coords, dtype=float)))
        return tuple(sorted(raw)) if raw else None

    def mark_inactive(self, key: SimplexKey) -> None:
        self._inactive.add(key)

    def mark_active(self, key: SimplexKey) -> None:
        self._inactive.discard(key)

    def is_active(self, key: SimplexKey) -> bool:
        return key not in self._inactive

    def _face_adjacent_simplices(self, key: SimplexKey) -> Set[SimplexKey]:
        """
        Return all simplices that share a (dim-1)-face with `key`.
        Two simplices are face-adjacent iff they share exactly dim vertices.
        """
        verts = set(key)
        adjacent = set()

        for raw in self._tri.simplices:
            other_key = tuple(sorted(raw))
            if other_key == key:
                continue
            shared = len(verts & set(other_key))
            if shared == self._dim:
                adjacent.add(other_key)

        return adjacent

    def reachable_simplices(
        self,
        key: SimplexKey,
        next_pt: np.ndarray,
        max_displacement: float,
    ) -> Set[SimplexKey]:
        """
        Find all simplices reachable from next_pt within max_displacement.

        BFS expansion from the simplex containing next_pt. A simplex σ' is
        included iff its distance to next_pt could be ≤ max_displacement:
        conservative check: ||centroid(σ') - next_pt|| ≤ max_displacement + radius(σ')
        """
        next_key = self._locate(next_pt)
        if next_key is None:
            return set()

        reachable = set()
        frontier = {next_key}

        while frontier:
            current = frontier.pop()
            if current in reachable:
                continue
            reachable.add(current)

            centroid_curr = self.centroid(current)
            dist_to_next = float(np.linalg.norm(centroid_curr - next_pt))
            radius_curr = self.radius(current)

            if dist_to_next > max_displacement + radius_curr:
                continue

            for neighbor in self._face_adjacent_simplices(current):
                if neighbor not in reachable:
                    frontier.add(neighbor)

        return reachable

    # --- Certificate -------------------------------------------------------

    def _cert_values_for(self, key: SimplexKey) -> Optional[List[float]]:
        verts = self.vertices_of(key)
        vals = [self._vertex_cert.get(tuple(v)) for v in verts]
        return None if any(v is None for v in vals) else vals

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
        vals = self._cert_values_for(key)
        if lam is None or vals is None:
            return None
        return float(np.dot(lam, vals))

    def gradient(self, key: SimplexKey) -> Optional[np.ndarray]:
        verts = self.vertices_of(key)
        vals = self._cert_values_for(key)
        if vals is None:
            return None
        edges = verts[1:] - verts[0]
        try:
            return np.linalg.solve(edges, np.array(vals)[1:] - vals[0])
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
        """Insert a new vertex; update certificate cache and stats dicts."""
        pt = np.asarray(point, dtype=float)
        coord = tuple(pt)
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

    def retraingulate_with_vertices(
        self, points: np.ndarray
    ) -> Tuple[Set[SimplexKey], Set[SimplexKey]]:
        """
        Insert multiple vertices at once via triangulation rebuild.

        Much faster than repeated add_vertex() when inserting many points.
        Reconstructs the triangulation from scratch with all existing + new points.
        """
        points_arr = np.asarray(points, dtype=float)
        if len(points_arr) == 0:
            return set(), set()

        all_new_points = np.vstack([self.pts, points_arr])

        for pt in points_arr:
            coord = tuple(pt)
            self._vertex_cert[coord] = float(self.true_certificate(pt))

        old_simplices = {tuple(sorted(s)) for s in self._tri.simplices}

        self._tri = Triangulation(all_new_points)
        self.pts = self._tri.vertices

        new_simplices = {tuple(sorted(s)) for s in self._tri.simplices}

        deleted = old_simplices - new_simplices
        added = new_simplices - old_simplices

        for key in deleted:
            self._stats.pop(key, None)
            self._inactive.discard(key)
        for key in added:
            self._stats[key] = SimplexStats()

        return deleted, added

    # --- Stats -------------------------------------------------------------

    def _compute_stats(
        self,
        key: SimplexKey,
        supersample: Optional["Supersample"] = None,
    ) -> SimplexStats:
        """
        Compute SimplexStats for ``key`` using the *local* (under-approximation)
        Lipschitz constant — i.e. just the gradient norm of this simplex.

        This is the default stats path.  For sound or global-Lipschitz bounds
        use ``sound_verify_mesh`` or ``global_verify_mesh`` respectively.
        """
        gnorm = self.gradient_norm(key)
        lipschitz = (self.dynamics_lipschitz + 1) * gnorm
        radius = self.radius(key)
        disc_error = radius * lipschitz

        if supersample is None:
            return SimplexStats(
                count=0,
                mean=float("inf"),
                gradient_norm=gnorm,
                lipschitz=lipschitz,
                disc_error=disc_error,
                ucb=float("inf"),
                lcb=float("-inf"),
                approx_error=float("inf"),
            )

        return SimplexStats.from_simplex(
            reward_value=supersample.reward_estimate,
            cert_true=supersample.true_cert_value,
            cert_interp=supersample.own_value,
            gradient_norm=gnorm,
            dynamics_lipschitz=self.dynamics_lipschitz,
            radius=radius,
        )

    def _compute_stats_global(
        self,
        key: SimplexKey,
        global_gradient_norm: float,
        supersample: Optional["Supersample"] = None,
    ) -> SimplexStats:
        """
        Compute SimplexStats using the *global* gradient norm (maximum over all
        active simplices) as the Lipschitz constant.  More conservative than the
        local estimate — never an under-approximation.
        """
        gnorm = self.gradient_norm(key)
        lipschitz = (self.dynamics_lipschitz + 1) * global_gradient_norm
        radius = self.radius(key)
        disc_error = radius * lipschitz

        if supersample is None:
            return SimplexStats(
                count=0,
                mean=float("inf"),
                gradient_norm=gnorm,
                lipschitz=lipschitz,
                disc_error=disc_error,
                ucb=float("inf"),
                lcb=float("-inf"),
                approx_error=float("inf"),
            )

        approx_error = (
            abs(supersample.true_cert_value - supersample.own_value)
            if np.isfinite(supersample.own_value)
            and np.isfinite(supersample.true_cert_value)
            else float("inf")
        )
        mean = float(supersample.reward_estimate)
        return SimplexStats(
            count=1,
            mean=mean,
            gradient_norm=gnorm,
            lipschitz=lipschitz,
            disc_error=disc_error,
            ucb=mean + disc_error,
            lcb=mean - disc_error,
            approx_error=approx_error,
            sound=True,
        )

    def stats(self, key: SimplexKey) -> SimplexStats:
        return self._stats.get(key, SimplexStats())

    def all_stats(self) -> Dict[SimplexKey, SimplexStats]:
        return {k: v for k, v in self._stats.items() if k not in self._inactive}

    def least_safe(self) -> Optional[SimplexKey]:
        """Highest-UCB active simplex."""
        active = self.active_simplices
        return (
            None
            if not active
            else max(active, key=lambda k: self._stats.get(k, SimplexStats()).ucb)
        )

    def __repr__(self) -> str:
        return (
            f"AdaptiveMesh(dim={self._dim}, "
            f"simplices={len(self.active_simplices)} active "
            f"/ {len(self._inactive)} inactive)"
        )


# ---------------------------------------------------------------------------
# TrackingAdaptiveMesh
# ---------------------------------------------------------------------------


class TrackingAdaptiveMesh(AdaptiveMesh):
    """
    Extends AdaptiveMesh with a supersample inverse index.

    Each active simplex holds at most one supersample, evaluated at its
    centroid. _simplex_to_supersamples[key] is the set of SupersampleIDs that
    have any tracked point (own position or next-state position) inside `key`.
    All iteration over these sets is explicitly sorted to eliminate hash-map
    order dependence.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.supersamples: Dict[SupersampleID, Supersample] = {}
        self._simplex_to_supersamples: Dict[SimplexKey, Set[SupersampleID]] = {}

    # --- Registration helpers ----------------------------------------------

    def _all_simplex_keys(self, ss: Supersample) -> Set[Optional[SimplexKey]]:
        return {ss.own_simplex, ss.next_simplex}

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
        next_position: Point,
        epsilon: float = 0.0,
        true_cert_value: Optional[float] = None,
    ) -> Optional[Supersample]:
        own_key = self._locate(position)
        if own_key is None:
            return None
        own_value = self._interpolate_in(own_key, position)
        if own_value is None:
            return None
        if true_cert_value is None:
            true_cert_value = float(self.true_certificate(position))

        next_pt = np.asarray(next_position, dtype=float)
        next_key = self._locate(next_pt)
        next_value = (
            self._interpolate_in(next_key, next_pt) if next_key is not None else None
        )

        ss = Supersample(
            id=ss_id,
            position=np.asarray(position, dtype=float),
            own_simplex=own_key,
            own_value=own_value,
            true_cert_value=true_cert_value,
            next_position=next_pt,
            next_simplex=next_key,
            next_value=next_value if next_value is not None else float("inf"),
            epsilon=epsilon,
        )
        self.supersamples[ss_id] = ss
        self._register(ss)
        return ss

    def add_supersamples_batch(
        self,
        entries: List[Tuple[SupersampleID, Point, Point]],
        epsilon: float = 0.0,
    ) -> List[Supersample]:
        """Add many supersamples (each a centroid + one next-state point), then refresh stats."""
        added = []
        dirty_own: Set[Optional[SimplexKey]] = set()
        for ss_id, pos, next_pos in entries:
            ss = self.add_supersample(ss_id, pos, next_pos, epsilon=epsilon)
            if ss is not None:
                added.append(ss)
                dirty_own.add(ss.own_simplex)
        self._refresh_stats(dirty_own)
        return added

    # --- Mesh update propagation -------------------------------------------

    def on_mesh_updated(self, removed: Set[SimplexKey], added: Set[SimplexKey]) -> None:
        """
        Called after every structural mesh change.

        Pass 1 — structural: for each affected supersample, re-locate any
        point whose containing simplex was destroyed and refresh its value.
        Pass 2 — stats: recompute SimplexStats for all newly-created simplices.
        """
        affected_ids: Set[SupersampleID] = set()
        for key in sorted(removed):
            affected_ids |= self._simplex_to_supersamples.pop(key, set())

        for ss_id in sorted(affected_ids):
            ss = self.supersamples[ss_id]
            self._unregister(ss)

            if ss.own_simplex in removed:
                new_key = self._locate(ss.position)
                ss.own_simplex = new_key
                val = self._interpolate_in(new_key, ss.position) if new_key else None
                ss.own_value = val if val is not None else float("inf")

            if ss.next_simplex in removed:
                new_key = self._locate(ss.next_position)
                ss.next_simplex = new_key
                val = (
                    self._interpolate_in(new_key, ss.next_position) if new_key else None
                )
                ss.next_value = val if val is not None else float("inf")

            self._register(ss)

        self._refresh_stats(added)

    def _refresh_stats(self, keys: Set[Optional[SimplexKey]]) -> None:
        """Recompute SimplexStats for the given simplex keys."""
        for key in sorted(k for k in keys if k is not None):
            own_ids = [
                sid
                for sid in sorted(self._simplex_to_supersamples.get(key, set()))
                if self.supersamples[sid].own_simplex == key
            ]
            ss = self.supersamples[own_ids[0]] if own_ids else None
            self._stats[key] = self._compute_stats(key, ss)

    def refresh_all_stats(self) -> None:
        self._refresh_stats(set(self.active_simplices))  # type: ignore[arg-type]

    # --- Override add_vertex -----------------------------------------------

    def add_vertex(self, point: np.ndarray) -> Tuple[Set[SimplexKey], Set[SimplexKey]]:
        deleted, added = super().add_vertex(point)
        self.on_mesh_updated(deleted, added)
        return deleted, added

    def retraingulate_with_vertices(
        self, points: np.ndarray
    ) -> Tuple[Set[SimplexKey], Set[SimplexKey]]:
        """Batch insert with supersample tracking."""
        deleted, added = super().retraingulate_with_vertices(points)
        self.on_mesh_updated(deleted, added)
        return deleted, added

    # --- Convenience -------------------------------------------------------

    def supersamples_in(self, key: SimplexKey) -> List[Supersample]:
        """All supersamples whose own simplex is `key`."""
        return [
            self.supersamples[sid]
            for sid in sorted(self._simplex_to_supersamples.get(key, set()))
            if self.supersamples[sid].own_simplex == key
        ]

    # --- High-level operations ---------------------------------------------

    def refine_simplex(
        self,
        key: SimplexKey,
        checker: Optional[Callable[[np.ndarray], bool]] = None,
    ) -> Tuple[Set[SimplexKey], Set[SimplexKey]]:
        """
        Refine a simplex by inserting its centroid and all edge midpoints.

        Parameters
        ----------
        key     : simplex to refine
        checker : if provided, marks new simplices inactive when it returns False
        """
        if key not in self.active_simplices:
            return set(), set()

        start = time.time()

        vertices = self.vertices_of(key)
        deleted, added = self.add_vertex(vertices.mean(axis=0))
        # center, radius = self._tri.circumscribed_circle(key)
        # center = np.array(center)
        # if self._hull

        # deleted, added = self.add_vertex(np.array(center))

        edge_added = set()
        edge_deleted = set()

        for i, j in combinations(range(len(vertices)), 2):
            d, a = self.add_vertex((vertices[i] + vertices[j]) / 2.0)
            edge_added |= a
            edge_deleted |= d

        added |= edge_added
        deleted |= edge_deleted

        start = time.time()

        if checker is not None:
            for key in list(added):
                if not checker(self.vertices_of(key)):
                    self.mark_inactive(key)

        deleted.add(key)
        return deleted, added

    def evaluate_simplices(
        self,
        keys: List[SimplexKey],
        transition_kernel: Callable[[np.ndarray, any], np.ndarray],
        epsilon: float = 0.0,
    ):
        """
        Evaluate the reward at the simplex centroid and update its statistics.

        Parameters
        ----------
        key               : simplex to evaluate
        transition_kernel : f(x, rng_key) -> x'  (deterministic — rng_key ignored)
        epsilon           : Lyapunov slack constant
        """
        # if key not in self.active_simplices:
        #     return self.stats(key)

        # if self.supersamples_in(key):
        #     return self.stats(key)

        xs = jnp.stack([self.centroid(k) for k in keys])
        next_pts = np.asarray(
            jax.vmap(transition_kernel, in_axes=(0, None))(xs, jax.random.key(0)),
            dtype=float,
        )
        self.add_supersamples_batch(
            [(_next_ss_id(), x, next_pt) for x, next_pt in zip(xs, next_pts)],
            epsilon=epsilon,
        )

    def __repr__(self) -> str:
        return (
            f"TrackingAdaptiveMesh(dim={self._dim}, "
            f"simplices={len(self.active_simplices)} active "
            f"/ {len(self._inactive)} inactive, "
            f"supersamples={len(self.supersamples)})"
        )

    # --- Sound verification ------------------------------------------------

    def sound_verify_mesh(
        self, transition_kernel: Callable[[np.ndarray, any], np.ndarray]
    ) -> Dict[SimplexKey, SimplexStats]:
        """
        Compute *sound local* Lipschitz bounds for all active simplices.

        For each simplex σ the reachable simplex set is found via BFS and the
        Lipschitz constant is:

            L = L_dyn * max_reachable_gradient_norm + local_gradient_norm

        Returns a Dict of sound SimplexStats without mutating the mesh.
        """
        sound_stats: Dict[SimplexKey, SimplexStats] = {}

        for key in self.active_simplices:
            gnorm = self.gradient_norm(key)
            radius = self.radius(key)

            x = self.centroid(key)
            next_pt = np.asarray(transition_kernel(x, jrn.key(0)), dtype=float)

            max_disp = self.dynamics_lipschitz * radius
            reachable = self.reachable_simplices(key, next_pt, max_disp)

            L_reachable = max(
                (self.gradient_norm(sk) for sk in reachable),
                default=gnorm,
            )
            L_reachable = max(L_reachable, gnorm)

            own_ids = [
                sid
                for sid in sorted(self._simplex_to_supersamples.get(key, set()))
                if self.supersamples[sid].own_simplex == key
            ]
            ss = self.supersamples[own_ids[0]] if own_ids else None

            if ss is None:
                sound_stats[key] = SimplexStats(
                    count=0,
                    mean=float("inf"),
                    gradient_norm=gnorm,
                    lipschitz=float("inf"),
                    disc_error=float("inf"),
                    ucb=float("inf"),
                    lcb=float("-inf"),
                    approx_error=float("inf"),
                    sound=True,
                )
            else:
                sound_stats[key] = SimplexStats.from_sound(
                    reward_value=ss.reward_estimate,
                    cert_true=ss.true_cert_value,
                    cert_interp=ss.own_value,
                    gradient_norm=gnorm,
                    reachable_gradient_norm=L_reachable,
                    dynamics_lipschitz=self.dynamics_lipschitz,
                    radius=radius,
                )

        return sound_stats

    def global_verify_mesh(self) -> Dict[SimplexKey, SimplexStats]:
        """
        Compute *global* Lipschitz bounds for all active simplices.

        Uses the maximum gradient norm across all active simplices as a single
        global Lipschitz constant.  Conservative but cheap — no BFS required.

        Returns a Dict of SimplexStats (sound=True) without mutating the mesh.
        """
        active = self.active_simplices
        if not active:
            return {}

        global_gnorm = max(self.gradient_norm(k) for k in active)

        global_stats: Dict[SimplexKey, SimplexStats] = {}
        for key in active:
            own_ids = [
                sid
                for sid in sorted(self._simplex_to_supersamples.get(key, set()))
                if self.supersamples[sid].own_simplex == key
            ]
            ss = self.supersamples[own_ids[0]] if own_ids else None
            global_stats[key] = self._compute_stats_global(key, global_gnorm, ss)

        return global_stats

    def _build_centroid_tree(self) -> Tuple[KDTree, List[SimplexKey], float]:
        """
        Build a cKDTree over active simplex centroids.

        Returns
        -------
        tree     : cKDTree over centroid coordinates
        keys     : simplex keys in the same order as tree rows
        r_max    : maximum simplex radius across all active simplices
        """
        active = self.active_simplices
        keys = list(active)
        centroids = np.array([self.centroid(k) for k in keys])
        r_max = max(self.radius(k) for k in keys)
        tree = KDTree(centroids)
        return tree, keys, r_max

    def sound_verify_mesh_kdtree(
        self,
        transition_kernel: Callable[[np.ndarray, any], np.ndarray],
        printing: bool = True,
    ) -> Dict[SimplexKey, SimplexStats]:
        """
        Compute sound local Lipschitz bounds for all active simplices using
        a cKDTree for reachable-simplex lookup instead of BFS.

        For each simplex σ the reachable set is found by querying a ball of
        radius (L_dyn * radius(σ) + r_max) around x' = f(centroid(σ)).
        L_reachable is the max gradient norm over that set.

        Returns a Dict of sound SimplexStats without mutating the mesh.
        """
        active = self.active_simplices
        if not active:
            return {}

        start = time.time()
        tree, keys, r_max = self._build_centroid_tree()
        if printing:
            print(f"KDTree built in {time.time() - start}")
        # key_index = {k: i for i, k in enumerate(keys)}

        # batch all centroid evaluations and next-state computations
        centroids = np.array([self.centroid(k) for k in keys])
        next_pts = np.asarray(
            jax.vmap(transition_kernel, in_axes=(0, None))(
                jnp.array(centroids), jax.random.key(0)
            ),
            dtype=float,
        )

        # precompute per-simplex quantities
        radii = np.array([self.radius(k) for k in keys])
        gnorms = np.array([self.gradient_norm(k) for k in keys])

        # batch ball queries — one query radius per simplex
        query_radii = self.dynamics_lipschitz * radii + r_max
        # query_ball_point accepts a scalar or per-point radius array
        neighbor_lists = tree.query_ball_point(next_pts, r=query_radii)

        sound_stats: Dict[SimplexKey, SimplexStats] = {}

        for i, key in enumerate(keys):
            neighbor_idxs = neighbor_lists[i]
            L_reachable = (
                float(np.max(gnorms[neighbor_idxs])) if neighbor_idxs else gnorms[i]
            )
            L_reachable = max(L_reachable, gnorms[i])

            gnorm = gnorms[i]
            radius = radii[i]

            own_ids = [
                sid
                for sid in sorted(self._simplex_to_supersamples.get(key, set()))
                if self.supersamples[sid].own_simplex == key
            ]
            ss = self.supersamples[own_ids[0]] if own_ids else None

            sound_stats[key] = SimplexStats.from_sound(
                reward_value=ss.reward_estimate,
                cert_true=ss.true_cert_value,
                cert_interp=ss.own_value,
                gradient_norm=gnorm,
                reachable_gradient_norm=L_reachable,
                dynamics_lipschitz=self.dynamics_lipschitz,
                radius=radius,
            )

        self._stats = sound_stats

        # if printing:
        #     _plot_state(self.hull, self)

        return sound_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _top_k_by_ucb(
    mesh: TrackingAdaptiveMesh,
    top_k: Optional[int] = None,
    priority_verify=False,
) -> List[SimplexKey]:
    """Return up to k active simplices with the highest UCB."""
    active = list(mesh.active_simplices)
    if not active:
        return []

    ucb = np.array([mesh.stats(key).ucb for key in active])

    mask = ucb > 0
    active = [key for key, m in zip(active, mask) if m]
    ucb = ucb[mask]

    if not top_k:
        top_k = len(active)

    if priority_verify:
        order = 1
    else:
        order = -1

    # deterministic tie-break
    order = np.lexsort((np.arange(len(active)), order * ucb))
    return [active[i] for i in order[:top_k]]


# ---------------------------------------------------------------------------
# Algorithm result
# ---------------------------------------------------------------------------


@dataclass
class AlgorithmResult:
    mesh: TrackingAdaptiveMesh
    status: Optional[bool] = None
    simplex: Optional[SimplexKey] = None
    violation: Optional[List[np.ndarray]] = None
    iteration: int = 0


# ---------------------------------------------------------------------------
# Shared internal helpers
# ---------------------------------------------------------------------------

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import os

physical_cores = os.cpu_count() // 2  # rough heuristic on most x86 CPUs
max_workers = physical_cores
ntasks = 2 * max_workers


def warmstart_learner(learner, loss_goal=None, npoints_goal=None):

    if loss_goal is None and npoints_goal is None:
        return

    # Build the goal callable from whichever targets were supplied.
    def goal(l):
        loss_done = (loss_goal is not None) and (l.loss() < loss_goal)
        points_done = (npoints_goal is not None) and (l.npoints >= npoints_goal)
        return loss_done or points_done

    # Use an Event so the monitor thread sees the stop signal reliably.
    stop_event = threading.Event()

    def monitor():
        last = 0
        while not stop_event.wait(timeout=5):  # wakes immediately on set()
            n = learner.npoints
            if n != last:
                print(f"Added {n - last} points (total={n})")
                last = n

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        adaptive.BlockingRunner(
            learner,
            executor=executor,
            ntasks=ntasks,
            goal=goal,
        )
    finally:
        stop_event.set()
        thread.join(timeout=2)
        executor.shutdown(wait=False, cancel_futures=True)


def _build_initial_mesh(
    hull,
    true_certificate: Callable,
    transition_kernel: Optional[Callable],
    dynamics_lipschitz: float,
    epsilon: float,
    warmstart_npoints: Optional[int],
    warmstart_loss: Optional[float],
    printing: bool,
    initial_vertices,
) -> Triangulation:
    """
    Construct (or reuse) a TrackingAdaptiveMesh.

    If ``warm_start_mesh`` is provided it is returned as-is — the caller is
    responsible for making sure its certificate / dynamics match.  Otherwise
    a fresh mesh is built from ``initial_vertices`` augmented by a LearnerND
    warm-start pass.
    """

    def disc_loss(verts, values, _):
        verts = np.array(verts)
        values = np.array(values, dtype=float)

        # Vectorised pairwise distances via broadcasting, no Python loop
        diffs = verts[1:] - verts[0]
        radius = np.max(np.linalg.norm(verts[:, None] - verts[None, :], axis=-1)) / 2

        grad = np.linalg.solve(diffs, values[1:] - values[0])
        return float(radius * np.linalg.norm(grad) * (1 + dynamics_lipschitz))

    def reward_loss(verts, values, _):
        verts = np.array(verts)
        values = np.array(values, dtype=float)
        mid = np.mean(verts, axis=0)
        next_pt = transition_kernel(mid, jrn.key(0))

        # Vectorised pairwise distances via broadcasting, no Python loop
        diffs = verts[1:] - verts[0]
        radius = np.max(np.linalg.norm(verts[:, None] - verts[None, :], axis=-1)) / 2

        grad = np.linalg.solve(diffs, values[1:] - values[0])
        return float(
            radius * np.linalg.norm(grad) * (1 + dynamics_lipschitz)
            + (true_certificate(next_pt) - true_certificate(mid) + epsilon)
        )

    learner = LearnerND(
        lambda x: true_certificate(x),
        hull,
        loss_per_simplex=(reward_loss if transition_kernel else disc_loss),
    )

    if printing:
        print("[VERIFY] Warming up.")
    learner.tell_many(initial_vertices, [true_certificate(x) for x in initial_vertices])
    warmstart_learner(learner, loss_goal=warmstart_loss, npoints_goal=warmstart_npoints)

    return learner._tri


# def _make_candidate_generator(verts, n_verts, max_batch):
#     """Build the candidate-point generator used by the refuter."""
#     verts_jax = jnp.array(verts)
#     deterministic_jax = jnp.array(np.vstack([verts, verts.mean(axis=0, keepdims=True)]))

#     def generate_candidates(_, batch):
#         batch = min(batch, max_batch)
#         n_det = deterministic_jax.shape[0]
#         if batch <= n_det:
#             return deterministic_jax[:batch], None
#         n_additional = batch - n_det
#         rng = jrn.key(0)
#         random_weights = jrn.dirichlet(rng, jnp.ones(n_verts), shape=(n_additional,))
#         random_points = random_weights @ verts_jax
#         return jnp.vstack([deterministic_jax, random_points])[:batch], None

#     return generate_candidates


# def _try_refute(
#     mesh: TrackingAdaptiveMesh,
#     unsafe_keys: List[SimplexKey],
#     refuter: Callable,
#     true_certificate: Callable,
#     transition_kernel: Callable,
#     epsilon: float,
# ) -> List[np.ndarray]:
#     """Run the refuter over ``unsafe_keys``; return confirmed violations."""
#     all_violations = []
#     for key in unsafe_keys:
#         verts = mesh.vertices_of(key)
#         n_verts = len(verts)
#         gen = _make_candidate_generator(verts, n_verts, max(n_verts * 10, 1000))
#         violation = refuter(gen)
#         if violation is not None and (
#             true_certificate(transition_kernel(violation, jrn.key(0)))
#             - true_certificate(violation)
#             + epsilon
#             > 0.0
#         ):
#             all_violations.append(violation)
#     return all_violations


def _try_refute(
    mesh,  # Assuming TrackingAdaptiveMesh
    unsafe_keys: List,  # Assuming SimplexKey
    batched_refuter: Callable,
    true_certificate: Callable,
    transition_kernel: Callable,
    epsilon: float,
    base_key: jnp.ndarray,
) -> List[np.ndarray]:
    """
    Run the vectorized refuter over all `unsafe_keys` simultaneously;
    return confirmed violations.
    """
    if not unsafe_keys:
        return []

    all_verts = jnp.stack([mesh.vertices_of(key) for key in unsafe_keys])
    n_simplices = len(unsafe_keys)
    keys = jrn.split(base_key, n_simplices)

    best_candidates, ys = batched_refuter(all_verts, keys)

    # 4. Check results for true violations
    all_violations = []
    for i in range(n_simplices):
        violation = best_candidates[i]
        if ys[i] > 0.0:
            all_violations.append(np.array(violation))

    return all_violations


# ---------------------------------------------------------------------------
# Strategy 1: Under-approximation (heuristic, fast)
# ---------------------------------------------------------------------------


def warm_start_under_approx(
    initial_vertices: np.ndarray,
    true_certificate: Callable[[np.ndarray], float],
    transition_kernel: Callable[[jnp.ndarray, jnp.ndarray], np.ndarray],
    checker: Callable[[np.ndarray], np.ndarray],
    refuter: Callable[[Callable], Optional[np.ndarray]],
    dynamics_lipschitz: float,
    epsilon: float = 0.0,
    K: Optional[int] = 1,
    max_iterations: int = 20000,
    warmstart_npoints: int = 1,
    warmstart_loss: Optional[float] = None,
    printing: bool = False,
    print_every: int = 10,
    plot_every: int = 100,
) -> AlgorithmResult:
    """

    Build a mesh using the **under-approximation** (heuristic) Lipschitz strategy.



    The discretization error bound uses only the local gradient norm of each

    simplex — this is an under-approximation of the true Lipschitz constant,

    so ``ucb <= 0`` is a *necessary* but not *sufficient* condition for safety.

    The result is fast to compute and useful as a warm-start for the sound methods.



    Parameters

    ----------

    initial_vertices   : (N, dim) convex-hull seed points

    true_certificate   : V(x) — neural-network certificate

    transition_kernel  : f(x, rng_key) -> x'  (rng_key is ignored; deterministic)

    checker            : returns True iff all vertices lie in the region of interest

    refuter            : attempts to find a true counterexample in a simplex

    dynamics_lipschitz : global Lipschitz constant of the dynamics

    epsilon            : Lyapunov slack

    K                  : simplices to refine per round

    max_iterations     : iteration budget

    npoints_warmstart  : LearnerND warm-start budget (ignored when warm_start_mesh given)

    printing           : enable progress output

    print_every        : print interval (iterations)

    plot_every         : plot interval (iterations)

    warm_start_mesh    : optional pre-built mesh to continue from

    """

    build_start = time.time()

    warm_start_start = time.time()

    hull = ConvexHull(initial_vertices)

    init_tri = _build_initial_mesh(
        hull=hull,
        initial_vertices=initial_vertices,
        true_certificate=true_certificate,
        transition_kernel=None,  # transition_kernel,
        dynamics_lipschitz=dynamics_lipschitz,
        epsilon=epsilon,
        warmstart_npoints=warmstart_npoints,
        warmstart_loss=warmstart_loss,
        printing=printing,
    )

    mesh = TrackingAdaptiveMesh(
        hull=hull,
        triangulation=init_tri,
        true_certificate=true_certificate,
        dynamics_lipschitz=dynamics_lipschitz,
    )

    mesh.evaluate_simplices(
        mesh.simplices, transition_kernel=transition_kernel, epsilon=epsilon
    )

    warm_start_time = time.time() - warm_start_start

    if printing:

        print(
            f"[VERIFY] Warm-up completed in {warm_start_time:.4f} seconds with {len(mesh._tri.simplices)} simplices.\n"
        )

        _plot_state(hull, mesh)

    pairs = list(combinations(range(mesh._dim + 1), 2))

    for iteration in range(max_iterations):

        # 1. Refine: insert centroid + edge midpoints for top-K candidates

        candidates = _top_k_by_ucb(mesh, top_k=K)

        valid_keys = [k for k in candidates if k in mesh.active_simplices]

        new_points = []

        for key in valid_keys:

            vertices = mesh.vertices_of(key)

            new_points.append(mesh.centroid(key))

            for i, j in pairs:

                new_points.append((vertices[i] + vertices[j]) / 2.0)

        points = np.array(new_points, dtype=float)

        deleted, added = mesh.retraingulate_with_vertices(points)

        # Deterministic ordering so vmap results align with zip

        new = sorted(added - deleted)

        if new:

            checks = jax.vmap(checker)(
                jnp.array([mesh.vertices_of(key) for key in new])
            )

            mesh.evaluate_simplices(
                new, transition_kernel=transition_kernel, epsilon=epsilon
            )

            for new_key, ok in zip(new, checks):

                if not ok:

                    mesh.mark_inactive(new_key)

        # 2. Check stats on the updated mesh

        active_stats = mesh.all_stats()

        all_safe = all(s.ucb <= 0 for s in active_stats.values())

        any_unsafe = any(s.lcb > 0 for s in active_stats.values())

        if all_safe:

            if printing:

                print(
                    f"[UNDER-APPROX] ✓ Verified at iteration {iteration} "
                    f"({len(mesh.active_simplices)} simplices, "
                    f"{time.time() - build_start:.2f}s)"
                )

                _plot_state(hull, mesh)

            return AlgorithmResult(mesh=mesh, status=True, iteration=iteration)

        if any_unsafe:

            unsafe_keys = [k for k, s in active_stats.items() if s.lcb > 0]

            violations = _try_refute(
                mesh,
                unsafe_keys,
                refuter,
                true_certificate,
                transition_kernel,
                epsilon,
                base_key=jrn.key(42),
            )

            if violations:

                if printing:

                    print(
                        f"[UNDER-APPROX] ✗ Found {len(violations)} violations at iteration {iteration}"
                    )

                    _plot_state(hull, mesh)

                return AlgorithmResult(
                    status=False, mesh=mesh, violation=violations, iteration=iteration
                )

            if printing:

                print(
                    f"[UNDER-APPROX] Refuter found nothing; refining {len(unsafe_keys)} unsafe simplices"
                )

            for key in unsafe_keys:

                if key in mesh.active_simplices:

                    mesh.refine_simplex(key, checker=lambda v: checker(v).item())

        if printing and iteration % print_every == 0:

            print(
                f"[UNDER-APPROX] Iter {iteration}: {len(mesh.active_simplices)} simplices"
            )

        if printing and (iteration + 1) % plot_every == 0:

            _plot_state(hull, mesh)

    return AlgorithmResult(mesh=mesh, iteration=max_iterations)


# ---------------------------------------------------------------------------

# Strategy 2: Sound local Lipschitz (reachable-set BFS)

# ---------------------------------------------------------------------------


def verify_mesh_sound_local(
    initial_vertices: np.ndarray,
    true_certificate: Callable[[np.ndarray], float],
    transition_kernel: Callable[[jnp.ndarray, jnp.ndarray], np.ndarray],
    checker: Callable[[np.ndarray], np.ndarray],
    refuter: Callable[[Callable], Optional[np.ndarray]],
    dynamics_lipschitz: float,
    epsilon: float = 0.0,
    max_iterations: int = 20000,
    printing: bool = True,
    print_every: int = 10,
    plot_every: int = 100,
) -> AlgorithmResult:

    build_start = time.time()

    mesh = TrackingAdaptiveMesh(
        true_certificate=true_certificate,
        dynamics_lipschitz=dynamics_lipschitz,
        points=initial_vertices,
    )

    mesh.evaluate_simplices(
        mesh.simplices, transition_kernel=transition_kernel, epsilon=epsilon
    )

    if printing:

        print(f"[SOUND-LOCAL] Initial mesh: {len(mesh.active_simplices)} simplices")

    pairs = list(combinations(range(mesh._dim + 1), 2))

    # --- Main loop (sound local Lipschitz) ---------------------------------

    for iteration in range(max_iterations):

        # 1. Refine: insert centroid + edge midpoints for all active candidates

        candidates = _top_k_by_ucb(mesh, top_k=None)

        valid_keys = [k for k in candidates if k in mesh.active_simplices]

        new_points = []

        for key in valid_keys:

            vertices = mesh.vertices_of(key)

            new_points.append(mesh.centroid(key))

            for i, j in pairs:

                new_points.append((vertices[i] + vertices[j]) / 2.0)

        points = np.array(new_points, dtype=float)

        deleted, added = mesh.retraingulate_with_vertices(points)

        # Deterministic ordering so vmap results align with zip

        new = sorted(added - deleted)

        if new:

            checks = jax.vmap(checker)(
                jnp.array([mesh.vertices_of(key) for key in new])
            )

            mesh.evaluate_simplices(
                new, transition_kernel=transition_kernel, epsilon=epsilon
            )

            for new_key, ok in zip(new, checks):

                if not ok:

                    mesh.mark_inactive(new_key)

        # 2. Check sound stats on the updated mesh

        active_stats = mesh.sound_verify_mesh_kdtree(
            transition_kernel, printing=printing
        )

        all_safe = all(s.ucb <= 0 for s in active_stats.values())

        any_unsafe = any(s.lcb > 0 for s in active_stats.values())

        if all_safe:

            if printing:

                print(f"[SOUND-LOCAL] ✓ Verified at iteration {iteration}")

            return AlgorithmResult(status=True, mesh=mesh, iteration=iteration)

        if any_unsafe:

            unsafe_keys = [k for k, s in active_stats.items() if s.lcb > 0]

            violations = _try_refute(
                mesh,
                unsafe_keys,
                refuter,
                true_certificate,
                transition_kernel,
                epsilon,
                base_key=jrn.key(42),
            )

            if violations:

                if printing:

                    print(
                        f"[SOUND-LOCAL] ✗ Found {len(violations)} violations at iteration {iteration}"
                    )

                return AlgorithmResult(
                    status=False, mesh=mesh, violation=violations, iteration=iteration
                )

            if printing:

                print(
                    f"[SOUND-LOCAL] Refuter found nothing; refining {len(unsafe_keys)} unsafe simplices"
                )

        if printing and iteration % print_every == 0:

            print(
                f"[SOUND-LOCAL] Iter {iteration}: {len(mesh.active_simplices)} simplices"
            )

    return AlgorithmResult(mesh=mesh, iteration=max_iterations)


# ---------------------------------------------------------------------------
# Strategy 3: Sound global Lipschitz
# ---------------------------------------------------------------------------


def build_mesh_sound_global(
    initial_vertices: np.ndarray,
    true_certificate: Callable[[np.ndarray], float],
    transition_kernel: Callable[[jnp.ndarray, jnp.ndarray], np.ndarray],
    checker: Callable[[np.ndarray], np.ndarray],
    refuter: Callable[[Callable], Optional[np.ndarray]],
    dynamics_lipschitz: float,
    epsilon: float = 0.0,
    K: int = 1,
    max_iterations: int = 20000,
    npoints_warmstart: int = 10,
    printing: bool = True,
    print_every: int = 10,
    plot_every: int = 100,
    warm_start_mesh: Optional[TrackingAdaptiveMesh] = None,
) -> AlgorithmResult:
    """
    Build a mesh using the **sound global** Lipschitz strategy.

    A single global Lipschitz constant is derived from the maximum gradient
    norm across all active simplices and applied uniformly:

        L_global = (L_dyn + 1) * max_{σ active} ||∇V||_{σ}

    This is always sound (conservative) and cheaper than the local BFS
    approach, but may require more refinement because the bound is coarser.

    Parameters mirror ``build_mesh_under_approx``.
    """
    build_start = time.time()

    mesh = _build_initial_mesh(
        initial_vertices,
        true_certificate,
        transition_kernel,
        dynamics_lipschitz,
        checker,
        epsilon,
        npoints_warmstart,
        printing,
        warm_start_mesh,
    )

    # --- Heuristic warm-up phase -------------------------------------------
    for iteration in range(max_iterations):

        candidates = _top_k_by_ucb(mesh, K)
        if not candidates:
            break

        for key in list(candidates):
            if key not in mesh.active_simplices:
                continue
            _, added = mesh.refine_simplex(key, checker=lambda v: checker(v).item())
            for new_key in sorted(added):
                if new_key in mesh.active_simplices:
                    mesh.sample_simplex(
                        new_key, transition_kernel=transition_kernel, epsilon=epsilon
                    )

        active_stats = mesh.all_stats()
        all_safe = all(s.ucb <= 0 for s in active_stats.values())
        any_unsafe = any(s.lcb > 0 for s in active_stats.values())

        if all_safe:
            if printing:
                print(
                    f"\n[SOUND-GLOBAL] Heuristic phase passed at iteration {iteration}; starting sound check."
                )
            break

        if any_unsafe:
            unsafe_keys = [k for k, s in active_stats.items() if s.lcb > 0]
            violations = _try_refute(
                mesh, unsafe_keys, refuter, true_certificate, transition_kernel, epsilon
            )
            if violations:
                if printing:
                    print(
                        f"[SOUND-GLOBAL] Heuristic phase found {len(violations)} violations"
                    )
                return AlgorithmResult(
                    status=False, mesh=mesh, violation=violations, iteration=iteration
                )

            if printing:
                print(
                    f"[SOUND-GLOBAL] Refuter found nothing (heuristic); refining {len(unsafe_keys)} simplices"
                )
            for key in unsafe_keys:
                if key in mesh.active_simplices:
                    mesh.refine_simplex(key, checker=lambda v: checker(v).item())

        if printing and iteration % print_every == 0:
            print(
                f"[SOUND-GLOBAL] Heuristic iter {iteration}: {len(mesh.active_simplices)} simplices"
            )

    # --- Sound verification phase (global Lipschitz) -----------------------
    sound_iteration = 0
    while True:
        if printing:
            print(f"\n[SOUND-GLOBAL] Sound iteration {sound_iteration}")

        global_stats = mesh.global_verify_mesh()
        if printing:
            print(f"[SOUND-GLOBAL] Bounds computed for {len(global_stats)} simplices")

        sound_safe = [
            k
            for k, s in global_stats.items()
            if s.lcb <= 0 and s.disc_error != float("inf")
        ]
        sound_unsafe = [k for k, s in global_stats.items() if s.lcb > 0]
        sound_unknown = [
            k for k, s in global_stats.items() if not np.isfinite(s.disc_error)
        ]

        if printing:
            print(
                f"[SOUND-GLOBAL] {len(sound_safe)} safe, {len(sound_unsafe)} unsafe, {len(sound_unknown)} unknown"
            )

        if not sound_unsafe and not sound_unknown:
            build_time = time.time() - build_start
            if printing:
                print(
                    "[SOUND-GLOBAL] ✓ Verification PASSED — all simplices are sound-safe"
                )
                print(f"Build time: {build_time:.2f}s")
                _plot_state(
                    mesh,
                )
            return AlgorithmResult(status=True, mesh=mesh, iteration=sound_iteration)

        if sound_unsafe:
            if printing:
                print(
                    f"[SOUND-GLOBAL] Found {len(sound_unsafe)} sound-unsafe simplices"
                )
            violations = _try_refute(
                mesh,
                sound_unsafe,
                refuter,
                true_certificate,
                transition_kernel,
                epsilon,
            )
            if violations:
                if printing:
                    print(f"[SOUND-GLOBAL] ✗ Found {len(violations)} counterexamples")
                    _plot_state(
                        mesh,
                    )
                return AlgorithmResult(
                    status=False,
                    mesh=mesh,
                    violation=violations,
                    iteration=sound_iteration,
                )

            if printing:
                print(
                    f"[SOUND-GLOBAL] Refuter found nothing; refining {len(sound_unsafe)} unsafe simplices"
                )
            for key in sound_unsafe:
                if key in mesh.active_simplices:
                    _, added = mesh.refine_simplex(
                        key, checker=lambda v: checker(v).item()
                    )
                    for new_key in sorted(added):
                        if new_key in mesh.active_simplices:
                            mesh.sample_simplex(
                                new_key,
                                transition_kernel=transition_kernel,
                                epsilon=epsilon,
                            )

        if sound_unknown:
            if printing:
                print(
                    f"[SOUND-GLOBAL] Found {len(sound_unknown)} sound-unknown simplices; refining"
                )
            for key in sound_unknown:
                if key in mesh.active_simplices:
                    _, added = mesh.refine_simplex(
                        key, checker=lambda v: checker(v).item()
                    )
                    for new_key in sorted(added):
                        if new_key in mesh.active_simplices:
                            mesh.sample_simplex(
                                new_key,
                                transition_kernel=transition_kernel,
                                epsilon=epsilon,
                            )

        sound_iteration += 1
        if printing:
            print(
                f"[SOUND-GLOBAL] After refinement: {len(mesh.active_simplices)} total simplices"
            )

    return AlgorithmResult(mesh=mesh, iteration=sound_iteration)


# ---------------------------------------------------------------------------
# Plotting helper (unchanged)
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt
from src.plotting.tilings import plot_mesh_ucb, plot_mesh_lcb  # noqa: F401
from src.plotting.heatmaps import plot_heatmap  # noqa: F401
from src.plotting.utils import create_figure_layout  # noqa: F401


def _plot_state(
    hull,
    mesh: TrackingAdaptiveMesh,
) -> None:
    import math

    fig, [[hist_ucb, hist_lcb, hist_lip], [ax_ucb, ax_lcb, ax_heat]] = (
        create_figure_layout([[None, None, None], [None, None, None]], figsize=(14, 10))
    )

    ucb_vals = [v.ucb for v in mesh.all_stats().values() if math.isfinite(v.ucb)]
    lcb_vals = [v.lcb for v in mesh.all_stats().values() if math.isfinite(v.lcb)]
    lip_vals = [
        v.lipschitz for v in mesh.all_stats().values() if math.isfinite(v.lipschitz)
    ]

    hist_ucb.hist(ucb_vals, bins=100)
    hist_ucb.set_title("UCB Distribution")
    hist_ucb.set_xlabel("UCB")
    hist_ucb.set_ylabel("Frequency")

    hist_lcb.hist(lcb_vals, bins=100)
    hist_lcb.set_title("LCB Distribution")
    hist_lcb.set_xlabel("LCB")
    hist_lcb.set_ylabel("Frequency")

    hist_lip.hist(lip_vals, bins=100)
    hist_lip.set_title("Lipschitz Constant Distribution")
    hist_lip.set_xlabel("Lipschitz")
    hist_lip.set_ylabel("Frequency")

    plot_mesh_ucb(mesh, ax=ax_ucb)
    plot_mesh_lcb(mesh, ax=ax_lcb)

    hull_pts = hull.points[hull.vertices]

    min_xy = np.min(hull_pts, axis=0).tolist()
    max_xy = np.max(hull_pts, axis=0).tolist()

    bbox = zip(min_xy, max_xy)

    plot_heatmap(mesh.certificate, bbox, ax=ax_heat)
    plt.show()
