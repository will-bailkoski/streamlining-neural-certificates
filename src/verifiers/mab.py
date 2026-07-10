"""
Multi-armed-bandit (UCB) sample-based verification engine.

A *sound* sample-only verifier. The domain is covered by axis-aligned boxes;
each box keeps a confidence interval on the worst-case drift built from (a) a
statistical term — a cell-wise STITCHED EMPIRICAL-BERNSTEIN radius (predictable
empirical variance, valid for all sample counts simultaneously) with an
opened-cell confidence allocation delta_C = 6 delta/(pi^2 j^2) that union-bounds
the TOTAL failure probability across every cell the search ever opens — and
(b) a discretisation term (the box diameter times a rigorous Lipschitz bound on
the drift). A box is proven safe when its UCB <= 0 and flagged as a
counterexample when its LCB > 0; boxes are split where discretisation error
dominates. The discretisation term uses a *true* Lipschitz upper bound, so a
safe verdict holds with probability >= 1 - significance GLOBALLY (not per box).
See the internal notes "Cell-wise stitched empirical-Bernstein bounds" for the
exact bound and constants (eta=2, m0=1, s=1.4).

It checks the SAME canonical drift as every other engine,

    g(x) = E_w[ V(f(x, w)) ] - V(x) + epsilon,    over  domain \\ equilibrium,

so it slots into `compare_verifiers` / the CEGIS loop with no special casing.

Rigorous drift-Lipschitz bound used for the discretisation term:

    Lip(g) <= Lip(V) * Lip_x(f) + Lip(V) = Lip(V) * (lip_f + 1),

with Lip(V) the spectral-norm product of the certificate weights and `env.lip_f`
the one-step state Lipschitz of the dynamics.

The heavy lifting (the JAX-native box state machine) lives in
`src.verifiers._mab_search.sample_efficient_method`; this module only
adapts it to the `(env, spec, params, epsilon, key) -> VerifierResult`
interface.
"""

from __future__ import annotations
import time
import itertools
import numpy as np

import jax.numpy as jnp
import jax.random as jrn

from src.verifiers.base import VerifierResult, register
from src.verifiers.drift import make_drift
from src.benchmarks.convex_sets import Hypersphere, Ellipsoid
from src.verifiers._mab_search import sample_efficient_method


def _uniform_grid(bounds: np.ndarray, m: int) -> np.ndarray:
    """Partition the bounding box into m^d axis-aligned cells. bounds: (d, 2)."""
    d = bounds.shape[0]
    cells_per_dim = []
    for i in range(d):
        edges = np.linspace(bounds[i, 0], bounds[i, 1], m + 1)
        cells_per_dim.append(np.stack([edges[:-1], edges[1:]], axis=1))  # (m, 2)
    grids = np.meshgrid(*[np.arange(m) for _ in range(d)], indexing="ij")
    idxs = np.stack([g.ravel() for g in grids], axis=1)  # (m^d, d)
    boxes = np.zeros((idxs.shape[0], d, 2), dtype=np.float64)
    for j in range(d):
        boxes[:, j, :] = cells_per_dim[j][idxs[:, j]]
    return boxes


def _ellipsoid_quad_bounds(M, c, lo, hi):
    """Sound interval [low, high] on ``(x-c)^T M (x-c)`` for x in each box [lo, hi].

    Interval arithmetic over the quadratic form (correlations dropped, so it is a
    valid over-approximation): true_min >= low and true_max <= high per box.
    lo, hi: (B, d)  ->  low, high: (B,).
    """
    d = lo.shape[1]
    dlo, dhi = lo - c, hi - c  # per-dim offset intervals [dlo_i, dhi_i]
    low = np.zeros(lo.shape[0])
    high = np.zeros(lo.shape[0])
    for i in range(d):
        for j in range(d):
            m = float(M[i, j])
            if m == 0.0:
                continue
            if i == j:  # square interval [0 or min^2, max^2]
                spans0 = (dlo[:, i] <= 0) & (dhi[:, i] >= 0)
                a2, b2 = dlo[:, i] ** 2, dhi[:, i] ** 2
                p_lo = np.where(spans0, 0.0, np.minimum(a2, b2))
                p_hi = np.maximum(a2, b2)
            else:  # product of two intervals
                prods = np.stack([dlo[:, i] * dlo[:, j], dlo[:, i] * dhi[:, j],
                                  dhi[:, i] * dlo[:, j], dhi[:, i] * dhi[:, j]], axis=1)
                p_lo, p_hi = prods.min(axis=1), prods.max(axis=1)
            if m > 0:
                low += m * p_lo
                high += m * p_hi
            else:
                low += m * p_hi
                high += m * p_lo
    return low, high


def _make_checker(env):
    """(B, d, 2) -> (B,) bool: keep a box if it can contain a valid state.

    Keep a box unless it is provably (a) fully OUTSIDE the domain or (b) fully
    INSIDE the equilibrium — in both cases nothing there constrains the drift.
    Both tests are SOUND (never drop a box that touches domain\\eq): for an
    ellipsoid they use interval bounds on the quadratic; for a sphere/box the
    exact nearest/farthest-corner tests. Handling ellipsoid DOMAINS is essential
    for the non-normal envs (van der Pol, pendulum) whose invariant set is an
    ellipsoid — the box grid's out-of-ellipsoid corner cells would otherwise get
    zero valid samples and drive an unbounded split cascade.

    Pure NumPy on purpose: runs on the host between XLA dispatches, on the live-box
    arrays that already live in NumPy, never differentiated (a jitted version would
    recompile on every distinct batch size and add a host<->device round trip).
    """
    dom, eq = env.domain, env.equilibrium

    # -- domain: keep boxes that MIGHT intersect the domain --
    if isinstance(dom, Ellipsoid):
        dc, dM = np.asarray(dom.center, float), np.asarray(dom.M, float)

        def domain_keep(lo, hi):
            low, _ = _ellipsoid_quad_bounds(dM, dc, lo, hi)
            return low <= 1.0
    else:  # HyperRectangle (or anything with a box `bounds`)
        b = np.asarray(dom.bounds, float)
        dl, dh = b[:, 0], b[:, 1]

        def domain_keep(lo, hi):
            return np.all((lo <= dh) & (hi >= dl), axis=-1)

    # -- equilibrium: drop boxes proven ENTIRELY inside it (condition not required) --
    if isinstance(eq, Hypersphere):
        ec, er2 = np.asarray(eq.center, float), float(eq.radius) ** 2

        def eq_drop(lo, hi):
            far = np.maximum(np.abs(lo - ec), np.abs(hi - ec))
            return np.sum(far ** 2, axis=-1) <= er2
    elif isinstance(eq, Ellipsoid):
        ec, eM = np.asarray(eq.center, float), np.asarray(eq.M, float)

        def eq_drop(lo, hi):
            _, high = _ellipsoid_quad_bounds(eM, ec, lo, hi)
            return high <= 1.0
    else:  # EmptySet / other: nothing to exclude

        def eq_drop(lo, hi):
            return np.zeros(lo.shape[0], dtype=bool)

    def checker(boxes):
        boxes = np.asarray(boxes)
        if boxes.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        lo, hi = boxes[..., 0], boxes[..., 1]
        return domain_keep(lo, hi) & ~eq_drop(lo, hi)

    return checker


def _make_point_validity(env):
    """Jittable (N, d) -> (N,) bool: is each point in domain \\ equilibrium?

    Points outside the domain or inside the equilibrium do not constrain the
    drift, so the mab search masks them out of every box's statistics. Essential
    for (a) continuous-noise envs, where the drift is inherently positive inside
    the noise-floor equilibrium, and (b) ellipsoid-domain envs, whose bounding-box
    grid samples points outside the true (ellipsoidal) domain.

    Membership is inlined as BATCH jnp ops (broadcast over the N points) rather
    than `jax.vmap(set.contains)`: vmapping the single-point method inside the
    already-vmapped+jitted sample kernel is ~100x slower (it would not fuse).
    """
    dom, eq = env.domain, env.equilibrium

    if isinstance(dom, Ellipsoid):
        dc, dM = jnp.asarray(dom.center), jnp.asarray(dom.M)

        def in_domain(x):  # (N, d) -> (N,)
            dq = x - dc
            return jnp.einsum("ni,ij,nj->n", dq, dM, dq) <= 1.0
    else:  # HyperRectangle (or any box-bounded domain)
        b = jnp.asarray(np.asarray(dom.bounds))
        dl, dh = b[:, 0], b[:, 1]

        def in_domain(x):
            return jnp.all((x >= dl) & (x <= dh), axis=-1)

    if isinstance(eq, Hypersphere):
        ec, er2 = jnp.asarray(eq.center), float(eq.radius) ** 2

        def in_eq(x):
            d = x - ec
            return jnp.sum(d * d, axis=-1) < er2
    elif isinstance(eq, Ellipsoid):
        ec, eM = jnp.asarray(eq.center), jnp.asarray(eq.M)

        def in_eq(x):
            dq = x - ec
            return jnp.einsum("ni,ij,nj->n", dq, eM, dq) < 1.0
    else:  # EmptySet / other
        in_eq = None

    def valid(x):  # x: (N, d)
        v = in_domain(x)
        return v if in_eq is None else (v & ~in_eq(x))

    return valid


def find_counterexample(
    env,
    spec,
    params,
    epsilon: float,
    key=None,
    mc_samples: int = 1,
    grid_per_dim: int = 1,
    significance: float = 0.05,
    range_bound: float = 1.0,
    max_boxes: int = 2**18,
    max_samples: int = 2_000_000,
    timeout: float | None = None,  # wall-clock budget (s); None = run to a verdict
    k: int = 64,  # boxes processed per iteration (batched UCB selection)
    sample_per_select: int = 64,  # samples drawn per selected box per iteration
    lipschitz_method: str = "lipbab",  # "lipbab" (tight, default) | "spectral" (loose)
    lipbab_timeout: float = 20.0,
    progress_interval: int = 0,  # >0: one-line heartbeat every N search iterations
    record: bool | int = False,  # opt-in: per-iteration trace + box frames in stats
    **hp,
):
    if key is None:
        key = jrn.key(0)

    V = lambda z: spec.forward(params, z)
    single, _, _ = make_drift(env, V, epsilon, n=mc_samples)

    # reward_fn(x, key) -> one-sample drift at x; vmapped/diffed inside the engine.
    def reward_fn(x, k):
        return single(x, k)

    # Tighter L_V => smaller discretisation term => fewer box splits. LipBaB on the
    # raw net composed with the output activation's Lipschitz (e.g. hard-sigmoid 1/6).
    from src.verifiers.lipbab import lipschitz_v_bound

    lip_v = float(lipschitz_v_bound(spec, params, env.domain.bounds,
                                    method=lipschitz_method, timeout=lipbab_timeout))
    reward_lipschitz = lip_v * (float(env.lip_f) + 1.0)

    checker = _make_checker(env)
    valid_fn = _make_point_validity(env)
    initial_gridding = _uniform_grid(np.asarray(env.domain.bounds), grid_per_dim)

    # record=True -> default frame cadence; record=<int> -> a frame every <int>
    # iterations. The sink is caller-owned so partial traces survive a budget
    # RuntimeError (the run that dies mid-way is often the one worth plotting).
    sink = {} if record else None

    t0 = time.perf_counter()
    try:
        decision, cex_box, n_iter, n_samp = sample_efficient_method(
            initial_gridding,
            reward_fn,
            checker,
            reward_lipschitz,
            key,
            significance,
            k=k,
            M=range_bound,
            # a.s. reward range for the stitched empirical-Bernstein bound:
            # V in [0, range_bound] => single-draw drift V(f(x,w)) - V(x) + eps
            # lies in [eps - range_bound, eps + range_bound]
            reward_range=(epsilon - range_bound, epsilon + range_bound),
            max_boxes=max_boxes,
            max_samples=max_samples,
            timeout=timeout,
            sample_per_select=sample_per_select,
            valid_fn=valid_fn,
            printing=False,
            progress_interval=progress_interval,
            record_sink=sink,
            **({"record_every": int(record)} if isinstance(record, int)
               and not isinstance(record, bool) and record > 1 else {}),
        )
    except RuntimeError as e:
        # box/sample/time budget exhausted before a verdict (keep stats uniform with
        # the decided paths so callers can always read lip_v / reward_lipschitz).
        # "hit" names WHICH budget fired — the runners surface it as the
        # inconclusive reason; anything unrecognised (e.g. a swallowed
        # XlaRuntimeError) stays hit="error" with the message in "error".
        msg = str(e)
        hit = next((h for h in ("max_boxes", "max_samples", "timeout") if h in msg),
                   "error")
        stats = {
            "time": time.perf_counter() - t0,
            "hit": hit,
            "error": msg,
            "reward_lipschitz": reward_lipschitz,
            "lip_v": lip_v,
            "lipschitz_method": lipschitz_method,
        }
        if sink:
            stats.update(sink)
        return VerifierResult(verified=None, stats=stats, engine="mab")

    elapsed = time.perf_counter() - t0
    stats = {
        "time": elapsed,
        "iterations": int(n_iter),
        "samples": int(n_samp),
        "reward_lipschitz": reward_lipschitz,
        "lip_v": lip_v,
        "lipschitz_method": lipschitz_method,
    }
    if sink:
        stats.update(sink)

    if decision:  # all boxes proven safe -> verified
        return VerifierResult(verified=True, violation=None, stats=stats, engine="mab")

    # counterexample box found: report its centre + the box itself (re-checked by
    # the caller; the box is the region the CEGIS fairness control samples within)
    cex = np.asarray(cex_box, dtype=np.float64)
    centre = 0.5 * (cex[:, 0] + cex[:, 1])
    return VerifierResult(
        verified=False, violation=centre[None, :], region=cex, stats=stats, engine="mab"
    )


class _MABEngine:
    name = "mab"
    find_counterexample = staticmethod(find_counterexample)


register("mab", _MABEngine())
