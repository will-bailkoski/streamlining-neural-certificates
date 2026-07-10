"""
A fully JAX-native implementation of the MAB verification algorithm.
Contains two versions of the algorithm, a precise and memory efficient method for experiments with requiring precise sample counts, and another optimised for time efficiency
All mutable state lives in a fixed-size array of shape (max_boxes, ...).
Free slots are tracked via a simple integer free-list so that pruned boxes can
be reused and the array never overflows unexpectedly.

"""

from __future__ import annotations
import time
from jax import vmap, grad, jit
import jax.numpy as jnp
import jax.random as jrn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors

# =============================================================================
# Free-list helpers  (pure NumPy, no JAX)
# =============================================================================


def _freelist_alloc(free_stack: list) -> int:
    """Pop one free box slot.  Raises RuntimeError if the array is full."""
    if not free_stack:
        raise RuntimeError(
            "max_boxes exceeded: the fixed-size box array is full. "
            "Increase max_boxes or reduce the problem size."
        )
    return free_stack.pop()


def _freelist_release(free_stack: list, idx: int) -> None:
    free_stack.append(idx)


# =============================================================================
# Confidence-bound helpers — cell-wise STITCHED EMPIRICAL-BERNSTEIN bounds
# (internal notes, 2026-07: "Cell-wise stitched empirical-Bernstein bounds").
# Pure NumPy: they run once per iteration on <= k live-box rows, and depend on
# per-cell state (predictable variance, opening index) held in NumPy anyway.
# =============================================================================

# Fixed stitching parameters (eta=2, m0=1, s=1.4) and their derived constants:
#   k1(eta) = (eta^{1/4} + eta^{-1/4}) / sqrt(2),  k2(eta) = (sqrt(eta)+1)/2.
_EB_S = 1.4
_EB_K1SQ = 2.0607
_EB_K2SQ = 1.4571
_EB_K2 = 1.2071
# opened-cell confidence allocation: delta_C = 6 delta_stat / (pi^2 j(C)^2),
# alpha_C = delta_C / 2; ell's constant folds zeta(1.4)/log^{1.4}(2) with it:
#   log( zeta(s) / (alpha_C log^s eta) ) = log( 17.0672 j^2 / delta_stat ).
_EB_OPENED_CONST = 17.0672


def _diameter_np(bounds):
    """Euclidean diameter per box.  bounds: (B, d, 2) -> (B,)."""
    return np.linalg.norm(bounds[:, :, 1] - bounds[:, :, 0], axis=1)


def _eb_radius(count, pvar, open_idx, c_range, delta_stat):
    """Stitched empirical-Bernstein statistical radius EB_C(n), vectorised.

    count:    (B,) valid-sample counts n (>= 1 for a meaningful bound)
    pvar:     (B,) predictable empirical variance sums \\hat V_C(n)
    open_idx: (B,) opening index j(C) (1-based, over ALL cells ever activated)
    c_range:  scalar reward range c_C = b_C - a_C (a.s. bound on the reward)
    delta_stat: total statistical failure probability across all cells/counts.

    EB = [ sqrt(k1^2 u ell + k2^2 c^2 ell^2) + k2 c ell ] / n,
    u = max(pvar, 1),  ell = s loglog(2u) + log(17.0672 j^2 / delta_stat).
    Valid simultaneously for all opened cells, all n >= 1, w.p. 1 - delta_stat
    (sum_j 6/(pi^2 j^2) = 1), so safe/violation verdicts inherit the SAME
    global confidence no matter how many cells the search opens.
    """
    n = np.maximum(count, 1.0)
    u = np.maximum(pvar, 1.0)
    ell = (_EB_S * np.log(np.log(2.0 * u))
           + np.log(_EB_OPENED_CONST * open_idx.astype(np.float64) ** 2 / delta_stat))
    return (np.sqrt(_EB_K1SQ * u * ell + _EB_K2SQ * (c_range * ell) ** 2)
            + _EB_K2 * c_range * ell) / n


def _pvar_update(n0, m0, v0, z):
    """Predictable-centre incremental update for ONE cell over a value batch.

    Starting from n0 samples with empirical mean m0 and predictable variance
    sum v0, fold in the ordered batch z (1-D): the centre for each z_k is the
    empirical mean BEFORE observing it (vectorised via prefix sums).
    Returns (n1, m1, v1).
    """
    B = z.shape[0]
    if B == 0:
        return n0, m0, v0
    ks = np.arange(1, B + 1, dtype=np.float64)
    prefix_mean = (n0 * m0 + np.cumsum(z, dtype=np.float64)) / (n0 + ks)
    centres = np.concatenate([[m0], prefix_mean[:-1]])
    return n0 + B, float(prefix_mean[-1]), v0 + float(np.sum((z - centres) ** 2))


# =============================================================================
# Sampling kernel
# =============================================================================


def _make_sample_kernel(reward_fn, n_samples, valid_fn=None, valid_oversample=8):
    """Return a vmapped, JIT-compiled kernel closed over reward_fn.

    Returns shapes:
        x:     (k, samples, d)
        r:     (k, samples)
        valid: (k, samples)   1.0 if the point lies in domain\\equilibrium, else 0.0

    `valid_fn(x) -> (samples,)` flags which sampled points actually constrain the
    drift; the caller masks the rest out of the box statistics. Defaults to
    "everything valid" so callers that do not pass it keep the old behaviour.

    `valid_oversample`: draw this many candidate points per slot and keep the
    FIRST valid one (fallback: the first candidate, masked as invalid if none
    hit). Boxes whose valid (domain\\eq) sliver is a small fraction p of their
    volume otherwise collect only ~p*n_samples useful draws per visit — the
    boxes pinned on a curved eq/domain boundary are exactly the last to resolve.
    Each kept valid point is still uniform over the box's valid region (every
    candidate is iid uniform; keeping the first valid one conditions on the
    subset without biasing within it), so the box statistics are unchanged in
    distribution — only the valid yield per visit improves (~min(T, 1/p)x).
    Membership tests are cheap batch ops; reward_fn evals stay at n_samples.
    """
    if valid_fn is None:
        valid_fn = lambda x: jnp.ones(x.shape[0])
        valid_oversample = 1

    def kernel(key, bounds):
        key_sample, key_reward = jrn.split(key)
        lo, hi = bounds[:, 0], bounds[:, 1]
        d = lo.shape[0]
        T = valid_oversample
        xc = jrn.uniform(key_sample, shape=(n_samples * T, d), minval=lo, maxval=hi)
        vc = valid_fn(xc).reshape(n_samples, T)
        xc = xc.reshape(n_samples, T, d)
        first = jnp.argmax(vc, axis=1)              # first valid slot (0 if none)
        rows = jnp.arange(n_samples)
        x = xc[rows, first]
        r = reward_fn(x, key_reward)
        valid = vc[rows, first].astype(r.dtype)
        return x, r, valid

    # jit the whole batched kernel: the per-iteration box count is padded to a
    # fixed k, so this compiles ONCE and then runs as a single fused XLA kernel.
    # Without the jit the kernel executes eagerly — env.step's split/bernoulli/
    # matmul get re-traced and dispatched op-by-op every iteration (~100x slower).
    return jit(vmap(kernel))


# =============================================================================
# Split helpers  (pure NumPy — these run on the host between XLA dispatches;
# routing a trivial bisection through jit would add a host->device->host round
# trip per split AND recompile every time the batch size changes).
# =============================================================================


def _split_bounds(bounds, split_dim, split_val):
    """
    Split a single box along split_dim at split_val.  bounds: (d, 2).
    Returns (child_lo_bounds, child_hi_bounds), each (d, 2).
    """
    child_lo = bounds.copy()
    child_hi = bounds.copy()
    child_lo[split_dim, 1] = split_val  # lo child keeps the lower half
    child_hi[split_dim, 0] = split_val  # hi child keeps the upper half
    return child_lo, child_hi


def _compute_split_params_batch(bounds):
    """
    For each box choose the widest dimension and split at its midpoint.
    bounds: (B, d, 2)  ->  split_dims: (B,), split_vals: (B,)
    """
    diffs = bounds[:, :, 1] - bounds[:, :, 0]
    split_dims = np.argmax(diffs, axis=1)
    row_inds = np.arange(bounds.shape[0])
    lows = bounds[row_inds, split_dims, 0]
    highs = bounds[row_inds, split_dims, 1]
    return split_dims, (lows + highs) / 2.0


# =============================================================================
# Combined box + sample state
# =============================================================================


class _State:
    """
    All mutable algorithm state in one object.

    Box arrays  -- fixed NumPy arrays indexed by slot in [0, max_boxes).
    Sample storage -- per-box chunk lists.  Each live box b owns two Python
                      lists, _xs_chunks[b] and _rs_chunks[b], each holding
                      a sequence of small NumPy arrays written during
                      append_samples calls.  The lists are concatenated into
                      one contiguous array only when samples_of() is called
                      (i.e. during a split), keeping append cost at O(1).

    Budget tracking -- _live_samples counts the total number of samples
                       across all *live* boxes.  It is incremented on
                       activate_with_samples / append_samples and decremented
                       on deactivate, so the budget is genuinely recycled when
                       boxes are pruned.

    Lipschitz tracking -- _last_x and _last_r store the most recent sample
                          for each live box, used for the empirical Lipschitz
                          lower-bound estimate.
    """

    def __init__(self, max_boxes, max_samples, n_dims, reward_mid=0.0):
        self.max_boxes = max_boxes
        self.max_samples = max_samples
        self.n_dims = n_dims
        # midpoint (a_C + b_C)/2 of the a.s. reward range — the m-hat(0)
        # convention of the stitched empirical-Bernstein bound
        self.reward_mid = float(reward_mid)

        # -- Box metadata arrays --
        self.bounds = np.zeros((max_boxes, n_dims, 2), dtype=np.float32)
        self.mean = np.full(max_boxes, self.reward_mid, dtype=np.float64)
        self.pvar = np.zeros(max_boxes, dtype=np.float64)   # predictable var sum
        self.count = np.zeros(max_boxes, dtype=np.float64)
        self.open_idx = np.zeros(max_boxes, dtype=np.int64)  # j(C), 1-based
        self._next_open = 1  # global opening counter (union-bound allocation)
        self.ucb = np.full(max_boxes, -np.inf, dtype=np.float64)
        self.lcb = np.full(max_boxes, -np.inf, dtype=np.float64)
        self.is_live = np.zeros(max_boxes, dtype=bool)

        # -- Per-box sample chunk lists (lazy: dict keyed by live slot) --
        # A preallocated list of max_boxes empty lists costs O(max_boxes) time and
        # ~2 Python objects per slot at construction (seconds + GBs for millions of
        # boxes). Only live slots ever hold samples, so key them lazily instead.
        self._xs_chunks = {}
        self._rs_chunks = {}

        # -- Last sample per box, for Lipschitz LB estimation --
        self._last_x = np.zeros((max_boxes, n_dims), dtype=np.float32)
        self._last_r = np.zeros(max_boxes, dtype=np.float32)
        self._has_last = np.zeros(max_boxes, dtype=bool)

        # Running count of samples held by all currently-live boxes.
        self._live_samples = 0

        # Geometry of boxes retired as PROVEN SAFE (UCB <= 0), kept only for the
        # 2-D tiling plot: safe cells are deactivated the moment they are proven,
        # so without this the plot loses every verified cell and shows an almost
        # empty domain. Populated only when printing (see _prune_safe record=).
        self.retired_safe = []

    # ------------------------------------------------------------------
    # Box lifecycle
    # ------------------------------------------------------------------

    def activate(self, idx, bounds):
        """Initialise a new box slot with no samples yet (fresh opening index)."""
        self.bounds[idx] = bounds
        self.mean[idx] = self.reward_mid
        self.pvar[idx] = 0.0
        self.count[idx] = 0.0
        self.open_idx[idx] = self._next_open
        self._next_open += 1
        self.ucb[idx] = np.inf
        self.lcb[idx] = -np.inf
        self.is_live[idx] = True
        self._xs_chunks[idx] = []
        self._rs_chunks[idx] = []
        self._has_last[idx] = False

    def activate_with_samples(self, idx, bounds, xs_chunk, rs_chunk):
        """
        Initialise a box slot pre-loaded with redistributed parent samples.
        Called exclusively by _split_box.

        The child's statistics are rebuilt by REPLAYING the inherited rewards in
        their original arrival order through the predictable-centre recursion
        (fresh m-hat(0) = reward midpoint): conditional on falling in the child,
        the parent's uniform draws are i.i.d. uniform on the child, so the
        replayed sequence is exactly the sample stream the stitched bound covers.
        """
        n = len(rs_chunk)
        if self._live_samples + n > self.max_samples:
            raise RuntimeError(
                f"max_samples ({self.max_samples}) exceeded in "
                f"activate_with_samples: live={self._live_samples}, "
                f"requested={n}.  Increase max_samples."
            )
        self.bounds[idx] = bounds
        _, m1, v1 = _pvar_update(0.0, self.reward_mid, 0.0,
                                 np.asarray(rs_chunk, dtype=np.float64))
        self.mean[idx] = m1
        self.pvar[idx] = v1
        self.count[idx] = float(n)
        self.open_idx[idx] = self._next_open
        self._next_open += 1
        self.ucb[idx] = np.inf
        self.lcb[idx] = -np.inf
        self.is_live[idx] = True
        self._xs_chunks[idx] = [np.asarray(xs_chunk, dtype=np.float32)]
        self._rs_chunks[idx] = [np.asarray(rs_chunk, dtype=np.float32)]
        self._live_samples += n
        if n > 0:
            self._last_x[idx] = xs_chunk[-1]
            self._last_r[idx] = float(rs_chunk[-1])
            self._has_last[idx] = True
        else:
            self._has_last[idx] = False

    def deactivate(self, idx):
        """
        Mark a box as dead and return its sample budget.
        """
        self.bounds[idx] = None
        self._live_samples -= int(self.count[idx])
        self._xs_chunks.pop(idx, None)
        self._rs_chunks.pop(idx, None)
        self.is_live[idx] = False
        self.ucb[idx] = -np.inf
        self.lcb[idx] = -np.inf
        self._has_last[idx] = False

    # ------------------------------------------------------------------
    # Sample append
    # ------------------------------------------------------------------

    def append_samples(self, indices, xs_new, rs_new, valid_new):
        """
        Append the new (x, r) pairs to each selected box and fold them into the
        predictable-centre statistics (mean + predictable variance sum) — but
        count ONLY samples in domain\\equilibrium (valid_new==1). Inside-
        equilibrium / out-of-domain points do not constrain the drift, so they
        are neither stored nor counted; the box mean therefore estimates the
        drift over the valid region only.

        xs_new: (current_k, sample_per_select, d)
        rs_new / valid_new: (current_k, sample_per_select)
        """
        vmask = valid_new.astype(bool)
        valid_counts = vmask.sum(axis=1)                 # (current_k,)
        total_new = int(valid_counts.sum())

        if self._live_samples + total_new > self.max_samples:
            raise RuntimeError(
                f"max_samples ({self.max_samples}) exceeded in append_samples: "
                f"live={self._live_samples}, requested={total_new}.  "
                f"Increase max_samples."
            )

        for i, box_idx in enumerate(indices):
            if not valid_counts[i]:
                continue  # no valid sample this round -> nothing to store/track
            xs_v = np.asarray(xs_new[i][vmask[i]], dtype=np.float32)
            rs_v = np.asarray(rs_new[i][vmask[i]], dtype=np.float32)
            self._xs_chunks[box_idx].append(xs_v)
            self._rs_chunks[box_idx].append(rs_v)
            n1, m1, v1 = _pvar_update(
                float(self.count[box_idx]), float(self.mean[box_idx]),
                float(self.pvar[box_idx]), rs_v.astype(np.float64))
            self.count[box_idx] = n1
            self.mean[box_idx] = m1
            self.pvar[box_idx] = v1
            # Track last valid sample for Lipschitz LB estimation
            self._last_x[box_idx] = xs_v[-1]
            self._last_r[box_idx] = float(rs_v[-1])
            self._has_last[box_idx] = True

        self._live_samples += total_new

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def live_indices(self):
        return np.where(self.is_live)[0]

    def samples_of(self, idx):
        """
        Return (xs, rs) as contiguous arrays for box idx.
        """
        chunks_x = self._xs_chunks.get(idx, [])
        chunks_r = self._rs_chunks.get(idx, [])
        if not chunks_x:
            return (
                np.empty((0, self.n_dims), dtype=np.float32),
                np.empty(0, dtype=np.float32),
            )
        xs = np.concatenate(chunks_x, axis=0)
        rs = np.concatenate(chunks_r, axis=0)
        return xs, rs


# =============================================================================
# Split helper  (redistributes parent samples into two child slots)
# =============================================================================


def _split_box(state, parent_idx, split_dim, split_val, lo_slot, hi_slot):
    """
    Partition parent_idx's samples between lo_slot and hi_slot and
    activate both children with the redistributed statistics.
    """
    child_lo, child_hi = _split_bounds(
        state.bounds[parent_idx], int(split_dim), float(split_val)
    )

    xs_p, rs_p = state.samples_of(parent_idx)

    if len(xs_p) == 0:
        state.activate(lo_slot, child_lo)
        state.activate(hi_slot, child_hi)
        return

    in_lo = xs_p[:, split_dim] <= split_val
    order = np.concatenate([np.where(in_lo)[0], np.where(~in_lo)[0]])
    xs_ordered = xs_p[order]
    rs_ordered = rs_p[order]
    n_lo = int(in_lo.sum())

    state.activate_with_samples(lo_slot, child_lo, xs_ordered[:n_lo], rs_ordered[:n_lo])
    state.activate_with_samples(hi_slot, child_hi, xs_ordered[n_lo:], rs_ordered[n_lo:])


# =============================================================================
# Pruning helpers  (NumPy)
# =============================================================================


def _prune_invalid(state, indices, checker, free_stack):
    if len(indices) == 0:
        return

    boxes = state.bounds[indices]
    mask = np.array(checker(boxes))

    for i, keep in enumerate(mask):
        if not keep:
            idx = int(indices[i])
            state.deactivate(idx)
            _freelist_release(free_stack, idx)

    return int(np.sum(mask))


def _prune_safe(state, free_stack, record=False):
    """Deactivate and free every live box with UCB <= 0 (proven safe).

    When `record`, stash the retired cells' bounds first so the tiling plot can
    still draw them as verified (they are gone from the live set afterwards).

    Returns (n_pruned, pruned_volume).
    """
    live = state.live_indices()
    slice = live[state.ucb[live] <= 0]
    vol = _boxes_volume(state.bounds[slice])
    if record and slice.shape[0]:
        state.retired_safe.append(state.bounds[slice].copy())
    for idx in slice:
        state.deactivate(int(idx))
        _freelist_release(free_stack, int(idx))

    return slice.shape[0], vol


def _boxes_volume(bounds: np.ndarray) -> float:
    """Total volume of an (N, d, 2) array of boxes."""
    if bounds.shape[0] == 0:
        return 0.0
    return float(np.prod(bounds[:, :, 1].astype(np.float64)
                         - bounds[:, :, 0].astype(np.float64), axis=1).sum())


# =============================================================================
# Main algorithm
# =============================================================================


def sample_efficient_method(
    initial_gridding,  # np.ndarray (N0, d, 2)
    reward_fn,
    checker,  # vmapped (los: (B,d), his: (B,d)) -> (B,) bool
    reward_lipschitz,  # float
    key,  # jax.random.PRNGKey
    significance,  # float in (0, 1): TOTAL failure prob delta_stat (union-bounded
                   # over all opened cells and sample counts via the opened-cell
                   # allocation delta_C = 6 delta / (pi^2 j^2))
    k=1,
    M=1.0,  # half-range fallback: rewards in [-M, M] when reward_range is None
    reward_range=None,  # (a, b): a.s. reward bounds — c_C = b - a, m-hat(0) = mid
    max_boxes=2**18,
    max_samples=2_000_000,
    timeout=None,  # seconds; the loop otherwise runs until a verdict or a budget
    sample_per_select=64,
    valid_fn=None,  # (samples, d) -> (samples,) 1/0 mask for domain\equilibrium
    printing=False,
    progress_interval=0,  # >0: print a cheap one-line heartbeat every N iterations
    print_interval=50,
    plot_interval=500,
    plot_dir=".",
    record_sink=None,  # dict: filled with plain-data "trace" rows + box "frames"
    record_every=25,   # frame snapshot cadence (trace rows are per-iteration)
):
    """
    Returns
    -------
    decision       : True | False
    counterexample : np.ndarray (d, 2) | None
    n_iterations   : int
    n_samples      : int  exact reward-function evaluation count
    """

    n_dims = initial_gridding.shape[1]
    n_initial = initial_gridding.shape[0]

    if n_initial > max_boxes:
        raise RuntimeError(
            f"Initial gridding ({n_initial} boxes) exceeds max_boxes ({max_boxes})."
        )

    if reward_range is None:
        reward_range = (-float(M), float(M))
    reward_range_c = float(reward_range[1] - reward_range[0])   # c_C
    reward_mid = 0.5 * (reward_range[0] + reward_range[1])      # m-hat(0)

    state = _State(max_boxes, max_samples, n_dims, reward_mid=reward_mid)
    free_stack = list(range(n_initial, max_boxes))

    for i in range(n_initial):
        state.activate(i, initial_gridding[i])

    _prune_invalid(state, np.arange(n_initial, dtype=np.int32), checker, free_stack)

    if state.live_indices().size == 0:
        return True, None, 0, 0

    batch_step = _make_sample_kernel(
        vmap(reward_fn, in_axes=(0, None)), sample_per_select, valid_fn
    )

    # ------------------------------------------------------------------
    # Optional plain-data recording (decoupled from `printing`): per-iteration
    # trace rows + periodic {los, his, status} box frames, written into the
    # caller-owned record_sink so they survive even a budget RuntimeError.
    # Powers visualisation/mab_dashboard.py and visualisation/verifier_grid.py.
    # ------------------------------------------------------------------
    rec = record_sink
    if rec is not None:
        rec["trace"] = []
        rec["frames"] = []
        rec["safe_volume"] = 0.0
        rec["initial_volume"] = _boxes_volume(state.bounds[state.live_indices()])
        _t_rec0 = time.perf_counter()

        def _rec_row(iteration, samples):
            live_ = state.live_indices()
            ucb_, lcb_ = state.ucb[live_], state.lcb[live_]
            f_ucb, f_lcb = ucb_[np.isfinite(ucb_)], lcb_[np.isfinite(lcb_)]
            rec["trace"].append(dict(
                iter=int(iteration), t=time.perf_counter() - _t_rec0,
                samples=int(samples), live=int(live_.size),
                live_volume=_boxes_volume(state.bounds[live_]),
                safe_volume=rec["safe_volume"],
                max_ucb=float(f_ucb.max()) if f_ucb.size else float("inf"),
                max_lcb=float(f_lcb.max()) if f_lcb.size else float("-inf"),
            ))

        def _rec_frame(iteration):
            live_ = state.live_indices()
            st = np.zeros(live_.size, dtype=np.int8)
            st[state.ucb[live_] <= 0] = 1
            st[state.lcb[live_] > 0] = 2
            lo = [state.bounds[live_, :, 0]]
            hi = [state.bounds[live_, :, 1]]
            sts = [st]
            if state.retired_safe:
                safe = np.concatenate(state.retired_safe, axis=0)
                lo.append(safe[:, :, 0])
                hi.append(safe[:, :, 1])
                sts.append(np.ones(safe.shape[0], dtype=np.int8))
            rec["frames"].append(dict(
                iter=int(iteration), los=np.concatenate(lo).copy(),
                his=np.concatenate(hi).copy(), status=np.concatenate(sts),
            ))

    # ------------------------------------------------------------------
    # Padding helpers
    # ------------------------------------------------------------------

    def _pad_bounds(arr):
        shortfall = k - arr.shape[0]
        if shortfall <= 0:
            return arr
        return np.concatenate(
            [arr, np.zeros((shortfall, n_dims, 2), dtype=arr.dtype)], axis=0
        )

    def _pad_1d(arr, fill=0.0):
        shortfall = k - arr.shape[0]
        if shortfall <= 0:
            return arr
        return np.concatenate([arr, np.full(shortfall, fill, dtype=arr.dtype)], axis=0)

    grad_step = vmap(
        lambda x, k: jnp.linalg.norm(grad(reward_fn, argnums=0)(x, k), ord=2)
    )

    if printing:
        import os

        os.makedirs(plot_dir, exist_ok=True)
        matplotlib.use("Agg")

        _is_2d = n_dims == 2

        if _is_2d:
            fig = plt.figure(figsize=(19, 9))
            gs = gridspec.GridSpec(
                2,
                3,
                figure=fig,
                hspace=0.42,
                wspace=0.36,
                left=0.06,
                right=0.97,
                top=0.91,
                bottom=0.09,
            )
            ax_boxes = fig.add_subplot(gs[0, 0])
            ax_ucb = fig.add_subplot(gs[0, 1])
            ax_lip = fig.add_subplot(gs[0, 2])
            ax_props = fig.add_subplot(gs[1, 0])
            ax_2d = fig.add_subplot(gs[1, 1:])
        else:
            fig = plt.figure(figsize=(16, 8))
            gs = gridspec.GridSpec(
                2,
                2,
                figure=fig,
                hspace=0.42,
                wspace=0.36,
                left=0.07,
                right=0.97,
                top=0.91,
                bottom=0.09,
            )
            ax_boxes = fig.add_subplot(gs[0, 0])
            ax_ucb = fig.add_subplot(gs[0, 1])
            ax_lip = fig.add_subplot(gs[1, 0])
            ax_props = fig.add_subplot(gs[1, 1])
            ax_2d = None

        # History containers
        history_iters = []
        history_n_boxes = []
        history_total_boxes = []  # cumulative boxes ever allocated
        history_max_ucb = []
        history_mean_ucb = []
        history_max_lcb = []
        history_verified_prop = []
        history_unknown_prop = []
        history_unsafe_prop = []
        history_lip_lb = []

    # ------------------------------------------------------------------
    # Running state for Lipschitz LB and box count
    # ------------------------------------------------------------------

    lip_lb = 0.0
    n_iterations = 0
    n_samples = 0
    master_key = key

    n_safe_pruned = 0
    _t_start = time.perf_counter()

    while True:
        live = state.live_indices()
        n_live = live.size

        # Wall-clock budget: without this the loop only ever exits via a verdict
        # or a box/sample RuntimeError — on SLURM a wall kill would leave the run
        # status=running with no stats at all, so time out INSIDE the budget.
        if timeout is not None and time.perf_counter() - _t_start > timeout:
            raise RuntimeError(
                f"timeout ({timeout:.0f}s) exceeded after {n_iterations} iterations, "
                f"{n_samples} samples, {n_live} live boxes."
            )

        if rec is not None:
            _rec_row(n_iterations, n_samples)
            if n_iterations % record_every == 0:
                _rec_frame(n_iterations)

        # Cheap heartbeat (unlike `printing` it costs no gradient pass and no
        # plot): with SLURM's unbuffered stdout this is what tells you where a
        # long verification call is when a job times out.
        if progress_interval and n_iterations % progress_interval == 0:
            f_ucb = state.ucb[live][np.isfinite(state.ucb[live])]
            f_lcb = state.lcb[live][np.isfinite(state.lcb[live])]
            ucb_s = f"{f_ucb.max():+.4f}" if f_ucb.size else "inf"
            lcb_s = f"{f_lcb.max():+.4f}" if f_lcb.size else "-inf"
            print(
                f"[mab] iter={n_iterations:7d}  samples={n_samples:12,d}  "
                f"live={n_live:9,d}  safe={n_safe_pruned:9,d}  "
                f"max_ucb={ucb_s}  max_lcb={lcb_s}",
                flush=True,
            )

        # ----------------------------------------------------------------
        # Termination: all boxes proven safe
        # ----------------------------------------------------------------

        if n_live == 0 or np.all(state.ucb[live] <= 0):
            if rec is not None:
                _rec_frame(n_iterations)
            if printing:
                print(
                    f"[Verify] DONE (safe) — iteration {n_iterations}, "
                    f"samples {n_samples}, live boxes {n_live}"
                )
                _save_plot(
                    fig,
                    (ax_boxes, ax_ucb, ax_lip, ax_props, ax_2d),
                    plot_dir,
                    n_iterations,
                    history_iters,
                    history_n_boxes,
                    history_total_boxes,
                    history_max_ucb,
                    history_mean_ucb,
                    history_max_lcb,
                    history_verified_prop,
                    history_unknown_prop,
                    history_unsafe_prop,
                    history_lip_lb,
                    reward_lipschitz,
                    state,
                    n_dims,
                )
            return True, None, n_iterations, n_samples

        # ----------------------------------------------------------------
        # Record history for plotting
        # ----------------------------------------------------------------

        if printing:
            live_ucbs = state.ucb[live]
            live_lcbs = state.lcb[live]
            finite_ucbs = live_ucbs[np.isfinite(live_ucbs)]
            finite_lcbs = live_lcbs[np.isfinite(live_lcbs)]

            total_boxes = n_live + n_safe_pruned

            if total_boxes > 0:
                v_prop = n_safe_pruned / total_boxes
                unsafe_mask = live_lcbs > 0.0
                u_prop = int(unsafe_mask.sum()) / total_boxes
            else:
                v_prop = 0.0
                u_prop = 0.0

            history_iters.append(n_iterations)
            history_n_boxes.append(n_live)
            history_total_boxes.append(total_boxes)
            history_max_ucb.append(
                float(finite_ucbs.max()) if finite_ucbs.size else np.nan
            )
            history_mean_ucb.append(
                float(finite_ucbs.mean()) if finite_ucbs.size else np.nan
            )
            history_max_lcb.append(
                float(finite_lcbs.max()) if finite_lcbs.size else np.nan
            )

            # ---> NEW: Use the updated proportions <---
            history_verified_prop.append(v_prop)
            history_unsafe_prop.append(u_prop)
            history_unknown_prop.append(1.0 - v_prop - u_prop)
            history_lip_lb.append(lip_lb)

            if n_iterations % print_interval == 0:
                max_ucb_str = (
                    f"{history_max_ucb[-1]:.4f}"
                    if np.isfinite(history_max_ucb[-1])
                    else "inf"
                )
                max_lcb_str = (
                    f"{history_max_lcb[-1]:.4f}"
                    if np.isfinite(history_max_lcb[-1])
                    else "-inf"
                )
                print(
                    f"[Verify] iter={n_iterations:6d} | samples={n_samples:9d} | "
                    f"total boxes={total_boxes:7d} | "
                    f"verified={100*history_verified_prop[-1]:5.1f}% | "
                    f"max_ucb={max_ucb_str} | max_lcb={max_lcb_str} | "
                    f"lip_lb={lip_lb:.4f}"
                )

            if n_iterations % plot_interval == 0 and n_iterations > 0:
                _save_plot(
                    fig,
                    (ax_boxes, ax_ucb, ax_lip, ax_props, ax_2d),
                    plot_dir,
                    n_iterations,
                    history_iters,
                    history_n_boxes,
                    history_total_boxes,
                    history_max_ucb,
                    history_mean_ucb,
                    history_max_lcb,
                    history_verified_prop,
                    history_unknown_prop,
                    history_unsafe_prop,
                    history_lip_lb,
                    reward_lipschitz,
                    state,
                    n_dims,
                )

        # ----------------------------------------------------------------
        # Select top-k live boxes by UCB
        # ----------------------------------------------------------------

        current_k = min(k, n_live)
        top_pos = np.argpartition(state.ucb[live], -current_k)[-current_k:]
        selected_indices = live[top_pos]

        # ----------------------------------------------------------------
        # Build padded JAX arrays and sample
        # ----------------------------------------------------------------

        sel_bounds = state.bounds[selected_indices]

        pad_bounds = jnp.array(_pad_bounds(sel_bounds))

        master_key, batch_key = jrn.split(master_key)
        subkeys = jrn.split(batch_key, k)

        x_new_jax, r_new_jax, valid_new_jax = batch_step(subkeys, pad_bounds)

        x_np = np.array(x_new_jax)[:current_k]
        r_np = np.array(r_new_jax)[:current_k]
        valid_np = np.array(valid_new_jax)[:current_k]

        # ----------------------------------------------------------------
        # Lipschitz lower-bound update (before append so _last is stale)
        # ----------------------------------------------------------------

        # The empirical Lipschitz lower bound is a DIAGNOSTIC only (it feeds the
        # plots; no verdict or bound depends on it). Computing it costs a backward
        # pass through reward_fn AND a float() host-sync every iteration, so do it
        # only when plotting. The key is still advanced so the sample stream — and
        # therefore the verdict — is byte-for-byte identical to before.
        master_key, grad_key = jrn.split(master_key)
        if printing:
            grad_subkeys = jrn.split(grad_key, k * sample_per_select)
            current_batch_max = jnp.max(
                grad_step(x_new_jax.reshape(k * sample_per_select, n_dims), grad_subkeys)
            )
            lip_lb = max(lip_lb, float(current_batch_max))

        # ----------------------------------------------------------------
        # Commit samples (valid-masked, predictable-centre statistics), then
        # compute the stitched empirical-Bernstein confidence bounds from the
        # UPDATED per-cell state. The disc term still bounds the sup over the
        # valid subset (it covers the whole box), so masking stays sound.
        # ----------------------------------------------------------------

        state.append_samples(selected_indices, x_np, r_np, valid_np)
        n_samples += int(current_k) * sample_per_select

        sel_count = state.count[selected_indices]
        stat_err_np = _eb_radius(sel_count,
                                 state.pvar[selected_indices],
                                 state.open_idx[selected_indices],
                                 reward_range_c, significance)
        disc_err_np = reward_lipschitz * _diameter_np(
            state.bounds[selected_indices].astype(np.float64))
        mean_sel = state.mean[selected_indices]
        new_ucbs_np = mean_sel + stat_err_np + disc_err_np
        new_lcbs_np = mean_sel - stat_err_np - disc_err_np

        # A box with no valid sample yet has no estimate: force UCB=inf / LCB=-inf
        # so it is never declared safe nor a counterexample — it keeps getting
        # sampled (or split) until it captures a domain\equilibrium point.
        no_data = sel_count <= 0.0
        new_ucbs_np = np.where(no_data, np.inf, new_ucbs_np)
        new_lcbs_np = np.where(no_data, -np.inf, new_lcbs_np)
        state.ucb[selected_indices] = new_ucbs_np
        state.lcb[selected_indices] = new_lcbs_np

        # ----------------------------------------------------------------
        # Termination: counterexample found
        # ----------------------------------------------------------------

        if np.any(new_lcbs_np > 0):
            violation_pos = int(np.argmax(new_lcbs_np))
            cex_box = state.bounds[selected_indices[violation_pos]]
            if rec is not None:
                _rec_row(n_iterations, n_samples)
                _rec_frame(n_iterations)
            if printing:
                print(
                    f"[Verify] COUNTEREXAMPLE found at iteration {n_iterations}, "
                    f"samples {n_samples}. Box: {cex_box}"
                )
                _save_plot(
                    fig,
                    (ax_boxes, ax_ucb, ax_lip, ax_props, ax_2d),
                    plot_dir,
                    n_iterations,
                    history_iters,
                    history_n_boxes,
                    history_total_boxes,
                    history_max_ucb,
                    history_mean_ucb,
                    history_max_lcb,
                    history_verified_prop,
                    history_unknown_prop,
                    history_unsafe_prop,
                    history_lip_lb,
                    reward_lipschitz,
                    state,
                    n_dims,
                )
            return False, cex_box.copy(), n_iterations, n_samples

        # ----------------------------------------------------------------
        # Prune newly-safe boxes (UCB <= 0)
        # ----------------------------------------------------------------

        n_new_safe, new_safe_vol = _prune_safe(state, free_stack,
                                               record=printing or rec is not None)
        n_safe_pruned += n_new_safe
        if rec is not None:
            rec["safe_volume"] += new_safe_vol
        remaining_live = state.live_indices()

        if printing and n_iterations % print_interval == 0:
            print(f"[Verify]   └─ {n_safe_pruned} safe, {remaining_live.size} live")

        if remaining_live.size == 0 or np.all(state.ucb[remaining_live] <= 0):
            if rec is not None:
                _rec_row(n_iterations, n_samples)
                _rec_frame(n_iterations)
            if printing:
                print(
                    f"[Verify] DONE (safe after prune) — iteration {n_iterations}, "
                    f"samples {n_samples}"
                )
                _save_plot(
                    fig,
                    (ax_boxes, ax_ucb, ax_lip, ax_props, ax_2d),
                    plot_dir,
                    n_iterations,
                    history_iters,
                    history_n_boxes,
                    history_total_boxes,
                    history_max_ucb,
                    history_mean_ucb,
                    history_max_lcb,
                    history_verified_prop,
                    history_unknown_prop,
                    history_unsafe_prop,
                    history_lip_lb,
                    reward_lipschitz,
                    state,
                    n_dims,
                )
            return True, None, n_iterations, n_samples

        # ----------------------------------------------------------------
        # Split boxes that sampling cannot decide: (a) the width-driven disc
        # term alone pins their UCB above zero, or (b) they have never captured
        # a single valid (domain\eq) sample so no estimate exists at all.
        # ----------------------------------------------------------------

        still_live = state.is_live[selected_indices]
        # Force-split ONLY boxes with no drift estimate at all (zero valid
        # samples EVER, i.e. no_data): with no estimate they can neither verify
        # nor flag, so splitting toward prunable/estimable children is the only
        # move. Boxes that DO have an estimate are left to accumulate valid
        # samples — their stat term keeps their UCB high, so the bandit keeps
        # selecting them. (The previous rule force-split any box with
        # < sample_per_select//4 valid draws THIS round; on a curved eq rim the
        # valid sliver fraction shrinks with the width, so rim-pinned boxes
        # re-triggered it every generation — an infinite split cascade to
        # float32 dust at the rim, measured as ~8 retired/iter with ~40 live
        # boxes for 20k+ iterations on doublewell_mab while the whole rest of
        # the domain was verified.)
        mostly_eq = no_data
        # Split only boxes that sampling alone CANNOT decide: once stat has been
        # driven below disc, a box with mean + disc <= 0 will verify with more
        # samples (its UCB -> mean + disc as stat -> 0) — splitting it instead
        # quadruples the tiling around the entire margin surface (measured ~4-5x
        # total boxes on doublewell_mab). mean + disc > 0 means the width-driven
        # disc term pins the UCB positive regardless of sample count, so only
        # then is a split the way forward. Efficiency-only: verdicts still come
        # exclusively from the sound UCB/LCB tests above.
        undecidable = (mean_sel + disc_err_np) > 0.0
        # Endgame stall-breaker: a box with mean + disc <= 0 verifies by sampling
        # alone, but at the stat ~ 1/sqrt(n) rate — when mean + disc ~ 0^- that
        # wait is unbounded (measured on linstoch2D: ONE live box holding 15M
        # samples, UCB crawling +1e-3 -> +1e-4 over 245k iterations while the
        # whole rest of the domain was verified). Once stat has been driven
        # below disc/4, sampling has had a fair chance: if the UCB is STILL
        # positive, split — halving disc makes both children decisively
        # negative (or isolates a genuine violation), and the replayed parent
        # samples mean each child starts half-resolved. Boxes with
        # |mean + disc| > disc/4 verify by sampling before this gate opens, so
        # the doublewell anti-cascade property of `undecidable` is preserved.
        stalled = (stat_err_np < 0.25 * disc_err_np) & (new_ucbs_np > 0.0)
        should_split = (((stat_err_np < disc_err_np) & undecidable)
                        | stalled | mostly_eq) & still_live

        n_splits = int(should_split.sum())
        if n_splits > 0:
            split_sel = selected_indices[should_split]
            split_dims, split_vals = _compute_split_params_batch(
                state.bounds[split_sel]
            )

            if printing and n_iterations % print_interval == 0:
                print(f"[Verify]   └─ splitting {n_splits} box(es)")

            new_slots = []
            for parent_idx, sdim, sval in zip(split_sel, split_dims, split_vals):
                lo_slot = _freelist_alloc(free_stack)
                hi_slot = _freelist_alloc(free_stack)

                _split_box(
                    state, int(parent_idx), int(sdim), float(sval), lo_slot, hi_slot
                )

                state.deactivate(int(parent_idx))
                _freelist_release(free_stack, int(parent_idx))
                new_slots.append(lo_slot)
                new_slots.append(hi_slot)

            # Validity-check every child created this iteration in ONE call: the
            # checker has fixed per-call overhead, so a single batch of 2*n_splits
            # is far cheaper than 2-box calls per parent.
            _prune_invalid(
                state, np.array(new_slots, dtype=np.int32), checker, free_stack
            )

        n_iterations += 1


# =============================================================================
# Plot helper — saves to disk (no display required)
# =============================================================================

# Palette
_COL_VERIFIED = "#2a9d8f"  # teal-green  — proven safe
_COL_UNKNOWN = "#e9c46a"  # amber       — unresolved
_COL_UNSAFE = "#e76f51"  # coral-red   — LCB > 0
_COL_INF_UCB = "#adb5bd"  # grey        — never sampled (inf UCB)
_COL_LIP_LB = "#4361ee"  # indigo      — empirical Lip LB
_COL_LIP_UB = "#e63946"  # crimson     — rigorous Lip UB


def _ffill(arr):
    """Forward-fill NaN values for continuous line plots."""
    out = np.asarray(arr, dtype=float)
    mask = np.isnan(out)
    if not mask.any() or mask.all():
        return out
    idx = np.where(~mask, np.arange(len(out)), 0)
    np.maximum.accumulate(idx, out=idx)
    out[mask] = out[idx[mask]]
    return out


def _save_plot(
    fig,
    axes,
    plot_dir,
    n_iterations,
    iters,
    n_boxes,
    total_boxes,
    max_ucb,
    mean_ucb,
    max_lcb,
    verified_prop,
    unknown_prop,
    unsafe_prop,
    lip_lb,
    reward_lipschitz,
    state,
    n_dims,
):
    import os

    fig.suptitle(
        f"MAB Verification – Progress for Iteration {n_iterations}",
        fontsize=13,
        fontweight="bold",
    )
    ax_boxes, ax_ucb, ax_lip, ax_props, ax_2d = axes
    it = np.asarray(iters)

    # ------------------------------------------------------------------ #
    # 1. Live boxes + cumulative total                                    #
    # ------------------------------------------------------------------ #
    ax_boxes.cla()
    ax_boxes.plot(it, n_boxes, color="steelblue", lw=1.8, label="Live boxes")
    ax_boxes.plot(
        it,
        total_boxes,
        color="#adb5bd",
        lw=1.2,
        linestyle="--",
        label="Total ever created",
    )
    ax_boxes.set_xlabel("Iteration", fontsize=9)
    ax_boxes.set_ylabel("Box count", fontsize=9)
    ax_boxes.set_title("Active Boxes", fontsize=10, fontweight="bold")
    ax_boxes.legend(fontsize=8)
    ax_boxes.grid(True, alpha=0.35)
    # Annotate latest total
    if len(total_boxes):
        ax_boxes.annotate(
            f"Total: {total_boxes[-1]:,}",
            xy=(it[-1], total_boxes[-1]),
            xytext=(0, 6),
            textcoords="offset points",
            fontsize=8,
            color="#555",
        )

    # ------------------------------------------------------------------ #
    # 2. UCB / LCB over iterations (finite values only, forward-filled)  #
    # ------------------------------------------------------------------ #
    ax_ucb.cla()
    ff_max_ucb = _ffill(max_ucb)
    ff_mean_ucb = _ffill(mean_ucb)
    ff_max_lcb = _ffill(max_lcb)
    ax_ucb.plot(it, ff_max_ucb, color=_COL_UNSAFE, lw=1.8, label="Max UCB")
    ax_ucb.plot(
        it, ff_mean_ucb, color="#f4a261", lw=1.2, linestyle="--", label="Mean UCB"
    )
    ax_ucb.plot(it, ff_max_lcb, color=_COL_LIP_LB, lw=1.8, label="Max LCB")
    ax_ucb.axhline(0, color="black", lw=0.9, linestyle=":", label="Desicion threshold")
    ax_ucb.set_xlabel("Iteration", fontsize=9)
    ax_ucb.set_ylabel("Bound value", fontsize=9)
    ax_ucb.set_title(
        "Confidence Bounds",
        fontsize=10,
        fontweight="bold",
    )
    ax_ucb.legend(fontsize=8)
    ax_ucb.grid(True, alpha=0.35)

    # ------------------------------------------------------------------ #
    # 3. Empirical Lipschitz lower bound vs rigorous upper bound         #
    # ------------------------------------------------------------------ #
    ax_lip.cla()
    ax_lip.plot(
        it, lip_lb, color=_COL_LIP_LB, lw=1.8, label="Empirical LB (running max)"
    )
    ax_lip.axhline(
        reward_lipschitz,
        color=_COL_LIP_UB,
        lw=1.5,
        linestyle="--",
        label=f"Rigorous UB = {reward_lipschitz:.4g}",
    )
    ax_lip.set_xlabel("Iteration", fontsize=9)
    ax_lip.set_ylabel("Lipschitz estimate", fontsize=9)
    ax_lip.set_title(
        "Lipschitz Constant\n(LB saturates toward true value)",
        fontsize=10,
        fontweight="bold",
    )
    ax_lip.legend(fontsize=8)
    ax_lip.grid(True, alpha=0.35)
    # Shade the gap between LB and UB
    ax_lip.fill_between(
        it,
        lip_lb,
        reward_lipschitz,
        color=_COL_LIP_LB,
        alpha=0.08,
        label="_nolegend_",
    )

    # ------------------------------------------------------------------ #
    # 4. Box-state proportions: verified / unknown / unsafe              #
    # ------------------------------------------------------------------ #
    ax_props.cla()
    vp = np.asarray(verified_prop)
    up = np.asarray(unknown_prop)
    sp = np.asarray(unsafe_prop)
    ax_props.stackplot(
        it,
        vp,
        up,
        sp,
        labels=["Verified (UCB ≤ 0)", "Unknown", "Unsafe (LCB > 0)"],
        colors=[_COL_VERIFIED, _COL_UNKNOWN, _COL_UNSAFE],
        alpha=0.85,
    )
    ax_props.set_xlabel("Iteration", fontsize=9)
    ax_props.set_ylabel("Proportion", fontsize=9)
    ax_props.set_title("Box States", fontsize=10, fontweight="bold")
    ax_props.set_ylim(0, 1)
    ax_props.legend(fontsize=8, loc="upper right")
    ax_props.grid(True, alpha=0.35)

    # ------------------------------------------------------------------ #
    # 5. 2-D state space (only when n_dims == 2)                         #
    # ------------------------------------------------------------------ #
    if ax_2d is not None and n_dims == 2:
        _draw_2d_state(ax_2d, state, n_iterations)

    fig.savefig(
        os.path.join(plot_dir, f"mab_verify_iter{n_iterations:07d}.png"),
        dpi=110,
        bbox_inches="tight",
    )
    print(
        f"[Verify]   └─ plot saved → "
        f"{os.path.join(plot_dir, f'mab_verify_iter{n_iterations:07d}.png')}"
    )


# Maximum boxes to render in the 2-D state plot; randomly subsampled if exceeded.
_MAX_2D_RENDER = 60_000


def _rects(bounds):
    """List of matplotlib Rectangles from a (N, 2, 2) [box, dim, lo/hi] array."""
    return [
        Rectangle(
            (bounds[i, 0, 0], bounds[i, 1, 0]),
            bounds[i, 0, 1] - bounds[i, 0, 0],
            bounds[i, 1, 1] - bounds[i, 1, 0],
        )
        for i in range(len(bounds))
    ]


def _draw_2d_state(ax, state, n_iterations):
    """Render the 2-D state-space tiling.

    Verified cells are the union of those already retired as safe (the bulk of a
    finished tiling) and any live cell whose UCB has just crossed <= 0; the rare
    unknown / unsafe / unsampled live cells are drawn on top and never subsampled
    away, so a nearly-complete verification still reads as a domain full of green.
    """
    ax.cla()

    n_dims = state.n_dims
    retired = (
        np.concatenate(state.retired_safe, axis=0)
        if state.retired_safe
        else np.empty((0, n_dims, 2))
    )

    live = state.live_indices()
    live_bounds = state.bounds[live]
    ucbs = state.ucb[live]
    lcbs = state.lcb[live]
    v_live = ucbs <= 0.0
    unsafe = lcbs > 0.0
    inf_ucb = np.isinf(ucbs) & ~v_live & ~unsafe
    unknown = ~v_live & ~unsafe & ~inf_ucb

    # true (un-subsampled) counts for the legend
    n_verified = len(retired) + int(v_live.sum())
    n_unknown, n_unsafe, n_unsamp = int(unknown.sum()), int(unsafe.sum()), int(inf_ucb.sum())
    grand = n_verified + n_unknown + n_unsafe + n_unsamp

    if grand == 0:
        ax.text(0.5, 0.5, "No cells yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=11)
        ax.set_title("State-Space Tiling", fontsize=10, fontweight="bold")
        return

    green = mcolors.to_rgba(_COL_VERIFIED, alpha=0.75)
    red = mcolors.to_rgba(_COL_UNSAFE, alpha=0.80)
    grey = mcolors.to_rgba(_COL_INF_UCB, alpha=0.50)
    cmap_unk = plt.get_cmap("YlOrBr")

    # Keep every "interesting" (non-verified) cell; spend the rest of the render
    # budget on the verified bulk, which is what gets subsampled.
    verified_bounds = np.concatenate([retired, live_bounds[v_live]], axis=0)
    v_budget = max(_MAX_2D_RENDER - (n_unknown + n_unsafe + n_unsamp), 1000)
    if len(verified_bounds) > v_budget:
        sel = np.random.default_rng(0).choice(len(verified_bounds), v_budget, replace=False)
        verified_bounds = verified_bounds[sel]

    bounds_parts = [verified_bounds]
    rgba_parts = [np.tile(green, (len(verified_bounds), 1))]

    if n_unknown:
        uv = ucbs[unknown]
        lo_v, hi_v = uv.min(), uv.max()
        span = hi_v - lo_v if hi_v > lo_v else 1.0
        bounds_parts.append(live_bounds[unknown])
        rgba_parts.append(cmap_unk(0.25 + 0.6 * (uv - lo_v) / span))
    if n_unsamp:
        bounds_parts.append(live_bounds[inf_ucb])
        rgba_parts.append(np.tile(grey, (n_unsamp, 1)))
    if n_unsafe:
        bounds_parts.append(live_bounds[unsafe])
        rgba_parts.append(np.tile(red, (n_unsafe, 1)))

    all_bounds = np.concatenate(bounds_parts, axis=0)
    all_rgba = np.concatenate(rgba_parts, axis=0)

    pc = PatchCollection(_rects(all_bounds), facecolor=all_rgba,
                         edgecolor="white", linewidth=0.05)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("x₀", fontsize=9)
    ax.set_ylabel("x₁", fontsize=9)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=_COL_VERIFIED, alpha=0.75, label=f"Verified  ({n_verified:,})"),
        Patch(facecolor=_COL_UNKNOWN, alpha=0.85, label=f"Unknown  ({n_unknown:,})"),
        Patch(facecolor=_COL_INF_UCB, alpha=0.55, label=f"Unsampled ({n_unsamp:,})"),
        Patch(facecolor=_COL_UNSAFE, alpha=0.80, label=f"Unsafe   ({n_unsafe:,})"),
    ]
    ax.legend(handles=legend_elements, fontsize=7.5, loc="upper right", framealpha=0.85)

    shown = len(all_bounds)
    sub_note = f" (showing {shown:,} of {grand:,} cells)" if shown < grand else ""
    ax.set_title(f"State-Space Tiling{sub_note}", fontsize=10, fontweight="bold")
