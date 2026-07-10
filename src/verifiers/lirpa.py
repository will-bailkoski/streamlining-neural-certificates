"""
autoLiRPA verification engine (bound propagation + branch and bound).

Builds a torch module for the drift d(x) = E_w[V(f(x,w))] - V(x) + eps via
`env.lirpa_drift_module(V, eps, noise_disc)`, then bounds d over boxes covering
the domain:

    upper bound <= 0 on every box       -> verified (sound)
    lower bound  > 0 on some valid box  -> counterexample (re-checked by caller)
    budget exhausted                    -> inconclusive

Continuous noise is handled exactly like the symbolic engines: it is binned into
`noise_disc` cells per dimension, and each cell is a noise box that the bound
propagation maximises over (a sound over-approximation of E[V]). Larger
`noise_disc` => tighter over-approximation. Discrete/switched dynamics carry no
noise input (`noise_box is None`).

Validity geometry (shared with the mab engine's sound checker): boxes provably
OUTSIDE the domain or provably INSIDE the equilibrium are discarded — essential
for the ellipsoid-domain envs (vanderpol, pendulum_lqr), whose bounding-box
tiling otherwise contains out-of-domain boxes where the drift really is
positive, so the engine "proves" violations at points the certificate is not
required to cover (this spuriously refuted every valid cert on those envs).
A counterexample is only reported from a violating box that contains a VALID
witness point: drift_lo > 0 holds for every point of the box, so any
domain\\eq point inside it is a genuine violator; a violating box with no
valid witness is left to split — its provably-invalid children get pruned.
"""

from __future__ import annotations
import time
import numpy as np

from src.verifiers.base import VerifierResult, register
from src.certificates.encoders import build_torch
from src.verifiers.mab import _make_checker


def _split_all_dims(los, his):
    """Split every box along ALL dimensions -> 2**d children per box.

    Aggressive: halves every side at once. Fast refinement in low dimension,
    but the box count grows by 2**d per level, so it explodes as d rises.
    """
    N, d = los.shape
    mids = (los + his) / 2
    masks = ((np.arange(2**d)[:, None] >> np.arange(d)[None, :]) & 1).astype(bool)
    mask = masks[:, None, :]
    lo_c = np.where(mask, mids[None], los[None])
    hi_c = np.where(mask, his[None], mids[None])
    new_los = lo_c.transpose(1, 0, 2).reshape(N * 2**d, d)
    new_his = hi_c.transpose(1, 0, 2).reshape(N * 2**d, d)
    return new_los, new_his


def _split_longest_dim(los, his):
    """Bisect each box along its WIDEST dimension -> 2 children per box.

    Conservative: the box count only doubles per level regardless of d, so it
    scales to higher dimension at the cost of slower per-level refinement.
    """
    los, his = np.asarray(los), np.asarray(his)
    n = los.shape[0]
    j = np.argmax(his - los, axis=1)          # widest side of each box
    mids = (los + his) / 2
    rows = np.arange(n)
    lo_lo, hi_lo = los.copy(), his.copy()     # lower half along j
    lo_hi, hi_hi = los.copy(), his.copy()     # upper half along j
    hi_lo[rows, j] = mids[rows, j]
    lo_hi[rows, j] = mids[rows, j]
    return np.concatenate([lo_lo, lo_hi]), np.concatenate([hi_lo, hi_hi])


_SPLITTERS = {"all": _split_all_dims, "longest": _split_longest_dim}


def _auto_device(torch) -> str:
    """Use cuda only if a *compatible* GPU is present, else cpu.

    `torch.cuda.is_available()` can be True for a GPU whose compute capability
    this torch build wasn't compiled for (e.g. a GTX 1080 Ti, sm_61, on a torch
    that ships sm_70+). Running a kernel there crashes with "no kernel image is
    available", so check the device arch against the build's arch list first.
    """
    if not torch.cuda.is_available():
        return "cpu"
    try:
        major, minor = torch.cuda.get_device_capability()
        if f"sm_{major}{minor}" in torch.cuda.get_arch_list():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def find_counterexample(
    env,
    spec,
    params,
    epsilon: float,
    key=None,
    bound_method: str = "CROWN",
    noise_disc: int = 1,
    split: str = "all",          # "all" -> split every dim, "longest" -> bisect widest
    device: str | None = None,   # None -> cuda if available, else cpu
    max_depth: int = 18,
    max_boxes: int = 200_000,
    timeout: float | None = None,  # wall-clock budget (s); None = run to max_depth.
    batch_size: int = 2048,
    record: bool = False,   # capture per-depth box exploration (for visualisation)
    **hp,
):
    import torch
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

    if device is None:
        device = _auto_device(torch)
    splitter = _SPLITTERS[split]

    V_module = build_torch(spec, params)
    module, noise_box = env.lirpa_drift_module(V_module, epsilon, noise_disc=noise_disc)
    module.eval()

    dim = env.dim
    if noise_box is None:
        bounded = BoundedModule(module, torch.empty((1, dim)), device=device)
        w_lo = w_hi = None
    else:
        w_lo, w_hi = np.asarray(noise_box[0], float), np.asarray(noise_box[1], float)
        bounded = BoundedModule(
            module, (torch.empty((1, dim)), torch.empty((1, w_lo.size))), device=device
        )

    def bounds_batch(los, his):
        x_L = torch.tensor(los, dtype=torch.float32, device=device)
        x_U = torch.tensor(his, dtype=torch.float32, device=device)
        x_bt = BoundedTensor((x_L + x_U) / 2, PerturbationLpNorm(x_L=x_L, x_U=x_U))
        if noise_box is None:
            lb, ub = bounded.compute_bounds(x=(x_bt,), method=bound_method)
        else:
            n = len(los)
            wl = torch.tensor(np.tile(w_lo, (n, 1)), dtype=torch.float32, device=device)
            wu = torch.tensor(np.tile(w_hi, (n, 1)), dtype=torch.float32, device=device)
            w_bt = BoundedTensor((wl + wu) / 2, PerturbationLpNorm(x_L=wl, x_U=wu))
            lb, ub = bounded.compute_bounds(x=(x_bt, w_bt), method=bound_method)
        return lb.detach().cpu().numpy().reshape(-1), ub.detach().cpu().numpy().reshape(-1)

    # Sound validity pruning + witness test (same geometry the mab engine uses).
    import jax
    import jax.numpy as jnp

    checker = _make_checker(env)
    _valid_mask = jax.jit(jax.vmap(env.is_valid_domain))
    _witness_rng = np.random.default_rng(0)

    def _valid_witness(lo, hi, tries=1024):
        """A domain\\eq point inside [lo, hi], or None if sampling finds none."""
        pts = lo + _witness_rng.random((tries, dim)) * (hi - lo)
        mask = np.asarray(_valid_mask(jnp.asarray(pts)))
        return pts[mask][0] if mask.any() else None

    los = np.asarray(env.domain.bounds[:, 0], dtype=float)[None].copy()
    his = np.asarray(env.domain.bounds[:, 1], dtype=float)[None].copy()

    total_boxes = 0      # cumulative boxes bounded (total work)
    peak_boxes = 0       # max boxes held at once (memory pressure)
    n_bound_calls = 0    # compute_bounds invocations (batches)
    frames: list = []    # per-depth box exploration (only when record=True)
    trace: list = []     # (cumulative_boxes, elapsed_s, resid_bound) per depth -> anytime curve
    _trace_on = bool(hp.get("trace", False))
    t0 = time.perf_counter()

    def _stats(**extra):
        elapsed = time.perf_counter() - t0
        s = {
            "time": round(elapsed, 4),
            "method": bound_method,
            "split": split,
            "device": device,
            "noise_disc": noise_disc,
            "total_boxes": total_boxes,
            "peak_boxes": peak_boxes,
            "n_bound_calls": n_bound_calls,
            "boxes_per_sec": round(total_boxes / elapsed, 1) if elapsed > 0 else 0.0,
            **extra,
        }
        if record:
            s["frames"] = frames
        if trace:
            s["trace"] = trace
        return s

    resid_bound = None  # tightest still-undecided drift UPPER bound -> "how close to verified"
    for depth in range(max_depth):
        keep = checker(np.stack([los, his], axis=2))
        los, his = los[keep], his[keep]
        if len(los) == 0:
            return VerifierResult(verified=True, stats=_stats(depth=depth), engine="lirpa")
        if len(los) > max_boxes:
            return VerifierResult(verified=None,
                                  stats=_stats(depth=depth, hit="max_boxes", resid_bound=resid_bound),
                                  engine="lirpa")

        n = len(los)
        total_boxes += n
        peak_boxes = max(peak_boxes, n)
        dl = np.empty(n)
        du = np.empty(n)
        for i in range(0, n, batch_size):
            # Between-batch wall-clock check: on SLURM the alternative is the wall
            # killing the job mid-call, which leaves the run status=running with no
            # stats at all — time out INSIDE the budget so the verdict is recorded.
            if timeout is not None and time.perf_counter() - t0 > timeout:
                return VerifierResult(
                    verified=None,
                    stats=_stats(depth=depth, hit="timeout", resid_bound=resid_bound),
                    engine="lirpa",
                )
            n_bound_calls += 1
            bl, bu = bounds_batch(los[i : i + batch_size], his[i : i + batch_size])
            dl[i : i + len(bl)] = bl
            du[i : i + len(bu)] = bu

        if record:
            # 2 = counterexample (lb>0), 1 = verified (ub<=0), 0 = undecided (split)
            status = np.where(dl > 0.0, 2, np.where(du <= 0.0, 1, 0))
            frames.append({"los": los.copy(), "his": his.copy(), "status": status.copy()})

        pos = dl > 0.0
        if np.any(pos):
            # dl > 0 proves EVERY point of the box violates, so any valid point
            # inside is a genuine witness. A violating box with no valid point
            # (e.g. fully outside an ellipsoid domain but kept by the
            # conservative checker) must NOT be reported — leave it to split.
            for j in np.argsort(-dl):
                if dl[j] <= 0.0:
                    break
                witness = _valid_witness(los[j], his[j])
                if witness is not None:
                    region = np.stack([los[j], his[j]], axis=1)  # (d, 2)
                    return VerifierResult(
                        verified=False, violation=witness[None, :], region=region,
                        stats=_stats(depth=depth, drift_lo=float(dl[j])), engine="lirpa",
                    )

        undecided = du > 0.0
        resid_bound = float(du.max())     # worst undecided drift upper bound this round
        if _trace_on:
            trace.append((total_boxes, round(time.perf_counter() - t0, 4), resid_bound))
        if not np.any(undecided):
            return VerifierResult(verified=True, stats=_stats(depth=depth), engine="lirpa")

        los, his = splitter(los[undecided], his[undecided])
        los = np.maximum(los, env.domain.bounds[:, 0])
        his = np.minimum(his, env.domain.bounds[:, 1])

    return VerifierResult(verified=None,
                          stats=_stats(depth=max_depth, hit="max_depth", resid_bound=resid_bound),
                          engine="lirpa")


# Each auto_LiRPA bound method is exposed as its OWN engine, so `--engine crown`,
# `--engine ibp`, etc. select the bound directly (the engine *is* the method).
# Three rungs of the tightness/cost ladder are kept: IBP (loosest, cheapest) ->
# CROWN (tight, moderate) -> alpha-CROWN (tightest, optimised, expensive).
LIRPA_METHODS = (
    "ibp",
    "crown",
    "alpha-crown",
)


class _LiRPAEngine:
    """autoLiRPA drift verifier pinned to a single bound method (its engine name)."""

    def __init__(self, method: str):
        self.name = method
        self._method = method

    def find_counterexample(self, env, spec, params, epsilon, **hp):
        hp["bound_method"] = self._method  # the engine name fixes the bound method
        return find_counterexample(env, spec, params, epsilon, **hp)


for _m in LIRPA_METHODS:
    register(_m, _LiRPAEngine(_m))

# Backwards-compatible alias: the old generic name defaults to CROWN.
register("lirpa", _LiRPAEngine("crown"))
