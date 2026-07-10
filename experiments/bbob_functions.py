"""
BBOB test functions (JAX, batched, NEGATED so refuters MAXIMISE them:
the global maximum is f_opt = 0 at x_opt, everywhere else f < 0).

All five are core formulas of BBOB/COCO noiseless-suite functions (Hansen et
al.; we apply a fixed optimum shift but not the suite's random rotations /
asymmetry transforms), chosen to span the landscape axes that separate
refuter strategies:

  rastrigin   (BBOB f3/f15) — regular grid of local optima:
              punishes pure exploitation (gradient gets trapped).
  rosenbrock  (BBOB f8)     — narrow curved valley, ill-conditioned:
              punishes undirected search (random/grid crawl the valley).
  schaffers_f7(BBOB f17)    — irregular ripple multimodality at all radii:
              needs exploration AND fine exploitation.
  schwefel    (BBOB f20)    — deceptive: the best basins sit near the domain
              BOUNDARY, far from the centre: punishes centre-biased
              initialisation and premature convergence.
  gallagher   (BBOB f21)    — 101 random Gaussian peaks, ONE optimal and
              narrow: needle-among-distractors, punishes gradient methods
              (they climb the nearest distractor) and rewards coverage.

Each entry exposes f(xs) for xs of shape (B, d), the global maximum value
`f_opt` (= 0), the optimum `x_opt`, and a search `domain`.
"""

from dataclasses import dataclass
from typing import Callable
import jax.numpy as jnp


@dataclass
class TestFunction:
    name: str
    f: Callable  # (B, d) -> (B,)   minimised
    domain: jnp.ndarray  # (d, 2)
    x_opt: jnp.ndarray
    f_opt: float = 0.0


def _box(dim, lo, hi):
    return jnp.stack([jnp.full((dim,), lo), jnp.full((dim,), hi)], axis=1)


def make_suite(dim: int = 2):
    shift_a = jnp.array([1.3, -0.8] + [0.0] * (dim - 2))[:dim]
    shift_b = jnp.array([-1.5, 0.7] + [0.0] * (dim - 2))[:dim]

    def rastrigin(xs):
        z = xs - shift_a
        return -(10.0 * dim + jnp.sum(z**2 - 10.0 * jnp.cos(2.0 * jnp.pi * z), axis=1))

    def rosenbrock(xs):
        # classic (unshifted): optimum at ALL-ONES — x_opt must record that,
        # not a shift vector (the old entry stored shift_a: any distance-to-
        # optimum metric computed from it was wrong).
        z = xs
        return -jnp.sum(
            100.0 * (z[:, 1:] - z[:, :-1] ** 2) ** 2 + (z[:, :-1] - 1.0) ** 2, axis=1
        )

    def schwefel(xs):
        # classic Schwefel: optimum at 420.9687·1, near the +boundary of
        # [-500, 500]; the second-best basin is FAR away (deceptive).
        z = xs
        return -(418.9828872724339 * dim
                 - jnp.sum(z * jnp.sin(jnp.sqrt(jnp.abs(z))), axis=1))

    # BBOB f21 core (Gallagher's Gaussian 101-me peaks): value = the highest of
    # 101 Gaussian peaks; peak 0 has height 10 and is the NARROWEST, the 100
    # distractors have heights 1.1..9.1 and random widths. Centres/widths are
    # generated once with a fixed seed -> deterministic across processes.
    import numpy as _np

    _rng = _np.random.default_rng(21)
    _n_peaks = 101
    _centers = jnp.asarray(_rng.uniform(-4.0, 4.0, size=(_n_peaks, dim)))
    _heights = jnp.asarray(_np.concatenate([[10.0],
                                            _np.linspace(1.1, 9.1, _n_peaks - 1)]))
    # quadratic sharpness per peak: the GLOBAL peak is the sharpest (narrow
    # needle, basin ~1% of the domain); distractors are broad.
    _sharp = jnp.asarray(_np.concatenate([[8.0],
                                          _rng.uniform(0.5, 2.0, _n_peaks - 1)]))

    def gallagher(xs):
        d2 = jnp.sum((xs[:, None, :] - _centers[None, :, :]) ** 2, axis=2)  # (B, K)
        peaks = _heights[None, :] * jnp.exp(-_sharp[None, :] * d2 / (2.0 * dim))
        return jnp.max(peaks, axis=1) - 10.0

    def schaffers_f7(xs):
        # BBOB f17 core: s_i = sqrt(x_i^2 + x_{i+1}^2);
        # f = [mean_i( sqrt(s_i) * (1 + sin^2(50 s_i^{1/5})) )]^2, opt 0.
        z = xs - shift_b
        s = jnp.sqrt(z[:, :-1] ** 2 + z[:, 1:] ** 2)
        term = jnp.sqrt(s) * (1.0 + jnp.sin(50.0 * s ** 0.2) ** 2)
        return -(jnp.mean(term, axis=1) ** 2)

    return [
        TestFunction("rastrigin", rastrigin, _box(dim, -5.0, 5.0), shift_a),
        TestFunction("rosenbrock", rosenbrock, _box(dim, -5.0, 5.0), jnp.ones(dim)),
        TestFunction("schaffers_f7", schaffers_f7, _box(dim, -5.0, 5.0), shift_b),
        TestFunction("schwefel", schwefel, _box(dim, -500.0, 500.0),
                     jnp.full((dim,), 420.9687462275036)),
        TestFunction("gallagher", gallagher, _box(dim, -5.0, 5.0), _centers[0]),
    ]
