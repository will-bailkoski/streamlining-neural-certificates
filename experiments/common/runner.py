"""
Shared helpers for the per-combo runner executables (experiments/runners/*).

A runner declares its core identity flags with argparse and lets every other
`--flag value` flow through as a component hyperparameter. `parse_extras` routes
those leftovers: bare flags are engine hyperparameters, `--r_<key>` flags are
refuter hyperparameters (so an engine and a refuter that share a name — e.g.
`batch_size` — never collide on the command line).
"""

from __future__ import annotations


def coerce(s: str):
    """Best-effort scalar type from a CLI string: bool -> int -> float -> str."""
    if isinstance(s, bool):
        return s
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def parse_extras(extras: list[str], refuter_prefix: str = "r_") -> tuple[dict, dict]:
    """argparse leftovers -> (engine_hp, refuter_hp).

    `['--noise_disc', '1', '--r_a_max', '2.0']` -> ({'noise_disc': 1}, {'a_max': 2.0}).
    A flag with no following value is a bare bool flag (True).
    """
    engine_hp: dict = {}
    refuter_hp: dict = {}
    i = 0
    while i < len(extras):
        tok = extras[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key = tok[2:]
        if i + 1 < len(extras) and not extras[i + 1].startswith("--"):
            val = coerce(extras[i + 1])
            i += 2
        else:
            val = True
            i += 1
        if key.startswith(refuter_prefix):
            refuter_hp[key[len(refuter_prefix):]] = val
        else:
            engine_hp[key] = val
    return engine_hp, refuter_hp


def expand_cex(violation, history_len, key, env, region=None, ball_radius=0.05):
    """Counterexamples fed to the CEGIS trainer per round (fairness control).

    Returns an (m, d) array with 1 <= m <= history_len of VALID points — every
    point is rejection-filtered through env.is_valid_domain (domain minus
    equilibrium). The filtering is load-bearing: the drift condition is only
    required on domain\\eq, and inside the equilibrium it is UNSATISFIABLE for a
    bounded certificate (the eq covers the invariant noise floor, so V has a
    floor there). The box engines (mab / lirpa) report boxes that hug the eq
    boundary — exactly where the margin is tightest — and the box CENTRE itself
    can sit inside eq (or, for ellipsoid domains, outside the true domain, since
    `region` is only clipped to the bounding box). One such point in the
    persistent CEGIS buffer pins the training loss > 0 forever.

    Extra points (history_len - 1 of them) are drawn uniformly:
      * within the offending box `region` (∩ domain) when the engine localised
        one (the box engines mab / lirpa), or
      * within a ball of `ball_radius` around the point (∩ domain) for the point
        engines and refuters.
    This equalises the counterexample yield per round across methods, so the
    refuter-first vs verifier-only comparison is sample-fair.
    """
    import numpy as np
    import jax
    import jax.numpy as jnp
    import jax.random as jrn

    base = np.asarray(violation, dtype=float).reshape(-1)
    d = base.shape[0]
    dom = np.asarray(env.domain.bounds, dtype=float)
    if region is not None:
        box = np.asarray(region, dtype=float)
        lo = np.maximum(box[:, 0], dom[:, 0])
        hi = np.minimum(box[:, 1], dom[:, 1])
    else:
        lo = np.maximum(base - ball_radius, dom[:, 0])
        hi = np.minimum(base + ball_radius, dom[:, 1])

    is_valid = jax.vmap(env.is_valid_domain)
    base_valid = bool(np.asarray(is_valid(jnp.asarray(base[None, :])))[0])
    if history_len <= 1 and base_valid:
        return base[None, :]

    # oversample the region uniformly, keep only domain\eq points
    budget = max(512, 64 * history_len)
    u = np.asarray(jrn.uniform(key, (budget, d)))
    cands = lo + u * (hi - lo)
    valid = cands[np.asarray(is_valid(jnp.asarray(cands)))]

    need = history_len - (1 if base_valid else 0)
    out = np.vstack(([base[None, :]] if base_valid else [np.zeros((0, d))])
                    + [valid[:need]])
    if out.shape[0] == 0:
        # the region's valid sliver was too thin to hit — fall back to the raw
        # point so the loop still makes progress (old behaviour, worst case)
        return base[None, :]
    return out
